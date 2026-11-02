"""Language models behind one small interface.

Claude runs through the official Anthropic SDK whenever it can be reached. A
local model runs through Ollama so a post with no internet still gets briefs.
The agents talk only to the interface below, so which model answered is a
deployment detail rather than something the agent code branches on.

Each conversation keeps its own transcript in its provider's native format
and only ever appends to it. Claude's replies are stored exactly as returned
(thinking blocks included), which is what the API requires for a valid
continuation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import httpx

logger = logging.getLogger("backend.agent.models")

T = TypeVar("T")

# Server-side refusal fallback: if Claude declines a request on policy
# grounds, the API retries it on the model Anthropic recommends instead of
# returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Haiku has no server-side fallback; sending the parameter would do nothing.
_NO_FALLBACK_PREFIXES = ("claude-haiku",)


class ModelUnavailable(RuntimeError):
    """The model cannot be reached right now: offline, no credentials,
    overloaded, or the local model is not installed. Callers move on to the
    next model rather than failing the brief."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], object]


@dataclass
class ToolCall:
    call_id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    # done | tool_use | refusal | max_tokens
    stop: str
    model: str


class Conversation(Protocol):
    def send(self, text: str) -> Turn: ...
    def send_tool_results(self, results: list[ToolResult]) -> Turn: ...


class ChatModel(Protocol):
    name: str
    # claude | cloud | local: shown to operators, so they know whether an
    # answer came from the cloud or from the machine at the post.
    kind: str
    def start(self, system: str, tools: list[ToolSpec]) -> Conversation: ...


# --- Claude -------------------------------------------------------------------

class ClaudeModel:
    kind = "claude"

    def __init__(self, model: str, effort: str = "medium", timeout_sec: float = 60.0, client=None):
        self.name = model
        self.effort = effort
        self.timeout_sec = timeout_sec
        self._client = client

    def client(self):
        if self._client is None:
            import anthropic

            # One retry, not the SDK's default two: when the post is offline
            # the local model should take over within seconds.
            client = anthropic.Anthropic(timeout=self.timeout_sec, max_retries=1)
            # With nothing configured the SDK only fails once a request is
            # built, and with a TypeError rather than an API error. Checking
            # here turns "no key on this machine" into an ordinary fallback.
            if not any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")):
                raise ModelUnavailable("no Claude credentials on this machine (set ANTHROPIC_API_KEY)")
            self._client = client
        return self._client

    def start(self, system: str, tools: list[ToolSpec]) -> "ClaudeConversation":
        return ClaudeConversation(self, system, tools)


class ClaudeConversation:
    def __init__(self, model: ClaudeModel, system: str, tools: list[ToolSpec]):
        self._model = model
        self._system = system
        # strict keeps every tool call schema-valid; each schema sets
        # additionalProperties: false and lists all its properties.
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
            for t in tools
        ]
        self._messages: list[dict] = []

    def send(self, text: str) -> Turn:
        self._messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        # All results for one turn go back together in a single message.
        self._messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]})
        return self._call()

    def _call(self) -> Turn:
        import anthropic

        kwargs = dict(
            model=self._model.name, max_tokens=16000, system=self._system,
            messages=self._messages, output_config={"effort": self._model.effort},
        )
        if self._tools:
            kwargs["tools"] = self._tools
        if not self._model.name.startswith(_NO_FALLBACK_PREFIXES):
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            response = self._model.client().beta.messages.create(**kwargs)
        except anthropic.BadRequestError:
            raise  # our request is wrong; falling back would hide the bug
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ModelUnavailable(f"Claude credentials rejected: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ModelUnavailable("Claude rate limit reached") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelUnavailable(f"Claude service error {exc.status_code}") from exc
            raise
        except anthropic.APIConnectionError as exc:
            raise ModelUnavailable("Claude unreachable (no network?)") from exc
        except anthropic.AnthropicError as exc:
            # Raised before any request when no credentials are configured.
            raise ModelUnavailable(f"Claude not configured: {exc}") from exc

        # Stored exactly as returned, thinking blocks and all.
        self._messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            return self._call()
        if response.stop_reason == "refusal":
            return Turn(text="", tool_calls=[], stop="refusal", model=response.model)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        if calls:
            stop = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop = "max_tokens"
        else:
            stop = "done"
        return Turn(text=text, tool_calls=calls, stop=stop, model=response.model)


# --- Local model through Ollama -------------------------------------------------

class OllamaModel:
    kind = "local"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 timeout_sec: float = 180.0, http: httpx.Client | None = None):
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._http = http or httpx.Client(timeout=timeout_sec)

    def start(self, system: str, tools: list[ToolSpec]) -> "OllamaConversation":
        return OllamaConversation(self, system, tools)


class OllamaConversation:
    def __init__(self, model: OllamaModel, system: str, tools: list[ToolSpec]):
        self._model = model
        self._tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in tools
        ]
        self._messages: list[dict] = [{"role": "system", "content": system}]
        self._call_count = 0

    def send(self, text: str) -> Turn:
        self._messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        for r in results:
            self._messages.append({"role": "tool", "tool_name": r.name, "content": r.content})
        return self._call()

    def _call(self) -> Turn:
        # A cap on reply length: on a laptop CPU every token costs time, and a
        # brief or an answer never needs more than a few sentences.
        payload = {"model": self._model.name, "messages": self._messages, "stream": False,
                   "options": {"num_predict": 400}}
        if self._tools:
            payload["tools"] = self._tools
        try:
            response = self._model._http.post(f"{self._model.base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"local model unreachable at {self._model.base_url} (is Ollama running?)") from exc
        if response.status_code == 404:
            raise ModelUnavailable(f"local model {self._model.name!r} not installed (run: ollama pull {self._model.name})")
        if response.status_code >= 400:
            raise ModelUnavailable(f"local model error {response.status_code}: {response.text[:200]}")

        message = response.json().get("message") or {}
        self._messages.append(message)
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                # Small models sometimes send arguments as a JSON string.
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            self._call_count += 1
            calls.append(ToolCall(f"local_{self._call_count}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        return Turn(
            text=(message.get("content") or "").strip(), tool_calls=calls,
            stop="tool_use" if calls else "done", model=self._model.name,
        )


# --- Hosted models with free tiers (Groq, Gemini, OpenRouter, ...) ----------------
#
# Several providers offer free, rate-limited access through the same widely
# used chat-completions request format, so one adapter covers them all.
# Model names on these services change often; the operator names one with
# DRISHTI_HOSTED_MODEL rather than this file guessing.

HOSTED_PROVIDERS = {
    # provider: (base URL, environment variable holding the key, default model)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", None),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", None),
    # OpenRouter's own router that picks among its free models.
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter/free"),
}


class HostedModel:
    kind = "cloud"

    def __init__(self, provider: str, model: str, base_url: str, api_key: str | None,
                 key_env: str, timeout_sec: float = 60.0, http: httpx.Client | None = None):
        self.provider = provider
        self.name = f"{provider}:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.key_env = key_env
        self._http = http or httpx.Client(timeout=timeout_sec)

    def start(self, system: str, tools: list[ToolSpec]) -> "HostedConversation":
        if not self.api_key:
            raise ModelUnavailable(f"no {self.provider} key on this machine (set {self.key_env})")
        return HostedConversation(self, system, tools)


class HostedConversation:
    def __init__(self, model: HostedModel, system: str, tools: list[ToolSpec]):
        self._model = model
        self._tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in tools
        ]
        self._messages: list[dict] = [{"role": "system", "content": system}]

    def send(self, text: str) -> Turn:
        self._messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        for r in results:
            self._messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
        return self._call()

    def _call(self) -> Turn:
        m = self._model
        payload: dict = {"model": m.model, "messages": self._messages}
        if self._tools:
            payload["tools"] = self._tools
            payload["tool_choice"] = "auto"
        try:
            response = m._http.post(f"{m.base_url}/chat/completions", json=payload,
                                    headers={"Authorization": f"Bearer {m.api_key}"})
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"{m.provider} unreachable (no network?)") from exc
        status = response.status_code
        if status in (401, 403):
            raise ModelUnavailable(f"{m.provider} rejected the key in {m.key_env}")
        if status == 429:
            raise ModelUnavailable(f"{m.provider} free-tier limit reached; try again later")
        if status >= 400:
            # A free model that cannot take tools, or one that was withdrawn,
            # answers 400/404. The next model in the chain should take over
            # rather than the brief failing on a service we do not control.
            raise ModelUnavailable(f"{m.provider} error {status}: {response.text[:200]}")

        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        kept = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            kept["tool_calls"] = raw_calls
        self._messages.append(kept)

        calls = []
        for raw in raw_calls:
            fn = raw.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append(ToolCall(raw.get("id") or f"call_{len(calls)}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        if calls:
            stop = "tool_use"
        elif choice.get("finish_reason") == "length":
            stop = "max_tokens"
        else:
            stop = "done"
        return Turn(text=(message.get("content") or "").strip(), tool_calls=calls, stop=stop, model=m.name)


def hosted_model_from_config(config) -> HostedModel | None:
    provider = (config.hosted_provider or "").strip().lower()
    if not provider:
        return None
    if provider == "custom":
        if not config.hosted_url:
            raise ValueError("DRISHTI_HOSTED_PROVIDER=custom needs DRISHTI_HOSTED_URL")
        base_url, key_env, default_model = config.hosted_url, "DRISHTI_HOSTED_API_KEY", None
    elif provider in HOSTED_PROVIDERS:
        base_url, key_env, default_model = HOSTED_PROVIDERS[provider]
    else:
        raise ValueError(f"unknown DRISHTI_HOSTED_PROVIDER {provider!r}: use {', '.join(HOSTED_PROVIDERS)} or custom")
    model = config.hosted_model or default_model
    if not model:
        raise ValueError(f"set DRISHTI_HOSTED_MODEL to a model {provider} currently offers")
    return HostedModel(provider, model, base_url, os.environ.get(key_env), key_env,
                       timeout_sec=config.cloud_timeout_sec)


# --- Choosing a model -------------------------------------------------------------

class ModelChain:
    """Try models in order, remembering which ones just failed.

    A model that raised ModelUnavailable is tried last for retry_after_sec,
    so an offline post goes straight to the local model instead of waiting
    on the cloud timeout for every incident. It is never skipped outright:
    if everything has failed recently, everything is tried again.
    """

    def __init__(self, models: list[ChatModel], retry_after_sec: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        if not models:
            raise ValueError("ModelChain needs at least one model")
        self.models = models
        self.retry_after_sec = retry_after_sec
        self._clock = clock
        self._failed_at: dict[str, float] = {}

    def _order(self) -> list[ChatModel]:
        now = self._clock()
        healthy = [m for m in self.models if now - self._failed_at.get(m.name, float("-inf")) >= self.retry_after_sec]
        resting = [m for m in self.models if m not in healthy]
        return healthy + resting

    def run(self, task: Callable[[ChatModel], T]) -> tuple[T, ChatModel]:
        problems = []
        for model in self._order():
            try:
                result = task(model)
            except ModelUnavailable as exc:
                logger.warning("model %s unavailable: %s", model.name, exc)
                self._failed_at[model.name] = self._clock()
                problems.append(f"{model.name}: {exc}")
                continue
            self._failed_at.pop(model.name, None)
            return result, model
        raise ModelUnavailable("; ".join(problems))


def build_chain(config) -> ModelChain:
    """Best model first, always ending with the local one.

    Claude, when this machine has credentials for it; then a hosted free-tier
    model, when one is configured; then the local model, which needs no key
    and no internet. A model without its key fails instantly and is skipped,
    so a post can run any subset of these.
    """
    models: list[ChatModel] = [
        ClaudeModel(config.cloud_model, effort=config.cloud_effort, timeout_sec=config.cloud_timeout_sec),
    ]
    hosted = hosted_model_from_config(config)
    if hosted is not None:
        models.append(hosted)
    models.append(OllamaModel(config.local_model, base_url=config.local_url, timeout_sec=config.local_timeout_sec))
    return ModelChain(models, retry_after_sec=config.retry_after_sec)
