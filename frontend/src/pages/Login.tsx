import { useState } from "react";
import { GoogleLogin, GoogleOAuthProvider } from "@react-oauth/google";
import { Eye, Loader2, ShieldCheck } from "lucide-react";
import { ApiError } from "../api/client";
import { authApi, type SessionState } from "../api/auth";

// Set in frontend/.env (see .env.example). Empty = Google sign-in is hidden
// and the password form below is the only way in.
const GOOGLE_CLIENT_ID = (import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined)?.trim() ?? "";

/**
 * Sign-in, and first-run administrator creation.
 *
 * The two share a screen because they are the same moment from the operator's
 * point of view: this system needs to know who you are. Which one is shown
 * depends on whether any account exists yet, so a fresh install cannot be left
 * accidentally open - there is no state in which the API is reachable without
 * an account.
 */
export function Login({ session, onSignedIn }: { session: SessionState; onSignedIn: () => void }) {
  const firstRun = !session.configured;
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tooShort = firstRun && password.length > 0 && password.length < 8;
  const mismatch = firstRun && confirm.length > 0 && confirm !== password;
  const canSubmit =
    username.trim().length > 0 &&
    password.length > 0 &&
    (!firstRun || (password.length >= 8 && confirm === password));

  async function onGoogle(credential: string | undefined) {
    if (!credential) {
      setError("Google did not return a credential");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await authApi.google(credential);
      onSignedIn();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Google sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || busy) return;
    setBusy(true);
    setError(null);
    try {
      if (firstRun) await authApi.bootstrap(username.trim(), password);
      else await authApi.login(username.trim(), password);
      onSignedIn();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={submit} className="w-full max-w-sm">
        <div className="mb-7 flex flex-col items-center text-center">
          <div className="mb-3 grid h-12 w-12 place-items-center rounded-xl border border-accent-blue/30 bg-accent-blue/10 shadow-[0_0_28px_-6px_var(--color-accent-blue)]">
            <Eye size={22} strokeWidth={1.75} className="text-accent-blue" />
          </div>
          <div className="text-lg font-semibold tracking-[0.1em] text-text-primary">DRISHTI</div>
          <div className="mt-0.5 text-[12px] text-text-tertiary">Video intelligence</div>
        </div>

        <div className="surface rounded-xl border border-border-1 p-6">
          <h1 className="mb-1 text-[15px] font-bold text-text-primary">
            {firstRun ? "Create administrator" : "Sign in"}
          </h1>
          <p className="mb-4 text-[12px] leading-relaxed text-text-tertiary">
            {firstRun
              ? "No accounts exist yet. This first account has full control, including cameras, evidence and users."
              : "Camera feeds, evidence and the audit trail require a signed-in account."}
          </p>

          {GOOGLE_CLIENT_ID ? (
            <>
              <div className="flex justify-center">
                <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
                  <GoogleLogin
                    onSuccess={(res) => void onGoogle(res.credential)}
                    onError={() => setError("Google sign-in was cancelled or failed")}
                    theme="filled_black"
                    shape="pill"
                    size="large"
                    text="signin_with"
                  />
                </GoogleOAuthProvider>
              </div>
              <div className="my-4 flex items-center gap-3 text-[11px] text-text-tertiary">
                <span className="h-px flex-1 bg-border-1" />
                or use a local account
                <span className="h-px flex-1 bg-border-1" />
              </div>
            </>
          ) : (
            <div className="mb-4 rounded border border-border-2 bg-bg-3 px-3 py-2 text-[11.5px] text-text-tertiary">
              Google sign-in isn&apos;t configured (set VITE_GOOGLE_CLIENT_ID). Use a local account.
            </div>
          )}

          <label className="mb-1 block text-[11.5px] font-semibold text-text-secondary">Username</label>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            className="mb-3 w-full rounded border border-border-2 bg-bg-3 px-3 py-2 text-[13px] text-text-primary focus:border-accent-blue focus:outline-none"
          />

          <label className="mb-1 block text-[11.5px] font-semibold text-text-secondary">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={firstRun ? "new-password" : "current-password"}
            className="w-full rounded border border-border-2 bg-bg-3 px-3 py-2 text-[13px] text-text-primary focus:border-accent-blue focus:outline-none"
          />
          {tooShort && (
            <div className="mt-1 text-[11px] text-accent-amber">At least 8 characters.</div>
          )}

          {firstRun && (
            <>
              <label className="mb-1 mt-3 block text-[11.5px] font-semibold text-text-secondary">
                Confirm password
              </label>
              <input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
                className="w-full rounded border border-border-2 bg-bg-3 px-3 py-2 text-[13px] text-text-primary focus:border-accent-blue focus:outline-none"
              />
              {mismatch && (
                <div className="mt-1 text-[11px] text-accent-red">Passwords do not match.</div>
              )}
            </>
          )}

          {error && (
            <div className="mt-3 rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!canSubmit || busy}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded bg-accent-blue py-2.5 text-[13px] font-semibold text-on-accent disabled:opacity-40"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <ShieldCheck size={14} />}
            {firstRun ? "Create administrator" : "Sign in"}
          </button>
        </div>
      </form>
    </div>
  );
}
