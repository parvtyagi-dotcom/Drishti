"""Zero-DCE++ low-light enhancement on the CPU.

Zero-DCE++ (Li, Guo & Loy, "Learning to Enhance Low-Light Image via
Zero-Reference Deep Curve Estimation", TPAMI 2021) does not paint a new
image. A 7-layer network looks at the frame and predicts, for every pixel
and colour channel, a curve parameter alpha in [-1, 1]; the frame is then
brightened by applying the same quadratic curve eight times:

    x <- x + alpha * (x^2 - x)

(That is the authors' code's form - the paper writes x + alpha*x*(1 - x),
the same curve with alpha's sign flipped. The released weights were trained
with the code's form, so here a negative alpha brightens.)

How it is made cheap enough for live video, measured on a Ryzen 5 5500U:

- The network only predicts how much each region should be lifted, so it
  runs on a copy of the frame no larger than 320px and its alpha map is
  scaled back up (the paper's own scale_factor trick). Through OpenVINO,
  when installed, it takes ~25ms against ~72ms through torch, same output.
- The eight curve steps are the expensive part at full resolution (~36ms
  of float maths on a 960x540 frame). But the result depends only on a
  pixel's value (256 levels) and its alpha, so it is precomputed once as a
  128x256 table and each frame is a single table lookup (~12ms). Against
  the exact full-resolution computation the output differs by 2.4-4.9/255
  on average over the dark demo clips.

The network below is written from the paper and the authors' reference
code; only the trained weights (models/zero_dce_pp.pth, 52 KB, the
authors' released Epoch99 snapshot from github.com/Li-Chongyi/
Zero-DCE_extension) come from outside, loaded with weights_only=True so the
file is read as tensors and can never run code.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

CURVE_ITERATIONS = 8
ALPHA_LEVELS = 128  # rows of the curve table; level index << 8 | pixel must fit in uint16
_FEATURES = 32


def _build_network():
    import torch
    from torch import nn

    class _DepthwiseSeparable(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.depth_conv = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
            self.point_conv = nn.Conv2d(c_in, c_out, 1)

        def forward(self, x):
            return self.point_conv(self.depth_conv(x))

    class _CurveNet(nn.Module):
        """Layer names match the released state dict (e_conv1..e_conv7)."""

        def __init__(self):
            super().__init__()
            f = _FEATURES
            self.e_conv1 = _DepthwiseSeparable(3, f)
            self.e_conv2 = _DepthwiseSeparable(f, f)
            self.e_conv3 = _DepthwiseSeparable(f, f)
            self.e_conv4 = _DepthwiseSeparable(f, f)
            self.e_conv5 = _DepthwiseSeparable(f * 2, f)
            self.e_conv6 = _DepthwiseSeparable(f * 2, f)
            self.e_conv7 = _DepthwiseSeparable(f * 2, 3)

        def forward(self, x):
            relu = torch.relu
            x1 = relu(self.e_conv1(x))
            x2 = relu(self.e_conv2(x1))
            x3 = relu(self.e_conv3(x2))
            x4 = relu(self.e_conv4(x3))
            x5 = relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = relu(self.e_conv6(torch.cat([x2, x5], 1)))
            return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    return _CurveNet()


def apply_curves(image01: np.ndarray, alpha: np.ndarray, iterations: int = CURVE_ITERATIONS) -> np.ndarray:
    """The light-enhancement curve, applied in place: x + alpha*(x^2 - x),
    `iterations` times. Float32 arrays of the same shape; image values in
    [0, 1]. With alpha in [-1, 1] every step maps [0, 1] into itself, so the
    result needs no clipping."""
    x = image01
    tmp = np.empty_like(x)
    for _ in range(iterations):
        np.multiply(x, x, out=tmp)
        tmp -= x
        tmp *= alpha
        x += tmp
    return x


def build_curve_table(levels: int = ALPHA_LEVELS) -> np.ndarray:
    """table[level << 8 | pixel] = the curve's output for that pixel value
    at alpha = -1 + 2*level/(levels-1), as uint8."""
    alphas = np.linspace(-1.0, 1.0, levels, dtype=np.float32)[:, None]
    pixels = np.tile(np.arange(256, dtype=np.float32)[None] / 255.0, (levels, 1))
    out = apply_curves(pixels, np.broadcast_to(alphas, pixels.shape).copy())
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).ravel()


class ZeroDCE:
    """Loaded once and shared; enhance() is thread-safe (each live camera
    runs its own analysis thread)."""

    def __init__(self, weights_path: Path, work_max_dim: int = 320, use_openvino: bool = True):
        import torch

        self.work_max_dim = work_max_dim
        self._net = _build_network()
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        self._net.load_state_dict(state)
        self._net.eval()
        self._table = build_curve_table()
        self._lock = threading.Lock()
        # OpenVINO compiles for one input shape; a camera's frames all have
        # the same one, so this holds one entry per camera resolution.
        self._ov_requests: dict[tuple[int, int], object] = {}
        self._ov_core = None
        if use_openvino:
            try:
                import openvino

                self._ov_core = openvino.Core()
            except Exception:
                self._ov_core = None
        self.backend = "openvino" if self._ov_core is not None else "torch"

    def _small_alpha(self, small_rgb01_nchw: np.ndarray) -> np.ndarray:
        """Runs the network: 1x3xhxw float32 in, hxwx3 alpha out."""
        if self._ov_core is not None:
            shape = small_rgb01_nchw.shape[2:]
            request = self._ov_requests.get(shape)
            if request is None:
                import openvino
                import torch

                model = openvino.convert_model(self._net, example_input=torch.from_numpy(small_rgb01_nchw))
                compiled = self._ov_core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
                request = self._ov_requests[shape] = compiled.create_infer_request()
            out = request.infer({0: small_rgb01_nchw})[0]
        else:
            import torch

            with torch.inference_mode():
                out = self._net(torch.from_numpy(small_rgb01_nchw)).numpy()
        return out[0].transpose(1, 2, 0)

    def alpha_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel curve parameters at the frame's full size, HxWx3
        float32 in the frame's own BGR channel order."""
        h, w = image_bgr.shape[:2]
        scale = min(1.0, self.work_max_dim / max(h, w))
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        with self._lock:
            alpha = self._small_alpha(np.ascontiguousarray(x))
        alpha = np.ascontiguousarray(alpha[:, :, ::-1])  # RGB -> BGR
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        alpha = self.alpha_map(image_bgr)
        half = (ALPHA_LEVELS - 1) / 2
        index = cv2.convertScaleAbs(alpha, alpha=half, beta=half).astype(np.uint16)  # rounded alpha level
        index <<= 8
        index |= image_bgr
        return np.take(self._table, index)


_shared: dict[tuple[Path, int], ZeroDCE] = {}
_shared_lock = threading.Lock()


def shared_model(weights_path: Path, work_max_dim: int) -> ZeroDCE:
    """One loaded model per weights file and working size, for the whole
    process."""
    key = (Path(weights_path), work_max_dim)
    with _shared_lock:
        model = _shared.get(key)
        if model is None:
            model = _shared[key] = ZeroDCE(weights_path, work_max_dim)
        return model
