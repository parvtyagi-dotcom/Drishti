"""Drone-sound classifier features, shared by training and the tests.

The phone sensor page (backend/static/sensor.html) runs the trained model in
the browser. It cannot run Python, so these features are chosen to be exactly
what a Web Audio AnalyserNode gives: Blackman-windowed FFT frames in dB. The
page computes the same numbers in JavaScript; this module is the reference the
model is trained on (models/drone_audio/train.py) and checked against
(backend/tests/test_drone_audio.py).

Features for a ~1 s window of frames:
  * the spectral shape: 24 mel-spaced bands from 100 to 7000 Hz, in dB,
    minus that frame's mean so phone gain and distance cancel out, averaged
    over the window and its spread over time;
  * tonality: the share of 100-4000 Hz power in the 8 strongest bins (the old
    hum rule's measure), averaged and its spread;
  * how steady the overall level is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_BANDS = 24
F_LO, F_HI = 100.0, 7000.0
TONAL_LO, TONAL_HI, TONAL_TOP = 100.0, 4000.0, 8
N_FEATURES = 2 * N_BANDS + 3

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "drone_audio" / "model.json"


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def band_weights(sr: float, n_fft: int) -> np.ndarray:
    """Triangular mel filters, shape (N_BANDS, n_fft // 2)."""
    freqs = np.arange(n_fft // 2) * sr / n_fft
    edges = 700.0 * (10 ** (np.linspace(_mel(F_LO), _mel(F_HI), N_BANDS + 2) / 2595.0) - 1.0)
    w = np.zeros((N_BANDS, n_fft // 2))
    for b in range(N_BANDS):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        w[b] = np.clip(np.minimum(up, down), 0.0, None)
    return w


def frames_db(audio: np.ndarray, sr: float, n_fft: int, hop: int) -> np.ndarray:
    """dB spectra as AnalyserNode.getFloatFrequencyData reports them
    (smoothingTimeConstant 0): Blackman window, |X| / N, 20 log10."""
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = np.arange(n_fft)
    a = 0.16
    win = (1 - a) / 2 - 0.5 * np.cos(2 * np.pi * n / n_fft) + a / 2 * np.cos(4 * np.pi * n / n_fft)
    starts = range(0, len(x) - n_fft + 1, hop)
    fr = np.stack([x[s:s + n_fft] * win for s in starts])
    mag = np.abs(np.fft.rfft(fr, axis=1))[:, : n_fft // 2] / n_fft
    return 20 * np.log10(np.maximum(mag, 1e-12))


def frame_stats(db: np.ndarray, sr: float, n_fft: int) -> np.ndarray:
    """Per-frame (shape..., tonal, level), shape (T, N_BANDS + 2)."""
    db = np.maximum(db, -160.0)
    p = 10 ** (db / 10)
    bands = 10 * np.log10(p @ band_weights(sr, n_fft).T + 1e-16)
    shape = bands - bands.mean(axis=1, keepdims=True)
    hz = sr / n_fft
    lo, hi = int(np.floor(TONAL_LO / hz)), int(np.ceil(TONAL_HI / hz))
    tp = p[:, lo:hi + 1]
    total = tp.sum(axis=1) + 1e-16
    top = -np.sort(-tp, axis=1)[:, :TONAL_TOP].sum(axis=1)
    level = 10 * np.log10(total)
    return np.column_stack([shape, top / total, level])


def window_features(stats: np.ndarray) -> np.ndarray:
    """Collapse per-frame stats over a window into the model's input."""
    shape, tonal, level = stats[:, :N_BANDS], stats[:, N_BANDS], stats[:, N_BANDS + 1]
    return np.concatenate([shape.mean(0), shape.std(0), [tonal.mean(), tonal.std(), level.std()]])


def clip_features(audio: np.ndarray, sr: float, n_fft: int = 1024, hop: int | None = None) -> np.ndarray:
    hop = hop or int(round(0.06 * sr))
    return window_features(frame_stats(frames_db(audio, sr, n_fft, hop), sr, n_fft))


class DroneModel:
    """The exported MLP: standardise, one hidden ReLU layer, sigmoid."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.mu = np.asarray(spec["mu"])
        self.sd = np.asarray(spec["sd"])
        self.w1 = np.asarray(spec["w1"])
        self.b1 = np.asarray(spec["b1"])
        self.w2 = np.asarray(spec["w2"])
        self.b2 = float(spec["b2"])
        self.threshold = float(spec["threshold"])

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DroneModel":
        return cls(json.loads(Path(path).read_text()))

    def prob(self, feats: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(feats) - self.mu) / self.sd
        h = np.maximum(z @ self.w1.T + self.b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))
