"""The phone drone classifier: Python reference features, the exported model,
and that the JavaScript copy in sensor.html computes the same score."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.live import drone_audio as da

PAGE = Path(__file__).resolve().parents[1] / "static" / "sensor.html"
SR = 16000


def _hum(seconds=1.0, f0=180.0, noise=0.01, seed=0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    return 0.1 * x + noise * np.random.default_rng(seed).standard_normal(len(t))


def test_features_have_the_model_input_size_and_ignore_gain():
    x = _hum()
    f = da.clip_features(x, SR)
    assert f.shape == (da.N_FEATURES,)
    # Spectral shape, tonality and level spread are all gain-free.
    assert np.allclose(f, da.clip_features(x * 20, SR), atol=1e-6)


def test_a_steady_hum_is_more_tonal_than_white_noise():
    noise = 0.1 * np.random.default_rng(1).standard_normal(SR)
    tonal = lambda x: da.clip_features(x, SR)[2 * da.N_BANDS]  # noqa: E731
    assert tonal(_hum()) > 0.5 > tonal(noise)


def test_band_filters_cover_100_to_7000_hz():
    w = da.band_weights(48000, 2048)
    hz = 48000 / 2048
    used = np.nonzero(w.sum(0))[0]
    assert used.min() * hz >= da.F_LO and used.max() * hz <= da.F_HI
    assert (w.sum(1) > 0).all()


def test_exported_model_loads_and_scores_probabilities():
    m = da.DroneModel.load()
    assert len(m.mu) == da.N_FEATURES and m.w1.shape[1] == da.N_FEATURES
    p = m.prob(np.vstack([da.clip_features(_hum(seed=s), SR) for s in range(3)]))
    assert ((p >= 0) & (p <= 1)).all()
    assert 0.5 <= m.threshold < 1
    for key in ("precision", "recall"):
        assert 0 <= m.spec["metrics"]["test_fft1024"][key] <= 1


def _inline_model():
    page = PAGE.read_text(encoding="utf-8")
    return page, json.loads(re.search(r"/\*DRONE_MODEL\*/(.*?)/\*END_DRONE_MODEL\*/", page, re.S).group(1))


def test_sensor_page_carries_the_exported_model():
    _, inline = _inline_model()
    assert inline == json.loads(da.MODEL_PATH.read_text())


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_javascript_matches_python_reference():
    page, _ = _inline_model()
    start = page.index("  const DRONE_MODEL =")
    end = page.index("\n  }\n", page.index("function droneClassifier")) + 4
    sr, nfft = 48000, 2048
    x = _hum()
    x = np.interp(np.arange(sr) / sr, np.arange(len(x)) / SR, x)  # 1 s at 48 kHz
    db = da.frames_db(x, sr, nfft, int(0.06 * sr))
    want = da.DroneModel.load().prob(da.window_features(da.frame_stats(db, sr, nfft)))[0]
    js = page[start:end] + (
        f"\nconst c = droneClassifier({sr}, {nfft});"
        f"\nconst db = {json.dumps(db.round(4).tolist())};"
        "\nconsole.log(c.prob(db.map((f) => c.frame(f))));")
    got = float(subprocess.run(["node"], input=js, capture_output=True, text=True, check=True).stdout)
    assert got == pytest.approx(want, abs=1e-3)
