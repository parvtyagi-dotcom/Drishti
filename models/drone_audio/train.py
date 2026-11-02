"""Train the phone drone-sound classifier and export it for the sensor page.

Data (gitignored, in models/drone_audio/data/): DADS, "Drone Audio Detection
Samples" (huggingface.co/datasets/geronimobasso/drone-audio-detection-samples,
MIT on the hub; it bundles other datasets, each with its own terms).
  * drone clips: parquet shards 04, 12 and 30 (~275 MB, 13,871 clips);
  * no-drone clips: parquet shard 02 (529 MB, 4,624 clips: street, park,
    indoor and other scenes, up to 3 s used from each).

Held out: every drone clip in shard 30 and every 5th block of 50 no-drone
rows, so test clips never share a shard or block with training clips.

    python models/drone_audio/train.py

writes models/drone_audio/model.json and the copy inlined in
backend/static/sensor.html.
"""
from __future__ import annotations

import io
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.live.drone_audio import N_FEATURES, clip_features  # noqa: E402

DATA = ROOT / "models" / "drone_audio" / "data"
OUT = ROOT / "models" / "drone_audio" / "model.json"
PAGE = ROOT / "backend" / "static" / "sensor.html"
SR = 16000
WIN = SR  # 1 s, what the page looks at
rng = np.random.default_rng(0)


def clips(shard: str, label: int = 1) -> list[np.ndarray]:
    import pyarrow.parquet as pq
    out = []
    for row in pq.read_table(DATA / f"s{shard}.parquet").to_pylist():
        assert row["label"] == label
        with wave.open(io.BytesIO(row["audio"]["bytes"])) as w:
            assert w.getframerate() == SR and w.getsampwidth() == 2
            out.append(np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0)
    return out


def windows(clip: np.ndarray, most: int) -> list[np.ndarray]:
    if len(clip) < SR // 2:
        return []
    starts = list(range(0, max(1, len(clip) - WIN + 1), WIN))[:most]
    return [clip[s:s + WIN] for s in starts]


def mix(drone: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, len(drone))
    pd, pn = np.mean(drone ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return drone + noise * np.sqrt(pd / pn / 10 ** (snr_db / 10))


def feats(clips, n_fft=1024) -> np.ndarray:
    return np.array([clip_features(c, SR, n_fft) for c in clips]) if clips else np.zeros((0, N_FEATURES))


def main() -> None:
    neg = clips("02", label=0)
    block = np.arange(len(neg)) // 50
    neg_split = np.where(block % 5 == 4, "test", np.where(block % 5 == 3, "val", "train"))
    negs = {s: [w for a, sp in zip(neg, neg_split) if sp == s for w in windows(a, 3)]
            for s in ("train", "val", "test")}

    train_d = clips("04") + clips("12")
    cut = int(len(train_d) * 0.8)  # contiguous, so val clips are not neighbours of train clips
    pos = {"train": [w for c in train_d[:cut] for w in windows(c, 2)],
           "val": [w for c in train_d[cut:] for w in windows(c, 2)],
           "test": [w for c in clips("30") for w in windows(c, 2)]}

    # Training positives also come mixed into background noise, 0-20 dB SNR,
    # and frames come at two FFT sizes, since a phone at 48 kHz frames differently.
    noisy = [mix(p, negs["train"][rng.integers(len(negs["train"]))], rng.uniform(0, 20))
             for p in pos["train"][::2]]
    Xp = np.vstack([feats(pos["train"]), feats(noisy, 512)])
    Xn = np.vstack([feats(negs["train"]), feats(negs["train"][::2], 512)])
    X = np.vstack([Xp, Xn]).astype(np.float32)
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(N_FEATURES, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(Xn) / len(Xp)))
    for _ in range(1500):
        opt.zero_grad()
        loss = lossf(net(Xt).squeeze(1), yt)
        loss.backward()
        opt.step()

    def prob(F):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(((F - mu) / sd).astype(np.float32))).squeeze(1)).numpy()

    def pr(pp, pn, t):
        tp, fp = (pp >= t).sum(), (pn >= t).sum()
        return float(tp / max(tp + fp, 1)), float(tp / len(pp)), float(fp / len(pn))

    # Threshold: the lowest that keeps validation precision at 98 % - a false
    # drone alarm wakes the post, so precision matters more than recall here.
    vp, vn = prob(feats(pos["val"])), prob(feats(negs["val"]))
    threshold = next(t for t in np.arange(0.5, 0.995, 0.01) if pr(vp, vn, t)[0] >= 0.98)

    metrics = {"threshold": round(float(threshold), 2)}
    for split in ("val", "test"):
        for n_fft in (1024, 512):
            pp, pn = prob(feats(pos[split], n_fft)), prob(feats(negs[split], n_fft))
            p, r, f = pr(pp, pn, threshold)
            metrics[f"{split}_fft{n_fft}"] = {"drone_windows": len(pp), "background_windows": len(pn),
                                              "precision": round(p, 3), "recall": round(r, 3),
                                              "false_alarm_rate": round(f, 4)}
    tp_noisy = prob(feats([mix(p, negs["test"][rng.integers(len(negs["test"]))], 10) for p in pos["test"]]))
    metrics["test_recall_drone_in_noise_10dB"] = round(float((tp_noisy >= threshold).mean()), 3)
    print(json.dumps(metrics, indent=1))

    w = net[0].weight.detach().numpy(), net[0].bias.detach().numpy(), net[2].weight.detach().numpy()[0], float(net[2].bias)
    r6 = lambda a: [round(float(v), 6) for v in np.ravel(a)]  # noqa: E731
    spec = {"version": 1, "sr_train": SR, "mu": r6(mu), "sd": r6(sd),
            "w1": [r6(row) for row in w[0]], "b1": r6(w[1]), "w2": r6(w[2]), "b2": round(w[3], 6),
            "threshold": metrics["threshold"], "metrics": metrics}
    OUT.write_text(json.dumps(spec))
    page = PAGE.read_text(encoding="utf-8")
    page, n = re.subn(r"(/\*DRONE_MODEL\*/).*?(/\*END_DRONE_MODEL\*/)",
                      lambda m: m.group(1) + json.dumps(spec, separators=(",", ":")) + m.group(2), page, flags=re.S)
    assert n == 1, "sensor.html is missing the DRONE_MODEL marker"
    PAGE.write_text(page, encoding="utf-8")
    print("wrote", OUT, "and", PAGE)


if __name__ == "__main__":
    main()
