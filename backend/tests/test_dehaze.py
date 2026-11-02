"""Fog mode: clears foggy frames, leaves clear ones alone."""
import numpy as np

from backend.core.dehaze import dehaze, fog_measures


def scene():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    img[60:150, 100:140] = 20   # a dark figure
    return img


def fogged(img, amount=0.6, grey=215):
    return (img * (1 - amount) + grey * amount).astype(np.uint8)


def test_clear_frame_passes_through_untouched():
    img = scene()
    out, info = dehaze(img)
    assert out is img and not info.foggy


def test_foggy_frame_gets_its_contrast_back():
    foggy = fogged(scene())
    out, info = dehaze(foggy)
    assert info.foggy and info.cleared
    assert fog_measures(out)[1] > fog_measures(foggy)[1] * 1.5   # contrast restored
    figure, background = out[60:150, 100:140].mean(), out[:, :80].mean()
    assert background - figure > foggy[:, :80].mean() - foggy[60:150, 100:140].mean()
