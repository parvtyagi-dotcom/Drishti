"""Satellite check: finds cleared vegetation, tracks and structures on synthetic pairs."""
import numpy as np

from backend.live import satellite
from backend.live.satellite import detect_changes


def terrain(seed=0, size=(200, 240)):
    rng = np.random.default_rng(seed)
    base = rng.normal(150, 6, size).astype(np.float32)
    img = np.dstack([base * 0.8, base * 0.95, base]).clip(0, 255).astype(np.uint8)
    return img


def bands(size=(200, 240)):
    red = np.full(size, 400, np.uint16)    # green scrub: low red, high NIR
    nir = np.full(size, 3500, np.uint16)
    return red, nir


def test_identical_images_have_no_changes():
    img = terrain()
    r = detect_changes(img, img.copy())
    assert r["changes"] == [] and r["changed_area_m2"] == 0


def test_new_track_is_found_with_area():
    before = terrain()
    after = before.copy()
    after[100:103, 20:200] = 235            # a bright 3-pixel-wide track, 180 px long
    r = detect_changes(before, after)
    assert r["method"] == "colour"
    c = r["changes"][0]
    assert c["kind"] == "track"
    x, y, w, h = c["box"]
    assert x <= 22 and x + w >= 198 and 97 <= y <= 101
    assert 40_000 < c["area_m2"] < 70_000
    assert "about" in c["reason"] and "m²" in c["reason"]


def test_new_dark_structure():
    before = terrain()
    after = before.copy()
    after[40:60, 150:175] = 40
    c = detect_changes(before, after)["changes"][0]
    assert c["kind"] == "structure"
    assert abs(c["area_m2"] - 20 * 25 * 100) < 1500


def test_vegetation_cleared_uses_ndvi():
    before, after = terrain(), terrain()
    br, bn = bands()
    ar, an = br.copy(), bn.copy()
    ar[50:80, 60:100], an[50:80, 60:100] = 1800, 2200   # bare soil where scrub was
    r = detect_changes(before, after, before_red=br, before_nir=bn, after_red=ar, after_nir=an)
    assert r["method"] == "ndvi+colour"
    assert len(r["changes"]) == 1
    c = r["changes"][0]
    assert c["kind"] == "vegetation_cleared"
    assert c["reason"].startswith("Vegetation cleared, about 120,000 m²")


def test_small_misregistration_is_aligned_away():
    rng = np.random.default_rng(3)
    big = rng.integers(60, 200, (260, 300, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(big[::4, ::4], 4, 0), 4, 1)[:260, :300]   # blocky fields
    before = big[20:220, 20:260]
    after = big[22:222, 23:263].copy()   # same place, content 3 px left and 2 px up
    r = detect_changes(before, after)
    assert abs(r["shift_px"][0] + 3) < 0.6 and abs(r["shift_px"][1] + 2) < 0.6
    assert r["changed_fraction"] < 0.03


def test_global_haze_is_not_change():
    before = terrain()
    after = (before * 0.7 + 70).astype(np.uint8)   # whole scene washed out
    assert detect_changes(before, after)["changes"] == []


def test_saved_pairs_and_bad_ids(tmp_path, monkeypatch):
    import cv2, json
    monkeypatch.setattr(satellite, "PAIRS_DIR", tmp_path)
    satellite._cache.clear()
    d = tmp_path / "demo"
    d.mkdir()
    before = terrain()
    after = before.copy()
    after[40:60, 150:175] = 40
    cv2.imwrite(str(d / "before_rgb.png"), before)
    cv2.imwrite(str(d / "after_rgb.png"), after)
    (d / "meta.json").write_text(json.dumps({"name": "Demo", "pixel_m": 10}))
    assert [p["id"] for p in satellite.list_pairs()] == ["demo"]
    r = satellite.run_check("demo")
    assert r["total_changes"] == 1 and satellite.list_pairs()[0]["checked"]
    assert satellite.run_check("../demo") is None
    assert satellite.image_path("demo", "before").name == "before_rgb.png"
    assert satellite.image_path("demo", "secret") is None


def test_location_maps_are_served_only_when_both_exist(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(satellite, "PAIRS_DIR", tmp_path)
    d = tmp_path / "demo"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "Demo", "pixel_m": 10}))
    assert satellite.list_pairs()[0]["has_map"] is False
    assert satellite.image_path("demo", "map_region") is None

    (d / "map_region.png").write_bytes(b"png")
    assert satellite.list_pairs()[0]["has_map"] is False  # the close-up is still missing
    (d / "map_local.png").write_bytes(b"png")
    assert satellite.list_pairs()[0]["has_map"] is True
    assert satellite.image_path("demo", "map_region").name == "map_region.png"
    assert satellite.image_path("demo", "map_local").name == "map_local.png"
    assert satellite.image_path("demo", "meta") is None
