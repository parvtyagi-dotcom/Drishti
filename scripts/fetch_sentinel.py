"""Fetch a before/after pair of Sentinel-2 images for the Satellite Check page.

Needs internet, so run it at sector HQ and carry the folder it writes to the
post. The download itself lives in backend/live/sentinel.py, which the page's
place search also uses when the machine is online.

    python scripts/fetch_sentinel.py --bbox 74.55,31.58,74.59,31.62 \
        --before 2025-09-01/2025-09-30 --after 2025-10-20/2025-11-15 \
        --name "Attari fields" --id attari

Writes samples/satellite/<id>/ with meta.json, the before/after images and
map_region/map_local.png street maps showing where the box is.
`--map-only <id> ...` adds just the maps to pairs already saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.live.sentinel import (  # noqa: E402
    ASK_ABOVE_MB, OUT_ROOT, FetchError, box_km, estimate_mb, fetch_pair, save_location_maps,
)


def main() -> int:
    if "--map-only" in sys.argv:
        # Add location maps to pairs fetched before maps existed:
        #   python scripts/fetch_sentinel.py --map-only khavda attari
        for pair_id in sys.argv[sys.argv.index("--map-only") + 1:]:
            meta = json.loads((OUT_ROOT / pair_id / "meta.json").read_text())
            print(pair_id)
            save_location_maps(meta["bbox"], OUT_ROOT / pair_id)
        return 0

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", required=True, help="west,south,east,north in degrees")
    ap.add_argument("--before", required=True, help="date range, YYYY-MM-DD/YYYY-MM-DD")
    ap.add_argument("--after", required=True, help="date range, YYYY-MM-DD/YYYY-MM-DD")
    ap.add_argument("--name", required=True)
    ap.add_argument("--id", required=True, help="folder name under samples/satellite")
    ap.add_argument("--note", default="", help="what to expect, shown on the page")
    ap.add_argument("--max-cloud", type=float, default=30, help="scene-wide cloud %% for the search")
    ap.add_argument("--yes", action="store_true", help="do not ask before a large download")
    args = ap.parse_args()

    bbox = [float(v) for v in args.bbox.split(",")]
    w_km, h_km = box_km(bbox)
    est_mb = estimate_mb(bbox)
    print(f"Box is {w_km:.1f} x {h_km:.1f} km. Expected download: about {est_mb:.0f} MB per attempt.")
    if est_mb > ASK_ABOVE_MB and not args.yes:
        if input(f"That is over {ASK_ABOVE_MB} MB. Continue? [y/N] ").strip().lower() != "y":
            return 1
    try:
        fetch_pair(bbox, args.before, args.after, args.name, args.id, args.note, args.max_cloud)
    except FetchError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
