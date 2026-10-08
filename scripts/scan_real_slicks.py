"""Scan recent real Sentinel-1 passes over Danish shipping lanes for ship-trail slick candidates.

For each (scene, AOI): skip if ERA5 wind at acquisition is outside [--wmin, --wmax] m/s (slicks hard to
judge), otherwise download the AOI, run detection + triage, and record components with the ship-trail
pattern. Output: outputs/scan_results.json. Everything is real data; nothing is labelled oil by this
script — it only shortlists scenes for an analyst.

  python scripts/scan_real_slicks.py --start 2026-08-01 --end 2026-09-24 --max-ingests 12
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import load_config  # noqa: E402
from app.environmental.forcing import point_wind  # noqa: E402
from app.sar.sentinel1 import search_scenes  # noqa: E402
from app.services import pipeline as P  # noqa: E402

LANES = {  # main traffic lanes (DMA coverage); ~0.6 x 0.4 deg AOIs
    "Skagen / Skagerrak approach": [10.3, 57.6, 10.9, 58.0],
    "Kattegat T-route north": [11.0, 56.9, 11.6, 57.3],
    "Kattegat T-route south": [11.1, 56.3, 11.7, 56.7],
    "Great Belt north": [10.7, 55.8, 11.2, 56.1],
    "Kiel Bay / Fehmarn Belt": [10.8, 54.4, 11.4, 54.8],
    "Arkona / Baltic approach": [12.8, 54.6, 13.6, 55.0],
    "Bornholm Gat": [13.8, 55.1, 14.6, 55.5],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--wmin", type=float, default=3.5); ap.add_argument("--wmax", type=float, default=10.0)
    ap.add_argument("--max-ingests", type=int, default=12)
    ap.add_argument("--res", type=float, default=20)
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "outputs" / "scan_results.json"
    results, ingests = [], 0
    for lane, aoi in LANES.items():
        scenes = [s for s in search_scenes(aoi, a.start, a.end) if s["aoi_coverage"] >= 0.95]
        print(f"{lane}: {len(scenes)} scene(s)", flush=True)
        for s in scenes:
            if ingests >= a.max_ingests:
                break
            t = datetime.fromisoformat(s["datetime"].replace("Z", "+00:00"))
            try:
                w = point_wind((aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2, t)["wind_speed_ms"]
            except Exception as exc:
                print("  wind unavailable", exc); continue
            if w is None or not (a.wmin <= w <= a.wmax):
                continue
            ingests += 1
            print(f"  [{ingests}] {s['id']} wind {w:.1f} m/s -> detecting", flush=True)
            an = P.run_detection(cfg, scene_id=s["id"], aoi=aoi, resolution_m=a.res)
            try:
                tri = an.load("triage.json")["components"]
            except FileNotFoundError:
                tri = []
            trails = [r for r in tri if r.get("ship_trail")]
            results.append({"lane": lane, "scene": s["id"], "time": s["datetime"], "wind_ms": w, "analysis_id": an.id,
                            "status": an.state.get("status"), "n_components": len(tri),
                            "ship_trails": [{k: r[k] for k in ("component_id", "label", "area_km2", "length_km",
                                                              "elongation", "contrast_db", "nearest_sar_target_km")}
                                            for r in trails]})
            print(f"      {len(tri)} components, {len(trails)} ship-trail pattern(s)", flush=True)
            out.write_text(json.dumps(results, indent=2))
    print("done", out)


if __name__ == "__main__":
    main()
