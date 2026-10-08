"""SIH26143 Oil Spill Investigation — command line interface.

All commands call the same service layer as the API (backend/app/services/pipeline.py).

  python main.py segment      --input IMG [--geo GEO.json] [--threshold 0.5]   -> creates a spill ID
  python main.py georeference --input IMG [--geo GEO.json]
  python main.py backtrack    --spill ID [--particles N] [--members M]
  python main.py ais          --spill ID          (AIS retrieval, filtering, scoring)
  python main.py score        --spill ID          (same stage; prints ranked candidates)
  python main.py report       --spill ID
  python main.py run          --input IMG [...]   (full pipeline on a local GeoTIFF; top triage slick)
  python main.py scenes       --bbox W,S,E,N --start YYYY-MM-DD --end YYYY-MM-DD   (real Sentinel-1 search)
  python main.py detect       --scene ITEM_ID --aoi W,S,E,N [--res 20]              (download AOI, detect, triage)
  python main.py triage       --spill ID                                            (show triage table)
  python main.py investigate  --spill ID [--components C01,C04]                     (drift + AIS + score + report)
  python main.py compare      --a ID1 --b ID2      (multi-temporal comparison of two analyses)
  python main.py list
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
for _s in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.core.config import load_config  # noqa: E402
from app.core.errors import PipelineError  # noqa: E402
from app.services import pipeline as P  # noqa: E402



def _print_event(ev):
    print(f"  [{ev['status']:<24}] {ev['message']}")


def _geo(args):
    if getattr(args, "geo", None):
        return json.loads(Path(args.geo).read_text())
    if getattr(args, "bbox", None):
        g = {"bbox": [float(v) for v in args.bbox.split(",")], "provenance": "LOCAL_FILE"}
        if args.timestamp:
            g["timestamp"] = args.timestamp
        return g
    if getattr(args, "timestamp", None):
        return {"timestamp": args.timestamp}
    return None


def _summary(an):
    d = an.repo.dir(an.id)
    print(f"\nAnalysis {an.id}: {an.state.get('outcome') or an.state['status']}")
    for w in an.state.get("warnings", []):
        print(f"  WARNING: {w}")
    if an.state.get("error"):
        e = an.state["error"]
        print(f"  {e['code']}: {e['message']}\n  hint: {e.get('hint', '')}")
    print(f"  artifacts: {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26143 oil spill investigation pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--forcing", choices=["netcdf", "open_meteo"],
                    help="override environment.provider (open_meteo = live, no API key)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("segment", "georeference", "run"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--geo", help="georeference JSON (bbox/transform, crs, timestamp)")
        p.add_argument("--bbox", help="west,south,east,north (north-up image)")
        p.add_argument("--timestamp", help="acquisition time ISO-8601 UTC")
        p.add_argument("--threshold", type=float)
        p.add_argument("--particles", type=int)
        p.add_argument("--members", type=int)
    for name in ("backtrack", "ais", "score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--spill", required=True)
        p.add_argument("--particles", type=int)
        p.add_argument("--members", type=int)
    p = sub.add_parser("scenes")
    p.add_argument("--bbox", required=True); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--scene", required=True); p.add_argument("--aoi", required=True)
    p.add_argument("--res", type=float, default=None); p.add_argument("--threshold", type=float)
    p = sub.add_parser("triage")
    p.add_argument("--spill", required=True)
    p = sub.add_parser("investigate")
    p.add_argument("--spill", required=True); p.add_argument("--components")
    p.add_argument("--ais", choices=["auto", "real", "synthetic"], help="AIS mode (default: config ais.mode)")
    p.add_argument("--particles", type=int); p.add_argument("--members", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--max-speed-kmh", type=float, default=3.0)
    sub.add_parser("list")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cfg = load_config(args.config, {"environment": {"provider": args.forcing}} if args.forcing else None)

    try:
        if args.cmd == "georeference":
            from app.geospatial.georeference import read_scene
            sc = read_scene(args.input, _geo(args), None, cfg.geospatial.sidecar_suffix)
            print(json.dumps({"georef_status": sc.georef_status, "georef": sc.georef.to_dict() if sc.georef else None,
                              "timestamp": sc.timestamp.isoformat() if sc.timestamp else None,
                              "timestamp_source": sc.timestamp_source, "meta": sc.meta}, indent=2))
            if not sc.georef:
                print("\nGeoreferencing unavailable. Supply a GeoTIFF, a <image>.geo.json sidecar, --geo or --bbox.")
            return 0
        if args.cmd == "segment":
            an = P.Analysis(cfg, on_event=_print_event)
            res = P.stage_segment(an, args.input, _geo(args), args.threshold)
            s = res.get("spill")
            print(json.dumps({"spill_id": an.id, "segmentation": res["segmentation"],
                              "spill": {k: s[k] for k in ("timestamp", "centroid", "area_m2", "perimeter_m", "bbox",
                                                          "n_components", "detection_confidence")} if s else None},
                             indent=2, default=str))
            _summary(an)
            return 0
        if args.cmd == "run":
            an = P.run_full(cfg, args.input, _geo(args), on_event=_print_event, threshold=args.threshold,
                            particles=args.particles, members=args.members)
            _summary(an); _print_candidates(an)
            return 0 if an.state["status"] != "FAILED" else 1
        if args.cmd == "scenes":
            from app.sar.sentinel1 import search_scenes
            rows = search_scenes([float(v) for v in args.bbox.split(",")], args.start, args.end)
            for r in rows:
                print(f"{r['id']}  {r['datetime'][:19]}  {r['platform']:<12} {r['orbit_state']:<10} AOI coverage {100 * r['aoi_coverage']:.0f}%")
            print(f"{len(rows)} scene(s) (Sentinel-1 GRD IW, Microsoft Planetary Computer, no key)")
            return 0
        if args.cmd == "detect":
            an = P.run_detection(cfg, on_event=_print_event, threshold=args.threshold, scene_id=args.scene,
                                 aoi=[float(v) for v in args.aoi.split(",")],
                                 resolution_m=args.res or cfg.sentinel1.default_resolution_m)
            _summary(an); _print_triage(an)
            return 0 if an.state["status"] != "FAILED" else 1
        if args.cmd == "investigate":
            comps = args.components.split(",") if args.components else None
            an = P.run_investigation(cfg, args.spill, comps, args.particles, args.members, on_event=_print_event,
                                     ais_mode=args.ais)
            _summary(an); _print_candidates(an)
            return 0 if an.state["status"] != "FAILED" else 1
        if args.cmd == "triage":
            _print_triage(P.Analysis(cfg, args.spill))
            return 0
        if args.cmd == "list":
            from app.database.repository import build_repository
            for a in build_repository(cfg).list_analyses():
                print(f"{a['analysis_id']:<32} {a['status']:<12} {a.get('input')}")
            return 0
        if args.cmd == "compare":
            from app.multitemporal.compare import compare
            from app.database.repository import build_repository
            repo = build_repository(cfg)
            sa, sb = (repo.load_json(x, "spill.json")["spill"] for x in (args.a, args.b))
            pred = None
            try:
                pred = repo.load_json(args.a, "drift/forward_summary.json").get("end_center")
            except FileNotFoundError:
                pass
            print(json.dumps(compare(sa, sb, pred, max_speed_kmh=args.max_speed_kmh), indent=2, default=str))
            return 0

        an = P.Analysis(cfg, args.spill, on_event=_print_event)
        if not an.repo.exists(args.spill):
            print(f"Unknown spill/analysis id {args.spill}. Use `python main.py list`."); return 2
        if args.cmd == "backtrack":
            s = P.stage_drift(an, args.particles, args.members)
            print(json.dumps({"release_window": s["release_window"], "source_regions": {k: round(v["area_km2"], 2)
                              for k, v in s["source_regions"].items()}, "combined_center": s["combined_center"]}, indent=2))
        elif args.cmd in ("ais", "score"):
            r = P.stage_ais_and_score(an)
            if args.cmd == "ais":
                print(json.dumps(r["ais"], indent=2, default=str))
                for v in r["filtered_out"]:
                    print(f"  filtered: {v['name'] or v['mmsi']}: {v['reason']}")
            _print_candidates(an)
        elif args.cmd == "report":
            P.stage_report(an)
            print(f"Report: {an.repo.dir(an.id) / 'report.html'}")
        _summary(an)
        return 0
    except PipelineError as exc:
        print(f"\nERROR {exc.code}: {exc.message}\nhint: {exc.hint}")
        return 1


def _print_triage(an, n=15):
    try:
        t = an.load("triage.json")
    except FileNotFoundError:
        return
    w = t.get("wind_at_acquisition") or {}
    print(f"\nSlick triage (wind at acquisition {w.get('wind_speed_ms', 'n/a')} m/s, "
          f"{t['sar_ship_detection']['n_detections']} SAR point targets):")
    print(f"  {'#':<4}{'component':<34}{'label':<18}{'score':>6}{'km2':>8}{'elong':>7}{'dB':>6}{'ship km':>9}")
    for r in t["components"][:n]:
        c = r["contrast_db"]; sk = r["nearest_sar_target_km"]
        print(f"  {r['triage_rank']:<4}{r['component_id']:<34}{r['label']:<18}{r['triage_score']:>6.2f}"
              f"{r['area_km2']:>8.2f}{r['elongation']:>7.1f}{(c if c is not None else float('nan')):>6.1f}"
              f"{(sk if sk is not None else float('nan')):>9.2f}")
    if len(t["components"]) > n:
        print(f"  ... {len(t['components']) - n} more")
    print("  Investigate with: python main.py investigate --spill", an.id, "--components <id>[,<id>]")


def _print_candidates(an):
    try:
        r = an.load("candidates.json")
    except FileNotFoundError:
        return
    print(f"\n{r['message']}")
    if r["candidates"]:
        print("\nCandidate vessels requiring investigation (ranked by Evidence Correlation Score):")
        print(f"  {'#':<3}{'Vessel':<26}{'Type':<14}{'Score':>7}{'Dist km':>9}{'Δt h':>7}")
        for c in r["candidates"]:
            dt = c["time_difference_hours"]
            print(f"  {c['rank']:<3}{(c['vessel']['name'] or c['vessel']['mmsi'])[:25]:<26}{c['vessel']['type']:<14}"
                  f"{c['score']:>7.1f}{c['source_distance_km']:>9.1f}{(dt if dt is not None else float('nan')):>7.1f}")
        print(f"\n  {r['disclaimer']}")


if __name__ == "__main__":
    sys.exit(main())
