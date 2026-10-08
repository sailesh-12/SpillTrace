"""Analysis orchestrator — the single service layer used by BOTH the CLI and the API.

State machine (spec §30A):
QUEUED → SCENE_INSPECTION → GEOREFERENCING → PREPROCESSING → SEGMENTATION → POLYGONIZATION
→ ENVIRONMENTAL_DATA → BACKWARD_DRIFT → SOURCE_ESTIMATION → AIS_PROCESSING → AIS_GAP_ANALYSIS
→ SAR_SHIP_DETECTION → SAR_AIS_MATCHING → LOOKALIKE_ANALYSIS → MULTI_TEMPORAL_ANALYSIS
→ CANDIDATE_SCORING → REPORT → COMPLETED | PARTIAL (a required input was unavailable) | FAILED

Each stage function reads what it needs from the repository and writes its
artifacts back, so the CLI can run stages individually (`backtrack --spill ID`).
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.core.config import Config
from app.core.errors import (ForcingUnavailable, GeoreferenceUnavailable, PipelineError)
from app.core.provenance import Provenance
from app.database.repository import build_repository

log = logging.getLogger(__name__)

STAGES = ["QUEUED", "SCENE_ACQUISITION", "SCENE_INSPECTION", "GEOREFERENCING", "PREPROCESSING", "SEGMENTATION", "POLYGONIZATION",
          "ENVIRONMENTAL_DATA", "BACKWARD_DRIFT", "SOURCE_ESTIMATION", "AIS_PROCESSING", "AIS_GAP_ANALYSIS",
          "SAR_SHIP_DETECTION", "SAR_AIS_MATCHING", "LOOKALIKE_ANALYSIS", "MULTI_TEMPORAL_ANALYSIS",
          "CANDIDATE_SCORING", "REPORT", "COMPLETED"]
# Phase 1 (acquire -> detect -> triage) ends in AWAITING_SLICK_SELECTION; the investigator then picks slick(s).
TERMINAL = {"COMPLETED", "PARTIAL", "FAILED", "AWAITING_SLICK_SELECTION"}

ProgressFn = Callable[[dict], None]


class Analysis:
    """Holds state for one analysis and emits progress events."""

    def __init__(self, cfg: Config, analysis_id: str | None = None, on_event: ProgressFn | None = None,
                 repo=None):
        self.cfg = cfg
        self.repo = repo or build_repository(cfg)
        self.id = analysis_id or f"SPILL_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4].upper()}"
        self.on_event = on_event
        self.state = self.repo.load_state(self.id) if self.repo.exists(self.id) else {
            "analysis_id": self.id, "status": "QUEUED", "stage": "QUEUED", "events": [], "warnings": [],
            "created_at": datetime.now(timezone.utc).isoformat(), "mode": cfg.get("mode", "live")}

    # --- state helpers -------------------------------------------------------------------------------
    def emit(self, stage: str, message: str, status: str | None = None, **extra):
        self.state["stage"] = stage
        self.state["status"] = status or stage
        ev = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, "status": self.state["status"],
              "message": message, "progress": round(STAGES.index(stage) / (len(STAGES) - 1), 3)
              if stage in STAGES else None, **extra}
        self.state["events"].append(ev)
        self.repo.save_state(self.id, self.state)
        log.info("[%s] %s: %s", self.id, stage, message)
        if self.on_event:
            self.on_event(ev)

    def warn(self, msg: str):
        if msg not in self.state["warnings"]:
            self.state["warnings"].append(msg)

    def save(self, name, obj):
        return self.repo.save_json(self.id, name, obj)

    def load(self, name):
        return self.repo.load_json(self.id, name)


# ======================================================================================================
# Stage 1-6: scene → segmentation → spill polygon
# ======================================================================================================
def stage_segment(an: Analysis, input_path: str, geo_override: dict | None = None,
                  threshold: float | None = None) -> dict:
    from app.geospatial.georeference import read_scene
    from app.geospatial.polygon import build_spill_record, pixel_to_wgs84
    from app.segmentation.inference import run_inference
    from app.segmentation.model_loader import load_model
    from app.segmentation.postprocess import clean_mask, components, mask_to_pixel_polygons

    cfg, s = an.cfg, an.cfg.segmentation
    an.state["input"] = str(input_path)
    an.emit("SCENE_INSPECTION", f"Reading {Path(input_path).name}")
    scene = read_scene(input_path, geo_override, dict(s.get("geotiff_scaling", {})), cfg.geospatial.sidecar_suffix)
    an.emit("GEOREFERENCING", "Georeference: " + (f"{scene.georef.source} ({scene.georef.provenance})"
                                                   if scene.georef else "UNAVAILABLE"))
    an.emit("PREPROCESSING", "RGB uint8 → float32/255, CHW (training preprocessing, no mean/std normalisation)")
    thr = float(threshold if threshold is not None else s.threshold)
    model = load_model(cfg.path(s.model_path), s.encoder_name, s.device)
    an.emit("SEGMENTATION", f"DeepLabV3+/ResNet34 on {model.device}, threshold {thr}")
    seg = run_inference(model, scene.rgb_u8, thr, s.tile_size, s.tile_overlap)
    sea = _sea_mask_for(scene)
    if sea is not None:
        seg["probability_mask"][~sea] = 0.0          # land is never oil
        seg["binary_mask"] &= sea
        seg["oil_fraction"] = float(seg["binary_mask"].mean())
    min_px = int(s.min_component_area_pixels)
    if scene.georef and scene.georef.pixel_size_m and s.get("min_component_area_m2"):
        min_px = max(min_px, int(s.min_component_area_m2 / scene.georef.pixel_size_m ** 2))
    mask = clean_mask(seg["binary_mask"], min_px, s.morphology_kernel)
    comps = components(mask, seg["probability_mask"])

    # artifacts for UI: SAR image, probability map, mask overlay
    disp = _display_size(scene.rgb_u8.shape[:2])        # browser-friendly overlays (analysis uses full resolution)
    Image.fromarray(scene.rgb_u8).resize(disp, Image.BILINEAR).convert("L").save(
        an.repo.artifact_path(an.id, "scene.jpg"), quality=88)                    # display only
    Image.fromarray((seg["probability_mask"] * 255).astype(np.uint8)).resize(disp, Image.BILINEAR).save(
        an.repo.artifact_path(an.id, "probability.png"))
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (255, 80, 0, 170)
    Image.fromarray(rgba).resize(disp, Image.NEAREST).save(an.repo.artifact_path(an.id, "mask.png"))
    np.save(an.repo.artifact_path(an.id, "mask.npy"), mask)

    conf = float(seg["probability_mask"][mask].mean()) if mask.any() else 0.0
    seg_info = {"threshold": thr, "raw_oil_fraction": seg["oil_fraction"], "clean_oil_fraction": float(mask.mean()),
                "n_components": len(comps), "detection_confidence": conf, "device": model.device,
                "model": {k: v for k, v in model.metadata.items() if k != "path"},
                "min_component_area_pixels": min_px,
                "image_size": list(scene.rgb_u8.shape[:2])}
    scene_info = {"path": scene.path, "georef_status": scene.georef_status,
                  "georef": scene.georef.to_dict() if scene.georef else None,
                  "timestamp": scene.timestamp.isoformat() if scene.timestamp else None,
                  "timestamp_source": scene.timestamp_source, "meta": scene.meta}
    if scene.georef:
        a, b, c, d, e, f = scene.georef.transform
        h, w = scene.rgb_u8.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        scene_info["image_corners_lonlat"] = [[a * x + b * y + c, d * x + e * y + f] for x, y in corners] \
            if scene.georef.crs.upper() == "EPSG:4326" else None
    result = {"analysis_id": an.id, "scene": scene_info, "segmentation": seg_info, "spill": None}

    if not comps:
        an.save("spill.json", result)
        an.state["outcome"] = "COMPLETED"
        an.emit("POLYGONIZATION", "No oil components above the minimum area — nothing to investigate.")
        return result
    if scene.georef is None:
        result["error"] = GeoreferenceUnavailable(
            "Georeferencing unavailable. Cannot perform geographic drift/AIS analysis. Georeferencing required.",
            "Supply a GeoTIFF, or a <image>.geo.json sidecar / geo override with bbox or transform.").to_dict()
        # pixel-space polygons are still useful for display
        result["pixel_components"] = [{"area_px": c["area_px"], "bbox_px": c["bbox_px"],
                                       "mean_probability": c["mean_probability"]} for c in comps]
        an.save("spill.json", result)
        an.state["error"] = result["error"]
        an.state["outcome"] = "PARTIAL"
        an.emit("POLYGONIZATION", result["error"]["message"])
        return result

    polys, stats = [], []
    for c in comps:
        for p in mask_to_pixel_polygons(c["mask"], c.get("offset", (0, 0))):
            polys.append(pixel_to_wgs84(p, scene.georef))
            stats.append(c)
    spill = build_spill_record(an.id, polys, stats, scene.timestamp, conf)
    spill["georef_provenance"] = scene.georef.provenance
    result["spill"] = spill
    result["sea_mask_applied"] = sea is not None
    an.save("spill.json", result)
    an.save("spill.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": comp["geometry"],
         "properties": {k: comp[k] for k in ("component_id", "area_m2", "perimeter_m", "mean_probability")}}
        for comp in spill["components"]]})
    an.emit("POLYGONIZATION", f"{spill['n_components']} dark-feature component(s), total {spill['area_m2'] / 1e6:.3f} km²")
    stage_triage(an, scene, sea, stats)
    return result


def _display_size(hw, max_px: int = 4096) -> tuple[int, int]:
    h, w = hw
    s = min(1.0, max_px / max(h, w))
    return max(int(w * s), 1), max(int(h * s), 1)


def _sea_mask_for(scene):
    """Sea mask from the ingest sidecar (<tif>_sea.npy), or computed from the GSHHG coastline for north-up WGS84 grids."""
    p = Path(scene.path)
    side = p.with_name(p.stem + "_sea.npy")
    if side.exists():
        m = np.load(side)
        if m.shape == scene.rgb_u8.shape[:2]:
            return m
    g = scene.georef
    if g is not None and g.crs.upper() == "EPSG:4326" and g.transform[1] == 0 and g.transform[3] == 0:
        try:
            from app.sar.sentinel1 import sea_mask
            return sea_mask(g.bounds_wgs84, g.width, g.height)
        except Exception as exc:  # landmask unavailable -> report, do not guess
            log.warning("sea mask unavailable: %s", exc)
    return None


def stage_triage(an: Analysis, scene, sea, comp_masks: list):
    """SAR ship detection + per-slick look-alike triage. Ends phase 1 awaiting investigator selection."""
    import re
    from scipy import ndimage
    from app.environmental.forcing import point_wind
    from app.lookalike.triage import triage_components
    from app.sar_ship_detection.detector import CFARShipDetector, geolocate

    cfg = an.cfg
    spill = an.load("spill.json")["spill"]
    gray = scene.rgb_u8[..., 0]
    sd = cfg.sar_ship_detection
    anom_p = Path(scene.path).with_name(Path(scene.path).stem + "_anom.npy")
    if anom_p.exists():
        from app.sar_ship_detection.detector import DbPointTargetDetector
        det = DbPointTargetDetector(np.load(anom_p), float(sd.get("point_target_db", 10.0)),
                                    int(sd.get("point_target_min_px", 2)), sd.max_pixels)
        an.emit("SAR_SHIP_DETECTION", f"Point targets > {det.thr:g} dB above sea background (untrained baseline)")
    else:
        det = CFARShipDetector(sd.cfar_guard_px, sd.cfar_background_px, sd.cfar_k, sd.min_pixels, sd.max_pixels)
        an.emit("SAR_SHIP_DETECTION", "CA-CFAR bright point targets on 8-bit image (untrained baseline)")
    dets = det.detect(gray)
    if sea is not None:                                   # ignore coast/land edges
        inner = ndimage.binary_erosion(sea, iterations=int(sd.get("coast_exclusion_px", 8)))
        dets = [d for d in dets if inner[int(d["row"]), int(d["col"])]]
    dets = geolocate(dets, scene.georef)
    ships = {"available": True, "detector": det.name, "n_detections": len(dets), "detections": dets,
             "caveat": "Untrained CFAR detector: bright targets include ships, platforms, wind turbines and buoys; "
                       "absence of a detection is not evidence of absence."}
    wind = None
    if scene.timestamp is not None and cfg.lookalike.get("fetch_wind", True):
        try:
            wind = point_wind(spill["centroid"]["lat"], spill["centroid"]["lon"], scene.timestamp)
        except Exception as exc:
            an.warn(f"Wind at acquisition unavailable for look-alike triage: {exc}")
    tags = scene.meta.get("tags", {})
    db_per_level, contrast_src = 20.0 / 255, "8-bit proxy (assumed 20 dB span)"
    m = re.search(r"gray_per_db=([\d.]+)", tags.get("SCALING", ""))
    if m:
        db_per_level, contrast_src = 1.0 / float(m.group(1)), "normalised dB (relative to local sea background)"
    an.emit("LOOKALIKE_ANALYSIS", f"Triage of {len(comp_masks)} component(s): wind "
            f"{wind['wind_speed_ms'] if wind else 'n/a'} m/s, {len(dets)} SAR point target(s)")
    rows = triage_components(spill, comp_masks, gray, sea if sea is not None else np.ones_like(gray, bool),
                             db_per_level, dets, wind, cfg)
    n_oil = sum(r["label"] == "OIL_LIKELY" for r in rows)
    an.save("triage.json", {"wind_at_acquisition": wind, "sar_ship_detection": ships, "components": rows,
                            "contrast_source": contrast_src,
                            "note": "Indicators support analyst verification; they are not a trained look-alike classifier."})
    if scene.timestamp is None:
        an.state["error"] = {"code": "TIMESTAMP_UNAVAILABLE", "message": "Acquisition time unknown: drift/AIS impossible.",
                             "hint": "Provide the acquisition timestamp."}
        an.state["outcome"] = "PARTIAL"
        return
    an.state["outcome"] = "AWAITING_SLICK_SELECTION"
    an.emit("LOOKALIKE_ANALYSIS", f"{len(rows)} component(s) triaged ({n_oil} OIL_LIKELY). "
            "Select the slick(s) to investigate.", status="AWAITING_SLICK_SELECTION")


def select_components(an: Analysis, component_ids: list[str] | None) -> dict:
    """Build the investigated spill from selected components (default: the top triage-ranked one)."""
    from shapely.geometry import shape
    from app.geospatial.polygon import build_spill_record
    spill = an.load("spill.json")["spill"]
    if not component_ids:
        try:
            component_ids = [an.load("triage.json")["components"][0]["component_id"]]
            an.warn(f"No slick selected: investigating the top triage-ranked component {component_ids[0]} (ASSUMPTION).")
        except (FileNotFoundError, IndexError):
            component_ids = [c["component_id"] for c in spill["components"]]
    comps = [c for c in spill["components"] if c["component_id"] in set(component_ids)]
    if not comps:
        raise PipelineError(f"Unknown component id(s) {component_ids}", "Use component ids from the triage table.")
    sel = build_spill_record(an.id, [shape(c["geometry"]) for c in comps],
                             [{"area_px": c["area_px"], "mean_probability": c["mean_probability"]} for c in comps],
                             datetime.fromisoformat(spill["timestamp"]) if spill.get("timestamp") else None,
                             float(np.mean([c["mean_probability"] or 0 for c in comps])))
    sel["selected_component_ids"] = [c["component_id"] for c in comps]
    sel["georef_provenance"] = spill.get("georef_provenance")
    an.state["selected_components"] = sel["selected_component_ids"]
    an.save("spill_selected.json", sel)
    an.save("spill_selected.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": c["geometry"], "properties": {"component_id": c["component_id"]}} for c in comps]})
    return sel


# ======================================================================================================
# Stage 7-9: forcing → backward drift ensemble → source probability; optional forward drift
# ======================================================================================================
def _drift_bbox(spill: dict, hours: float, max_speed_ms: float = 1.0) -> list[float]:
    import math
    reach_km = max_speed_ms * hours * 3.6
    lat = spill["centroid"]["lat"]
    dlat = reach_km / 111.32
    dlon = reach_km / (111.32 * math.cos(math.radians(lat)))
    w, s, e, n = spill["bbox"]
    return [w - dlon, s - dlat, e + dlon, n + dlat]


def _frames(ens, max_particles: int = 1200, rng_seed: int = 0) -> dict:
    """Subsampled particle frames for animation: {times, frames[[lon,lat]...]}."""
    lon = ens.lon.reshape(-1, len(ens.times))
    lat = ens.lat.reshape(-1, len(ens.times))
    idx = np.random.default_rng(rng_seed).choice(lon.shape[0], min(max_particles, lon.shape[0]), replace=False)
    frames = []
    for j in range(len(ens.times)):
        pts = np.column_stack([lon[idx, j], lat[idx, j]])
        pts = pts[np.isfinite(pts).all(1)]
        frames.append(np.round(pts, 5).tolist())
    return {"times": [t.isoformat() for t in ens.times], "frames": frames, "direction": ens.direction,
            "n_members": int(ens.lon.shape[0]), "n_particles_per_member": int(ens.lon.shape[1])}


def stage_drift(an: Analysis, particles: int | None = None, members: int | None = None) -> dict:
    from app.drift import backtrack as bt
    from app.drift import source_probability as sp
    from app.environmental.forcing import build_provider

    cfg, d = an.cfg, an.cfg.drift
    seg = an.load("spill.json")
    if not seg.get("spill"):
        raise PipelineError("No georeferenced spill available for drift.", "Run segmentation on a georeferenced scene.")
    try:
        spill = an.load("spill_selected.json")
    except FileNotFoundError:
        spill = select_components(an, None)
    obs = datetime.fromisoformat(spill["timestamp"]) if spill.get("timestamp") else None
    if obs is None:
        raise PipelineError("Observation timestamp missing — backward drift needs the acquisition time.",
                            "Add 'timestamp' to the .geo.json sidecar or pass --timestamp.", "ENVIRONMENTAL_DATA")
    offsets = sorted(float(h) for h in d.release_offsets_hours)
    # Evidence-dependent start of the release window: a slick with a ship-trail pattern may be minutes old;
    # a detached slick implies the source has left, so releases < detached_min_offset_h are excluded (ASSUMPTION).
    try:
        rows = {r["component_id"]: r for r in an.load("triage.json")["components"]}
    except FileNotFoundError:
        rows = {}
    trail = any(rows.get(c, {}).get("ship_trail") for c in spill.get("selected_component_ids", []))
    min_off = 0.0 if trail else float(d.get("detached_min_offset_h", 1.0))
    offsets = [h for h in offsets if h >= min_off] or offsets
    window_basis = ("ship-trail pattern at acquisition: fresh release possible (window starts at T)" if trail else
                    f"detached slick (no vessel at its head): release assumed >= {min_off:g} h before acquisition")
    hours = max(offsets) + 1
    reach = float(cfg.environment.get("open_meteo", {}).get("reach_speed_ms", 1.0))
    if cfg.environment.provider == "open_meteo":
        # size the forcing domain from the actual drift speed at the slick (2 API calls) instead of a fixed
        # worst case: typically 3-4x fewer grid points against Open-Meteo's per-location rate limit
        from app.environmental.forcing import probe_drift_speed
        try:
            pr = probe_drift_speed(spill["centroid"]["lat"], spill["centroid"]["lon"], obs - timedelta(hours=hours),
                                   obs + timedelta(hours=d.forward.hours))
            if pr["p95_ms"]:
                # mean transport over 49 h is well below the peak hour; p95 x 1.3 bounds it with margin, and the
                # out-of-domain guard below enlarges the domain automatically if particles still reach the edge
                reach = min(reach, max(0.15, 1.3 * pr["p95_ms"]))
                an.emit("ENVIRONMENTAL_DATA", f"Drift-speed probe: p95 {pr['p95_ms']:.2f} m/s (max {pr['max_ms']:.2f}) "
                                              f"-> forcing domain sized for {reach:.2f} m/s")
        except ForcingUnavailable as exc:
            if cfg.environment.get("mode", "auto") == "real":
                raise
            log.warning("drift-speed probe unavailable (%s); synthetic fallback may be used", exc.message)
        except Exception as exc:  # probe is an optimisation only
            log.warning("drift-speed probe failed: %s", exc)
    reach_cap = float(cfg.environment.get("open_meteo", {}).get("reach_speed_ms", 1.0)) * 2
    for attempt in range(2):
        bbox = _drift_bbox(spill, hours, reach)
        result = _drift_attempt(an, spill, obs, hours, bbox, offsets, window_basis, particles, members)
        if result is not None or attempt == 1:
            break
        reach = min(reach * 1.6, reach_cap)
        an.emit("ENVIRONMENTAL_DATA", f"Particles reached the forcing-domain edge: enlarging domain to {reach:.2f} m/s")
    if result is None:
        raise ForcingUnavailable("Particles left the forcing domain even after enlarging it.",
                                 "Increase environment.open_meteo.reach_speed_ms and rerun.")
    return result


def _drift_attempt(an, spill, obs, hours, bbox, offsets, window_basis, particles, members):
    """One forcing download + backward ensemble + source estimation. Returns None if particles left the domain."""
    from app.drift import backtrack as bt
    from app.drift import source_probability as sp
    from app.environmental.forcing import build_provider
    cfg, d = an.cfg, an.cfg.drift
    fmode = cfg.environment.get("mode", "auto")        # auto: real, else SYNTHETIC fallback | real | synthetic
    t_from, t_to = obs - timedelta(hours=hours), obs + timedelta(hours=d.forward.hours)
    provider = None
    if fmode != "synthetic":
        an.emit("ENVIRONMENTAL_DATA", f"Loading forcing provider '{cfg.environment.provider}'")
        try:
            provider = build_provider(cfg, bbox, t_from, t_to, progress=lambda msg: an.emit("ENVIRONMENTAL_DATA", msg))
            provider.check_coverage(spill["bbox"], obs - timedelta(hours=hours), obs)
        except ForcingUnavailable as exc:
            if fmode == "real":
                raise
            an.warn(f"Real environmental forcing unavailable ({exc.message}) — using SYNTHETIC Indian-waters "
                    "climatology forcing for demonstration.")
            provider = None
    if provider is None:
        from app.environmental.synthetic_forcing import synthetic_provider
        provider = synthetic_provider(bbox, t_from, t_to, an.repo.dir(an.id) / "forcing")
        an.warn("Environmental forcing is SYNTHETIC (climatology-inspired analytic currents and wind) — drift "
                "results illustrate the method only.")
        an.emit("ENVIRONMENTAL_DATA", f"SYNTHETIC forcing generated: wind from {provider.regime['wind_from_deg']}° "
                                      f"~{provider.regime['wind_ms']} m/s, current toward "
                                      f"{provider.regime['current_to_deg']}° ~{provider.regime['current_ms']} m/s")
    coverage_bbox = spill["bbox"]
    provider.check_coverage(coverage_bbox, obs - timedelta(hours=hours), obs)
    forcing = provider.describe()
    at_spill = provider.sample(spill["centroid"]["lat"], spill["centroid"]["lon"], obs)

    oil = {"oil_type": d.oil_type,
           "oil_model_assumption": d.oil_model_assumption if str(d.oil_type).upper() == "UNKNOWN" else d.oil_type}

    n = int(particles or d.particles)
    m = int(members or d.ensemble_runs)
    an.emit("BACKWARD_DRIFT", f"OpenDrift {d.model}: {m} members × {n} particles, {hours:.0f} h backward, "
                              f"dt={d.timestep_seconds}s")
    ens = bt.run_ensemble(provider, spill["polygon"], obs, hours, d, "backward", n, m,
                          progress=lambda k, tot: an.emit("BACKWARD_DRIFT", f"single OpenDrift run with {tot} members: "
                                                                 + ("started" if k == 0 else "finished")))
    ens = bt.fill_stranded(ens)
    # guard: particles must stay inside the forcing grid (outside it OpenDrift would use fallback values)
    cov = provider.describe()["currents"][0]
    lo_, la_ = ens.lon[np.isfinite(ens.lon)], ens.lat[np.isfinite(ens.lat)]
    outside = float(np.mean((lo_ < cov["lon"][0]) | (lo_ > cov["lon"][1]) | (la_ < cov["lat"][0]) | (la_ > cov["lat"][1])))
    if outside > 0.001:
        log.warning("%.2f%% of particle positions outside forcing domain", 100 * outside)
        return None
    np.savez_compressed(an.repo.artifact_path(an.id, "drift/backward_ensemble.npz"), lon=ens.lon, lat=ens.lat,
                        times=np.array([t.timestamp() for t in ens.times]))
    an.save("drift/backward_particles.json", _frames(ens))

    an.emit("SOURCE_ESTIMATION", "Kernel density of particle positions per release time; HDR regions")
    grid = sp.build_grid(ens, spill["centroid"], offsets, obs, d.probability_grid_resolution_m, d.kde_bandwidth_m)
    levels = dict(d.region_levels)
    regs = sp.regions(grid, levels)
    an.save("drift/source_probability.geojson", sp.to_geojson(grid, regs))
    stranded = int(sum((s == "stranded").sum() for s in ens.status))
    summary = {
        "spill_id": an.id,
        "observation_time": obs.isoformat(),
        "release_window": {"start": (obs - timedelta(hours=max(offsets))).isoformat(),
                           "end": (obs - timedelta(hours=min(offsets))).isoformat(),
                           "offsets_hours": offsets,
                           "prior": "uniform across offsets (ASSUMPTION)", "basis": window_basis},
        "per_release_time": sp.per_release_summary(grid, levels),
        "combined_center": sp.weighted_center(grid, "combined"),
        "source_regions": {k: {kk: vv for kk, vv in v.items() if kk != "_shape"} for k, v in regs.items()},
        "model": ens.model, "model_config": ens.config, "members": ens.members,
        "particles_per_member": n, "n_members": m, "timestep_seconds": d.timestep_seconds,
        "stranded_particles": stranded,
        "forcing": forcing, "forcing_at_spill": at_spill, "oil": oil,
        "weathering": "disabled for backward runs (evaporation/emulsification/dispersion/biodegradation are not "
                      "reversible); transport only",
        "notes": ["The source region is where the observed oil is estimated to have been at each candidate release "
                  "time; it is not an exact origin point.",
                  "Stokes drift not modelled explicitly (no wave forcing); partly represented by the wind drift factor range."],
    }
    an.save("drift/summary.json", summary)
    an.emit("SOURCE_ESTIMATION", f"High-probability region {regs['high']['area_km2']:.1f} km² "
                                 f"(50% mass); 95% region {regs.get('low', regs['high'])['area_km2']:.1f} km²")

    if d.forward.enabled:
        stage_forward(an, provider=provider)
    return summary



def stage_forward(an: Analysis, hours: float | None = None, particles: int | None = None, provider=None) -> dict:
    """Forward drift (future flow) from the observed slick, with OpenOil weathering enabled."""
    from app.drift import backtrack as bt
    from app.drift import source_probability as sp
    from app.environmental.forcing import build_provider

    cfg, d = an.cfg, an.cfg.drift
    try:
        spill = an.load("spill_selected.json")
    except FileNotFoundError:
        raise PipelineError("No selected slick available for forward drift.", "Run backtrack first.")
    obs = datetime.fromisoformat(spill["timestamp"])
    H = float(hours or d.forward.hours)
    n = int(particles or d.forward.particles)
    m = max(2, int(d.ensemble_runs) // 2)
    try:
        provider = provider or build_provider(cfg, _drift_bbox(spill, H), obs, obs + timedelta(hours=H))
        provider.check_coverage(spill["bbox"], obs, obs + timedelta(hours=H))
        forcing = provider.describe()
        an.emit("BACKWARD_DRIFT", f"Forward drift: {m} members × {n} particles, {H:.0f} h")
        fe = bt.fill_stranded(bt.run_ensemble(provider, spill["polygon"], obs, H, d, "forward", n, m, seed=7))
        an.save("drift/forward_particles.json", _frames(fe))
        label = f"T-{-H:g}h"
        fgrid = sp.build_grid(fe, spill["centroid"], [-H], obs, d.probability_grid_resolution_m, d.kde_bandwidth_m)
        fregs = sp.regions(fgrid, {"affected_95": 0.95}, label)
        fwd = {"available": True, "hours": H, "model": fe.model, "members": m, "particles_per_member": n,
               "forecast_window": [obs.isoformat(), (obs + timedelta(hours=H)).isoformat()],
               "affected_region": {k: {kk: vv for kk, vv in v.items() if kk != "_shape"} for k, v in fregs.items()},
               "end_center": sp.weighted_center(fgrid, label),
               "stranded_particles": int(sum((s == "stranded").sum() for s in fe.status)),
               "weathering": "enabled (OpenOil NOAA weathering) for forward prediction",
               "forcing_provenance": forcing["provenance"]}
    except ForcingUnavailable as exc:
        fwd = {"available": False, "reason": "Forward prediction unavailable: " + exc.message}
    an.save("drift/forward_summary.json", fwd)
    return fwd


# ======================================================================================================
# Stage 10-16: AIS → gaps → SAR ships → SAR/AIS → look-alike → multi-temporal → scoring
# ======================================================================================================
def _load_corridor(an: Analysis, summary: dict):
    from shapely.geometry import shape
    from app.ais.corridor_filter import Corridor
    from app.drift.backtrack import EnsembleResult
    from app.geospatial.polygon import local_projection
    spill = an.load("spill_selected.json")
    z = np.load(an.repo.artifact_path(an.id, "drift/backward_ensemble.npz"))
    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in z["times"]]
    ens = EnsembleResult("backward", times[0], times, z["lon"], z["lat"], [])
    fwd, inv = local_projection(spill["centroid"]["lon"], spill["centroid"]["lat"])
    regs = {k: {**v, "_shape": shape(v["geometry"])} for k, v in summary["source_regions"].items()}
    rw = summary["release_window"]
    window = (datetime.fromisoformat(rw["start"]), datetime.fromisoformat(rw["end"]))
    return Corridor.from_ensemble(ens, fwd, inv, window, regs), spill


def stage_ais_and_score(an: Analysis, ais_mode: str | None = None) -> dict:
    from app.ais import corridor_filter as cf
    from app.ais.gaps import data_quality
    from app.ais.retrieval import search_window
    from app.ais.track_processing import clean_and_build
    from app.environmental.forcing import build_provider
    from app.lookalike.analysis import assess
    from app.sar_ship_detection.detector import CFARShipDetector, geolocate, sar_ais_mismatch
    from app.scoring.evidence_report import build_candidate_evidence
    from app.scoring.features import compute_features
    from app.scoring.scorer import DISCLAIMER, SCORE_NAME, feature_scores, total_score

    cfg, a = an.cfg, an.cfg.ais
    summary = an.load("drift/summary.json")
    cor, spill = _load_corridor(an, summary)
    seg = an.load("spill.json")
    obs = datetime.fromisoformat(spill["timestamp"])
    low = summary["source_regions"].get("low") or summary["source_regions"]["high"]
    from shapely.geometry import shape
    rb = list(shape(low["geometry"]).bounds)

    an.emit("AIS_PROCESSING", "Retrieving historical AIS for source region + release window")
    from app.core.errors import AISDataError
    from app.ais.selection import select_ais
    bbox, t0, t1 = search_window(rb, spill["bbox"], obs, max(summary["release_window"]["offsets_hours"]),
                                 a.search_margin_km, a.time_margin_hours)
    rw = summary["release_window"]

    def corridor_point(t):
        """Median backtracked-oil position at time t (lat, lon) — used ONLY to place synthetic scenario vessels."""
        c = cor.cloud_at(t.timestamp())
        if not len(c):
            return None
        lon_, lat_ = cor.inv.transform(float(np.median(c[:, 0])), float(np.median(c[:, 1])))
        return float(lat_), float(lon_)
    try:
        syn_scene = an.load("acquisition.json").get("synthetic_scene")
    except FileNotFoundError:
        syn_scene = None
    context = {"spill_centroid": spill["centroid"], "observation_time": obs, "corridor_point": corridor_point,
               "synthetic_scene": syn_scene,
               "release_window": (datetime.fromisoformat(rw["start"]), datetime.fromisoformat(rw["end"])),
               "seed_key": an.id}
    ais_error, selection = None, None
    import pandas as pd
    from app.ais.provider import CANONICAL
    provider = None
    try:
        raw, provider, selection = select_ais(cfg, bbox, t0, t1, context, ais_mode,
                                              progress=lambda msg: an.emit("AIS_PROCESSING", msg))
    except AISDataError as exc:
        ais_error = exc.to_dict()
        an.warn(f"AIS unavailable: {exc.message}")
        raw = pd.DataFrame(columns=CANONICAL)
    synthetic = bool(provider is not None and provider.provenance == Provenance.DEMO_SYNTHETIC.value)
    if synthetic:
        an.warn("AIS data are SYNTHETIC (generated for the spill region to demonstrate the algorithm, as permitted "
                "by SIH26143) — they are NOT real vessels.")
        if selection and selection.get("fallback_reason"):
            an.warn(f"Real AIS unavailable ({selection['fallback_reason']}) — using SYNTHETIC AIS.")
        an.save("ais/synthetic_truth.json", getattr(provider, "truth", {}))
    # coarse hourly presence (GFW) needs a longer AIS-gap threshold than raw AIS
    if provider is not None and getattr(provider, "gap_threshold_minutes", None):
        a = type(a)({**a, "gap_threshold_minutes": max(a.gap_threshold_minutes, provider.gap_threshold_minutes)})
    tracks, cleaning = clean_and_build(raw, a.max_speed_knots)
    ais_info = {"provider": provider.describe() if provider else {"provider": "none", "provenance": "UNAVAILABLE"},
                "query": {"bbox": bbox, "start": t0.isoformat(), "end": t1.isoformat()},
                "cleaning": cleaning, "error": ais_error, "selection": selection, "synthetic": synthetic}
    an.emit("AIS_PROCESSING", f"{cleaning['vessels']} vessel track(s) after cleaning "
                              f"(dropped: {cleaning['duplicates']} duplicates, {cleaning['invalid_coordinates']} invalid "
                              f"coords, {cleaning['impossible_speed']} impossible speeds)")

    matches = cf.match_tracks(tracks, cor, a)
    an.emit("AIS_GAP_ANALYSIS", "Measuring AIS continuity anomalies")

    # ---- SAR ship detection (from triage) + SAR/AIS mismatch ------------------------------------
    try:
        triage = an.load("triage.json")
    except FileNotFoundError:
        triage = {"sar_ship_detection": {"available": False, "detections": []}, "components": [], "wind_at_acquisition": None}
    sar_ships, mismatch = triage["sar_ship_detection"], {"available": False}
    if sar_ships.get("available"):
        sd = cfg.sar_ship_detection
        an.emit("SAR_AIS_MATCHING", f"{sar_ships['n_detections']} SAR point target(s); matching to AIS at acquisition time")
        mismatch = {"available": True, **sar_ais_mismatch(sar_ships["detections"], tracks, obs,
                                                          seg["scene"]["georef"]["bounds_wgs84"],
                                                          sd.match_distance_km, sd.match_time_minutes)}
    an.emit("LOOKALIKE_ANALYSIS", "Look-alike indicators for the selected slick(s)")
    sel_ids = set(spill.get("selected_component_ids") or [])
    rows = [r for r in triage["components"] if r["component_id"] in sel_ids]
    labels = {r["label"] for r in rows}
    lookalike = {"label": rows[0]["label"] if len(labels) == 1 else ("UNCERTAIN" if rows else "UNCERTAIN"),
                 "components": rows, "reasons": [x for r in rows for x in r["reasons"]] or
                 ["No triage indicators available for the selected component(s)."],
                 "segmentation_confidence": spill.get("detection_confidence"),
                 "wind_at_acquisition": triage.get("wind_at_acquisition"),
                 "method": "rule-based indicators (SOS training data has no look-alike labels; not a trained classifier)",
                 "uncertainty": "Heuristic triage aid; the investigator verified/selected the slick."}
    an.emit("MULTI_TEMPORAL_ANALYSIS", "Single observation: multi-temporal analysis requires ≥2 associated scenes "
                                       "(use `python main.py compare`)")
    multitemporal = {"available": False, "reason": "Only one observation of this spill was supplied."}

    # ---- candidate scoring ------------------------------------------------------------------------
    an.emit("CANDIDATE_SCORING", "Computing evidence features and Evidence Correlation Score")
    weights = dict(cfg.scoring.weights)
    ctx = {"ais_provenance": provider.provenance if provider else "UNAVAILABLE", "synthetic_ais": synthetic,
           "corridor_buffer_km": a.corridor_buffer_km,
           "high_region_area_km2": summary["source_regions"]["high"]["area_km2"],
           "n_members": summary["n_members"], "n_particles": summary["particles_per_member"],
           "release_window": summary["release_window"], "forcing_provenance": summary["forcing"]["provenance"],
           "gap_threshold_min": a.gap_threshold_minutes, "georef_provenance": spill.get("georef_provenance"),
           "sar_ais_note": "untrained CFAR baseline; matching tolerance "
                           f"{cfg.sar_ship_detection.match_distance_km} km / {cfg.sar_ship_detection.match_time_minutes} min"}
    # vessels observed by SAR+AIS at the selected slick at acquisition time (strongest observable link)
    attached = {}
    if mismatch.get("available"):
        from shapely.geometry import Point, shape as _shape
        sel_geom = _shape(spill["polygon"])
        link_km = cfg.lookalike.get("ship_link_km", 1.0)
        dets = sar_ships["detections"]
        for mt in mismatch["matches"]:
            d = dets[mt["detection"]]
            dkm = sel_geom.distance(Point(d["lon"], d["lat"])) * 111.32 * np.cos(np.radians(d["lat"]))
            if dkm <= link_km:
                attached[mt["mmsi"]] = {"matched": True, "target_to_slick_km": float(dkm),
                                        "ais_to_target_km": mt["distance_km"], "ais_time_offset_min": mt.get("time_offset_min", 0.0),
                                        "detection": {"lat": d["lat"], "lon": d["lon"]}}
    ctx["sar_attached_available"] = bool(attached)
    cands, filtered = [], []
    by_mmsi = {t.mmsi: t for t in tracks}
    for mmsi, m in matches.items():
        t = by_mmsi[mmsi]
        if mmsi in attached and not m.is_candidate:
            m.is_candidate = True
            m.reason = "Observed by SAR and AIS at the selected slick at acquisition time"
        if not m.is_candidate:
            filtered.append({"mmsi": mmsi, "name": t.name, "vessel_type": t.vessel_type, "reason": m.reason,
                             "min_source_distance_km": m.min_d_region_km if np.isfinite(m.min_d_region_km) else None})
            continue
        f = compute_features(t, m, cor, cfg)
        f["sar_attached"] = attached.get(mmsi, {"matched": False})
        dq = data_quality(t)
        f["n_positions"], f["median_report_interval_s"] = dq["n_positions"], dq["median_report_interval_s"]
        fs = feature_scores(f, cfg)
        cands.append({"vessel": {"mmsi": mmsi, "imo": t.imo, "name": t.name, "type": t.vessel_type},
                      "synthetic": synthetic,
                      "score": total_score(fs, weights), "feature_scores": fs, "features": f, "data_quality": dq})
    cands.sort(key=lambda c: c["score"], reverse=True)
    cands = cands[: cfg.scoring.max_candidates]
    ctx["n_candidates"] = len(cands)
    for r, c in enumerate(cands, 1):
        c["rank"] = r
        c["evidence"] = build_candidate_evidence(c, ctx)
        c["feature_scores_pct"] = {k: round(100 * v, 1) for k, v in c["feature_scores"].items()}
        c["uncertainties"] = c["evidence"]["uncertainties"]
        mmsi = c["vessel"]["mmsi"]
        c["source_distance_km"] = c["features"]["source_distance_km"]
        c["time_difference_hours"] = c["features"]["temporal"].get("time_difference_hours")
    from app.ais.track_processing import track_feature
    rank_of = {c["vessel"]["mmsi"]: c["rank"] for c in cands}
    # display tracks: ~20 m tolerance for candidates, ~80 m for context traffic (analysis used full tracks)
    track_feats = [track_feature(t, 0.0002 if t.mmsi in rank_of else 0.0008, is_candidate=t.mmsi in rank_of,
                                 rank=rank_of.get(t.mmsi), synthetic=synthetic) for t in tracks]

    separation = None
    if len(cands) >= 2:
        gap = cands[0]["score"] - cands[1]["score"]
        separation = {"top_two_score_gap": gap, "separable": gap >= 10,
                      "note": (f"The top two candidates differ by {gap:.1f} points. Differences below ~10 points are "
                               "within the uncertainty of this uncalibrated heuristic: both should be investigated "
                               "with equal priority." if gap < 10 else
                               f"The top candidate leads by {gap:.1f} points; this reflects stronger evidence "
                               "correlation, not proof.")}
    if ais_error:
        message = f"Drift analysis completed, but AIS is unavailable: {ais_error['message']}"
    elif synthetic and cands:
        message = f"{len(cands)} SYNTHETIC candidate vessel(s) — demonstration only, not real vessels"
    elif not tracks:
        message = "Drift analysis completed, but no AIS tracks were available for the requested region/time."
    elif not cands:
        message = "No vessels matched the configured spatial and temporal candidate criteria."
    else:
        message = f"{len(cands)} candidate vessel(s) requiring investigation"
    result = {"score_name": SCORE_NAME, "disclaimer": DISCLAIMER, "weights": weights, "message": message,
              "candidates": cands, "ranking_separation": separation, "filtered_out": filtered, "ais": ais_info,
              "sar_ship_detection": sar_ships, "sar_ais_mismatch": mismatch, "lookalike": lookalike,
              "multitemporal": multitemporal, "candidate_criteria": {
                  "max_source_distance_km": a.max_source_distance_km, "corridor_buffer_km": a.corridor_buffer_km,
                  "release_window": summary["release_window"]}}
    an.save("ais/tracks.geojson", {"type": "FeatureCollection", "features": track_feats})
    an.save("candidates.json", result)
    an.emit("CANDIDATE_SCORING", message)
    return result


def stage_impact(an: Analysis) -> dict | None:
    """Severity index, shoreline/receptor threat, response plan, next-port intercepts, ranking robustness and a
    POLREP draft. Advisory: a failure here never fails the investigation."""
    from app.impact.service import assess_impact
    an.emit("REPORT", "Assessing impact: severity index, shoreline threat, response plan")
    try:
        imp = assess_impact(an)
        sev = imp["severity"]
        an.emit("REPORT", f"Spill Severity & Impact Index {sev['score']:.0f}/100 ({sev['level']}); "
                          f"{len(imp['response']['actions'])} response actions suggested")
        return imp
    except Exception as exc:
        log.warning("impact assessment failed for %s: %s", an.id, exc, exc_info=True)
        an.warn(f"Impact assessment unavailable: {exc}")
        return None


def stage_seal(an: Analysis) -> dict | None:
    """Tamper-evident SHA-256 manifest of all evidence artifacts (chain of custody)."""
    from app.impact.service import seal_evidence
    try:
        m = seal_evidence(an)
        an.emit("REPORT", f"Evidence sealed: {m['n_files']} files, SHA-256 root {m['root'][:12]}…")
        return m
    except Exception as exc:
        log.warning("evidence seal failed for %s: %s", an.id, exc)
        return None


def stage_report(an: Analysis) -> dict:
    from app.reports.report import build_report, render_html, render_markdown
    an.emit("REPORT", "Generating investigator report")
    rep = build_report(an)
    an.save("report.json", rep)
    an.repo.artifact_path(an.id, "report.md").write_text(render_markdown(rep), encoding="utf-8")
    html_path = an.repo.artifact_path(an.id, "report.html")
    html_path.write_text(render_html(rep), encoding="utf-8")
    an.repo.index_results(an.id, {"spill": rep.get("spill"), "source_regions": (rep.get("drift") or {}).get("source_regions"),
                                  "candidates": rep.get("candidates"), "report_path": str(html_path)})
    return rep


# ======================================================================================================
def _guard(an: Analysis, fn):
    """Run fn() mapping failures onto the state machine with actionable messages."""
    try:
        fn()
    except PipelineError as exc:
        an.state["error"] = exc.to_dict()
        an.emit(an.state.get("stage", "QUEUED"), exc.message, status="FAILED", hint=exc.hint)
    except Exception as exc:  # unexpected: keep traceback for debugging
        an.state["error"] = {"code": "UNEXPECTED", "message": str(exc), "hint": "See server log.",
                             "traceback": traceback.format_exc(limit=6)}
        an.emit(an.state.get("stage", "QUEUED"), f"Unexpected error: {exc}", status="FAILED")
        log.exception("analysis %s failed", an.id)


def stage_acquire(an: Analysis, scene_id: str, aoi: list[float], resolution_m: float,
                  synthetic_time: datetime | None = None) -> str:
    from app.sar.sentinel1 import clip_aoi_to_scene, ingest_scene
    if scene_id == "SYNTHETIC":
        from app.sar.synthetic_scene import generate_scene
        t = synthetic_time or (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=0, minute=31, second=0,
                                                                                         microsecond=0)
        an.emit("SCENE_ACQUISITION", "No usable real Sentinel-1 scene selected: generating a SYNTHETIC SAR scene for "
                                     f"AOI {[round(v, 3) for v in aoi]} (demonstration data, not an observation)")
        an.warn("SAR scene is SYNTHETIC (generated for demonstration) — it is not a satellite observation.")
        info = generate_scene(aoi, t, an.repo.dir(an.id) / "scene", max(resolution_m, 20.0),
                              seed=int(__import__("hashlib").sha256(an.id.encode()).hexdigest()[:8], 16),
                              norm=dict(an.cfg.sentinel1.get("normalization", {})),
                              progress=lambda msg: an.emit("SCENE_ACQUISITION", msg))
        an.state["scene_request"] = {"scene_id": "SYNTHETIC", "aoi": aoi, "resolution_m": info["resolution_m"]}
        an.save("acquisition.json", info)
        an.emit("SCENE_ACQUISITION", f"SYNTHETIC scene ready: {info['size'][0]}x{info['size'][1]} px, "
                                     f"sea {100 * info['sea_fraction']:.0f}%")
        return info["path"]
    clipped, cover = clip_aoi_to_scene(scene_id, aoi)
    if clipped != aoi:
        an.warn(f"The selected real Sentinel-1 scene covers only {100 * cover:.0f}% of the drawn area; the analysis "
                "is limited to the covered part.")
        an.emit("SCENE_ACQUISITION", f"Scene covers {100 * cover:.0f}% of the AOI: area clipped to "
                                     f"{[round(v, 3) for v in clipped]}")
        aoi = clipped
    an.emit("SCENE_ACQUISITION", f"Sentinel-1 {scene_id}: AOI {[round(v, 3) for v in aoi]} at {resolution_m:g} m")
    info = ingest_scene(scene_id, aoi, an.repo.dir(an.id) / "scene", resolution_m,
                        progress=lambda msg: an.emit("SCENE_ACQUISITION", msg),
                        norm=dict(an.cfg.sentinel1.get("normalization", {})))
    an.state["scene_request"] = {"scene_id": scene_id, "aoi": aoi, "resolution_m": resolution_m}
    an.save("acquisition.json", info)
    an.emit("SCENE_ACQUISITION", f"Scene ready: {info['size'][0]}x{info['size'][1]} px, sea {100 * info['sea_fraction']:.0f}%")
    return info["path"]


def run_detection(cfg: Config, input_path: str | None = None, geo_override: dict | None = None,
                  analysis_id: str | None = None, on_event: ProgressFn | None = None, threshold: float | None = None,
                  scene_id: str | None = None, aoi: list[float] | None = None, resolution_m: float = 10.0,
                  repo=None, synthetic_time: datetime | None = None) -> Analysis:
    """Phase 1: (acquire Sentinel-1) -> segment -> polygons -> SAR targets -> triage -> AWAITING_SLICK_SELECTION."""
    an = Analysis(cfg, analysis_id, on_event, repo)
    an.emit("QUEUED", "Detection queued")

    def go():
        path = stage_acquire(an, scene_id, aoi, resolution_m, synthetic_time) if scene_id else input_path
        seg = stage_segment(an, path, geo_override, threshold)
        if not seg.get("spill") or an.state.get("outcome") != "AWAITING_SLICK_SELECTION":
            stage_report(an)
            an.emit("REPORT", "Report generated", status=an.state.get("outcome", "PARTIAL"))
    _guard(an, go)
    return an


def run_investigation(cfg: Config, analysis_id: str, component_ids: list[str] | None = None,
                      particles: int | None = None, members: int | None = None, on_event: ProgressFn | None = None,
                      repo=None, ais_mode: str | None = None) -> Analysis:
    """Phase 2: selected slick(s) -> forcing -> backward/forward drift -> AIS -> scoring -> report."""
    an = Analysis(cfg, analysis_id, on_event, repo)
    an.state.pop("error", None)

    def go():
        select_components(an, component_ids)
        try:
            stage_drift(an, particles, members)
        except (ForcingUnavailable, PipelineError) as exc:
            if isinstance(exc, ForcingUnavailable):
                exc.message = "Backward drift unavailable because environmental forcing data are unavailable. " + exc.message
            an.state["error"] = exc.to_dict()
            an.state["outcome"] = "PARTIAL"
            stage_report(an)
            an.emit("REPORT", exc.message, status="PARTIAL")
            return
        stage_ais_and_score(an, ais_mode)
        if an.cfg.get("impact", {}).get("enabled", True):
            stage_impact(an)
        an.state["outcome"] = "COMPLETED"
        stage_report(an)
        stage_seal(an)
        an.emit("COMPLETED", "Investigation complete", status="COMPLETED")
    _guard(an, go)
    return an


def run_full(cfg: Config, input_path: str | None = None, geo_override: dict | None = None,
             analysis_id: str | None = None, on_event: ProgressFn | None = None, threshold: float | None = None,
             particles: int | None = None, members: int | None = None, repo=None, component_ids=None,
             scene_id: str | None = None, aoi: list[float] | None = None, resolution_m: float = 20.0,
             ais_mode: str | None = None) -> Analysis:
    """Both phases (CLI convenience). Without component_ids the top triage-ranked slick is investigated."""
    an = run_detection(cfg, input_path, geo_override, analysis_id, on_event, threshold, scene_id, aoi,
                       resolution_m, repo)
    if an.state.get("status") != "AWAITING_SLICK_SELECTION":
        return an
    return run_investigation(cfg, an.id, component_ids, particles, members, on_event, repo, ais_mode)
