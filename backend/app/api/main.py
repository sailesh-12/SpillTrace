"""FastAPI application — the investigator backend (real data).

Run:  uvicorn app.api.main:app --app-dir backend --port 8000
All endpoints delegate to app.services.pipeline (same code path as the CLI).

Real-data workflow:
  GET  /api/v1/sources                          which data sources are active / need keys
  GET  /api/v1/scenes?bbox=&start=&end=         Sentinel-1 search (Planetary Computer, no key)
  POST /api/v1/analyses                         {scene_id, aoi, resolution_m, threshold} -> phase 1 job
  POST /api/v1/analyses/upload                  multipart GeoTIFF (+ optional sidecar) -> phase 1 job
  GET  /api/v1/analyses/{id}/events             SSE progress
  GET  /api/spill/{id}/triage                   slick triage table (after phase 1)
  POST /api/v1/analyses/{id}/investigate        {component_ids, particles, members} -> phase 2 job
Long jobs run in one background worker thread (GPU + OpenDrift are the bottleneck).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.schemas import AnalysisCreated, BacktrackRequest, CandidateOut, DriftOutput, ForwardRequest
from app.core.config import PROJECT_ROOT, get_config
from app.core.errors import PipelineError
from app.core.sources import load_dotenv, status as sources_status
from app.services import pipeline as P

load_dotenv()
log = logging.getLogger("sih.api")
logging.basicConfig(level=logging.INFO)
cfg = get_config()
app = FastAPI(title="SIH26143 Oil Spill Investigation API", version="2.0.0",
              description="Real Sentinel-1 → detection → triage → hindcast → AIS correlation → evidence report. "
                          "Candidate rankings are decision support, not attribution of responsibility.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
_queue: list[str] = []            # analysis ids queued or running, in execution order (single worker)
import threading as _threading
_qlock = _threading.Lock()


def _submit(aid: str, fn, *args):
    """Queue one job per analysis; duplicates are refused (repeated clicks used to queue the same job 3x)."""
    with _qlock:
        if aid in _queue:
            raise HTTPException(409, f"Analysis {aid} is already queued or running "
                                     f"(position {_queue.index(aid) + 1} of {len(_queue)}).")
        _queue.append(aid)

    def run():
        try:
            fn(*args)
        finally:
            with _qlock:
                if aid in _queue:
                    _queue.remove(aid)
    executor.submit(run)


def _warm_up():
    """Pay one-time costs at startup instead of inside the first investigation:
    OpenDrift/OpenOil import (~30 s), GSHHG landmask build (~20 s), model load (~5 s)."""
    import time
    t = time.time()
    try:
        from opendrift.models.openoil import OpenOil  # noqa: F401
        from opendrift.readers.reader_global_landmask import get_mask
        get_mask()
        from app.segmentation.model_loader import load_model
        load_model(cfg.path(cfg.segmentation.model_path), cfg.segmentation.encoder_name, cfg.segmentation.device)
        from app.impact.coast import _world       # GSHHG polygons for the impact assessment (~4 s)
        _world()
        log.info("warm-up done in %.0f s", time.time() - t)
    except Exception as exc:  # warm-up is an optimisation only
        log.warning("warm-up failed: %s", exc)


def _mark_interrupted():
    """Analyses left mid-stage by a previous server process can never finish: mark them FAILED with a clear
    message instead of showing a stage that looks 'stuck' forever."""
    repo = P.build_repository(cfg)
    for a in repo.list_analyses():
        if a.get("status") not in P.TERMINAL:
            an = P.Analysis(cfg, a["analysis_id"], repo=repo)
            an.state["error"] = {"code": "INTERRUPTED", "message": "The server restarted while this analysis was "
                                 f"running ({a.get('stage')}).", "hint": "Start the step again from the dashboard."}
            an.emit(an.state.get("stage", "QUEUED"), an.state["error"]["message"], status="FAILED")


_mark_interrupted()
executor.submit(_warm_up)      # first job on the single worker, so analyses never race it
UPLOADS = cfg.path(cfg.storage.output_dir) / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
ARTIFACTS = {"scene.jpg", "scene.png", "mask.png", "probability.png", "report.html", "report.md", "report.json", "spill.geojson",
             "spill.json", "spill_selected.geojson", "triage.json", "candidates.json", "acquisition.json",
             "drift/summary.json", "drift/forward_summary.json", "drift/backward_particles.json",
             "drift/forward_particles.json", "drift/source_probability.geojson", "ais/tracks.geojson",
             "impact.json", "evidence_manifest.json"}


class SceneAnalysisRequest(BaseModel):
    scene_id: str = Field(..., description='Sentinel-1 item id, or "SYNTHETIC" for a labelled synthetic demo scene')
    synthetic_time: Optional[str] = Field(None, description="ISO time for a SYNTHETIC scene (default: 2 days ago 00:31 UTC)")
    aoi: list[float] = Field(..., min_length=4, max_length=4, description="[west, south, east, north]")
    resolution_m: float = Field(20.0, ge=10, le=100)
    threshold: Optional[float] = Field(None, gt=0, lt=1)


class InvestigateRequest(BaseModel):
    component_ids: list[str] = []
    ais_mode: Optional[str] = Field(None, pattern="^(auto|real|synthetic)$",
                                    description="auto: real AIS if available else SYNTHETIC; real; synthetic")
    particles: Optional[int] = Field(None, ge=10, le=20000)
    members: Optional[int] = Field(None, ge=1, le=50)


def _analysis(aid: str) -> P.Analysis:
    an = P.Analysis(cfg, aid)
    if not an.repo.exists(aid):
        raise HTTPException(404, f"Unknown analysis/spill id '{aid}'")
    return an


def _load(aid: str, name: str):
    try:
        return _analysis(aid).load(name)
    except FileNotFoundError:
        raise HTTPException(404, f"{name} not available for {aid} (stage not run or unavailable)")


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_, exc: PipelineError):
    return JSONResponse(status_code=422, content={"error": exc.to_dict()})


# ---------------------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    import torch
    return {"status": "ok", "mode": cfg.get("mode"), "cuda": torch.cuda.is_available(),
            "model_present": cfg.path(cfg.segmentation.model_path).exists(), "storage": cfg.storage.backend,
            "forcing_provider": cfg.environment.provider, "ais_mode": cfg.ais.get("mode", "auto"),
            "ais_real_provider": cfg.ais.get("real_provider", "auto")}


@app.get("/api/v1/config")
def public_config():
    return {"segmentation": {k: cfg.segmentation[k] for k in ("threshold", "min_component_area_pixels", "tile_size")},
            "sentinel1": dict(cfg.sentinel1),
            "drift": {k: cfg.drift[k] for k in ("particles", "ensemble_runs", "release_offsets_hours", "oil_type",
                                                "oil_model_assumption", "timestep_seconds")},
            "ais": {k: cfg.ais.get(k) for k in ("mode", "real_provider", "corridor_buffer_km",
                                                "max_source_distance_km", "gap_threshold_minutes")},
            "scoring": {"weights": dict(cfg.scoring.weights)}, "stages": P.STAGES}


@app.get("/api/v1/sources")
def sources():
    return sources_status()


@app.get("/api/v1/scenes")
def scenes(bbox: str, start: str, end: str):
    from app.sar.sentinel1 import search_scenes
    try:
        b = [float(v) for v in bbox.split(",")]
        assert len(b) == 4 and b[0] < b[2] and b[1] < b[3]
    except Exception:
        raise HTTPException(400, "bbox must be 'west,south,east,north'")
    return search_scenes(b, start, end)


# ---------------------------------------------------------------------------------------------------
# Phase 1: detection jobs
# ---------------------------------------------------------------------------------------------------
@app.post("/api/v1/analyses", response_model=AnalysisCreated, status_code=202)
def create_scene_analysis(req: SceneAnalysisRequest):
    w, s, e, n = req.aoi
    if not (w < e and s < n):
        raise HTTPException(400, "aoi must be [west, south, east, north]")
    if max(e - w, n - s) > cfg.sentinel1.get("max_aoi_deg", 1.5):
        raise HTTPException(400, f"AOI too large (max {cfg.sentinel1.get('max_aoi_deg', 1.5)}° per side)")
    an = P.Analysis(cfg)
    an.emit("QUEUED", f"Queued: Sentinel-1 {req.scene_id}")
    from datetime import datetime as _dt
    syn_t = _dt.fromisoformat(req.synthetic_time.replace("Z", "+00:00")) if req.synthetic_time else None
    an.emit("QUEUED", "SYNTHETIC SAR scene requested (demonstration)" if req.scene_id == "SYNTHETIC"
            else f"Real Sentinel-1 scene {req.scene_id}")
    _submit(an.id, P.run_detection, cfg, None, None, an.id, None, req.threshold, req.scene_id, req.aoi,
            req.resolution_m, None, syn_t)
    return AnalysisCreated(analysis_id=an.id, status="QUEUED")


@app.post("/api/v1/analyses/upload", response_model=AnalysisCreated, status_code=202)
def create_upload_analysis(file: UploadFile = File(...), georef: Optional[UploadFile] = File(None),
                           bbox: Optional[str] = Form(None), timestamp: Optional[str] = Form(None),
                           threshold: Optional[float] = Form(None)):
    d = UPLOADS / uuid.uuid4().hex[:10]
    d.mkdir(parents=True)
    dst = d / Path(file.filename or "scene.tif").name
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if georef is not None:
        with open(d / (dst.name + cfg.geospatial.sidecar_suffix), "wb") as f:
            shutil.copyfileobj(georef.file, f)
    geo = {}
    if bbox:
        try:
            geo["bbox"] = [float(v) for v in bbox.split(",")]
        except ValueError:
            raise HTTPException(400, "bbox must be 'west,south,east,north'")
        geo["provenance"] = "LOCAL_FILE"
    if timestamp:
        geo["timestamp"] = timestamp
    an = P.Analysis(cfg)
    an.emit("QUEUED", f"Queued: upload {dst.name}")
    _submit(an.id, P.run_detection, cfg, str(dst), geo or None, an.id, None, threshold)
    return AnalysisCreated(analysis_id=an.id, status="QUEUED")


# Phase 2: investigation of selected slick(s)
@app.post("/api/v1/analyses/{aid}/investigate", response_model=AnalysisCreated, status_code=202)
def investigate(aid: str, req: InvestigateRequest):
    an = _analysis(aid)
    with _qlock:
        if aid in _queue:
            raise HTTPException(409, f"This investigation is already queued or running "
                                     f"(position {_queue.index(aid) + 1} of {len(_queue)}).")
    if not an.load("spill.json").get("spill"):
        raise HTTPException(422, "No georeferenced slick polygons; drift/AIS analysis impossible.")
    ahead = len(_queue)
    an.emit("ENVIRONMENTAL_DATA", f"Investigation queued for {len(req.component_ids) or 'top-ranked'} slick(s)"
            + (f" — waiting for {ahead} other analysis job(s) to finish" if ahead else ""))
    _submit(aid, P.run_investigation, cfg, aid, req.component_ids or None, req.particles, req.members, None, None,
            req.ais_mode)
    return AnalysisCreated(analysis_id=aid, status="QUEUED")


@app.get("/api/v1/queue")
def queue():
    """Jobs in execution order; the first one is running (single analysis worker)."""
    with _qlock:
        return {"running": _queue[0] if _queue else None, "waiting": _queue[1:]}


@app.get("/api/v1/analyses")
def list_analyses():
    return P.build_repository(cfg).list_analyses()


@app.get("/api/v1/analyses/{aid}")
def get_analysis(aid: str):
    return _analysis(aid).state


@app.get("/api/v1/analyses/{aid}/events")
async def analysis_events(aid: str, since: int = 0):
    """Server-Sent Events: replays events from `since`, then streams until a terminal/awaiting state."""
    _analysis(aid)
    repo = P.build_repository(cfg)

    async def gen():
        sent, idle = since, 0
        while True:
            try:
                st = repo.load_state(aid)
            except (FileNotFoundError, json.JSONDecodeError):
                await asyncio.sleep(0.3)
                continue
            evs = st.get("events", [])
            for ev in evs[sent:]:
                yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
            sent = len(evs)
            if st.get("status") in P.TERMINAL:
                yield f"event: done\ndata: {json.dumps({'status': st['status'], 'error': st.get('error'), 'n': sent})}\n\n"
                return
            idle += 1
            if idle % 30 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------------------------------
# Stage endpoints (spec §35) and results
# ---------------------------------------------------------------------------------------------------
@app.post("/api/drift/backtrack", response_model=DriftOutput)
def drift_backtrack(req: BacktrackRequest):
    an = _analysis(req.spill_id)
    s = executor.submit(P.stage_drift, an, req.particles, req.members).result()
    base = f"/api/spill/{req.spill_id}"
    return DriftOutput(spill_id=req.spill_id, release_window=s["release_window"], source_region=s["source_regions"],
                       probability_map=f"{base}/source-probability", particles=f"{base}/particles",
                       uncertainties=s["notes"] + [f"Forcing provenance: {s['forcing']['provenance']}"])


@app.post("/api/drift/forward")
def drift_forward(req: ForwardRequest):
    return executor.submit(P.stage_forward, _analysis(req.spill_id), req.hours, req.particles).result()


@app.get("/api/spill/{sid}")
def get_spill(sid: str):
    return _load(sid, "spill.json")


@app.get("/api/spill/{sid}/triage")
def get_triage(sid: str):
    return _load(sid, "triage.json")


@app.get("/api/spill/{sid}/source-probability")
def get_source_probability(sid: str):
    return {"summary": _load(sid, "drift/summary.json"), "geojson": _load(sid, "drift/source_probability.geojson")}


@app.get("/api/spill/{sid}/particles")
def get_particles(sid: str, direction: str = "backward"):
    if direction not in ("backward", "forward"):
        raise HTTPException(400, "direction must be backward|forward")
    return _load(sid, f"drift/{direction}_particles.json")


@app.get("/api/spill/{sid}/vessels", response_model=list[CandidateOut])
def get_vessels(sid: str):
    r = _load(sid, "candidates.json")
    return [CandidateOut(mmsi=c["vessel"]["mmsi"], imo=c["vessel"].get("imo"), vessel_name=c["vessel"].get("name"),
                         vessel_type=c["vessel"]["type"], evidence_score=c["score"], feature_scores=c["feature_scores_pct"],
                         source_distance_km=c.get("source_distance_km"), time_difference_hours=c.get("time_difference_hours"),
                         evidence_summary=c["evidence"]["summary"], uncertainties=c["uncertainties"], evidence=c["evidence"])
            for c in r["candidates"]]


@app.get("/api/spill/{sid}/report")
def get_report(sid: str, format: str = "json"):
    an = _analysis(sid)
    if format == "json":
        return _load(sid, "report.json")
    p = an.repo.dir(sid) / f"report.{'html' if format == 'html' else 'md'}"
    if not p.exists():
        raise HTTPException(404, "Report not generated yet")
    if format == "html":
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown",
                             headers={"Content-Disposition": f"attachment; filename={sid}_report.md"})


@app.get("/api/spill/{sid}/layers")
def get_layers(sid: str):
    """Everything the dashboard needs in one call (missing layers are null)."""
    an = _analysis(sid)
    base = f"/api/spill/{sid}/files"

    def opt(name):
        try:
            return an.load(name)
        except FileNotFoundError:
            return None
    seg = opt("spill.json") or {}
    return {"analysis": an.state, "scene": seg.get("scene"), "segmentation": seg.get("segmentation"),
            "spill": seg.get("spill"), "acquisition": opt("acquisition.json"), "triage": opt("triage.json"),
            "selected": opt("spill_selected.json"), "selected_geojson": opt("spill_selected.geojson"),
            "sar_image_url": f"{base}/scene.jpg" if (an.repo.dir(sid) / "scene.jpg").exists() else f"{base}/scene.png", "mask_url": f"{base}/mask.png", "probability_url": f"{base}/probability.png",
            "spill_geojson": opt("spill.geojson"), "drift": opt("drift/summary.json"),
            "forward": opt("drift/forward_summary.json"), "source_probability": opt("drift/source_probability.geojson"),
            "backward_particles": opt("drift/backward_particles.json"), "forward_particles": opt("drift/forward_particles.json"),
            "ais_tracks": opt("ais/tracks.geojson"), "candidates": opt("candidates.json"), "impact": opt("impact.json"),
            "evidence_seal": _seal_summary(an),
            "report_html_url": f"/api/spill/{sid}/report?format=html", "report_md_url": f"/api/spill/{sid}/report?format=md"}


def _seal_summary(an):
    try:
        m = an.load("evidence_manifest.json")
    except FileNotFoundError:
        return None
    return {k: m.get(k) for k in ("root", "chain", "version", "sealed_at", "n_files", "algorithm")}


# ---- novelty: impact & response, evidence integrity ---------------------------------------------------
@app.post("/api/spill/{sid}/impact")
def recompute_impact(sid: str):
    """(Re)compute the severity index, shoreline threat, response plan, intercepts and robustness; re-seal."""
    an = _analysis(sid)
    with _qlock:
        if sid in _queue:
            raise HTTPException(409, "This analysis is still running; impact is computed automatically at the end.")
    from app.impact.service import assess_impact, seal_evidence
    try:
        imp = assess_impact(an)
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc))
    seal_evidence(an)
    return imp


@app.get("/api/spill/{sid}/verify")
def verify_evidence(sid: str):
    """Recompute SHA-256 hashes of all artifacts and compare with the sealed manifest."""
    from app.impact.custody import verify
    an = _analysis(sid)
    return verify(an.repo.dir(sid))


@app.get("/api/spill/{sid}/polrep")
def get_polrep(sid: str):
    imp = _load(sid, "impact.json")
    return PlainTextResponse(imp["polrep"], headers={"Content-Disposition": f"attachment; filename={sid}_POLREP_draft.txt"})


@app.get("/api/spill/{sid}/files/{name:path}")
def get_file(sid: str, name: str):
    if name not in ARTIFACTS:
        raise HTTPException(404, "Unknown artifact")
    p = _analysis(sid).repo.dir(sid) / name
    if not p.exists():
        raise HTTPException(404, f"{name} not available")
    return FileResponse(p)


_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="dashboard")
