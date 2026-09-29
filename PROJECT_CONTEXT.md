# PROJECT_CONTEXT — SIH26143 oil-spill investigation system (checkpoint)

Hand-over notes so a new developer/AI can continue from this snapshot. Read with `README.md`.

## Problem
Smart India Hackathon 2026, problem **SIH26143**: detect marine oil spills in satellite SAR imagery, characterise
them, **hindcast** the slick back to a probable source region/time (and forecast forward), then correlate with
**historical AIS** vessel tracks to rank *candidate* vessels with explainable evidence. Output is decision support —
never "vessel X caused the spill". Scores are an uncalibrated **Evidence Correlation Score**, not a probability.

## Current state (what works)
- Runs on **real, key-free data**: Sentinel-1 GRD (Microsoft Planetary Computer STAC), Open-Meteo marine
  currents + ERA5/forecast wind, Danish Maritime Authority (DMA) historical AIS, GSHHG coastline.
- **Target region: Indian waters.** AIS source per investigation (`ais.mode`): `auto` = real AIS if available
  (Global Fishing Watch hourly presence with `GFW_API_TOKEN`; DMA for Danish scenes; local files), otherwise
  **SYNTHETIC AIS** generated around the spill region; `real` = never synthetic; `synthetic` = always. SIH26143 permits
  synthetic AIS: "Real AIS if available may be used else synthetic data can be prepared for the region of oil spill to
  demonstrate the functioning of the algorithm." Synthetic data is labelled SYNTHETIC everywhere (see README §4b).
- End-to-end verified via API/UI on real scenes: fresh area ≈ 10 min (mostly one-time AIS day downloads,
  ~0.5–0.7 GB each); repeat investigation in a cached area ≈ 1–1.5 min. 54 tests pass (offline fixtures).

## Architecture (modular monolith)
```
frontend/ React 19 + Vite + TS + Tailwind v4 + shadcn/ui (Base UI) + MapLibre   (UI only, no science)
   │ REST + SSE
backend/app/api/main.py  FastAPI; single background worker thread; job queue (duplicate jobs refused, /api/v1/queue);
   │                     startup warm-up (OpenDrift import, landmask, model); interrupted jobs marked FAILED on restart
backend/app/services/pipeline.py  the ONLY orchestrator (CLI main.py uses the same functions)
   ├─ sar/sentinel1.py, sar/cogread.py   STAC search; AOI extraction via GCPs; parallel COG tile reads (HTTP/1.1,
   │                                     timeouts); land mask; calibrated radiometric normalisation (sos_normalize)
   ├─ segmentation/ (model_loader, inference, postprocess)   DeepLabV3+/ResNet34, sliding window, cropped components
   ├─ geospatial/ (georeference, polygon)                    GeoTIFF/sidecar georef, WGS84 polygons, geodesic area
   ├─ sar_ship_detection/detector.py                         CA-CFAR point targets + SAR/AIS matching
   ├─ lookalike/triage.py                                    per-slick OIL_LIKELY/LOOKALIKE_LIKELY/UNCERTAIN indicators
   ├─ environmental/forcing.py   NetCDF + Open-Meteo providers (probe-sized domain, snapped cache reuse, pacing,
   │                             fail-fast on hourly quota), cached OpenDrift readers
   ├─ drift/ (opendrift_runner, backtrack, source_probability)  OpenOil; ALL ensemble members in ONE run;
   │                             backward (weathering off) + forward 24 h; KDE + 50/80/95 % HDR source regions
   ├─ ais/ selection.py (real-vs-synthetic choice) · gfw_provider.py (GFW 4Wings, Indian waters) ·
   │       synthetic_provider.py (Indian-waters presets, MMSI 419…, SYN- names, scenario roles) · dma_provider.py ·
   │       local_provider.py · track_processing, corridor_filter, gaps, retrieval
   │                             DMA daily zip -> Parquet cache; Arrow-side filtering/thinning; vectorised cleaning;
   │                             spatio-temporal corridor matching vs time-resolved particle cloud (vectorised)
   ├─ scoring/ (features, scorer, evidence_report)   feature scores, weighted score, epistemically-labelled statements
   ├─ reports/report.py   JSON/Markdown/HTML report       ├─ database/ File repo (default) + PostGIS repo (untested)
```

## Pipeline / state machine
Phase 1: `QUEUED → SCENE_ACQUISITION → SCENE_INSPECTION → GEOREFERENCING → PREPROCESSING → SEGMENTATION →
POLYGONIZATION → SAR_SHIP_DETECTION → LOOKALIKE_ANALYSIS → AWAITING_SLICK_SELECTION` (investigator ticks slick(s)).
Phase 2: `ENVIRONMENTAL_DATA → BACKWARD_DRIFT → SOURCE_ESTIMATION → AIS_PROCESSING → AIS_GAP_ANALYSIS →
SAR_AIS_MATCHING → CANDIDATE_SCORING → REPORT → COMPLETED | PARTIAL | FAILED`.
Key rules: release window offsets `[0,0.5,1,3,6,12,24,36,48] h`; starts at T only for ship-trail slicks, otherwise
≥ 1 h (assumption); SAR-attached vessel (AIS matched to SAR target touching slick) = strongest evidence (15 % weight).
Weights: spatial .25, temporal .25, trajectory .15, sar_attached .15, vessel_type .07, behaviour .08, continuity .05.

## Model
`models/best_oil_spill_deeplabv3_resnet34.pth` (Git LFS, 258 MB; torch zip format; includes optimizer state).
`smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=1)`, trained on the Refined Deep-SAR Oil Spill
(SOS) dataset (notebook: `models/training_notebook.ipynb`); input RGB/255 (no mean/std), 256×256 tiles, output logit
→ sigmoid → 0.5. Val Dice 0.746 / IoU 0.640 at epoch 3. Real Sentinel-1 must be normalised to SOS-like statistics
(`configs/config.yaml → sentinel1.normalization`: background grey 150, 40 grey per speckle σ, 10 km p60 background),
calibrated with `scripts/calibrate_radiometry.py` (clean-sea FP 15 % → 0.02 %; implanted −6 dB slicks 100 % recall).

## Important files
`configs/config.yaml` (every threshold/weight) · `backend/app/services/pipeline.py` · `backend/app/api/main.py` ·
`main.py` (CLI) · `frontend/src/App.tsx`, `frontend/src/components/{MapView,panels}.tsx` · `scripts/scan_real_slicks.py`
(shortlist real scenes) · `scripts/calibrate_radiometry.py` · `tests/` (fixtures in `tests/fixtures/`, app never reads them)
· `docs/ARCHITECTURE_AND_CONCEPTS.md`, `docs/SIH_PRESENTATION.md`.

## How to run
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or cu128/cpu
pip install -r requirements.txt
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130   # smp swaps in CPU torch
cd frontend; npm install; npm run build; cd ..
.\.venv\Scripts\python -m uvicorn app.api.main:app --app-dir backend --port 8000   # open http://localhost:8000
.\.venv\Scripts\python -m pytest -q tests
```
UI flow: draw AOI → search scenes → Acquire & detect → tick slick(s) in Triage → Investigate → Candidates → Evidence → Report.
CLI: `python main.py scenes|detect|triage|investigate|report|run|list` (see README §11).

## Real-first, SYNTHETIC-fallback at every step (Indian waters)
- Scene: real Sentinel-1 (partial scenes clipped to coverage) or **SYNTHETIC SAR scene** (`sar/synthetic_scene.py`,
  UI button / API `scene_id: "SYNTHETIC"`), labelled SYNTHETIC_DEMO; its discharging ship is planted in synthetic AIS.
- Forcing: Open-Meteo, else SYNTHETIC monsoon-climatology field (`environmental/synthetic_forcing.py`,
  `environment.mode: auto|real|synthetic`).
- Ship detection: dB-anomaly point targets (> +10 dB) when `_anom.npy` exists, else 8-bit CFAR (uploads).
- Default map/AOI now off Mumbai; UI shows real-scene coverage and offers the synthetic scene when coverage < 50 %.

## Candidate / traffic limits and UI
- At most 10 candidate vessels are ranked for real or synthetic AIS (`scoring.max_candidates: 10`).
- Synthetic AIS: at most 10 vessels (<= 2 fishing); lane ships use straight legs + smooth course alterations,
  trawlers use tow/haul/turn patterns (no random-walk zig-zags, no ruler-straight lines).
- Frontend: landing page at `/` (project brief, need, pipeline, real-vs-synthetic data, live endpoint list) and
  the investigation console at `/#/console` (`?step=search|triage|investigation|candidates|evidence`);
  NEXUS-style dark theme (particle backdrop, glass cards, step rail, UTC clock, LIVE pill).
  Landing motion: animated India ops map (GSHHG coastline, lanes, ships, ports, S-1 swath, slick + hindcast), word-by-word
  headline, typewriter, scroll reveals, count-ups, scroll-linked pipeline bar and ship on waves (components/motion.tsx,
  OceanMap.tsx, Waves.tsx; honours prefers-reduced-motion). Console: step header, collapsible notices/callouts,
  candidate cards, score ring, stage stepper, investigation HUD, collapsible legend.

## Novelty layer: impact & response (backend/app/impact/)
- stage_impact (after scoring) -> impact.json: SSII severity index + NOS-DCP tier, shoreline ETA (forward model or
  extrapolated), threatened receptors (gazetteer.py, approximate), response action plan (response.py), next-port
  intercepts (intercept.py, same-basin), ranking robustness (robustness.py), POLREP draft. stage_seal -> evidence_manifest.json
  (custody.py, SHA-256 + chain). Failures only add a warning. Console step 6 "Impact & response"; map layer "impact".
- Coast from GSHHG polygons (coast.py, cached per process, warmed at API start).

## AIS modes (real vs synthetic)
- Selection: `backend/app/ais/selection.py`; config `ais.mode`, `ais.real_provider`, `ais.synthetic.*`; per-run override
  via dashboard dropdown, API `ais_mode`, CLI `--ais`.
- Synthetic generator: deterministic per analysis id; lanes by region preset, per-type speeds, fishing loiter walks,
  60–360 s reporting, Poisson AIS gaps (15–90 min), noise + duplicates/invalid/spike defects; optional scenario
  (planted release tanker on the backtracked path, nearest-vessel trap, outside-window vessel) with roles in
  `ais/synthetic_truth.json` (never read by scoring). On the Mumbai test scene the planted tanker ranked 3rd of 15 and
  both decoys were filtered — busy synthetic lanes also cross the corridor, as in reality.
- Labelling: provenance SYNTHETIC_DEMO, `synthetic: true` on candidates/tracks, `[SYNTHETIC AIS]` evidence prefix,
  report assumption quoting SIH26143, dashboard banner + badges.

## Known limitations / open issues
- Model trained only 3 epochs on SOS, no look-alike negatives; still needs analyst triage; fine-tuning on real
  Sentinel-1 (with look-alike negatives) is the top next step (retraining only if the owner asks).
- No genuine real ship-discharge slick found yet in 16 scanned Danish scenes (Aug–Sep 2026).
- Open-Meteo free tier: ~600 locations/min, 5,000/hour, 10,000/day — heavy use hits the hourly limit (fails fast with
  a clear message). CMEMS/ERA5 NetCDF (`environment.provider: netcdf`) is the alternative.
- First-time AIS per new day is network-bound (~0.5–0.7 GB DMA zip + ~30 s indexing); cached afterwards.
- Single analysis worker (one job at a time). Docker/PostGIS path written but never run (no Docker here).
- GFW provider is implemented and mock-tested but not yet run against the live API (needs the owner's token);
  GFW is hourly presence, not raw AIS. Synthetic MMSIs use MID 419 and could coincide with real numbers (names are SYN-).
- Optional keyed sources (CMEMS, CDSE, Indian institutional AIS) listed in `/api/v1/sources` but not integrated.
- Windows gotchas: pandas 3 datetimes not ns (use `to_epoch_s`); MapLibre forces `position:relative` on its container.
