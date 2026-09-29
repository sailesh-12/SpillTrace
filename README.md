# SIH26143 — Oil Spill Detection, Hindcasting & Vessel Correlation (real data)

**Smart India Hackathon 2026 · Problem Statement SIH26143**

> *"Leveraging satellite imagery to determine oil spills at sea along with AIS data correlations to identify vessel responsible for the spill."*

> **Checkpoint snapshot.** See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for a concise hand-over (architecture, pipeline, key files, limitations). Major technologies: Python 3.14 · PyTorch 2.14 (CUDA) · segmentation-models-pytorch (DeepLabV3+/ResNet34) · OpenDrift/OpenOil · rasterio/GDAL · shapely/pyproj · pandas/pyarrow · FastAPI + SSE · React 19 + Vite + TypeScript + Tailwind v4 + shadcn/ui (Base UI) + MapLibre GL.

### What is in this repository / what is not
| Included | Not included (and why) |
|---|---|
| All backend, frontend, CLI, scripts, configs, tests, docs | `.venv/`, `frontend/node_modules/`, `frontend/dist/` — regenerate with §6 |
| Trained model `models/best_oil_spill_deeplabv3_resnet34.pth` (258 MB) via **Git LFS** | `data/ais/cache/` (~0.25 GB per day, several GB) — re-downloaded automatically from DMA |
| Training notebook `models/training_notebook.ipynb` | `data/forcing/cache/` — re-fetched automatically from Open-Meteo |
| 4 SOS sample tiles `data/sar/samples/` | `outputs/` (analysis results, ~1.3 GB) — produced by running the app |
| Synthetic test fixtures `tests/fixtures/` (used only by tests) | The SOS training dataset itself (not in this folder; third-party dataset) |
| `.env.example` (no values) | `.env` / any keys — none are required; never committed |

Clone with LFS installed (`git lfs install`) so the model file downloads (otherwise it is a small pointer file).

## 1. Project overview
A slick seen by a satellite is usually **not where the oil entered the water**: currents and wind have been moving it. This application works on **real data** and reconstructs where the oil *was*, then asks which vessels were there *at those times*:

```
Real Sentinel-1 SAR (Planetary Computer, no key)
  → AOI extraction via GCPs + land mask (GSHHG) + calibrated radiometric normalisation
  → DeepLabV3+/ResNet34 segmentation → dark-feature polygons (WGS84, area, centroid, time)
  → per-slick TRIAGE (ERA5 wind, ship-trail pattern, contrast, shape)  ← investigator selects slick(s)
  → live currents + wind (Open-Meteo) → OpenDrift/OpenOil backward ensemble + forward 24 h
  → source-probability surface + evidence-dependent release window
  → real historical AIS (Danish Maritime Authority, no key) → cleaning → spatio-temporal corridor
  → SAR/AIS match at acquisition → Evidence Correlation Score → ranked candidates + evidence report
  → investigator dashboard (React + shadcn/ui + MapLibre)
```
The output is **candidate vessels requiring investigation**, never a verdict of responsibility.

## 2. Problem statement
Detect/characterise spills; trace the slick back to origin point/time and predict future flow using oceanographic and meteorological data; reconstruct AIS traffic around the origin window; filter irrelevant traffic; score suspects on proximity, trajectory and behavioural anomalies; provide a visual interface.

## 3. Architecture (modular monolith)
```
frontend/ (React 19 + Vite + TypeScript + Tailwind v4 + shadcn/ui + MapLibre)   ← investigator UI, no science
        │ REST + SSE
backend/app/api (FastAPI)  → services/pipeline.py (two-phase state machine, shared with the CLI main.py)
        │
 sar/ (Sentinel-1 search+ingest)  segmentation/  geospatial/  lookalike/ (triage)  environmental/ (forcing)
 drift/ (OpenDrift/OpenOil)  ais/ (DMA + local providers)  scoring/  sar_ship_detection/  multitemporal/  reports/
        │
 database/ → FileRepository (default) | PostGISRepository (target) + filesystem artifacts
```
State machine: `QUEUED → SCENE_ACQUISITION → SCENE_INSPECTION → GEOREFERENCING → PREPROCESSING → SEGMENTATION → POLYGONIZATION → SAR_SHIP_DETECTION → LOOKALIKE_ANALYSIS → **AWAITING_SLICK_SELECTION**` → (investigator selects) → `ENVIRONMENTAL_DATA → BACKWARD_DRIFT → SOURCE_ESTIMATION → AIS_PROCESSING → AIS_GAP_ANALYSIS → SAR_AIS_MATCHING → CANDIDATE_SCORING → REPORT → COMPLETED | PARTIAL | FAILED`.

## 4. Data sources and API keys
**Nothing needs a key to run the full pipeline.** Key-free sources are fetched automatically; the *Data sources* panel in the UI (and `GET /api/v1/sources`) shows the status of each.

| Data | Source | Key | Used for | Coverage |
|---|---|---|---|---|
| Sentinel-1 GRD (VV) | Microsoft Planetary Computer STAC | **none** (anonymous token) | SAR imagery | global |
| Currents | Open-Meteo Marine API | **none** | drift forcing | global ~0.08° |
| 10 m wind | Open-Meteo (ERA5 archive / forecast) | **none** | drift + triage | global 0.25° |
| Historical AIS | Danish Maritime Authority (aisdata.ais.dk) | **none** | vessel tracks | Danish waters, daily files 2024-03 → present |
| Coastline | GSHHG full resolution (roaring-landmask) | none (bundled) | land mask, stranding | global |
| Real AIS, Indian waters | Global Fishing Watch 4Wings presence | free non-commercial token (`GFW_API_TOKEN`) | vessel tracks (hourly) | global incl. Indian EEZ |
| **Synthetic AIS** (fallback) | built-in generator, Indian-waters presets | **none** | vessel tracks when no real AIS (labelled SYNTHETIC) | any region |
| *Optional upgrades* | CMEMS, Copernicus Data Space, Indian institutional AIS | **you enter them** in `.env` | not required | — |

- Keys are never created or entered by the software. To use an optional source, register with the provider, copy `.env.example` to `.env`, fill the variable, restart the backend. Values are never returned by the API.
- **Indian waters:** Sentinel-1 and forcing work everywhere. For AIS the app uses Global Fishing Watch if `GFW_API_TOKEN` is set, otherwise it generates clearly labelled **SYNTHETIC AIS** for the spill region (see §4b). Danish scenes still use real DMA AIS automatically.

## 4b. AIS for Indian waters: real provider + SYNTHETIC fallback

All AIS sources implement the same `AISProvider` interface (`backend/app/ais/provider.py`) and return the same
columns, so **real and synthetic AIS feed the identical downstream pipeline**:
AIS → cleaning/filtering → spatio-temporal corridor → candidate vessels → evidence scoring → investigation report.
The choice is made in `backend/app/ais/selection.py`.

**Real AIS options for Indian waters**
| Option | Data | Access | Status in this project |
|---|---|---|---|
| **Global Fishing Watch 4Wings** (`public-global-presence:latest`) | AIS-derived presence of **all vessel types**: 1 position per vessel per hour, 0.01° (~1 km) cells, name/MMSI/type | Free **non-commercial** token → `GFW_API_TOKEN` in `.env` | Implemented (`ais/gfw_provider.py`); request/parse tested with a mocked response; not yet run against the live API (needs your token) |
| DG Shipping NAIS / INCOIS / Indian Navy IFC-IOR | Raw national AIS | Institutional agreement | Not available — add an `AISProvider` when access exists |
| Commercial (Spire/Kpler, MarineTraffic, exactEarth) | Raw terrestrial + satellite AIS | Paid | Not integrated |
| Danish Maritime Authority | Raw AIS, Danish waters only | Free | Implemented (`ais/dma_provider.py`), used automatically for Danish scenes |

GFW caveats: it is not raw AIS (speed/course are derived from hourly positions; AIS-gap detection uses a ≥ 3 h
threshold for this source), coverage near busy coasts depends on terrestrial/satellite reception, and the licence
requires attribution and non-commercial use.

**Why synthetic AIS is allowed.** The official SIH26143 statement says: *"Real AIS if available may be used else
synthetic data can be prepared for the region of oil spill to demonstrate the functioning of the algorithm."*
Free historical raw AIS for Indian waters does not exist, so the project generates labelled synthetic AIS when
real AIS is unavailable — it never depends on real Indian AIS being present.

**How synthetic AIS is generated** (`backend/app/ais/synthetic_provider.py`, deterministic per analysis id):
1. *Region preset* from the search area: Arabian Sea / west coast (lanes ≈160°, 175°, 95°; tankers, container,
   offshore supply near Mumbai High, heavy fishing), Bay of Bengal / east coast (≈20°, 35°, 100°), Sri Lanka
   east–west route (≈90°, 80°, 120°), or a generic Indian Ocean preset.
2. *Lane traffic*: lanes at the preset bearings offset across the area; straight legs joined by smooth course
   alterations (8–25°, spread over 15–35 min, alternating so ships keep to their lane) plus a ±1–2.5° meander;
   per-type speeds (tanker 11–14 kn, container 14–20, bulk 10–13, cargo 10–15, offshore supply 8–12, tug 6–10,
   passenger 14–18) with AR(1) variation.
3. *Fishing vessels* (at most 2): one 12–30 h session of trawl tows (1.5–5 h at 2.5–4.5 kn), hauling at 0.5–1.5 kn,
   slow ~180° turns (≤ 2.5°/min) onto the next tow, kept on a fishing ground.
   **At most 10 synthetic vessels in total** (scenario + background; `ais.synthetic.vessels`, capped in code).
4. *Reporting*: every 60–180 s (large ships) / 180–360 s (fishing, class-B-like); AIS gaps as Poisson episodes of
   15–90 min (1.5× more frequent offshore); 10 m position jitter, 0.3 kn SOG and 2° COG noise; a few duplicate
   reports, invalid coordinates and ~85 km position spikes so the real cleaning code is exercised.
5. *Identity*: 9-digit MMSIs with India's MID **419** (band 419900000–419999999), names prefixed **`SYN-`**, no IMO.
   A synthetic MMSI could coincide with a real Indian vessel's number — which is why the `SYN-` name and the
   SYNTHETIC label are mandatory and shown everywhere.
6. *Scenario* (`ais.synthetic.scenario: true`): a planted release tanker placed on the model's backtracked oil path at
   a random time in the release window (slowdown to 4 kn, AIS gap 1–3 h later), a container ship crossing the observed
   slick at image time (nearest-vessel trap), and a cargo ship crossing the source area before the window. Roles are
   saved to `outputs/analyses/<ID>/ais/synthetic_truth.json` and are never read by the scoring. Because the planted
   vessel uses the model's own backtracked path, this demonstrates the matching/scoring logic, not real-world skill.

**Labelling.** Provenance `SYNTHETIC_DEMO`; warnings in the analysis and report; the report's first assumption quotes
the SIH permission; every AIS evidence statement is prefixed `[SYNTHETIC AIS]`; candidates and tracks carry
`synthetic: true`; the dashboard shows a "SYNTHETIC AIS — not real vessels" banner and a per-row SYNTHETIC badge.

**Switching modes** (`configs/config.yaml → ais`):
```yaml
ais:
  mode: auto            # auto: real if available, else SYNTHETIC | real: never synthetic | synthetic: always
  real_provider: auto   # auto (DMA in Danish waters, else GFW if GFW_API_TOKEN set, else local files) | gfw | dma | local
  synthetic: {fallback: true, vessels: 10, scenario: true, gap_rate_per_day: 2.0, seed: 0}   # max 10
```
Per investigation: the **AIS source** dropdown next to *Investigate* in the dashboard, `ais_mode` in
`POST /api/v1/analyses/{id}/investigate`, or `python main.py investigate --spill <ID> --ais auto|real|synthetic`.
For real Indian AIS: put `GFW_API_TOKEN=...` in `.env`, restart the backend, use mode `auto` or `real`.

## 4c. Real data first, labelled SYNTHETIC fallback at every step (Indian waters)

| Step | Real source | When it is missing | Synthetic fallback (always labelled SYNTHETIC) |
|---|---|---|---|
| Area & scene | Sentinel-1 GRD (Planetary Computer) — footprints shown; a partially covering scene is **clipped** to the covered part | Sentinel-1 revisits Indian waters rarely; many boxes are covered only partly or not at all | **Use SYNTHETIC SAR scene** button: `backend/app/sar/synthetic_scene.py` — gamma speckle (ENL 4.4) × streaked wind texture (±1 dB) × incidence ramp; one ship (+20 dB point target with sidelobes) at the head of a 7–14 km discharge trail (6–8 dB damping, 80→450 m wide, meandering); a compact low-wind look-alike (−3.5 dB); 4–8 other ships; then the SAME normalisation as real scenes. Scenario saved in `acquisition.json`. |
| Detection & triage | model + triage on the scene | — | runs unchanged; on synthetic scenes the trail is found (OIL_LIKELY, ship-trail) and look-alikes are rejected |
| Hindcast | Open-Meteo currents + ERA5/forecast wind | offline / free-tier hourly limit | `backend/app/environmental/synthetic_forcing.py`: Indian-waters monsoon climatology (SW monsoon Jun–Sep from ~240° 9 m/s, NE monsoon Dec–Feb from ~40° 6 m/s; WICC/EICC/monsoon currents by season, M2 tide). `environment.mode: auto|real|synthetic` |
| Candidates / evidence | GFW (token) · DMA (Danish) · local files | no real AIS | synthetic AIS (§4b); with a synthetic scene, its discharging ship gets a matching AIS track, so SAR/AIS evidence is consistent |

Ship detection on Sentinel-1/synthetic scenes now uses the **unclipped dB anomaly** (clusters > +10 dB above the
local sea; `sar_ship_detection.point_target_db`) — the 8-bit CFAR could not fire after normalisation (threshold above 255).
Downloads retry 3× with fewer connections. Example (synthetic scene off Mumbai, real Open-Meteo forcing, synthetic AIS):
phase 1 ≈ 20 s, phase 2 ≈ 25 s; trail OIL_LIKELY with ship-trail pattern, window starts at T, #1 candidate is the
discharging vessel (SAR-attached).

## 4d. Novel features — from "who did it" to "what to do now"

Runs automatically after candidate scoring (`backend/app/impact/`, config `impact:`), is shown in the console step
**6 · Impact & response**, in the report (section 6b) and on the map. Every part is advisory and degrades gracefully.

| Feature | What it does | How |
|---|---|---|
| **Spill Severity & Impact Index (SSII, 0–100)** | Prioritises the response: LOW / MODERATE / HIGH / CRITICAL with a per-factor breakdown | size (log area) · coast proximity · beaching threat · most-sensitive receptor threatened (discounted by ETA) · oil persistence & wind dispersion; weights in config |
| **Volume range → NOS-DCP tier** | Planning estimate of spilled volume and response tier | area × assumed thickness (Bonn Agreement appearance codes 0.3–50 µm); Tier 1 < 700 t, Tier 2 700–10 000 t, Tier 3 > 10 000 t |
| **Shoreline ETA** | When and where oil reaches the coast | OpenOil forward forecast (≥ 1 % of particles within 0.5 km of the GSHHG shoreline), else extrapolation of the forecast drift vector until it meets land (labelled) |
| **Threatened sensitive receptors** | Mangroves, coral reefs, turtle-nesting beaches, lagoons, power-plant seawater intakes, ports, fishing harbours, tourism beaches — with ETA | 31-site Indian gazetteer (approximate reference points, **not official ESI maps**) |
| **Immediate response action plan** | Prioritised (P1–P3), time-boxed actions with the responsible agency | rules aligned with NOS-DCP roles (ICG MRCC Mumbai / Chennai / Port Blair, ports, State/UT, DG Shipping): notify, verify, sample for fingerprinting, protect receptors before ETA, contain (wind limit), dispersant guidance, inspect suspects, identify dark vessels, re-task satellite |
| **Next-port intercept** | Where to inspect each suspect vessel and when | last AIS course/speed → Indian port ahead (±40°, same sea basin), straight-line ETA; flags AIS silence inside the area |
| **Ranking robustness** | Does the #1 candidate depend on the (uncalibrated) weights? | 2 000 Dirichlet perturbations of the Evidence Correlation Score weights → P(#1), P(top-3), verdict ROBUST / LIKELY / CONTESTED |
| **Tamper-evident evidence seal** | Chain of custody for MARPOL prosecution | SHA-256 manifest of every artifact, root + chain hash across re-analyses; `GET /api/spill/{id}/verify` reports modified / missing files |
| **POLREP draft** | Ready-to-review pollution report for the MRCC | POLWARN / POLINF / POLFAC structure, nautical positions; `GET /api/spill/{id}/polrep` (never sent automatically) |

New endpoints: `POST /api/spill/{id}/impact` (recompute + re-seal, e.g. for older analyses), `GET /api/spill/{id}/verify`,
`GET /api/spill/{id}/polrep`. With SYNTHETIC inputs the plan and POLREP are marked *exercise only*.

## 5. Prerequisites & versions
Windows 10/11, Linux or macOS · **Python 3.14.3** (3.12+ fine) · **Node 24.14** (20+) · ~10 GB free disk (AIS cache ≈ 0.25 GB per day) · internet access · NVIDIA GPU optional (verified on RTX 5050, CUDA 13).

## 6. Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu130      # or cu128 / cpu
pip install -r requirements.txt
# segmentation-models-pytorch may replace CUDA torch with a CPU build; if so:
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd frontend; npm install; npm run build; cd ..
```

## 7. Model
`models/best_oil_spill_deeplabv3_resnet34.pth` (the `.pth.zip` you received *is* the PyTorch file). Verified: `smp.DeepLabV3Plus(resnet34, in_channels=3, classes=1)`, 22.46 M params, input RGB/255 without normalisation, 256×256 tiles (sliding window on larger scenes), output logit → sigmoid → threshold 0.5; SOS validation Dice 0.746 (epoch 3). The model never leaves the server.

## 8. Real-SAR preprocessing and calibration (important)
The model was trained on 8-bit SOS tiles whose radiometric processing is undocumented. Feeding real Sentinel-1 with a naive stretch caused **15 % of clean sea** to be flagged. The ingest now:
1. reads only the AOI window from the cloud-optimised GRD (GCP polynomial → window → GCP reprojection to EPSG:4326),
2. masks land with the GSHHG coastline,
3. converts to dB and normalises **relative to the local sea background** (60th percentile, 10 km window — also flattens incidence-angle/wind gradients),
4. maps background → grey 150 and speckle σ → 40 grey levels.

Step 4 was **calibrated** (`scripts/calibrate_radiometry.py`) on a clean real scene (Skagen, 8 m/s wind) with implanted −3/−6 dB slicks: false positives **15 % → 0.02 %**, recall 100 % (−6 dB) / ~98 % (−3 dB). Re-run it if you change resolution or sensor.

## 8b. Performance (measured on this machine, real data)
| Stage | Before | After | How |
|---|---|---|---|
| Sentinel-1 AOI download (0.6°×0.4°, 10 m) | 4–5+ min | ~45 s | tuned GDAL HTTP (4 MB chunks, merged ranges) + 16 parallel tile reads |
| Open-Meteo forcing (cold) | 38 s | 8 s | concurrent chunk requests + currents/wind in parallel |
| Backward ensemble + forward 24 h | several min | ~40 s | one OpenDrift run for all members (per-particle wind factor), cached readers, 30-min step (≤ 0.12 km change vs 15-min) |
| One-time OpenDrift import + GSHHG landmask | every job | once at server start | background warm-up; landmask shared with SAR sea mask |
| AIS retrieval (3 days, 1,600 vessels) | 55 s | ~18 s | filter + thin inside Arrow before pandas |
| AIS cleaning / corridor match / scoring | 106 s | ~21 s | vectorised cleaning, one KD-tree query per time step for all vessels, cached epochs, vectorised gap search |
| First-time AIS day download | sequential | parallel days | 4 concurrent downloads/indexing |

## 9. Configuration
`configs/config.yaml` holds every threshold and weight (segmentation, Sentinel-1 resolution & normalisation, Open-Meteo grid, drift ensemble & release offsets, AIS thresholds, scoring weights, CFAR, triage). Environment overrides: `MODEL_PATH`, `SIH_CONFIG`, `SIH_STORAGE=postgis`, `SIH_POSTGIS_DSN`.

## 10. Running the application
**Terminal 1 — backend + dashboard**
```powershell
.\.venv\Scripts\python -m uvicorn app.api.main:app --app-dir backend --port 8000
```
Open **http://localhost:8000** — a landing page (project brief, why it is needed, pipeline, data, live endpoint list) with *Launch console* → **http://localhost:8000/#/console** (API docs: http://localhost:8000/docs). Optional hot-reload UI: `cd frontend; npm run dev` → http://localhost:5173.

**Investigator workflow in the UI**
1. **Area & scene** — *Draw on map* (drag a box over water, ≤ 1° per side), choose dates, *Search scenes*, pick a Sentinel-1 pass (footprints shown on the map), *Acquire & detect* (10 m native recommended; ~2–6 min download for a 0.6°×0.4° AOI).
2. **Detection & triage** — every dark feature is coloured by triage label (orange oil-likely, yellow uncertain, green look-alike likely); the table shows wind at acquisition, ship-trail flag, contrast, elongation. Click a row to zoom; **tick the slick(s) you judge to be oil** → *Investigate*.
3. **Hindcast** — live forcing download, OpenOil ensemble backwards (and 24 h forward), source regions; press ▶ on the timeline to animate.
4. **Candidates** — real AIS downloaded (first time ~1–2 min per day, then cached), cleaned, corridor-filtered; ranked table with filtered-vessel reasons.
5. **Evidence** — per-vessel feature bars and statements labelled Observed / Model / Assumption / Inference; supporting / contradicting / missing evidence; uncertainty; *Report* opens the HTML report.

## 11. CLI (same service layer)
```powershell
.\.venv\Scripts\python main.py scenes --bbox 10.3,57.6,10.9,58.0 --start 2026-09-01 --end 2026-09-24
.\.venv\Scripts\python main.py detect --scene <ITEM_ID> --aoi 10.3,57.6,10.9,58.0 [--res 10]
.\.venv\Scripts\python main.py triage --spill <ID>
.\.venv\Scripts\python main.py investigate --spill <ID> --components <ID>_C12[,<ID>_C40] [--ais auto|real|synthetic]
.\.venv\Scripts\python main.py report --spill <ID>
.\.venv\Scripts\python main.py run --input my_scene.tif          # local GeoTIFF, both phases (top triage slick)
.\.venv\Scripts\python main.py list
.\.venv\Scripts\python scripts\scan_real_slicks.py --start 2026-07-01 --end 2026-09-24   # shortlist scenes with ship-trail patterns
```

## 12. Outputs (`outputs/analyses/<ID>/`)
`scene/…tif` (normalised AOI GeoTIFF + sea mask), `scene.jpg`/`mask.png` (display), `spill.json/.geojson` (all dark features), `triage.json`, `spill_selected.json`, `drift/summary.json`, `drift/source_probability.geojson`, `drift/*_particles.json`, `drift/forward_summary.json`, `ais/tracks.geojson` (display-simplified), `candidates.json`, `report.json/.md/.html`, `state.json`. AIS cache: `data/ais/cache/dma_YYYY-MM-DD.parquet`; forcing cache: `data/forcing/cache/`.

## 13. Tests
```powershell
.\.venv\Scripts\python -m pytest -q tests        # 54 tests, ~1.5 min, offline
```
The tests use deterministic synthetic fixtures (`tests/fixtures/`, a labelled twin experiment with a planted vessel). **The application never reads them**; they exist so the science can be verified offline and reproducibly.

## 14. Troubleshooting
| Symptom | Fix |
|---|---|
| CUDA false after installing requirements | reinstall torch from the CUDA index (§6) |
| "No DMA AIS file published for …" | DMA publishes with a few days' delay; pick an older scene |
| "No AIS coverage for this region" | DMA covers Danish waters only (§4) |
| Open-Meteo 429 / slow | free-tier rate limit; the client backs off and retries; results are cached |
| Many dark features at < 3 m/s wind | expected: natural look-alikes; triage flags low wind |
| Map blank offline | OSM basemap needs internet; all analysis layers still render |
| Download slow | AOI size drives download; use 20 m or a smaller AOI |

## 15. Scientific, data and attribution limitations
- Segmentation: SOS-trained (Dice 0.75, 3 epochs), no look-alike labels; real-scene behaviour depends on the calibrated normalisation (§8). A human analyst must confirm slicks (triage). Retraining/fine-tuning on real Sentinel-1 with look-alike negatives is the most valuable next step.
- Ship-attached dark streaks can also be a vessel's turbulent wake at low wind.
- Backtracking is transport-only (weathering is irreversible), no explicit Stokes drift; forcing is coarse (Open-Meteo); uncertainty is expressed as ensembles and HDR regions, not a point.
- Release time is not identifiable from one image: the window starts at T for ship-trail slicks and ≥ 1 h before the image for detached slicks (configurable ASSUMPTION).
- In dense traffic many vessels cross a 48 h corridor; the report states when candidates cannot be separated.
- The **Evidence Correlation Score** (spatial 25, temporal 25, trajectory 15, SAR-attached 15, type 7, behaviour 8, continuity 5) is an uncalibrated prototype heuristic, not a probability of responsibility. AIS gaps are "continuity anomalies", never "evasion".
- Docker/PostGIS (`docker-compose.yml`, `database/schema.sql`) is written but not verified here (Docker not installed).

Further: `docs/ARCHITECTURE_AND_CONCEPTS.md`, `docs/SIH_PRESENTATION.md`.
