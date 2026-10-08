# How the system works: concepts, simply first, then technically

Each item gives a **plain explanation** first, then the **technical detail**, then *where it lives in the code*.

---

### 1. Why SAR imagery?
**Simply:** radar sees through clouds and at night, and oil makes the sea look dark on radar.
**Technically:** C-band SAR (Sentinel-1) measures backscatter from centimetre-scale capillary waves (Bragg scattering). Oil raises surface viscosity and damps those waves, so an oil film shows up as a low-backscatter (dark) patch. SAR is active, so it works regardless of cloud cover and daylight. *(Look-alikes have the same effect; see #31 and the look-alike module.)*

### 2. What the segmentation model does
**Simply:** it colours in, pixel by pixel, the parts of the image that look like oil.
**Technically:** binary semantic segmentation: `f: R^{3×256×256} → R^{1×256×256}`. It produces one logit per pixel. It doesn't classify the whole image. → `segmentation/`

### 3. DeepLabV3+
**Simply:** a network that looks at the image at several zoom levels at once, then sharpens the edges.
**Technically:** an encoder–decoder. The **ASPP** (Atrous Spatial Pyramid Pooling) head applies dilated convolutions at several rates (12/24/36) plus image pooling, which captures multi-scale context at output stride 16. The **decoder** upsamples and fuses low-level encoder features (the 48-channel `block1`) to recover sharp boundaries. Our checkpoint's key names (`decoder.aspp.0.convs.{0..4}`, `decoder.block1/2`, `segmentation_head`) match `smp.DeepLabV3Plus` exactly.

### 4. ResNet34
**Simply:** the part that learns what textures and shapes mean.
**Technically:** a 34-layer residual CNN (encoder). Skip connections (`y = F(x) + x`) make deep networks trainable. It was ImageNet-pretrained and then fine-tuned on SOS SAR tiles. `encoder.conv1` is `(64,3,7,7)`, which is why the input has 3 channels: the greyscale SAR is replicated.

### 5. Segmentation mask
**Simply:** a black-and-white picture of the same size where white = oil.
**Technically:** `H×W` boolean array. We keep three versions: the probability mask (float), the binary mask (thresholded), and the cleaned mask (morphological open/close plus removal of components under `min_component_area_pixels`).

### 6. Logit
**Simply:** the model's raw "how oily" number, anywhere from −∞ to +∞.
**Technically:** the pre-activation output `z`. It equals `log(p/(1−p))`. Training used `BCEWithLogitsLoss` (+ Dice), so the network outputs logits, not probabilities.

### 7. Why sigmoid
**Simply:** it squashes the raw number into 0–1.
**Technically:** `σ(z) = 1/(1+e^{−z})` is the inverse of the logit and matches the binary cross-entropy training objective. **Caveat:** the resulting probability is *uncalibrated*. That's why we call the scene figure "detection confidence (mean sigmoid over detected pixels)", not "probability it is oil".

### 8. Thresholding
**Simply:** anything above 0.5 counts as oil.
**Technically:** `mask = σ(z) > τ`, with `τ` configurable (`segmentation.threshold`). A lower τ raises recall and false positives; a higher τ does the opposite. Tested in `test_threshold_configurable`.

### 9. Pixels → polygons
**Simply:** we trace the outline of each white blob.
**Technically:**
1. 8-connected component labelling (`scipy.ndimage.label`).
2. Size filter.
3. **Exact pixel-boundary tracing** (`rasterio.features.shapes`), with no smoothing, because shape is evidence.
4. Components stay separate and are combined as a MultiPolygon, never merged.

→ `segmentation/postprocess.py`

### 10. Georeferencing
**Simply:** knowing which latitude/longitude each pixel covers.
**Technically:** an affine transform `(a,b,c,d,e,f)`, where `X = a·col + b·row + c` and `Y = d·col + e·row + f`, plus a CRS. It comes from GeoTIFF metadata or GCPs (Sentinel-1 GRD), or an explicit sidecar. The model itself knows nothing about location. **If there's no georeferencing, we stop; we never guess.** → `geospatial/georeference.py`

### 11. Spill area
**Simply:** we measure the outline's area on the real curved Earth.
**Technically:** the pixel polygon is transformed to WGS84 and passed to `pyproj.Geod.geometry_area_perimeter` (geodesic, WGS84 ellipsoid). Check: the demo has 10,178 px × (40 m)² = 16.28 km², and the geodesic result is 16.20 km². The centroid is computed in a local azimuthal-equidistant projection.

### 12. Why the observed location isn't the source
**Simply:** by the time the satellite passes, the oil may have drifted tens of kilometres.
**Technically:** surface oil moves at roughly `u_current + α·u_wind10` with `α ≈ 2–4 %`, plus Stokes drift and diffusion. At 0.3 m/s that's about 26 km per day. In the demo, the backtracked centre at T‑48 h is 52 km WNW of the slick.

### 13. Ocean drift
**Simply:** the sea surface is a moving conveyor belt pushed by currents and wind.
**Technically:** Lagrangian transport, `dx/dt = u_c(x,t) + α u_w(x,t) + u_Stokes + random walk (diffusivity K)`.

### 14. OpenDrift
**Simply:** a well-established open-source simulator that moves virtual particles through real ocean and weather data.
**Technically:** a Lagrangian framework from MET Norway. **Readers** interpolate forcing fields (NetCDF, THREDDS) in space and time; **models** define how elements move. It supports negative time steps, which is how backward runs work. We don't implement any ocean physics ourselves.

### 15. OpenOil
**Simply:** OpenDrift's oil-specific model.
**Technically:** it adds the oil properties from the NOAA ADIOS database (`GENERIC MEDIUM CRUDE`, …), wind drift, and weathering (evaporation, emulsification, dispersion, biodegradation). **Backward:** weathering is disabled because it's irreversible, so the run is transport only. **Forward:** weathering is enabled.

### 16. Hindcasting
**Simply:** running the film backwards.
**Technically:** integration with `time_step = −900 s` from the observation time. Particle positions at `T−h` estimate where the observed oil was `h` hours earlier.

### 17. Why particles throughout the spill
**Simply:** every part of the slick has its own history.
**Technically:** we sample points uniformly (by area) inside the actual polygon(s) (`drift/backtrack.sample_points_in_polygon`). An elongated slick maps back to an elongated set of possible source positions. That's information a centroid throws away.

### 18. Why the centroid alone isn't enough
**Simply:** one dot can't represent a 10 km streak.
**Technically:** a centroid run collapses the initial spatial distribution and underestimates uncertainty. For a curved slick, the centroid may not even lie inside the oil.

### 19. Why the release time is a window
**Simply:** we don't know *when* the oil was released.
**Technically:** from a single image, release time isn't identifiable. We evaluate T−6/12/24/36/48 h (configurable) and combine them with a stated **uniform prior**. Each candidate then gets a *candidate-conditional* release time: the time its track best matches the backtracked oil.

### 20. Why an ensemble
**Simply:** the inputs are uncertain, so we run many plausible versions.
**Technically:** the members vary the wind-drift factor (2–4 %), the seed positions and OpenDrift's stochastic wind/current uncertainty and diffusivity. The spread of the ensemble *is* the uncertainty estimate. It grows from 4 km at T‑6 h to 9 km at T‑48 h in the demo.

### 21. Source probability
**Simply:** a heat map of where the oil probably was when released.
**Technically:**
1. A 2D kernel density of all ensemble particle positions at each release time, on a 1 km grid with a 1.5 km Gaussian kernel, normalised to sum to 1.
2. The combined map averages the per-release-time maps.
3. **Highest-density regions:** the smallest area holding 50 % / 80 % / 95 % of the mass, reported as high / medium / low.

It is not a single point.

### 22. Environmental forcing
**Simply:** the current and wind maps that push the particles.
**Technically:** gridded, time-varying fields (`x_sea_water_velocity`, `x_wind`, …). They sit behind `EnvironmentalForcingProvider`:
- NetCDF files (CMEMS, HYCOM, ERA5, demo);
- Open-Meteo, which is live with no key.

Coverage over space and the full window is checked; missing data raises `ENVIRONMENTAL_FORCING_UNAVAILABLE`, never a default value.

### 23. What AIS contains
**Simply:** ships broadcasting who they are and where they are.
**Technically:**
- **Dynamic messages:** MMSI, time, lat/lon, SOG, COG and heading, every 2 s to 3 min depending on speed.
- **Static messages:** name, IMO, call sign and ship type (80–89 tanker, 70–79 cargo, 30 fishing, …).
- Received by terrestrial stations and satellites, so there are coverage gaps.

### 24. MMSI
**Simply:** the ship's radio ID number.
**Technically:** Maritime Mobile Service Identity, 9 digits. The first 3 digits are the MID (country): 419 = India. The IMO number, by contrast, is permanent for the hull.

### 25. Vessel trajectories
**Simply:** joining a ship's dots in time order.
**Technically:**
1. Group by MMSI and sort by time.
2. Drop duplicates, invalid coordinates, missing times and fixes implying more than 45 kn.
3. Derive speed and course from positions.

Positions are interpolated **only** for evaluating distance at a query time, and never across gaps longer than 2 h. → `ais/track_processing.py`

### 26. How AIS tracks are filtered
**Simply:** keep ships that were near where the oil was *when* it was there.
**Technically:**
- On a 10‑min grid across the release window, compute `d_st(t)` = distance from the vessel at t to the nearest backtracked particle at the same t (particles interpolated between hourly outputs).
- A vessel is a candidate if `min d_region ≤ max_source_distance_km` **or** `min d_st ≤ corridor_buffer_km`.
- Every excluded vessel gets a stated reason. → `ais/corridor_filter.py`

### 27. Why nearest-vessel logic is insufficient
**Simply:** the ship next to the slick in the photo may simply be passing by.
**Technically:** the demo's container ship crosses the observed slick exactly at image time, but it wasn't anywhere near the backtracked oil during the release window, so it's filtered out ("nearest-vessel trap").

### 28. Candidate features
- **Spatial:** closest approach to the 50 % source region.
- **Temporal:** at the vessel's best-match point, `|t_vessel − t_oil|`, where `t_oil` is the *density-weighted* time the backtracked oil occupied that spot (± spread).
- **Trajectory:** minimum time-matched distance to particles.
- **Vessel type:** from AIS (contextual only).
- **Behaviour:** speed before, near and after; slowdown ratio; net course change; stop duration.
- **AIS continuity:** gaps longer than 30 min within ±6 h of the best match.

→ `scoring/features.py`

### 29. The Evidence Correlation Score
Each feature maps to 0–1:
- `exp(−d/5 km)`, `exp(−Δt/4 h)`, `exp(−d_st/3 km)`, a type lookup, a behaviour combination, and `gap/120 min`.

Score = `100 · Σ wᵢ sᵢ / Σ wᵢ` with weights 0.30/0.30/0.15/0.10/0.10/0.05 (configurable). It's fully transparent: every sub-score is shown.

### 30. Why the score isn't proof
The weights are expert heuristics, not fitted to labelled cases. The score isn't calibrated, and it only ranks vessels **that broadcast AIS** and were **retrieved**. A high score means strong spatio-temporal correlation, which is necessary but not sufficient. In the demo, two vessels score within 1.3 points, and the system says so.

### 31. AIS continuity anomaly
**Simply:** a stretch of time when a ship's AIS went quiet.
**Technically:** consecutive reports more than `gap_threshold_minutes` apart. We report start, end, duration, position before and after, and implied speed.

### 32. Why a gap isn't automatically evasion
Terrestrial range limits, satellite revisit, receiver slot collisions in busy areas, equipment faults and archive gaps all produce gaps. Intentional switch-off can't be told apart from these by the gap alone, so it gets only a 5 % weight and neutral wording.

### 33. How the dashboard combines evidence
One map with toggleable layers (SAR, mask, spill, centroid, particles, heatmap, source regions, AIS tracks, live vessel positions, forward region, SAR ships) sits on a single timeline from T‑49 h to T+24 h.

Selecting a candidate:
- highlights its track;
- jumps the timeline to its best-match time, so the vessel is shown inside or next to the oil cloud;
- shows its labelled evidence.

Provenance badges (LIVE / LOCAL / DEMO / UNAVAILABLE) are always visible.

### 34. What the system *can* establish
- Where the model detects oil-like dark features, and their size and shape.
- Given the stated forcing and assumptions, where that oil probably was at earlier times, as regions with uncertainty.
- Which AIS-broadcasting vessels were spatio-temporally consistent with that history, and why; which were excluded, and why.
- A projected affected region for the next 24 h.

### 35. What it *cannot* establish
- That any vessel caused the spill, since that needs oil fingerprinting, inspection and logs.
- That a dark patch is definitely oil, since look-alikes exist.
- The exact release point or time.
- Anything about vessels without AIS, beyond SAR detections.
- Anything more accurate than its forcing data allows.

---

## Real-data additions (what changed when we left synthetic data)

**Sentinel-1 ingest (`backend/app/sar/sentinel1.py`).** Simply: we cut the area we care about out of a huge satellite image stored in the cloud and put it on a map grid. Technically: STAC search → signed COG asset → 2nd-order polynomial from the product's GCPs gives the source pixel window → windowed read → reprojection with window-adjusted GCPs to EPSG:4326 → GSHHG land mask → dB → local-background normalisation.

**Why normalisation and calibration.** Simply: the model learned on images that "look" a certain way; real images must be made to look the same or it sees oil everywhere. Technically: background grey and speckle contrast were chosen by a negative/positive-control experiment (`scripts/calibrate_radiometry.py`): FP 15 % → 0.02 %, −6 dB recall 100 %.

**Triage (`backend/app/lookalike/triage.py`).** Simply: not every dark patch is oil; we give each patch clues and let the analyst decide. Technically: ERA5 wind at acquisition, ship-trail test (SAR point target within 1 km of an END of the slick's major axis AND elongation ≥ 4 — a blob merely next to a ship is explicitly *not* rewarded), damping contrast in dB vs. a sea ring, filament-network and low-wind-zone shape tests → OIL_LIKELY / LOOKALIKE_LIKELY / UNCERTAIN.

**Two-phase job.** Phase 1 ends in `AWAITING_SLICK_SELECTION`; phase 2 runs drift/AIS/scoring only on the selected slick(s). This mirrors operational practice and avoids backtracking hundreds of look-alikes.

**Evidence-dependent release window.** Ship-trail slick → window starts at T (fresh discharge possible); detached slick → release ≥ 1 h before the image (configurable ASSUMPTION). Without this, any ship crossing an old slick at image time would look perfect (the nearest-vessel fallacy).

**SAR-attached evidence.** An AIS vessel matched (≤ 1.5 km, ≤ 30 min) to a SAR point target that touches the selected slick at acquisition is the strongest observable link (15 % weight) and is always kept as a candidate.

**DMA AIS provider (`backend/app/ais/dma_provider.py`).** Daily 0.5 GB zips → streamed pyarrow CSV → zstd Parquet cache (≈ 0.25 GB/day) → predicate-pushdown bbox queries; thinned to ≤ 1 fix/min/vessel; static fields propagated per MMSI; vectorised spike removal. Display tracks are Douglas–Peucker simplified (49 MB → 1.7 MB for 1,600 vessels) while analysis uses full tracks.

## Architecture decisions, and why

| Decision | Reason |
|---|---|
| Modular monolith, one worker thread | GPU and OpenDrift are the bottleneck; microservices or Celery would add failure modes with no benefit at hackathon scale |
| One service layer for CLI and API | No duplicated science; the CLI is the debugging tool for the API |
| Repository interface (File / PostGIS) | Science never touches SQL; the demo runs offline; PostGIS is the target for querying spills/regions/candidates spatially |
| Large artifacts on disk | Particle arrays and rasters are big and write-once; the DB stores paths and geometries |
| SSE, not WebSocket | Progress is one-way; SSE is plain HTTP and auto-reconnects |
| Spatio-temporal matching, not static distance | Static region distance can't tell "was there when the oil was" from "was there days later" |
| Density-weighted oil time | The ensemble cloud is a streak; "nearest particle at any time" was ambiguous (we found and fixed this during testing) |
| Twin-experiment demo | No real AIS exists for unlocated SOS tiles. A planted, blind-recovered scenario validates the method honestly |

## How to modify
- **Weights and thresholds:** edit `configs/config.yaml` and rerun `main.py score --spill ID`. No drift rerun is needed.
- **A new forcing source:** subclass `EnvironmentalForcingProvider` (`environmental/forcing.py`) and register it in `build_provider`.
- **A new AIS source:** subclass `AISProvider` and register it in `build_ais_provider`.
- **A trained ship detector or look-alike classifier:** implement `ShipDetector.detect` or `lookalike.assess` with the same outputs.
- **A new feature:** add it in `scoring/features.py`, map it to 0–1 in `scorer.feature_scores`, add a weight, and add wording in `evidence_report.py`.

## How to debug
1. `python main.py segment --input …`: check `outputs/analyses/ID/mask.png` and `spill.geojson`.
2. `python main.py backtrack --spill ID --particles 100 --members 2`: look at `drift/summary.json` → `per_release_time`.
3. `python main.py ais --spill ID`: prints the AIS query, the cleaning counts and every filtered vessel's reason.
4. Look at `state.json` → `events`: the last stage and message show where it stopped.
5. Run `pytest -q tests -k <area>`.
