# SIH26143: presentation pack

Every claim below matches the implemented system, which runs on REAL data: Sentinel-1 (Planetary Computer), Open-Meteo currents/wind and Danish Maritime Authority AIS — all key-free.

## 30-second version
Satellites can spot an oil slick, but the slick has usually drifted far from where it was released, so the nearest ship is often the wrong answer. Our system detects the slick in radar imagery with a deep-learning model. It then runs the ocean backwards with the OpenDrift/OpenOil simulator to map where the oil was hours earlier, and checks historical AIS ship tracks for vessels that were in those places *at those times*. It ranks the candidates with a transparent evidence score and explains each one, including what we don't know.

## 1-minute version
Add to the 30 seconds:
- **Detection:** a DeepLabV3+/ResNet34 model trained on the Deep-SAR Oil Spill dataset (validation Dice 0.75) produces the slick polygons, area and centroid.
- **Hindcast:** we seed particles across the whole slick and run an ensemble backward through current and wind data for up to 48 h. That gives a *probability map* of the source region and a release-time window, not a single guessed point.
- **Forecast:** we also run forward 24 h to show where the oil is heading.
- **Correlation:** the AIS matcher compares each ship with where the oil was *at the same moment*, filters out irrelevant traffic with a stated reason for each, and scores the rest on proximity, timing, trajectory, behaviour and AIS continuity.
- **Output:** the investigator gets a map with animated drift, a ranked candidate table, and an evidence report that separates observed facts from model results, assumptions and inferences.

## 3-minute version (structure)
1. **Problem** (30 s): unattributed spills; the slick isn't at the source; the nearest-ship fallacy.
2. **Pipeline** (60 s): walk through the architecture diagram, from SAR to evidence.
3. **Live demo** (60 s): the demo script below, steps 2–7.
4. **Honesty and novelty** (30 s): spatio-temporal matching, uncertainty everywhere, safeguards, and the twin-experiment validation.

## Problem explanation
Many marine oil spills are never attributed to a source. Detection is only the first step. Attribution needs the slick's history, which depends on currents and wind, and then a comparison with vessel traffic in that space-time window. Coverage gaps, many ships, look-alike slicks and uncertain timing all make this hard.

## Architecture explanation
- React/MapLibre investigator dashboard, talking over REST and SSE to FastAPI.
- One orchestrator (a state machine with 17 stages) that the CLI shares.
- Independent modules: segmentation, geospatial, forcing, drift, AIS, scoring, reports, SAR ships, look-alike and multi-temporal.
- Storage is file-based for offline demo; PostGIS is the target and its schema is included.
- The model runs only on the server.

## End-to-end demo script (≈3 min)
Pre-run one investigation before the session (download ≈ 5–10 min the first time; AIS/forcing are cached afterwards).
1. "Everything here is real and live-sourced — the badges say LIVE. No API keys: Sentinel-1 from Microsoft Planetary Computer, currents and ERA5 wind from Open-Meteo, historical AIS from the Danish Maritime Authority."
2. **Area & scene**: draw a box over the Kattegat/Skagerrak, search: "these footprints are the actual Sentinel-1 passes in that window." Pick one → *Acquire & detect*. The stage chips stream live (SSE).
3. **Detection & triage**: "The model flags dark features. Real sea has many look-alikes, so each feature gets transparent indicators — ERA5 wind at the acquisition time, whether a SAR-detected vessel sits at the end of a narrow trail, damping contrast, shape. The analyst confirms which slick to investigate — the same human-in-the-loop step EMSA CleanSeaNet uses."
4. Tick the slick → *Investigate*. "Live currents and wind are downloaded, OpenOil runs an ensemble backwards and 24 h forwards." Press ▶: particles flow back in time; the heatmap follows the release time.
5. **Candidates**: "Real AIS for those days — here about 1,600 vessels — is cleaned and matched against where the oil was *at the same time*. Every excluded vessel has a stated reason; the table says when the top candidates cannot be separated."
6. **Evidence**: click the top vessel — the map flies to its best-match position and time; statements are labelled Observed / Model / Assumption / Inference with supporting, contradicting and missing evidence. Open the HTML report.
7. Close honestly: "In dense lanes with low wind, attribution from drift alone is weak — the system says so rather than inventing certainty. The strongest evidence is a vessel caught by SAR and AIS at the head of a fresh trail."

## Novelty
- **Spatio-temporal correlation:** vessels are matched against the *time-resolved* backtracked oil cloud, not a static radius.
- **Uncertainty-first:** polygon-wide seeding, ensembles, HDR source regions, a release-time window with candidate-conditional release times, and a ranking-separation warning.
- **Explainability:** a transparent weighted score, per-feature bars, supporting/contradicting/missing evidence, and an epistemic label on every statement.
- **Scientific safeguards built in:** AIS gaps are called "continuity anomalies"; filtered vessels get stated reasons; "no candidate" is a valid result; provenance badges are always shown.
- **A twin-experiment validation harness:** a planted scenario, recovered blind, is part of the automated tests.

## Feasibility
It runs today on a laptop with an RTX 5050: the full pipeline takes about 50 s (6 members × 500 particles, 49 h backward and 24 h forward). There are 30 automated tests. Everything uses open tools (PyTorch, OpenDrift, FastAPI, MapLibre) and open data sources: Sentinel-1, CMEMS/HYCOM/ERA5, Open-Meteo (live, verified), and MarineCadastre/DMA AIS.

## Scalability
- Stateless API workers plus PostGIS spatial indexes on spills, regions and AIS positions.
- Drift members are independent, so they parallelise across processes or nodes.
- Sliding-window inference already handles full scenes.
- Provider interfaces let you plug in national AIS feeds (e.g. the DG Shipping / Indian Coast Guard network) and operational INCOIS/CMEMS forcing without touching the science.

## Limitations (say them before the judges do)
- The model was trained 3 epochs on SOS tiles (Dice 0.75) without look-alike negatives. On real Sentinel-1 a naive image stretch flagged 15 % of clean sea; our calibrated radiometric normalisation reduced this to 0.02 % on a clean scene while still detecting implanted −3/−6 dB slicks — but an analyst still confirms slicks (triage), and fine-tuning on real scenes is the priority next step.
- Look-alike screening is rule-based (wind, ship-trail, contrast, shape), not a trained classifier.
- Free historical AIS exists for Danish (and US) waters, not Indian waters; India needs an institutional/commercial AIS provider (plug-in interface ready).
- Open-Meteo forcing is real but coarse; no explicit Stokes drift; release time is a window, not a point.
- The Evidence Correlation Score is an uncalibrated heuristic; in busy lanes many vessels cross a 48 h corridor.
- PostGIS/Docker path written but not run in our environment.

## Future work
- Validate on real georeferenced Sentinel-1 scenes of documented spills, with real AIS (MarineCadastre / DMA / Indian AIS).
- Train a look-alike classifier (CleanSeaNet-style labels) and a SAR ship detector (e.g. on SSDD/HRSID).
- Calibrate the score on historical adjudicated cases (it would then become a probability model).
- Use multi-temporal scenes to constrain the release time; integrate INCOIS operational forcing and wave/Stokes drift; add oil-thickness and age estimation.

## Likely judge questions, with accurate answers
**Q: How do you know it's oil and not a natural slick?**
We don't, with certainty. The model flags dark oil-like patterns (Dice 0.75 on SOS). Our look-alike module reports indicators such as wind speed, contrast, shape and full-scene coverage, and usually says UNCERTAIN, because the dataset has no look-alike labels to train on. We present it as triage, not confirmation.

**Q: Why not just take the nearest ship?**
Because the oil moves. In our demo the ship nearest the slick at image time is actually excluded: it wasn't near the backtracked oil at any time in the release window.

**Q: How accurate is the backtracking?**
It's only as good as the forcing. That's why we use an ensemble and report probability regions whose spread grows with time (4 km at 6 h, 9 km at 48 h in the demo) instead of a point. With real forcing, errors of several km per day are typical.

**Q: How do you know when the oil was released?**
From one image, you can't. We evaluate a window (6–48 h) with a uniform prior. For each candidate we estimate the time its track best matches the oil. Two images of the same spill would narrow this, and the multi-temporal module computes the drift validation error.

**Q: Is 80/100 the probability that the tanker did it?**
No. It's an Evidence Correlation Score: a transparent ranking heuristic that isn't calibrated. We show every component, and when candidates are within about 10 points we explicitly say they can't be separated.

**Q: What if the ship turned off its AIS?**
It then can't be ranked from AIS. We report AIS gaps as continuity anomalies with neutral wording, and the SAR ship-detection interface compares radar-visible ships with AIS positions to flag mismatches. The current detector is a baseline.

**Q: Is the data real?**
Yes. Sentinel-1 GRD scenes (Planetary Computer), Open-Meteo marine currents and ERA5 wind, and the Danish Maritime Authority's historical AIS archive are downloaded live, and every layer carries a provenance badge. Only the automated tests use a synthetic, labelled fixture (a planted-vessel twin experiment) so the science can be checked offline.

**Q: What does OpenOil do in backward mode?**
Transport only. Weathering (evaporation, emulsification, dispersion) can't be reversed, so it's disabled when running backward and enabled when running forward.

**Q: Why DeepLabV3+?**
Multi-scale context (ASPP) handles slicks from hundreds of metres to tens of kilometres, and the decoder recovers the thin, streaky boundaries typical of ship discharges. ResNet34 is a good accuracy/speed trade-off for a laptop GPU.

**Q: How would this be deployed for India?**
Plug the provider interfaces into national AIS feeds and INCOIS or CMEMS forcing, ingest Sentinel-1 automatically for the EEZ, store results in PostGIS, and give Coast Guard analysts the dashboard. The architecture doesn't change.

**Q: Do we need API keys?**
No. Every source in the working pipeline is key-free. Optional upgrades (CMEMS currents, Copernicus Data Space, Global Fishing Watch, Indian AIS) take keys the operator enters in a local `.env`; the app shows which are configured and never exposes the values.

**Q: Why did you need to "calibrate" the input?**
The SOS training tiles have undocumented 8-bit processing. We measured their statistics, normalised real Sentinel-1 relative to the local sea background, and chose the mapping with a controlled experiment: a clean real scene (negative control) plus implanted −3/−6 dB slicks (positive control). False positives fell from 15 % to 0.02 % with no retraining.
