"""Impact & response assessment for an investigated slick (runs after candidate scoring, or on demand).

Produces impact.json: threat analysis (coast distance, beaching ETA, threatened sensitive sites), the Spill
Severity & Impact Index, an immediate response action plan, next-port intercepts for the top candidates, a
ranking-robustness analysis and a draft POLREP. Every part degrades gracefully when inputs are missing."""
from __future__ import annotations

import logging
from datetime import datetime

from app.impact.coast import Coast
from app.impact.intercept import next_port
from app.impact.response import action_plan, polrep
from app.impact.robustness import ranking_robustness
from app.impact.severity import DEFAULTS, severity_index, threat_analysis

log = logging.getLogger(__name__)


def _opt(an, name):
    try:
        return an.load(name)
    except FileNotFoundError:
        return None


def _cfg(an) -> dict:
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    user = an.cfg.get("impact", {}) or {}
    for k, v in dict(user).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(dict(v))
        else:
            c[k] = v
    return c


def assess_impact(an, coast: Coast | None = None) -> dict:
    spill = _opt(an, "spill_selected.json")
    if not spill:
        raise FileNotFoundError("No investigated slick yet — run the investigation first.")
    cfg = _cfg(an)
    drift = _opt(an, "drift/summary.json")
    forward = _opt(an, "drift/forward_summary.json")
    fpart = _opt(an, "drift/forward_particles.json") if forward and forward.get("available") else None
    cand_doc = _opt(an, "candidates.json") or {}
    tracks = _opt(an, "ais/tracks.geojson") or {"features": []}
    acq = _opt(an, "acquisition.json") or {}
    c = spill["centroid"]
    coast = coast or Coast.around(c["lon"], c["lat"], 3.0)

    threat = threat_analysis(spill, forward, fpart, coast, cfg)
    sev = severity_index(spill, drift, threat, cfg)

    cands = cand_doc.get("candidates") or []
    qend, qbox = None, None
    try:
        qend = datetime.fromisoformat(cand_doc["ais"]["query"]["end"])
        qbox = cand_doc["ais"]["query"].get("bbox")
    except Exception:
        pass
    feats = {f["properties"]["mmsi"]: f for f in tracks.get("features", [])}
    intercepts = {}
    for cd in cands[:5]:
        f = feats.get(cd["vessel"]["mmsi"])
        if f:
            try:
                intercepts[cd["vessel"]["mmsi"]] = next_port(f, qend, qbox)
            except Exception as exc:                         # never fail the assessment on one track
                log.warning("intercept failed for %s: %s", cd["vessel"]["mmsi"], exc)
    robust = ranking_robustness(cands, cand_doc.get("weights") or {}) if cands else {"available": False, "reason": "no candidates"}

    synthetic = {"SAR scene": acq.get("provenance") == "SYNTHETIC_DEMO",
                 "currents & wind": str(((drift or {}).get("forcing") or {}).get("provenance", "")).startswith("SYNTHETIC"),
                 "AIS": bool((cand_doc.get("ais") or {}).get("synthetic"))}
    provenance = {"sar_image": "SYNTHETIC scene" if synthetic["SAR scene"] else f"Sentinel-1 {acq.get('item_id', '')}".strip(),
                  "environmental_forcing": ((drift or {}).get("forcing") or {}).get("provenance", "UNAVAILABLE"),
                  "ais": (((cand_doc.get("ais") or {}).get("provider")) or {}).get("provenance", "UNAVAILABLE")}
    plan = action_plan(spill, drift, forward, threat, sev, cands, intercepts, cand_doc, synthetic)
    text = polrep(spill, drift, forward, threat, sev, cands, plan, provenance, synthetic)
    out = {"generated_at": datetime.now().astimezone().isoformat(), "severity": sev, "threat": threat,
           "response": plan, "intercepts": intercepts, "robustness": robust, "polrep": text, "synthetic_inputs": synthetic,
           "gazetteer_note": "Sensitive-site coordinates are approximate reference points, not official ESI maps."}
    an.save("impact.json", out)
    return out


def seal_evidence(an) -> dict:
    from app.impact.custody import seal
    return seal(an.repo.dir(an.id), an.id)
