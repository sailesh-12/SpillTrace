"""Novelty features: severity index, shoreline threat, response plan, intercepts, robustness, evidence seal."""
from datetime import datetime, timedelta, timezone

import shapely

from app.impact.coast import Coast
from app.impact.custody import seal, verify
from app.impact.intercept import next_port
from app.impact.response import action_plan, polrep
from app.impact.robustness import ranking_robustness
from app.impact.severity import DEFAULTS, severity_index, threat_analysis

OBS = datetime(2026, 9, 27, 0, 30, tzinfo=timezone.utc)
# synthetic "coast": land east of 72.9 E (like the Mumbai coastline), no GSHHG load needed
LAND = shapely.box(72.9, 17.0, 75.0, 21.0)


def _spill(lon=72.5, lat=18.9):
    return {"timestamp": OBS.isoformat(), "centroid": {"lon": lon, "lat": lat}, "area_m2": 4.0e6,
            "polygon": shapely.geometry.mapping(shapely.box(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01))}


def _forward(lon0, lat0, dlon_per_h, hours=24, n=200):
    times = [(OBS + timedelta(hours=h)).isoformat() for h in range(hours + 1)]
    frames = [[[lon0 + dlon_per_h * h + 0.002 * (i % 10), lat0 + 0.002 * (i // 10 % 10)] for i in range(n)] for h in range(hours + 1)]
    fwd = {"available": True, "hours": hours, "end_center": {"lon": lon0 + dlon_per_h * hours, "lat": lat0}}
    return fwd, {"times": times, "frames": frames}


DRIFT = {"oil": {"oil_type": "heavy fuel oil"}, "forcing_at_spill": {"wind_speed": 5.0, "wind_from_deg": 240,
                                                                       "current_speed": 0.2, "current_to_deg": 80}}


def test_threat_forward_beaching_and_severity():
    coast = Coast(LAND, 72.5, 18.9)
    fwd, fp = _forward(72.5, 18.9, 0.03)                      # ~3 km/h east -> reaches the coast in < 24 h
    th = threat_analysis(_spill(), fwd, fp, coast, DEFAULTS)
    assert 35 < th["distance_to_coast_km"] < 45
    assert th["beaching"]["method"] == "forward_model" and th["beaching"]["eta_hours"] < 24
    assert any(s["id"] == "mumbai_port" and s["threatened"] for s in th["sites"])
    sev = severity_index(_spill(), DRIFT, th, DEFAULTS)
    assert sev["level"] in ("HIGH", "CRITICAL") and 0 <= sev["score"] <= 100
    assert sev["response_tier"]["central"] == "Tier 1"
    assert abs(sum(c["points"] for c in sev["components"].values()) - sev["score"]) < 1e-6


def test_threat_offshore_moving_away_is_low():
    coast = Coast(LAND, 71.0, 18.9)
    fwd, fp = _forward(71.0, 18.9, -0.02)                     # drifting west, away from the coast
    th = threat_analysis(_spill(71.0, 18.9), fwd, fp, coast, DEFAULTS)
    assert th["beaching"]["eta_hours"] is None
    sev = severity_index(_spill(71.0, 18.9), {"oil": {"oil_type": "marine diesel"}, "forcing_at_spill": {"wind_speed": 12}}, th, DEFAULTS)
    assert sev["level"] in ("LOW", "MODERATE")


def test_action_plan_and_polrep():
    coast = Coast(LAND, 72.5, 18.9)
    fwd, fp = _forward(72.5, 18.9, 0.03)
    th = threat_analysis(_spill(), fwd, fp, coast, DEFAULTS)
    sev = severity_index(_spill(), DRIFT, th, DEFAULTS)
    cands = [{"rank": 1, "score": 80.0, "synthetic": True, "vessel": {"mmsi": "419900001", "name": "SYN-TANKER-01"}}]
    plan = action_plan(_spill(), DRIFT, fwd, th, sev, cands, {}, {}, {"AIS": True})
    titles = [a["title"] for a in plan["actions"]]
    assert plan["mrcc"]["name"] == "MRCC Mumbai"
    assert any("Report the detection" in t for t in titles) and any("Inspect SYN-TANKER-01" in t for t in titles)
    assert any("dispersants" in t.lower() for t in titles)
    assert [a["priority"] for a in plan["actions"]] == sorted(a["priority"] for a in plan["actions"])
    txt = polrep(_spill(), DRIFT, fwd, th, sev, cands, plan, {"sar_image": "x", "environmental_forcing": "y", "ais": "z"}, {"AIS": True})
    assert "NOT SENT" in txt and "SYNTHETIC" in txt and "18°54.0'N 072°30.0'E" in txt


def test_next_port_intercept_and_silence():
    t0 = int(OBS.timestamp())
    coords = [[72.0 + 0.02 * i, 18.94] for i in range(10)]            # heading east toward Mumbai at ~6.8 kn
    f = {"geometry": {"coordinates": coords}, "properties": {"t": [t0 + 600 * i for i in range(10)]}}
    r = next_port(f, OBS + timedelta(hours=8))
    assert r["port"] in ("Mumbai", "JNPT (Nhava Sheva)") and 6 < r["speed_kn"] < 7.5
    assert r["ais_silence"] and r["eta_hours_from_last_fix"] < 8


def test_ranking_robustness():
    w = {"spatial": .25, "temporal": .25, "trajectory": .15, "sar_attached": .15, "vessel_type": .07, "behavior": .08, "continuity": .05}
    strong = {k: 0.95 for k in w}
    weak = {k: 0.2 for k in w}
    cands = [{"rank": 1, "vessel": {"mmsi": "1", "name": "A"}, "feature_scores": strong},
             {"rank": 2, "vessel": {"mmsi": "2", "name": "B"}, "feature_scores": weak}]
    r = ranking_robustness(cands, w, n=500)
    assert r["verdict"] == "ROBUST" and r["candidates"][0]["p_rank1"] > 0.99


def test_evidence_seal_detects_tampering(tmp_path):
    (tmp_path / "candidates.json").write_text('{"a": 1}')
    (tmp_path / "drift").mkdir()
    (tmp_path / "drift" / "summary.json").write_text("{}")
    (tmp_path / "state.json").write_text("{}")
    m = seal(tmp_path, "X")
    assert m["n_files"] == 2 and verify(tmp_path)["ok"]
    (tmp_path / "candidates.json").write_text('{"a": 2}')
    v = verify(tmp_path)
    assert not v["ok"] and v["modified"] == ["candidates.json"]
    m2 = seal(tmp_path, "X")
    assert m2["version"] == 2 and m2["previous"]["root"] == m["root"] and verify(tmp_path)["ok"]
