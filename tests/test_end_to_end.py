"""API endpoints and the full two-phase chain on the synthetic twin-experiment fixtures:
SAR -> segmentation -> triage -> (selected slicks) -> drift -> AIS -> score -> report."""
import json

import pytest

from app.services import pipeline as P
from conftest import FIX, FIXTURE_SCENE


@pytest.fixture(scope="module")
def e2e(cfg, demo_ready, model):
    an = P.run_detection(cfg, str(FIXTURE_SCENE))
    assert an.state["status"] == "AWAITING_SLICK_SELECTION", an.state.get("error")
    comps = [c["component_id"] for c in an.load("spill.json")["spill"]["components"]]
    return P.run_investigation(cfg, an.id, comps, particles=80, members=3)


def test_phase1_triage_table(e2e):
    t = e2e.load("triage.json")
    assert t["components"] and {"label", "triage_score", "ship_trail", "reasons"} <= set(t["components"][0])
    assert all(r["label"] in ("OIL_LIKELY", "LOOKALIKE_LIKELY", "UNCERTAIN") for r in t["components"])


def test_full_pipeline_recovers_planted_vessel_and_excludes_trap(e2e, cfg):
    assert e2e.state["status"] == "COMPLETED", e2e.state.get("error")
    cands = e2e.load("candidates.json")
    names = [c["vessel"]["name"] for c in cands["candidates"]]
    truth = json.loads((FIX / "ais" / "demo_scenario_truth.json").read_text())
    role = {v["role"]: v["name"] for v in truth["vessels"]}
    assert role["planted_source"] in names[:2]
    # detached slick -> window starts >= 1 h before the image, so the vessel crossing it at T is excluded
    assert role["nearest_vessel_trap"] not in names and role["temporal_mismatch"] not in names
    assert "detached" in e2e.load("drift/summary.json")["release_window"]["basis"]


def test_report_has_epistemic_labels_and_safeguards(e2e):
    rep = e2e.load("report.json")
    md = (e2e.repo.dir(e2e.id) / "report.md").read_text(encoding="utf-8")
    for label in ("OBSERVED_FACT", "MODEL_DERIVED", "CANDIDATE_INFERENCE", "ASSUMPTION"):
        assert label in md
    assert "does not establish" in md and "Evidence Correlation Score" in md
    assert "probability of responsibility" not in md.lower()
    assert rep["forward"]["available"] is True


def test_no_georef_gives_partial_with_actionable_message(cfg, model):
    an = P.run_full(cfg, str(cfg.path("data/sar/samples/1.png")))
    assert an.state["status"] == "PARTIAL"
    assert an.state["error"]["code"] == "GEOREFERENCING_UNAVAILABLE"


def test_no_forcing_gives_partial(cfg, model, demo_ready):
    from app.core.config import load_config
    c2 = load_config(overrides={**cfg, "environment": {**cfg["environment"], "mode": "real",
                                                       "current_files": ["tests/none.nc"]}})
    an = P.run_full(c2, str(FIXTURE_SCENE))
    assert an.state["status"] == "PARTIAL"
    assert "environmental forcing data are unavailable" in an.state["error"]["message"]


def test_api_endpoints(cfg, e2e, monkeypatch):
    from fastapi.testclient import TestClient
    import app.api.main as api
    monkeypatch.setattr(api, "cfg", cfg)
    client = TestClient(api.app)
    assert client.get("/api/health").json()["status"] == "ok"
    src = client.get("/api/v1/sources").json()
    assert any(s["id"] == "sentinel1_pc" and not s["key_required"] for s in src)
    assert all("value" not in s for s in src)                       # secrets never exposed
    sid = e2e.id
    assert client.get(f"/api/spill/{sid}/triage").json()["components"]
    v = client.get(f"/api/spill/{sid}/vessels").json()
    assert v and {"mmsi", "evidence_score", "feature_scores", "evidence_summary"} <= set(v[0])
    assert client.get(f"/api/spill/{sid}/source-probability").json()["geojson"]["type"] == "FeatureCollection"
    assert client.get(f"/api/spill/{sid}/report?format=html").status_code == 200
    lay = client.get(f"/api/spill/{sid}/layers").json()
    assert lay["selected"]["selected_component_ids"] and lay["triage"]
    assert client.get("/api/spill/NOPE/layers").status_code == 404
    assert client.get(f"/api/spill/{sid}/files/../../secret").status_code == 404
    assert client.post("/api/v1/analyses", json={"scene_id": "x", "aoi": [1, 1, 0, 0]}).status_code == 400
    assert "event: done" in client.get(f"/api/v1/analyses/{sid}/events").text


def test_synthetic_ais_investigation_is_labelled_end_to_end(cfg, e2e):
    an = P.run_investigation(cfg, e2e.id, None, particles=60, members=2, ais_mode="synthetic")
    assert an.state["status"] == "COMPLETED", an.state.get("error")
    c = an.load("candidates.json")
    assert c["ais"]["synthetic"] is True and c["ais"]["provider"]["provenance"] == "SYNTHETIC_DEMO"
    assert c["candidates"] and all(x["synthetic"] and x["vessel"]["name"].startswith("SYN-") for x in c["candidates"])
    assert all(s["text"].startswith("[SYNTHETIC AIS]") for x in c["candidates"] for s in x["evidence"]["statements"]
               if s["section"] in ("AIS", "Assessment"))
    md = (an.repo.dir(an.id) / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in md and "not real" in md.lower()
    assert any("SYNTHETIC" in w for w in an.state["warnings"])
    assert an.load("ais/synthetic_truth.json")["vessels"]


def test_impact_assessment_and_evidence_seal_end_to_end(cfg, e2e, monkeypatch):
    imp = e2e.load("impact.json")
    assert 0 <= imp["severity"]["score"] <= 100 and imp["severity"]["level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    assert imp["response"]["actions"] and imp["polrep"].startswith("POLREP")
    assert imp["robustness"]["available"]
    md = (e2e.repo.dir(e2e.id) / "report.md").read_text(encoding="utf-8")
    assert "Spill Severity & Impact Index" in md and "Immediate actions" in md
    from fastapi.testclient import TestClient
    import app.api.main as api
    monkeypatch.setattr(api, "cfg", cfg)
    client = TestClient(api.app)
    assert client.get(f"/api/spill/{e2e.id}/verify").json()["ok"] is True
    assert "NOT SENT" in client.get(f"/api/spill/{e2e.id}/polrep").text
    lay = client.get(f"/api/spill/{e2e.id}/layers").json()
    assert lay["impact"]["severity"] and lay["evidence_seal"]["root"]
