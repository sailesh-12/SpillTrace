"""Real/synthetic AIS selection and the SYNTHETIC generator (Indian waters)."""
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.ais.gfw_provider import GFWPresenceAISProvider
from app.ais.selection import select_ais
from app.ais.synthetic_provider import SyntheticAISProvider, region_preset
from app.ais.track_processing import clean_and_build
from app.core.config import load_config
from app.core.errors import AISDataError

OBS = datetime(2026, 6, 10, 5, 30, tzinfo=timezone.utc)
MUMBAI = [71.3, 18.4, 72.3, 19.2]
CTX = {"spill_centroid": {"lat": 18.8, "lon": 71.8}, "observation_time": OBS,
       "corridor_point": lambda t: (18.8 + 0.002 * (OBS - t).total_seconds() / 3600,
                                    71.8 - 0.004 * (OBS - t).total_seconds() / 3600),
       "release_window": (OBS - timedelta(hours=48), OBS - timedelta(hours=1)), "seed_key": "T"}


def _gen(**kw):
    p = SyntheticAISProvider({"vessels": 30, **kw}, CTX, "T")
    return p, p.get_tracks(MUMBAI, OBS - timedelta(hours=54), OBS + timedelta(hours=6))


def test_presets_cover_indian_waters():
    assert region_preset(MUMBAI) == "arabian_sea_west_coast"
    assert region_preset([80.0, 13.0, 81.0, 14.0]) == "bay_of_bengal_east_coast"
    assert region_preset([79.0, 5.3, 81.5, 6.3]) == "sri_lanka_east_west_route"


def test_synthetic_ais_is_realistic_labelled_and_deterministic():
    p, df = _gen()
    assert set(df.columns) >= {"mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_name", "vessel_type"}
    assert df["mmsi"].str.fullmatch(r"419\d{6}").all()            # Indian MID 419
    assert df["vessel_name"].str.startswith("SYN-").all() and df["imo"].isna().all()
    assert {"Tanker", "Fishing", "Container"} <= set(df["vessel_type"])
    assert p.provenance == "SYNTHETIC_DEMO" and p.describe()["synthetic"] is True
    roles = {v["role"] for v in p.truth["vessels"]}
    assert roles == {"planted_release", "outside_release_window", "nearest_vessel_trap"}
    tracks, rep = clean_and_build(df, 45)
    assert rep["duplicates"] > 0 and rep["invalid_coordinates"] > 0 and rep["impossible_speed"] > 0
    assert sum((t.df["dt_s"] > 1800).sum() for t in tracks) > 0     # AIS gaps present
    assert df.equals(_gen()[1])                                     # reproducible per analysis


def test_auto_falls_back_to_synthetic_in_indian_waters(monkeypatch):
    monkeypatch.delenv("GFW_API_TOKEN", raising=False)
    cfg = load_config(overrides={"ais": {"mode": "auto", "real_provider": "auto", "files": []}})
    df, prov, info = select_ais(cfg, MUMBAI, OBS - timedelta(hours=10), OBS, CTX)
    assert info["source"] == "synthetic" and prov.provenance == "SYNTHETIC_DEMO" and len(df)
    assert "GFW_API_TOKEN" in info["attempts"][0]["outcome"] or info["fallback_reason"]


def test_real_mode_never_falls_back(monkeypatch):
    monkeypatch.delenv("GFW_API_TOKEN", raising=False)
    cfg = load_config(overrides={"ais": {"mode": "real", "real_provider": "auto", "files": []}})
    with pytest.raises(AISDataError):
        select_ais(cfg, MUMBAI, OBS - timedelta(hours=10), OBS, CTX)


def test_gfw_presence_parsing_and_request(monkeypatch):
    payload = {"entries": [{"public-global-presence:v3.0": [
        {"date": "2026-06-09T14", "lat": 18.81, "lon": 71.75, "mmsi": "419000123", "shipName": "REAL SHIP",
         "vesselType": "CARGO", "hours": 1},
        {"date": "2026-06-09T15", "lat": 18.83, "lon": 71.80, "mmsi": "419000123", "shipName": "REAL SHIP",
         "vesselType": "CARGO", "hours": 1}]}]}
    df = GFWPresenceAISProvider.parse(payload)
    assert len(df) == 2 and df["timestamp"].iloc[0].hour == 14 and df["vessel_type"].iloc[0] == "CARGO"
    seen = {}

    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        seen.update(params=params, headers=headers, body=json)
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    p = GFWPresenceAISProvider(token="test-token")
    out = p.get_tracks(MUMBAI, datetime(2026, 6, 9, tzinfo=timezone.utc), datetime(2026, 6, 9, 23, tzinfo=timezone.utc))
    assert len(out) == 2 and seen["params"]["datasets[0]"] == "public-global-presence:latest"
    assert seen["params"]["temporal-resolution"] == "HOURLY" and seen["headers"]["Authorization"] == "Bearer test-token"
    assert p.provenance == "LIVE_EXTERNAL" and p.describe()["synthetic"] is False


def test_gfw_without_token_is_actionable(monkeypatch):
    monkeypatch.delenv("GFW_API_TOKEN", raising=False)
    with pytest.raises(AISDataError) as e:
        GFWPresenceAISProvider()
    assert "GFW_API_TOKEN" in e.value.message


def test_synthetic_traffic_capped_and_smooth():
    import numpy as np
    for key in ("T", "U", "V"):
        p = SyntheticAISProvider({"vessels": 40}, CTX, key)          # config above the cap is clamped to 10
        df = p.get_tracks(MUMBAI, OBS - timedelta(hours=54), OBS + timedelta(hours=6))
        assert df["mmsi"].nunique() <= 10
        assert (df["vessel_type"].groupby(df["mmsi"]).first() == "Fishing").sum() <= 2
        tracks, _ = clean_and_build(df, 45.0)
        for t in tracks:
            if len(t.df) < 20 or t.vessel_type == "Fishing":
                continue
            # transiting vessels: no zig-zag — course over ground changes < 5 deg/min on average
            c = np.unwrap(np.radians(t.df["cog"].to_numpy()))
            dt_min = np.diff(t.df["timestamp"].astype("int64").to_numpy()) / 6e10
            rate = np.degrees(np.abs(np.diff(c))) / np.maximum(dt_min, 1)
            assert np.median(rate) < 5, (t.name, float(np.median(rate)))
            # and not a ruler-straight line: heading varies by a few degrees along the track
            assert np.degrees(np.ptp(c)) > 2


def test_candidates_capped_at_ten():
    assert load_config().scoring.max_candidates == 10
