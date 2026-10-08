"""10-13: AIS parsing, trajectory construction, cleaning, corridor filtering, evidence scoring."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from app.ais.corridor_filter import Corridor, match_track
from app.ais.gaps import find_gaps
from app.ais.local_provider import LocalAISProvider
from app.ais.track_processing import clean_and_build, normalize_type
from app.drift.backtrack import EnsembleResult
from app.geospatial.polygon import local_projection
from app.scoring.features import compute_features
from app.scoring.scorer import feature_scores, total_score

T0 = datetime(2025, 6, 10, 0, 0, tzinfo=timezone.utc)


def _csv(tmp_path, rows, header="mmsi,timestamp,lat,lon,sog,cog,vessel_name,imo,vessel_type", prov=None):
    p = tmp_path / "ais.csv"
    txt = (f"# PROVENANCE: {prov}\n" if prov else "") + header + "\n" + "\n".join(rows) + "\n"
    p.write_text(txt)
    return p


def test_parse_marinecadastre_columns(tmp_path):
    p = _csv(tmp_path, ["123,2025-06-10T00:00:00,18.0,72.0,10,90,SHIP A,IMO1,80",
                        "123,2025-06-10T00:10:00,18.0,72.05,10,90,SHIP A,IMO1,80"],
             header="MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName,IMO,VesselType")
    prov = LocalAISProvider([p], "marinecadastre")
    df = prov.get_tracks([71, 17, 73, 19], T0 - timedelta(hours=1), T0 + timedelta(hours=1))
    assert len(df) == 2 and prov.get_vessel_metadata("123")["name"] == "SHIP A"
    assert prov.provenance == "LOCAL_FILE"


def test_vessel_type_normalisation():
    assert normalize_type(84) == "tanker" and normalize_type("70") == "cargo" and normalize_type("30") == "fishing"
    assert normalize_type("Bulk Carrier") == "bulk_carrier" and normalize_type(None) == "unknown"


def test_cleaning_reports_every_problem():
    ts = [T0 + timedelta(minutes=10 * i) for i in range(6)]
    df = pd.DataFrame({"mmsi": ["1"] * 6, "timestamp": ts, "lat": [18.0, 18.0, 18.01, 95.0, 18.02, 18.03],
                       "lon": [72.0, 72.0, 72.01, 72.0, 73.5, 72.03], "sog": 10.0, "cog": 45.0,
                       "vessel_name": "X", "imo": None, "vessel_type": "tanker"})
    df.loc[1, "timestamp"] = ts[0]                                   # duplicate
    df = pd.concat([df, pd.DataFrame({"mmsi": ["1"], "timestamp": [pd.NaT], "lat": [18.0], "lon": [72.0]})])
    tracks, rep = clean_and_build(df, max_speed_kn=45)
    assert rep["duplicates"] == 1 and rep["invalid_coordinates"] == 1 and rep["missing_timestamp"] == 1
    assert rep["impossible_speed"] == 1                              # 73.5 E jump in 10 min
    t = tracks[0]
    assert list(t.df["lat"]) == [18.0, 18.01, 18.03] and t.df["timestamp"].is_monotonic_increasing
    assert t.df["speed_kn"].iloc[1] == pytest.approx(1.535 / 1.852 * 3, rel=0.02)   # 0.01 deg lat+lon (1.535 km) in 20 min


def test_gap_is_continuity_anomaly_not_evasion():
    ts = [T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=95), T0 + timedelta(minutes=100)]
    df = pd.DataFrame({"mmsi": "1", "timestamp": ts, "lat": [18, 18.01, 18.2, 18.21], "lon": 72.0, "sog": 10.0,
                       "cog": 0.0, "vessel_name": "X", "imo": None, "vessel_type": None})
    tracks, _ = clean_and_build(df)
    gaps = find_gaps(tracks[0], 30)
    assert len(gaps) == 1 and gaps[0]["duration_min"] == 90 and gaps[0]["label"] == "AIS continuity anomaly"


def _synthetic_corridor():
    """Oil moving due east at ~1 km/h along 18.7N between T0-10h and T0."""
    fwd, inv = local_projection(72.0, 18.7)
    times = [T0 - timedelta(hours=h) for h in range(0, 11)]
    n = 40
    rng = np.random.default_rng(0)
    lon = np.zeros((1, n, len(times))); lat = np.zeros_like(lon)
    for j, t in enumerate(times):
        h = (T0 - t).total_seconds() / 3600
        x = -h * 1000 + rng.normal(0, 200, n)
        y = rng.normal(0, 200, n)
        lo, la = inv.transform(x, y)
        lon[0, :, j], lat[0, :, j] = lo, la
    ens = EnsembleResult("backward", T0, times, lon, lat, [])
    region_m = Point(-6000, 0).buffer(2500)
    from shapely.ops import transform as tr
    regs = {"high": {"_shape": tr(inv.transform, region_m)}}
    return Corridor.from_ensemble(ens, fwd, inv, (T0 - timedelta(hours=10), T0 - timedelta(hours=1)), regs)


def _track(lat0, lon0, lat1, lon1, t_start, hours, mmsi, vtype="tanker"):
    ts = [t_start + timedelta(minutes=5 * i) for i in range(int(hours * 12) + 1)]
    f = np.linspace(0, 1, len(ts))
    df = pd.DataFrame({"mmsi": mmsi, "timestamp": ts, "lat": lat0 + f * (lat1 - lat0), "lon": lon0 + f * (lon1 - lon0),
                       "sog": 10.0, "cog": 0.0, "vessel_name": f"V{mmsi}", "imo": None, "vessel_type": vtype})
    return clean_and_build(df)[0][0]


def test_corridor_filter_and_scoring_rank_spacetime_match_first(cfg):
    cor = _synthetic_corridor()
    # oil passes x=-6 km (lon ~71.943) at T0-6h. Vessel A crosses there at T0-6h (N->S);
    # vessel B crosses the same place 9 h too late? (outside window) -> filtered; vessel C crosses 15 km away.
    lon_x = cor.inv.transform(-6000, 0)[0]
    a = _track(18.80, lon_x, 18.60, lon_x, T0 - timedelta(hours=7), 2, "A")
    b = _track(18.80, lon_x, 18.60, lon_x, T0 + timedelta(hours=1), 2, "B")
    c_lon = cor.inv.transform(-6000 + 30000, 0)[0]
    c = _track(18.80, c_lon, 18.60, c_lon, T0 - timedelta(hours=7), 2, "C")
    ma, mb, mc = (match_track(t, cor, cfg.ais) for t in (a, b, c))
    assert ma.is_candidate and ma.min_d_st_km < 1.0
    assert not mb.is_candidate and "release window" in mb.reason
    assert not mc.is_candidate
    fa = compute_features(a, ma, cor, cfg)
    assert fa["temporal"]["time_difference_hours"] < 1.5
    fs = feature_scores(fa, cfg)
    assert all(0 <= v <= 1 for v in fs.values())
    s = total_score(fs, dict(cfg.scoring.weights))
    assert 0 <= s <= 100 and s > 60


def test_weights_are_configurable(cfg):
    fs = {"spatial": 1, "temporal": 0, "trajectory": 0, "vessel_type": 0, "behavior": 0, "ais_continuity": 0}
    w = dict(cfg.scoring.weights)
    assert total_score(fs, w) == pytest.approx(100 * w["spatial"] / sum(w.values()))
    assert total_score(fs, {**dict(cfg.scoring.weights), "spatial": 0.0}) == pytest.approx(0.0)
