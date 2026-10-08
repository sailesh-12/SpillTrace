"""Open-Meteo quota handling: cache reuse and fail-fast on hourly limits (no network needed)."""
from datetime import datetime, timezone

import httpx
import pytest

from app.core.errors import ForcingUnavailable
from app.environmental import forcing as F


def test_covering_cache_file_is_reused(tmp_path):
    f = tmp_path / "om_currents_10.00_56.00_12.50_57.50_2026061700_2026062100.nc"
    f.write_bytes(b"x")
    t0, t1 = datetime(2026, 6, 18, tzinfo=timezone.utc), datetime(2026, 6, 20, tzinfo=timezone.utc)
    assert F.OpenMeteoForcingProvider._covering(tmp_path, "currents", [10.5, 56.0, 12.0, 57.0], t0, t1) == f
    assert F.OpenMeteoForcingProvider._covering(tmp_path, "currents", [9.5, 56.0, 12.0, 57.0], t0, t1) is None
    assert F.OpenMeteoForcingProvider._covering(tmp_path, "wind", [10.5, 56.0, 12.0, 57.0], t0, t1) is None


def test_hourly_limit_fails_fast_with_actionable_message(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return httpx.Response(429, json={"error": True, "reason": "Hourly API request limit exceeded."},
                              request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ForcingUnavailable) as e:
        F.OpenMeteoForcingProvider._get("https://example.test", {"latitude": "1,2"})
    assert len(calls) == 1                           # no multi-minute retry loop
    assert "Hourly" in e.value.message and "netcdf" in e.value.hint
