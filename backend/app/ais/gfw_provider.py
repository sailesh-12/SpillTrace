"""Real AIS-derived vessel presence for Indian waters: Global Fishing Watch 4Wings API.

Dataset `public-global-presence:latest` — ALL vessel types, one AIS position per vessel per hour, gridded to
0.01 deg (~1 km) with spatial-resolution=HIGH, grouped by vessel. It is NOT raw AIS: speed/course are derived
downstream from consecutive positions, positions are cell centres, and gaps < ~1 h cannot be resolved.
Access: free token for NON-COMMERCIAL use (https://globalfishingwatch.org/our-apis/tokens), set as
GFW_API_TOKEN in `.env`. Attribution to Global Fishing Watch is required.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.ais.provider import CANONICAL, AISProvider
from app.core.errors import AISDataError
from app.core.provenance import Provenance

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-presence:latest"


def _parse_ts(ts) -> pd.Timestamp:
    """GFW hourly bins look like '2024-01-13T14' (no minutes); full ISO timestamps also accepted."""
    s = str(ts)
    if "T" in s and len(s) == 13:
        s += ":00"
    return pd.to_datetime(s, utc=True, errors="coerce")


class GFWPresenceAISProvider(AISProvider):
    name = "Global Fishing Watch 4Wings AIS vessel presence (hourly, 0.01 deg)"
    provenance = Provenance.LIVE.value
    # hourly presence: consecutive fixes are ~60 min apart by construction, so AIS-gap detection must use a
    # longer threshold than for raw AIS
    gap_threshold_minutes = 180

    def __init__(self, token: str | None = None, progress=None, timeout_s: float = 120):
        self.token = token or os.environ.get("GFW_API_TOKEN")
        if not self.token:
            raise AISDataError("Global Fishing Watch token not configured (GFW_API_TOKEN).",
                               "Register at https://globalfishingwatch.org/our-apis/tokens (non-commercial) and add "
                               "GFW_API_TOKEN=... to .env, or use ais.mode: synthetic.")
        self.progress, self.timeout = progress, timeout_s
        self._meta: dict = {}

    def describe(self) -> dict:
        return {"provider": self.name, "provenance": self.provenance, "synthetic": False,
                "resolution": "one position per vessel per hour, 0.01 deg cells (not raw AIS messages)",
                "license": "Global Fishing Watch API, non-commercial use, attribution required"}

    def get_vessel_metadata(self, mmsi: str) -> dict:
        return {"mmsi": mmsi, **self._meta.get(str(mmsi), {})}

    # ---------------------------------------------------------------------------------------------
    def _request(self, bbox, day0: datetime, day1: datetime) -> dict:
        import httpx
        w, s, e, n = bbox
        region = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}
        params = {"datasets[0]": DATASET, "format": "JSON", "temporal-resolution": "HOURLY",
                  "spatial-resolution": "HIGH", "group-by": "VESSEL_ID", "spatial-aggregation": "false",
                  "date-range": f"{day0:%Y-%m-%d},{day1:%Y-%m-%d}"}
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            r = httpx.post(URL, params=params, headers=headers, json={"geojson": region}, timeout=self.timeout)
            if r.status_code in (400, 422):          # docs show the geojson as a JSON-encoded string
                r = httpx.post(URL, params=params, headers=headers, json={"geojson": json.dumps(region)},
                               timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AISDataError(f"Global Fishing Watch request failed: {exc}", "Check internet access.") from exc
        if r.status_code == 401:
            raise AISDataError("Global Fishing Watch rejected the token (401).", "Check GFW_API_TOKEN in .env.")
        if r.status_code == 429:
            raise AISDataError("Global Fishing Watch rate limit reached (429).", "Retry later.")
        if r.status_code >= 400:
            raise AISDataError(f"Global Fishing Watch error {r.status_code}: {r.text[:200]}", "See GFW API docs.")
        return r.json()

    @staticmethod
    def parse(payload: dict) -> pd.DataFrame:
        """4Wings report JSON -> CANONICAL rows. Handles both `entries: [row,...]` and
        `entries: [{dataset_id: [row,...]}]` layouts."""
        rows = []
        for ent in payload.get("entries", []) or []:
            items = ent if isinstance(ent, list) else (
                [r for v in ent.values() if isinstance(v, list) for r in v] if not ("lat" in ent) else [ent])
            for it in items:
                ts = it.get("entryTimestamp") or it.get("date")
                if ts is None or it.get("lat") is None:
                    continue
                t = _parse_ts(ts)
                rows.append({"mmsi": str(it.get("mmsi") or it.get("vesselId") or ""), "timestamp": t,
                             "lat": float(it["lat"]), "lon": float(it["lon"]), "sog": None, "cog": None,
                             "vessel_name": it.get("shipName"), "imo": it.get("imo"),
                             "vessel_type": it.get("vesselType")})
        df = pd.DataFrame(rows, columns=CANONICAL)
        return df.dropna(subset=["timestamp"])

    def get_tracks(self, bbox, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        day = datetime(start_time.year, start_time.month, start_time.day, tzinfo=timezone.utc)
        frames = []
        while day <= end_time:
            nxt = day + timedelta(days=1)
            if self.progress:
                self.progress(f"Global Fishing Watch presence {day:%Y-%m-%d}")
            frames.append(self.parse(self._request(bbox, day, nxt)))
            day = nxt
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL)
        df = df[(df["timestamp"] >= pd.Timestamp(start_time)) & (df["timestamp"] <= pd.Timestamp(end_time))]
        for m, g in df.groupby("mmsi"):
            self._meta[m] = {"name": g["vessel_name"].dropna().iloc[0] if g["vessel_name"].notna().any() else None,
                             "imo": None, "type_raw": g["vessel_type"].dropna().iloc[0] if g["vessel_type"].notna().any() else None}
        return df.reset_index(drop=True)
