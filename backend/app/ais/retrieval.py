"""Build the AIS query (space/time window) from the backtracking result."""
from __future__ import annotations

import math
from datetime import datetime, timedelta


def search_window(region_bounds: list[float], spill_bbox: list[float], obs: datetime, max_offset_h: float,
                  margin_km: float, time_margin_h: float) -> tuple[list[float], datetime, datetime]:
    """bbox covering source region + spill + margin; time covering release window + margin up to obs."""
    w = min(region_bounds[0], spill_bbox[0]); s = min(region_bounds[1], spill_bbox[1])
    e = max(region_bounds[2], spill_bbox[2]); n = max(region_bounds[3], spill_bbox[3])
    dlat = margin_km / 111.32
    dlon = margin_km / (111.32 * math.cos(math.radians((s + n) / 2)))
    bbox = [w - dlon, s - dlat, e + dlon, n + dlat]
    return bbox, obs - timedelta(hours=max_offset_h + time_margin_h), obs + timedelta(hours=time_margin_h)


def retrieve(provider, bbox, start, end):
    return provider.get_tracks(bbox, start, end)
