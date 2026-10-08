"""Coastline geometry for impact analysis (GSHHG full-resolution land polygons via roaring_landmask — the
same coastline OpenDrift uses for stranding). Geometry is clipped to a regional box and projected to a local
equirectangular km grid so distances are in kilometres."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

KM = 111.32


@lru_cache(maxsize=1)
def _world():
    import roaring_landmask as rl
    import shapely.wkb as wkb
    g = wkb.loads(bytes(rl.Shapes.wkb(rl.LandmaskProvider.Gshhg)))
    polys = np.array(list(g.geoms), dtype=object)
    return polys, shapely.STRtree(polys)


class Coast:
    """Land geometry around (lon0, lat0) in a local km frame."""

    def __init__(self, land_lonlat: BaseGeometry, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = KM * math.cos(math.radians(lat0))
        self.land_ll = land_lonlat
        self.land = shapely.transform(land_lonlat, lambda xy: np.c_[(xy[:, 0] - lon0) * self.kx, (xy[:, 1] - lat0) * KM])
        shapely.prepare(self.land)
        self.shore = self.land.boundary

    @classmethod
    def around(cls, lon: float, lat: float, half_deg: float = 3.0) -> "Coast":
        polys, tree = _world()
        b = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
        idx = tree.query(shapely.box(*b))
        parts = [p for p in shapely.clip_by_rect(polys[idx], *b) if not p.is_empty] if len(idx) else []
        land = shapely.union_all(parts) if parts else shapely.Polygon()
        return cls(land, lon, lat)

    # ---- coordinate helpers --------------------------------------------------------------------
    def to_km(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * KM

    def to_ll(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / KM

    def geom_km(self, g_lonlat: BaseGeometry) -> BaseGeometry:
        return shapely.transform(g_lonlat, lambda xy: np.c_[(xy[:, 0] - self.lon0) * self.kx, (xy[:, 1] - self.lat0) * KM])

    # ---- queries ---------------------------------------------------------------------------------
    @property
    def empty(self) -> bool:
        return self.land.is_empty

    def is_land(self, lon, lat) -> np.ndarray:
        if self.empty:
            return np.zeros(np.shape(lon), bool)
        x, y = self.to_km(lon, lat)
        return shapely.contains_xy(self.land, x, y)

    def distance_km(self, g_lonlat: BaseGeometry) -> tuple[float | None, tuple[float, float] | None]:
        """Shortest distance from a geometry to the shoreline, and the nearest shore point (lon, lat)."""
        if self.empty:
            return None, None
        g = self.geom_km(g_lonlat)
        if g.intersects(self.land):
            p = g.intersection(self.land).representative_point()
            lo, la = self.to_ll(p.x, p.y)
            return 0.0, (float(lo), float(la))
        line = shapely.shortest_line(g, self.shore)
        (x1, y1), (x2, y2) = line.coords
        lo, la = self.to_ll(x2, y2)
        return float(math.hypot(x2 - x1, y2 - y1)), (float(lo), float(la))

    def ray_to_shore(self, lon: float, lat: float, bearing_deg: float, max_km: float = 400.0, step_km: float = 0.5):
        """March along a bearing until land; returns (distance_km, (lon, lat)) or (None, None)."""
        if self.empty:
            return None, None
        x0, y0 = self.to_km(lon, lat)
        d = np.arange(step_km, max_km + step_km, step_km)
        b = math.radians(bearing_deg)
        xs, ys = x0 + d * math.sin(b), y0 + d * math.cos(b)
        hit = shapely.contains_xy(self.land, xs, ys)
        if not hit.any():
            return None, None
        i = int(np.argmax(hit))
        lo, la = self.to_ll(xs[i], ys[i])
        return float(d[i]), (float(lo), float(la))


def nearest_place(lon: float, lat: float) -> tuple[str, float]:
    from app.impact.gazetteer import PLACES
    kx = KM * math.cos(math.radians(lat))
    best = min(PLACES, key=lambda p: ((p[1] - lon) * kx) ** 2 + ((p[2] - lat) * KM) ** 2)
    return best[0], math.hypot((best[1] - lon) * kx, (best[2] - lat) * KM)
