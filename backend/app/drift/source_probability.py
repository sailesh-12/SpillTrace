"""Source probability surface from backward-ensemble particle positions.

For each candidate release time T-h we take every ensemble particle's position
at T-h, build a kernel density on a metric grid, and normalise it to a
probability mass per cell. The combined map averages the per-release-time maps
(a uniform prior over the configured release window — an explicit ASSUMPTION).

Regions are highest-density regions (HDR): the smallest set of cells holding
50 % / 80 % / 95 % of the probability mass -> high / medium / low regions.
We deliberately do not report a single "source point".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from app.geospatial.polygon import geod, local_projection


@dataclass
class ProbabilityGrid:
    x_edges: np.ndarray
    y_edges: np.ndarray
    fwd: object
    inv: object
    maps: dict = field(default_factory=dict)       # label -> 2D probability array (rows = y)
    release_times: dict = field(default_factory=dict)

    @property
    def centers(self):
        xc = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        yc = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        return np.meshgrid(xc, yc)

    def lonlat_centers(self):
        X, Y = self.centers
        lon, lat = self.inv.transform(X, Y)
        return np.asarray(lon), np.asarray(lat)

    def density_at(self, label: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Probability density (per km^2) of map `label` at arbitrary points (0 outside grid)."""
        x, y = self.fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        ix = np.searchsorted(self.x_edges, x) - 1
        iy = np.searchsorted(self.y_edges, y) - 1
        ok = (ix >= 0) & (iy >= 0) & (ix < len(self.x_edges) - 1) & (iy < len(self.y_edges) - 1)
        out = np.zeros(np.shape(x))
        cell_km2 = (self.x_edges[1] - self.x_edges[0]) * (self.y_edges[1] - self.y_edges[0]) / 1e6
        out[ok] = self.maps[label][iy[ok], ix[ok]] / cell_km2
        return out


def build_grid(ens, spill_centroid: dict, offsets_h: list[float], obs: datetime,
               resolution_m: float = 1000, bandwidth_m: float = 1500, pad_m: float = 8000) -> ProbabilityGrid:
    fwd, inv = local_projection(spill_centroid["lon"], spill_centroid["lat"])
    pts = {}
    for h in offsets_h:
        lon, lat = ens.positions_at(obs - timedelta(hours=h))
        ok = np.isfinite(lon) & np.isfinite(lat)
        x, y = fwd.transform(lon[ok], lat[ok])
        pts[h] = (np.asarray(x), np.asarray(y))
    allx = np.concatenate([p[0] for p in pts.values()])
    ally = np.concatenate([p[1] for p in pts.values()])
    xe = np.arange(allx.min() - pad_m, allx.max() + pad_m + resolution_m, resolution_m)
    ye = np.arange(ally.min() - pad_m, ally.max() + pad_m + resolution_m, resolution_m)
    grid = ProbabilityGrid(xe, ye, fwd, inv)
    sigma = bandwidth_m / resolution_m
    for h, (x, y) in pts.items():
        H, _, _ = np.histogram2d(y, x, bins=[ye, xe])
        K = gaussian_filter(H, sigma=sigma, mode="constant")
        label = f"T-{h:g}h"
        grid.maps[label] = K / K.sum() if K.sum() > 0 else K
        grid.release_times[label] = obs - timedelta(hours=h)
    grid.maps["combined"] = np.mean([grid.maps[f"T-{h:g}h"] for h in offsets_h], axis=0)
    return grid


def hdr_masks(p: np.ndarray, levels: dict) -> dict:
    """Highest-density-region masks for each named mass fraction."""
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    out = {}
    for name, frac in levels.items():
        k = int(np.searchsorted(csum, frac)) + 1
        thr = flat[order[min(k, len(order)) - 1]]
        out[name] = p >= thr
    return out


def _mask_polygon(mask: np.ndarray, grid: ProbabilityGrid):
    from rasterio import features
    from rasterio.transform import Affine
    res_x = grid.x_edges[1] - grid.x_edges[0]
    res_y = grid.y_edges[1] - grid.y_edges[0]
    aff = Affine(res_x, 0, grid.x_edges[0], 0, res_y, grid.y_edges[0])
    polys = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=aff) if v == 1]
    if not polys:
        return None
    return shp_transform(grid.inv.transform, unary_union(polys))


def regions(grid: ProbabilityGrid, levels: dict, label: str = "combined") -> dict:
    masks = hdr_masks(grid.maps[label], levels)
    out = {}
    for name, m in masks.items():
        poly = _mask_polygon(m, grid)
        if poly is None:
            continue
        area, _ = geod().geometry_area_perimeter(poly)
        out[name] = {"level": name, "mass_fraction": levels[name], "geometry": mapping(poly),
                     "area_km2": abs(area) / 1e6, "_shape": poly}
    return out


def weighted_center(grid: ProbabilityGrid, label: str) -> dict:
    X, Y = grid.centers
    p = grid.maps[label]
    x, y = float((X * p).sum()), float((Y * p).sum())
    sx = float(np.sqrt(((X - x) ** 2 * p).sum()))
    sy = float(np.sqrt(((Y - y) ** 2 * p).sum()))
    lon, lat = grid.inv.transform(x, y)
    return {"lon": float(lon), "lat": float(lat), "spread_km": float(np.hypot(sx, sy) / 1000)}


def to_geojson(grid: ProbabilityGrid, regs: dict, min_prob: float = 1e-5) -> dict:
    """GeoJSON: probability cells (points with probability/lat/lon/release_time) + HDR polygons."""
    lon, lat = grid.lonlat_centers()
    feats = []
    for label, p in grid.maps.items():
        rt = grid.release_times.get(label)
        idx = np.nonzero(p > min_prob)
        for i, j in zip(*idx):
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [round(float(lon[i, j]), 5), round(float(lat[i, j]), 5)]},
                          "properties": {"kind": "probability_cell", "map": label, "probability": float(p[i, j]),
                                         "lat": round(float(lat[i, j]), 5), "lon": round(float(lon[i, j]), 5),
                                         "release_time": rt.isoformat() if rt else "window"}})
    for name, r in regs.items():
        feats.append({"type": "Feature", "geometry": r["geometry"],
                      "properties": {"kind": "source_region", "level": name, "mass_fraction": r["mass_fraction"],
                                     "area_km2": round(r["area_km2"], 2)}})
    return {"type": "FeatureCollection", "features": feats}


def per_release_summary(grid: ProbabilityGrid, levels: dict) -> list[dict]:
    rows = []
    for label, rt in grid.release_times.items():
        c = weighted_center(grid, label)
        r = regions(grid, {"high": levels["high"]}, label)
        rows.append({"label": label, "release_time": rt.isoformat(), "center": {"lat": c["lat"], "lon": c["lon"]},
                     "spread_km": round(c["spread_km"], 2),
                     "high_region_area_km2": round(r["high"]["area_km2"], 2) if "high" in r else None})
    return rows
