"""Real Sentinel-1 acquisition (Microsoft Planetary Computer, anonymous — no API key).

search_scenes()  STAC search of the `sentinel-1-grd` collection (IW, VV) by bbox + dates.
ingest_scene()   Reads only the area of interest from the cloud-optimised GRD GeoTIFF:
                 1. fit a 2nd-order polynomial lon/lat -> row/col from the product's GCPs,
                 2. read the covering source window, decimated to the target resolution,
                 3. reproject with the (window-adjusted) GCPs onto a regular EPSG:4326 grid,
                 4. DN -> dB (DN^2) -> 2-98 percentile stretch over SEA pixels -> uint8,
                 5. write a GeoTIFF with ACQUISITION_START_TIME + provenance tags and a sea mask.
Step 4 is an ASSUMPTION: the SOS tiles' exact radiometric processing is undocumented, so we match their
MEASURED statistics (background ~130, speckle std ~50 grey levels, native-resolution speckle) instead.
GRD DNs are not calibrated sigma0; the local-background normalisation removes the calibration offset.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.errors import InputImageError, PipelineError

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-grd"


def _client():
    try:
        import planetary_computer as pc
        import pystac_client
    except ImportError as exc:
        raise PipelineError("pystac-client / planetary-computer not installed",
                            "pip install pystac-client planetary-computer") from exc
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def search_scenes(bbox: list[float], start: str, end: str, limit: int = 60) -> list[dict]:
    try:
        items = list(_client().search(collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
                                      max_items=limit).items())
    except Exception as exc:
        raise PipelineError(f"Sentinel-1 search failed: {exc}", "Check internet access to planetarycomputer.microsoft.com")
    from shapely.geometry import box, shape
    aoi = box(*bbox)
    out = []
    for it in items:
        p = it.properties
        if p.get("sar:instrument_mode") != "IW" or "VV" not in (p.get("sar:polarizations") or []):
            continue
        fp = shape(it.geometry)
        out.append({"id": it.id, "datetime": p["datetime"], "platform": p.get("platform"),
                    "orbit_state": p.get("sat:orbit_state"), "relative_orbit": p.get("sat:relative_orbit"),
                    "polarizations": p.get("sar:polarizations"), "footprint": it.geometry,
                    "aoi_coverage": round(fp.intersection(aoi).area / aoi.area, 3),
                    "thumbnail": it.assets["thumbnail"].href if "thumbnail" in it.assets else None,
                    "provenance": "LIVE_EXTERNAL", "source": "Sentinel-1 GRD via Microsoft Planetary Computer"})
    out.sort(key=lambda s: s["datetime"], reverse=True)
    return out


def _grid(aoi: list[float], res_m: float) -> tuple[int, int]:
    lat_c = (aoi[1] + aoi[3]) / 2
    h = int(round((aoi[3] - aoi[1]) * 111320 / res_m))
    w = int(round((aoi[2] - aoi[0]) * 111320 * math.cos(math.radians(lat_c)) / res_m))
    if h * w > 60e6:
        raise InputImageError(f"AOI too large at {res_m} m ({w}x{h} px)", "Choose a smaller area or a coarser resolution.")
    return w, h


def sea_mask(aoi: list[float], w: int, h: int, coast_buffer_px: int = 2) -> np.ndarray:
    """True = sea (GSHHG full-resolution coastline via roaring_landmask), land dilated by a small buffer."""
    from scipy import ndimage
    from opendrift.readers.reader_global_landmask import get_mask
    lm = get_mask()            # process-wide cached GSHHG landmask (shared with OpenDrift; ~20 s to build once)
    lon = aoi[0] + (np.arange(w) + 0.5) * (aoi[2] - aoi[0]) / w
    lat = aoi[3] - (np.arange(h) + 0.5) * (aoi[3] - aoi[1]) / h
    LO, LA = np.meshgrid(lon, lat)
    land = lm.contains_many(LO.ravel(), LA.ravel()).reshape(h, w)
    if coast_buffer_px:
        land = ndimage.binary_dilation(land, iterations=coast_buffer_px)
    return ~land


def sos_normalize(db: np.ndarray, sea: np.ndarray, res_m: float, window_km: float = 10.0, percentile: float = 60.0,
                  target_bg: float = 150.0, target_std: float = 40.0) -> tuple[np.ndarray, dict]:
    """Map dB backscatter onto the radiometry of the SOS training tiles.

    Measured on the SOS tiles (oil-free pixels): background grey level ~130, speckle std ~50.
    1. local sea background = `percentile` of dB in ~500 m blocks, smoothed over `window_km`
       (flattens incidence-angle and large-scale wind gradients; standard in SAR oil detection),
    2. anomaly = dB - background; robust speckle std from the MAD of non-dark sea pixels,
    3. grey = target_bg + anomaly * target_std / std.  Land/no-data -> target_bg (neutral).
    """
    from scipy import ndimage
    h, w = db.shape
    f = max(int(round(500 / res_m)), 1)
    H, W = -(-h // f) * f, -(-w // f) * f
    pad = np.full((H, W), np.nan, np.float32)
    pad[:h, :w] = np.where(sea, db, np.nan)
    blocks = pad.reshape(H // f, f, W // f, f).transpose(0, 2, 1, 3).reshape(H // f, W // f, f * f)
    valid = np.isfinite(blocks).sum(-1) >= max(4, f * f // 4)
    with np.errstate(all="ignore"):
        coarse = np.where(valid, np.nanpercentile(blocks, percentile, axis=-1), np.nan)
    if not np.isfinite(coarse).any():
        raise InputImageError("No valid sea backscatter to normalise.", "Choose an AOI over open water.")
    idx = ndimage.distance_transform_edt(~np.isfinite(coarse), return_distances=False, return_indices=True)
    coarse = coarse[tuple(idx)]                                   # fill land/no-data with nearest sea value
    k = max(int(round(window_km * 1000 / (f * res_m))), 1)
    coarse = ndimage.median_filter(coarse, size=k, mode="nearest")
    bg = ndimage.zoom(coarse, (H / coarse.shape[0], W / coarse.shape[1]), order=1)[:h, :w]
    anom = db - bg
    core = sea & (anom > -3)                                     # exclude dark features from the speckle estimate
    mad = np.median(np.abs(anom[core] - np.median(anom[core])))
    s = max(1.4826 * mad, 0.3)
    gray_per_db = target_std / s
    u8 = np.clip(target_bg + anom * gray_per_db, 0, 255)
    u8[~sea] = target_bg
    sos_normalize.last_anomaly = np.where(sea, anom, 0).astype(np.float16)   # dB above local sea (ship detection)
    return u8.astype(np.uint8), {"mode": "sos_matched", "background_percentile": percentile, "window_km": window_km,
                                 "speckle_std_db": float(s), "gray_per_db": float(gray_per_db)}


def clip_aoi_to_scene(item_id: str, aoi: list[float], min_fraction: float = 0.05) -> tuple[list[float], float]:
    """Bounding box of (AOI ∩ scene footprint) and the covered fraction. Partially covering scenes are cropped
    instead of downloading and processing large no-data areas."""
    from shapely.geometry import box, shape
    item = _client().get_collection(COLLECTION).get_item(item_id)
    if item is None:
        raise InputImageError(f"Sentinel-1 item {item_id} not found", "Search scenes again.")
    a = box(*aoi)
    inter = shape(item.geometry).intersection(a)
    frac = inter.area / a.area if a.area else 0.0
    if frac < min_fraction:
        raise InputImageError(f"The selected scene covers only {100 * frac:.0f}% of the area of interest.",
                              "Pick a scene with more coverage, move the box inside a footprint, or use a SYNTHETIC "
                              "scene for demonstration.")
    if frac > 0.95:
        return aoi, frac
    return [round(v, 5) for v in inter.bounds], frac


def ingest_scene(item_id: str, aoi: list[float], out_dir: Path, res_m: float = 10.0,
                 polarization: str = "vv", progress=None, norm: dict | None = None) -> dict:
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    from rasterio.windows import Window

    item = _client().get_collection(COLLECTION).get_item(item_id)
    if item is None:
        raise InputImageError(f"Sentinel-1 item {item_id} not found", "Search scenes again.")
    from shapely.geometry import box, shape
    if not shape(item.geometry).intersects(box(*aoi)):
        raise InputImageError("The selected scene does not cover the area of interest.", "Pick a scene whose footprint overlaps the AOI.")
    w, h = _grid(aoi, res_m)
    href = item.assets[polarization].href
    from app.sar.cogread import GDAL_ENV, read_window
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            gcps, gcrs = src.gcps
            src_h, src_w = src.height, src.width
    G = np.array([[g.row, g.col, g.x, g.y] for g in gcps])
    feats = lambda x, y: np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])
    A = feats(G[:, 2], G[:, 3])
    cr = np.linalg.lstsq(A, G[:, 0], rcond=None)[0]
    cc = np.linalg.lstsq(A, G[:, 1], rcond=None)[0]
    xs = np.array([aoi[0], aoi[2], aoi[2], aoi[0], (aoi[0] + aoi[2]) / 2])
    ys = np.array([aoi[1], aoi[1], aoi[3], aoi[3], (aoi[1] + aoi[3]) / 2])
    rr, ccol = feats(xs, ys) @ cr, feats(xs, ys) @ cc
    r0, r1 = max(int(rr.min()) - 64, 0), min(int(rr.max()) + 64, src_h)
    c0, c1 = max(int(ccol.min()) - 64, 0), min(int(ccol.max()) + 64, src_w)
    if r1 <= r0 or c1 <= c0:
        raise InputImageError("AOI falls outside the scene raster.", "Pick another scene.")
    f = max(int(round(res_m / 10.0)), 1)                          # GRDH pixel spacing is 10 m
    if progress:
        progress(f"Streaming {c1 - c0}x{r1 - r0} px window of {item_id} (parallel tile reads)")
    arr, _ = read_window(href, r0, c0, r1 - r0, c1 - c0, f, progress=progress)
    sub = [GroundControlPoint(row=(g.row - r0) / f, col=(g.col - c0) / f, x=g.x, y=g.y) for g in gcps]
    dst = np.zeros((h, w), np.float32)
    transform = from_bounds(*aoi, w, h)
    reproject(arr, dst, gcps=sub, src_crs=gcrs, dst_crs="EPSG:4326", dst_transform=transform,
              resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    sea = sea_mask(aoi, w, h) & (dst > 0)
    if sea.sum() < 1000:
        raise InputImageError("Almost no valid sea pixels in the AOI for this scene.", "Choose an area over water.")
    db = (10 * np.log10(np.where(dst > 0, dst, 1.0) ** 2)).astype(np.float32)
    if progress:
        progress("Radiometric normalisation to SOS training statistics")
    u8, radiometry = sos_normalize(db, sea, res_m, **(norm or {}))
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{item_id}_{polarization}_{res_m:g}m.tif"
    acq = item.properties.get("start_datetime") or item.properties["datetime"]
    with rasterio.open(tif, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=transform, compress="deflate") as ds:
        ds.write(u8, 1)
        ds.update_tags(ACQUISITION_START_TIME=item.properties["datetime"], SCENE_START=acq, SOURCE_ITEM=item_id,
                       PLATFORM=str(item.properties.get("platform")), POLARIZATION=polarization.upper(),
                       PROVENANCE="LIVE_EXTERNAL",
                       SCALING=f"sos_matched gray_per_db={radiometry['gray_per_db']:.3f} speckle_std_db={radiometry['speckle_std_db']:.2f}",
                       PIXEL_SPACING_M=str(res_m))
    np.save(out_dir / (tif.stem + "_sea.npy"), sea)
    np.save(out_dir / (tif.stem + "_anom.npy"), sos_normalize.last_anomaly)     # unclipped dB anomaly
    return {"path": str(tif), "sea_mask": str(out_dir / (tif.stem + "_sea.npy")), "item_id": item_id,
            "acquired": item.properties["datetime"], "platform": item.properties.get("platform"),
            "orbit_state": item.properties.get("sat:orbit_state"), "aoi": aoi, "resolution_m": res_m,
            "size": [w, h], "sea_fraction": float(sea.mean()), "radiometry": radiometry,
            "provenance": "LIVE_EXTERNAL"}
