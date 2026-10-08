"""Scene reading + georeferencing.

The segmentation model knows nothing about latitude/longitude. Geolocation comes
only from:
  1. GeoTIFF metadata (affine transform + CRS, or GCPs as in Sentinel-1 GRD), or
  2. an explicit external georeference (sidecar <image>.geo.json or API/CLI input).
A plain PNG/JPG without either is reported as GEOREFERENCING UNAVAILABLE — we never
invent coordinates.

Sidecar format (all keys except bbox/transform optional):
{
  "bbox": [west, south, east, north],          # north-up image, OR
  "transform": [a, b, c, d, e, f],             # rasterio/GDAL affine (+ "crs")
  "crs": "EPSG:4326",
  "timestamp": "2025-06-10T13:04:00Z",         # satellite acquisition time (UTC)
  "provenance": "DEMO_ASSUMED" | "LOCAL_FILE",
  "sensor": "Sentinel-1A", "note": "..."
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.errors import InputImageError, TimestampUnavailable
from app.core.provenance import Provenance

RASTER_EXT = {".tif", ".tiff", ".geotiff"}
PLAIN_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
TIME_TAGS = ("ACQUISITION_START_TIME", "ACQUISITION_TIME", "SENSING_TIME", "TIFFTAG_DATETIME",
             "first_line_time", "timestamp")


@dataclass
class Georef:
    transform: tuple            # affine a,b,c,d,e,f
    crs: str
    width: int
    height: int
    provenance: str
    source: str
    bounds_wgs84: list = field(default_factory=list)   # [west, south, east, north]
    pixel_size_m: float | None = None

    def to_dict(self) -> dict:
        return {"transform": list(self.transform), "crs": self.crs, "width": self.width,
                "height": self.height, "provenance": self.provenance, "source": self.source,
                "bounds_wgs84": self.bounds_wgs84, "pixel_size_m": self.pixel_size_m}


@dataclass
class Scene:
    path: str
    rgb_u8: np.ndarray
    georef: Georef | None
    timestamp: datetime | None
    timestamp_source: str
    georef_status: str
    meta: dict = field(default_factory=dict)


def parse_time(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().replace("Z", "+00:00")
        for fmt in (None, "%Y:%m:%d %H:%M:%S", "%Y%m%dT%H%M%S"):
            try:
                dt = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_u8(band: np.ndarray, cfg: dict) -> np.ndarray:
    """Map raw SAR backscatter to 0..255 like the 8-bit training tiles (an ASSUMPTION)."""
    a = band.astype(np.float64)
    valid = np.isfinite(a) & (a > 0) if cfg.get("to_db", True) else np.isfinite(a)
    if cfg.get("to_db", True) and valid.any() and np.nanmax(a[valid]) < 50:   # looks linear sigma0
        a = np.where(valid, 10 * np.log10(np.where(valid, a, 1)), np.nan)
    lo, hi = np.nanpercentile(a[valid], [cfg.get("low_percentile", 2), cfg.get("high_percentile", 98)])
    out = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255
    return np.nan_to_num(out, nan=0).astype(np.uint8)


def _bounds_wgs84(transform, crs: str, w: int, h: int) -> list:
    from pyproj import Transformer
    xs = [0, w, w, 0]
    ys = [0, 0, h, h]
    a, b, c, d, e, f = transform
    X = [a * x + b * y + c for x, y in zip(xs, ys)]
    Y = [d * x + e * y + f for x, y in zip(xs, ys)]
    if crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        X, Y = tr.transform(X, Y)
    return [float(min(X)), float(min(Y)), float(max(X)), float(max(Y))]


def _pixel_size_m(transform, crs: str, bounds: list) -> float:
    a, b, _, d, e, _ = transform
    if crs.upper() in ("EPSG:4326", "OGC:CRS84"):
        from pyproj import Geod
        lat = (bounds[1] + bounds[3]) / 2
        g = Geod(ellps="WGS84")
        _, _, dx = g.inv(bounds[0], lat, bounds[0] + abs(a), lat)
        _, _, dy = g.inv(bounds[0], lat, bounds[0], lat + abs(e))
        return float((dx + dy) / 2)
    return float((abs(a) + abs(e)) / 2)


def georef_from_dict(geo: dict, width: int, height: int, source: str) -> Georef:
    crs = geo.get("crs", "EPSG:4326")
    if "transform" in geo:
        t = tuple(float(v) for v in geo["transform"][:6])
    elif "bbox" in geo:
        w, s, e, n = [float(v) for v in geo["bbox"]]
        if not (w < e and s < n):
            raise InputImageError(f"Invalid bbox {geo['bbox']}", "bbox must be [west, south, east, north].")
        if crs.upper() == "EPSG:4326" and not (-180 <= w <= 180 and -90 <= s <= 90 and -180 <= e <= 180 and -90 <= n <= 90):
            raise InputImageError(f"bbox outside valid lat/lon range: {geo['bbox']}", "Check coordinate order.")
        t = ((e - w) / width, 0.0, w, 0.0, -(n - s) / height, n)
    else:
        raise InputImageError("Georeference must contain 'bbox' or 'transform'.", "See georeference.py docstring.")
    bounds = _bounds_wgs84(t, crs, width, height)
    return Georef(transform=t, crs=crs, width=width, height=height,
                  provenance=geo.get("provenance", Provenance.LOCAL.value), source=source,
                  bounds_wgs84=bounds, pixel_size_m=_pixel_size_m(t, crs, bounds))


def read_scene(path: str | Path, geo_override: dict | None = None, scaling_cfg: dict | None = None,
               sidecar_suffix: str = ".geo.json") -> Scene:
    path = Path(path)
    if not path.exists():
        raise InputImageError(f"SAR image not found: {path}", "Check the --input path.")
    ext = path.suffix.lower()
    georef, ts, ts_src, meta = None, None, "none", {}

    if ext in RASTER_EXT:
        import rasterio
        try:
            src = rasterio.open(path)
        except Exception as exc:
            raise InputImageError(f"Cannot open GeoTIFF: {exc}", "Is this a valid GeoTIFF?") from exc
        with src:
            data = src.read()
            tags = {**src.tags(), **{k: v for i in range(1, src.count + 1) for k, v in src.tags(i).items()}}
            meta = {"driver": src.driver, "bands": src.count, "dtype": str(src.dtypes[0]),
                    "tags": {k: v for k, v in tags.items() if k in ("SCALING", "SOURCE_ITEM", "PLATFORM", "PROVENANCE",
                                                                  "POLARIZATION", "PIXEL_SPACING_M", "SCENE_START")}}
            if src.crs and not src.transform.is_identity:
                t = tuple(src.transform)[:6]
                crs = src.crs.to_string()
                georef = Georef(t, crs, src.width, src.height, tags.get("PROVENANCE", Provenance.LOCAL.value),
                                "geotiff_transform")
            elif src.gcps and src.gcps[0]:
                from rasterio.transform import from_gcps
                gcps, gcrs = src.gcps
                t = tuple(from_gcps(gcps))[:6]
                georef = Georef(t, gcrs.to_string() if gcrs else "EPSG:4326", src.width, src.height,
                                Provenance.LOCAL.value, "geotiff_gcps_affine_approx")
            for k in TIME_TAGS:
                if k in tags and parse_time(tags[k]):
                    ts, ts_src = parse_time(tags[k]), f"geotiff_tag:{k}"
                    break
        if data.dtype == np.uint8:
            rgb = np.transpose(data[:3], (1, 2, 0)) if data.shape[0] >= 3 else np.repeat(data[0][..., None], 3, 2)
        else:
            u8 = _to_u8(data[0], scaling_cfg or {})
            rgb = np.repeat(u8[..., None], 3, axis=2)
            meta["scaling"] = "ASSUMPTION: dB conversion + percentile stretch to 0..255"
    elif ext in PLAIN_EXT:
        from PIL import Image
        try:
            rgb = np.array(Image.open(path).convert("RGB"))   # identical to training loader
        except Exception as exc:
            raise InputImageError(f"Cannot decode image: {exc}", "Use PNG/JPG/GeoTIFF.") from exc
    else:
        raise InputImageError(f"Unsupported image format '{ext}'", "Supported: .tif/.tiff, .png, .jpg")

    h, w = rgb.shape[:2]
    sidecar = path.with_name(path.name + sidecar_suffix)
    if not sidecar.exists():
        sidecar = path.with_suffix(sidecar_suffix)
    geo = geo_override
    geo_src = "override"
    if geo is None and sidecar.exists():
        geo = json.loads(sidecar.read_text(encoding="utf-8"))
        geo_src = f"sidecar:{sidecar.name}"
    if geo:
        if "bbox" in geo or "transform" in geo:
            georef = georef_from_dict(geo, w, h, geo_src)
        if geo.get("timestamp") and ts is None:
            ts, ts_src = parse_time(geo["timestamp"]), geo_src
        meta.update({k: geo[k] for k in ("sensor", "note") if k in geo})
    if georef is not None and not georef.bounds_wgs84:
        georef.bounds_wgs84 = _bounds_wgs84(georef.transform, georef.crs, w, h)
        georef.pixel_size_m = _pixel_size_m(georef.transform, georef.crs, georef.bounds_wgs84)

    status = "OK" if georef else "GEOREFERENCING_UNAVAILABLE"
    return Scene(str(path), rgb, georef, ts, ts_src, status, meta)


def require_timestamp(scene: Scene) -> datetime:
    if scene.timestamp is None:
        raise TimestampUnavailable(
            "Satellite acquisition timestamp is unknown.",
            "Provide 'timestamp' in the .geo.json sidecar or --timestamp; drift and AIS need it.")
    return scene.timestamp
