"""SAR ship detection (independent of AIS) + SAR/AIS visibility mismatch.

Detector: cell-averaging CFAR — ships are bright point targets against sea
clutter. A pixel is a detection if it exceeds mean + k·std of a background ring
(excluding a guard window). Clusters are filtered by size. This is a classical,
untrained baseline; it is exposed behind `ShipDetector` so a trained detector
(e.g. a YOLO model on SAR ship chips) can replace it. On 8-bit, speckled,
downsampled tiles with ~40 m assumed pixels, small vessels may be invisible —
absence of detections is NOT evidence of absence of ships.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class ShipDetector:
    name = "abstract"

    def detect(self, gray: np.ndarray) -> list[dict]:
        raise NotImplementedError


class CFARShipDetector(ShipDetector):
    name = "CA-CFAR baseline (untrained)"

    def __init__(self, guard=4, background=12, k=5.0, min_px=3, max_px=400, min_grey=0):
        self.guard, self.bg, self.k, self.min_px, self.max_px = guard, background, k, min_px, max_px
        self.min_grey = min_grey      # absolute brightness floor (normalised scenes: ships saturate near 255)

    def detect(self, gray: np.ndarray) -> list[dict]:
        img = gray.astype(np.float64)
        outer, inner = 2 * self.bg + 1, 2 * self.guard + 1
        s_o = ndimage.uniform_filter(img, outer) * outer ** 2
        s_i = ndimage.uniform_filter(img, inner) * inner ** 2
        q_o = ndimage.uniform_filter(img ** 2, outer) * outer ** 2
        q_i = ndimage.uniform_filter(img ** 2, inner) * inner ** 2
        n = outer ** 2 - inner ** 2
        mu = (s_o - s_i) / n
        sd = np.sqrt(np.clip((q_o - q_i) / n - mu ** 2, 1e-9, None))
        det = (img > np.minimum(mu + self.k * sd, 254.5)) & (img >= self.min_grey)
        lab, nlab = ndimage.label(det)
        out = []
        # per-object slices (a full-image comparison per label was O(labels x pixels) and stalled on busy scenes)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lab[sl] == i)
            if self.min_px <= len(ys) <= self.max_px:
                ys, xs = ys + sl[0].start, xs + sl[1].start
                out.append({"row": float(ys.mean()), "col": float(xs.mean()), "n_pixels": int(len(ys)),
                            "peak": float(img[ys, xs].max()),
                            "contrast_sigma": float(((img[ys, xs] - mu[ys, xs]) / sd[ys, xs]).max())})
        return out


class DbPointTargetDetector(ShipDetector):
    """Bright point targets on the UNCLIPPED backscatter anomaly (dB above the local sea background).
    Ships are typically +15..30 dB brighter than sea clutter; Sentinel-1 speckle rarely exceeds ~+6 dB, so a
    +10 dB threshold on compact clusters separates them. Used when the ingest saved the anomaly map (real and
    synthetic Sentinel-1 scenes); 8-bit uploads fall back to CFARShipDetector (8-bit clipping makes saturated
    speckle and ships indistinguishable)."""
    name = "Point-target detector on dB anomaly (untrained baseline)"

    def __init__(self, anomaly_db: np.ndarray, threshold_db=10.0, min_px=2, max_px=400):
        self.anom, self.thr, self.min_px, self.max_px = anomaly_db, threshold_db, min_px, max_px

    def detect(self, gray: np.ndarray | None = None) -> list[dict]:
        a = self.anom.astype(np.float32)
        lab, _ = ndimage.label(a > self.thr, structure=np.ones((3, 3)))
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lab[sl] == i)
            if self.min_px <= len(ys) <= self.max_px:
                ys, xs = ys + sl[0].start, xs + sl[1].start
                v = a[ys, xs]
                out.append({"row": float(ys.mean()), "col": float(xs.mean()), "n_pixels": int(len(ys)),
                            "peak_db_above_sea": float(v.max()), "contrast_sigma": None})
        return out


def geolocate(dets: list[dict], georef) -> list[dict]:
    if georef is None:
        return dets
    a, b, c, d, e, f = georef.transform
    for x in dets:
        X, Y = a * (x["col"] + 0.5) + b * (x["row"] + 0.5) + c, d * (x["col"] + 0.5) + e * (x["row"] + 0.5) + f
        if georef.crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
            from pyproj import Transformer
            X, Y = Transformer.from_crs(georef.crs, "EPSG:4326", always_xy=True).transform(X, Y)
        x["lon"], x["lat"] = float(X), float(Y)
    return dets


def sar_ais_mismatch(dets: list[dict], tracks: list, obs_time, bounds, match_km: float, match_min: float) -> dict:
    """Compare SAR detections with AIS positions interpolated to the acquisition time."""
    from app.ais.track_processing import haversine_km
    te = obs_time.timestamp()
    w, s, e, n = bounds
    ais_pts = []
    for t in tracks:
        tt = t.t_epoch
        if tt.min() - match_min * 60 <= te <= tt.max() + match_min * 60:
            j = int(np.argmin(np.abs(tt - te)))
            if abs(tt[j] - te) <= match_min * 60:
                lat, lon, _ = t.interpolate(np.array([te]), max_gap_s=match_min * 120)
                lat, lon = (float(lat[0]), float(lon[0])) if np.isfinite(lat[0]) else (float(t.df.lat.iat[j]), float(t.df.lon.iat[j]))
                if w <= lon <= e and s <= lat <= n:
                    ais_pts.append({"mmsi": t.mmsi, "name": t.name, "lat": lat, "lon": lon,
                                    "time_offset_min": round((tt[j] - te) / 60, 1)})
    matches, used = [], set()
    for di, dd in enumerate(dets):
        if "lat" not in dd:
            continue
        best = min(((haversine_km(dd["lat"], dd["lon"], a["lat"], a["lon"]), ai) for ai, a in enumerate(ais_pts)
                    if ai not in used), default=(None, None))
        if best[0] is not None and best[0] <= match_km:
            used.add(best[1])
            matches.append({"detection": di, "mmsi": ais_pts[best[1]]["mmsi"], "name": ais_pts[best[1]]["name"],
                            "distance_km": round(float(best[0]), 2), "time_offset_min": ais_pts[best[1]]["time_offset_min"]})
    sar_only = [i for i in range(len(dets)) if i not in {m["detection"] for m in matches}]
    ais_only = [a for i, a in enumerate(ais_pts) if i not in used]
    return {
        "label": "SAR/AIS visibility mismatch",
        "matches": matches,
        "sar_detections_without_ais": [dets[i] for i in sar_only],
        "ais_vessels_without_sar_detection": ais_only,
        "possible_explanations": ["temporal mismatch between AIS report and acquisition", "spatial matching tolerance",
                                  "SAR detector limitations / false alarms (bright sea clutter, speckle)",
                                  "AIS coverage limitations", "vessel too small for image resolution",
                                  "data quality"],
        "note": "Mismatches are not interpreted as intentional AIS disabling.",
    }
