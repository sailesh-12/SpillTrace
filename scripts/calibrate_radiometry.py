"""Empirical calibration of the Sentinel-1 -> model input mapping (no model retraining).

Negative control: a real, clean Sentinel-1 sea area (moderate wind) -> false-positive fraction.
Positive control: synthetic dark streaks implanted at -3 dB and -6 dB relative to local background
(typical oil damping), sizes 150-400 m wide -> recall of implanted pixels.
Grid over target background grey level, contrast (grey per speckle-sigma) and multilook factor.

  python scripts/calibrate_radiometry.py --tif <ingested scene .tif>
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.segmentation.model_loader import load_model  # noqa
from app.segmentation.inference import run_inference  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--tif", required=True); ap.add_argument("--crop", type=int, default=2048)
a = ap.parse_args()
with rasterio.open(a.tif) as ds:
    u8 = ds.read(1).astype(np.float32); k0 = float(re.search(r"gray_per_db=([\d.]+)", ds.tags()["SCALING"]).group(1))
sea = np.load(a.tif.replace(".tif", "_sea.npy"))
h, w = u8.shape; c = a.crop
r0, c0 = 0, max(0, w - c)                                      # top-right: open sea, no coast
anom = (u8[r0:r0 + c, c0:c0 + c] - 130.0) / k0                 # dB anomaly w.r.t. local background
ok = sea[r0:r0 + c, c0:c0 + c]
print("crop sea fraction", ok.mean())
rng = np.random.default_rng(0)
imp = np.zeros_like(anom, bool); depth = np.zeros_like(anom)
yy, xx = np.mgrid[0:c, 0:c]
for i, (d_db, wid) in enumerate([(-6, 25), (-6, 15), (-3, 40), (-3, 20), (-6, 40), (-3, 30)]):
    y0 = 200 + i * 300; ang = rng.uniform(-0.4, 0.4)
    line = np.abs((yy - y0) - np.tan(ang) * (xx - 300)) < wid / 2
    line &= (xx > 300) & (xx < 1700)
    imp |= line; depth[line] = d_db
m = load_model(ROOT / "models/best_oil_spill_deeplabv3_resnet34.pth")
res = []
for ml in (1, 2):
    base = ndimage.uniform_filter(anom, ml) if ml > 1 else anom        # multilook in dB (approximation)
    sd = 1.4826 * np.median(np.abs(base[ok] - np.median(base[ok])))
    for bg in (130, 150, 170, 190):
        for per_sigma in (30, 40, 50):
            k = per_sigma / sd
            clean = np.clip(bg + base * k, 0, 255).astype(np.uint8)
            withs = np.clip(bg + (base + depth) * k, 0, 255).astype(np.uint8)
            fp = run_inference(m, np.repeat(clean[..., None], 3, 2))["binary_mask"][ok].mean()
            det = run_inference(m, np.repeat(withs[..., None], 3, 2))["binary_mask"]
            rec6 = det[imp & (depth == -6)].mean(); rec3 = det[imp & (depth == -3)].mean()
            res.append({"multilook": ml, "bg": bg, "grey_per_sigma": per_sigma, "fp": round(float(fp), 4),
                        "recall_-6dB": round(float(rec6), 3), "recall_-3dB": round(float(rec3), 3)})
            print(res[-1], flush=True)
best = max(res, key=lambda r: r["recall_-6dB"] - 4 * r["fp"])
print("BEST", best)
(ROOT / "outputs" / "radiometry_calibration.json").write_text(json.dumps({"results": res, "best": best, "tif": a.tif}, indent=2))
