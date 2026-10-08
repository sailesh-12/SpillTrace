"""Phase 1 check: load checkpoint, run inference on sample tiles, save overlays."""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import load_config
from app.segmentation.model_loader import load_model
from app.segmentation.inference import run_inference
from app.segmentation.postprocess import clean_mask, components

cfg = load_config()
s = cfg.segmentation
t = time.time()
m = load_model(cfg.path(s.model_path), s.encoder_name, s.device)
print(f"loaded on {m.device} in {time.time()-t:.1f}s  meta={m.metadata}")
out = ROOT / "outputs" / "phase1"; out.mkdir(parents=True, exist_ok=True)
for p in sorted((ROOT / "data/sar/samples").glob("*.png")):
    rgb = np.array(Image.open(p).convert("RGB"))
    r = run_inference(m, rgb, s.threshold, s.tile_size, s.tile_overlap)
    cm = clean_mask(r["binary_mask"], s.min_component_area_pixels, s.morphology_kernel)
    comps = components(cm, r["probability_mask"])
    pm = r["probability_mask"]
    print(f"{p.name}: shape={rgb.shape} prob[min,max,mean]=[{pm.min():.3f},{pm.max():.3f},{pm.mean():.3f}] "
          f"oil_frac={r['oil_fraction']:.3f} conf={r['confidence']:.3f} components_after_clean={len(comps)} "
          f"areas={[c['area_px'] for c in comps]}")
    ov = rgb.copy(); ov[cm] = (0.5 * ov[cm] + 0.5 * np.array([255, 60, 0])).astype(np.uint8)
    heat = (np.stack([pm, pm * 0.3, 1 - pm], -1) * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb, heat, ov], 1)).save(out / f"{p.stem}_panel.png")
print("panels written to", out)
