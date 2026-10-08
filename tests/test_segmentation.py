"""1-4: model loading, output shape, thresholding, polygon extraction."""
import numpy as np
import pytest
from PIL import Image

from app.core.errors import ModelError
from app.segmentation.inference import predict_logits, run_inference, to_model_input
from app.segmentation.model_loader import inspect_checkpoint, load_model
from app.segmentation.postprocess import clean_mask, components, mask_to_pixel_polygons


def test_model_loads_with_verified_structure(model):
    md = model.metadata
    assert md["format"] == "checkpoint_dict[model_state_dict]"
    assert md["in_channels"] == 3 and md["classes"] == 1 and md["n_tensors"] == 277
    assert not model.model.training          # eval mode


def test_missing_model_is_actionable(tmp_path):
    with pytest.raises(ModelError) as e:
        load_model(tmp_path / "nope.pth")
    assert "MODEL_PATH" in e.value.hint


def test_inspect_raw_state_dict():
    import torch
    info, sd = inspect_checkpoint({"encoder.conv1.weight": torch.zeros(64, 3, 7, 7),
                                   "segmentation_head.0.weight": torch.zeros(1, 256, 1, 1),
                                   "decoder.aspp.0.x": torch.zeros(1)})
    assert info["format"] == "raw_state_dict" and info["in_channels"] == 3


def test_preprocessing_matches_training():
    rgb = np.full((8, 8, 3), 255, np.uint8)
    x = to_model_input(rgb)
    assert x.shape == (3, 8, 8) and x.dtype == np.float32 and x.max() == 1.0   # /255, no normalisation


def test_output_shape_single_tile_and_sliding_window(model):
    x = np.random.default_rng(0).random((3, 256, 256), dtype=np.float32)
    assert predict_logits(model, x).shape == (256, 256)
    x2 = np.random.default_rng(0).random((3, 300, 520), dtype=np.float32)
    assert predict_logits(model, x2, tile=256, overlap=32).shape == (300, 520)


def test_threshold_configurable(model, cfg):
    rgb = np.array(Image.open(cfg.path("data/sar/samples/1.png")).convert("RGB"))
    lo = run_inference(model, rgb, 0.3)
    hi = run_inference(model, rgb, 0.8)
    assert lo["binary_mask"].sum() >= hi["binary_mask"].sum()
    assert 0 <= lo["probability_mask"].min() and lo["probability_mask"].max() <= 1
    assert lo["oil_fraction"] > 0.05          # the known slick tile is detected
    with pytest.raises(ValueError):
        run_inference(model, rgb, 1.5)


def test_clean_mask_removes_small_components_and_keeps_regions_separate():
    m = np.zeros((64, 64), bool)
    m[2:4, 2:4] = True                       # 4 px noise
    m[10:30, 10:30] = True                   # 400 px region A
    m[40:60, 40:60] = True                   # 400 px region B (disconnected)
    c = clean_mask(m, min_area_px=100, morph_kernel=0)
    comps = components(c)
    assert len(comps) == 2 and all(k["area_px"] == 400 for k in comps)
    polys = mask_to_pixel_polygons(comps[0]["mask"])
    assert len(polys) == 1 and abs(polys[0].area - 400) < 1e-6   # exact pixel-boundary tracing
