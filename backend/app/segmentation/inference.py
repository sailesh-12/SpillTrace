"""Segmentation inference.

Preprocessing replicates the training notebook exactly (cell 16):
    RGB uint8 -> float32 / 255 -> CHW, no mean/std normalization.
Scenes larger than the 256 px training tile are processed with an overlapping
sliding window; overlapping logits are averaged with a smooth (Hann) weight so
tile seams do not appear in the mask.
"""
from __future__ import annotations

import numpy as np

from app.core.errors import InputImageError
from app.segmentation.model_loader import LoadedModel


def to_model_input(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW in [0,1] (training preprocessing)."""
    if rgb_u8.ndim == 2:
        rgb_u8 = np.repeat(rgb_u8[..., None], 3, axis=2)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise InputImageError(f"Expected HxWx3 image, got shape {rgb_u8.shape}",
                              "Provide a single-band or 3-channel SAR image.")
    return np.transpose(rgb_u8.astype(np.float32) / 255.0, (2, 0, 1))


def _window_weights(size: int) -> np.ndarray:
    w = np.hanning(size + 2)[1:-1]
    return np.clip(np.outer(w, w), 1e-3, None).astype(np.float32)


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def predict_logits(loaded: LoadedModel, chw: np.ndarray, tile: int = 256, overlap: int = 32,
                   batch_size: int = 8) -> np.ndarray:
    import torch

    _, h, w = chw.shape
    # pad small images up to one tile (model needs H,W divisible by 16)
    ph, pw = max(h, tile), max(w, tile)
    if (ph, pw) != (h, w):
        chw = np.pad(chw, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    stride = tile - overlap
    ys, xs = _positions(ph, tile, stride), _positions(pw, tile, stride)
    acc = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    weight = _window_weights(tile) if len(ys) * len(xs) > 1 else np.ones((tile, tile), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    with torch.inference_mode():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([chw[:, y:y + tile, x:x + tile] for y, x in chunk])
            out = loaded.model(torch.from_numpy(batch).to(loaded.device)).float().cpu().numpy()[:, 0]
            for (y, x), logit in zip(chunk, out):
                acc[y:y + tile, x:x + tile] += logit * weight
                wsum[y:y + tile, x:x + tile] += weight
    return (acc / wsum)[:h, :w]


def run_inference(loaded: LoadedModel, rgb_u8: np.ndarray, threshold: float = 0.5,
                  tile: int = 256, overlap: int = 32) -> dict:
    """Return probability mask, binary mask and a scalar confidence.

    confidence = mean probability over predicted-oil pixels (0 if none). It is a
    model-derived quantity, not a calibrated probability that the slick is oil.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    logits = predict_logits(loaded, to_model_input(rgb_u8), tile=tile, overlap=overlap)
    from scipy.special import expit
    prob = expit(logits)                            # sigmoid (overflow-safe)
    binary = prob > threshold
    conf = float(prob[binary].mean()) if binary.any() else 0.0
    return {"probability_mask": prob.astype(np.float32), "binary_mask": binary,
            "confidence": conf, "threshold": threshold,
            "oil_fraction": float(binary.mean())}
