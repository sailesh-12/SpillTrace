"""Load the trained DeepLabV3+ / ResNet34 oil-spill segmentation checkpoint.

Verified against the actual checkpoint (Phase 0):
  * torch zip-serialized dict with keys
    epoch, model_state_dict, optimizer_state_dict, val_loss, val_dice, val_iou
  * 277 tensors / 22,458,718 params, key layout == smp.DeepLabV3Plus(resnet34)
  * conv1 (64,3,7,7) -> 3 input channels; segmentation_head (1,256,1,1) -> 1 logit
Raw state_dicts (no wrapper) are also accepted.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import ModelError


@dataclass
class LoadedModel:
    model: object
    device: str
    metadata: dict = field(default_factory=dict)


_cache: dict[str, LoadedModel] = {}
_lock = threading.Lock()


def inspect_checkpoint(ckpt: object) -> tuple[dict, dict]:
    """Describe a loaded checkpoint object without assuming its format. Returns (info, state_dict)."""
    if not isinstance(ckpt, dict):
        return {"format": type(ckpt).__name__}, {}
    if "model_state_dict" in ckpt:
        fmt, sd = "checkpoint_dict[model_state_dict]", ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        fmt, sd = "checkpoint_dict[state_dict]", ckpt["state_dict"]
    else:
        fmt, sd = "raw_state_dict", ckpt
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "state_dict", "optimizer_state_dict")
            and isinstance(v, (int, float, str))} if fmt != "raw_state_dict" else {}
    info = {"format": fmt, "n_tensors": len(sd), "metadata": meta}
    conv1 = sd.get("encoder.conv1.weight")
    head = sd.get("segmentation_head.0.weight")
    if conv1 is not None:
        info["in_channels"] = int(conv1.shape[1])
    if head is not None:
        info["classes"] = int(head.shape[0])
    info["has_aspp"] = any(k.startswith("decoder.aspp") for k in sd)
    return info, sd


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        raise ModelError("CUDA requested but not available.", "Set segmentation.device: auto or cpu.")
    return "cpu"


def load_model(model_path: str | Path, encoder_name: str = "resnet34", device: str = "auto") -> LoadedModel:
    """Load (and cache) the segmentation model in eval mode."""
    model_path = Path(model_path)
    key = f"{model_path.resolve()}::{device}"
    with _lock:
        if key in _cache:
            return _cache[key]
        if not model_path.exists():
            raise ModelError(
                f"Model checkpoint not found: {model_path}",
                "Place the .pth file at segmentation.model_path in configs/config.yaml or set MODEL_PATH.",
            )
        try:
            import torch
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ModelError(f"Deep-learning dependencies missing: {exc}",
                             "pip install torch segmentation-models-pytorch") from exc

        try:
            # weights_only=True is safe: the checkpoint contains only tensors, dicts and numbers.
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # corrupted / not a torch file
            raise ModelError(f"Could not read checkpoint {model_path.name}: {exc}",
                             "Verify the file is the torch.save() output from training.") from exc

        info, state_dict = inspect_checkpoint(ckpt)
        if not info.get("has_aspp") or "in_channels" not in info:
            raise ModelError(
                "Checkpoint does not look like a segmentation_models_pytorch DeepLabV3+ model.",
                f"Inspected structure: {info}",
            )
        # Architecture matches training notebook cell 21 (encoder_weights irrelevant: we load weights).
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None,
                                  in_channels=info["in_channels"], classes=info["classes"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ModelError(
                f"Incompatible architecture: {len(missing)} missing / {len(unexpected)} unexpected keys.",
                f"First missing: {missing[:3]}, first unexpected: {unexpected[:3]}",
            )
        dev = resolve_device(device)
        model.to(dev).eval()
        loaded = LoadedModel(model=model, device=dev, metadata={**info, "path": str(model_path)})
        _cache[key] = loaded
        return loaded
