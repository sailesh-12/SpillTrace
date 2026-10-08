"""Binary mask -> cleaned connected components -> pixel-space polygons.

Polygons are extracted with exact pixel-boundary tracing (rasterio.features
.shapes) so the slick shape is preserved; no smoothing/simplification beyond
removing collinear vertices. Disconnected regions stay separate.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def clean_mask(binary: np.ndarray, min_area_px: int = 100, morph_kernel: int = 3) -> np.ndarray:
    mask = binary.astype(bool)
    if morph_kernel and morph_kernel > 1:
        st = np.ones((morph_kernel, morph_kernel), bool)
        mask = ndimage.binary_opening(mask, structure=st)   # remove speckle
        mask = ndimage.binary_closing(mask, structure=st)   # close pin-holes
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(mask)
    areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = areas >= min_area_px
    return keep[labels]


def components(mask: np.ndarray, prob: np.ndarray | None = None) -> list[dict]:
    """Label connected components. Each component keeps a CROPPED mask plus its offset
    (full-scene masks per component would need gigabytes on large real scenes)."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == i
        out.append({
            "label": i,
            "area_px": int(comp.sum()),
            "bbox_px": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
            "mean_probability": float(prob[sl][comp].mean()) if prob is not None else None,
            "mask": comp,
            "offset": (int(sl[0].start), int(sl[1].start)),       # (row, col) of the crop in the scene
        })
    return out


def full_mask(comp: dict, shape: tuple) -> np.ndarray:
    m = np.zeros(shape, bool)
    r, c = comp.get("offset", (0, 0))
    h, w = comp["mask"].shape
    m[r:r + h, c:c + w] = comp["mask"]
    return m


def mask_to_pixel_polygons(comp_mask: np.ndarray, offset: tuple = (0, 0)) -> list:
    """Trace exact pixel boundaries of one (cropped) component -> shapely Polygons in scene pixel coords."""
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import shape

    tr = Affine.translation(offset[1], offset[0])
    polys = []
    for geom, val in features.shapes(comp_mask.astype(np.uint8), mask=comp_mask, connectivity=8, transform=tr):
        if val == 1:
            polys.append(shape(geom).simplify(0))   # simplify(0) only drops collinear vertices
    return polys
