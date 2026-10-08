"""Choose the AIS source for an analysis: real provider(s) first, synthetic fallback when allowed.

ais.mode (config, or per-investigation override):
  auto       try real AIS for the region; if unavailable or empty -> SYNTHETIC fallback (clearly labelled)
  real       real AIS only; never falls back (no AIS -> "no AIS available" result)
  synthetic  always generate SYNTHETIC AIS for the spill region
ais.real_provider: auto | gfw | dma | local
  auto -> DMA if the search area is in Danish waters, else Global Fishing Watch if GFW_API_TOKEN is set,
          else local files if configured.
Whatever is chosen returns the same CANONICAL DataFrame, so cleaning, corridor filtering, scoring and the
report are identical for real and synthetic AIS.
"""
from __future__ import annotations

import os

import pandas as pd

from app.ais.provider import CANONICAL
from app.core.errors import AISDataError

MODES = ("auto", "real", "synthetic")


def _real_candidates(cfg, bbox, progress):
    a = cfg.ais
    choice = a.get("real_provider") or (a.get("provider") if a.get("provider") in ("dma", "gfw", "local") else "auto")
    from app.ais.local_provider import LocalAISProvider

    def dma():
        from app.ais.dma_provider import DMAAISProvider
        p = DMAAISProvider(cfg.path(a.get("cache_dir", "data/ais/cache")), a.get("thin_seconds", 60), progress)
        if not p.covers(bbox):
            raise AISDataError("Search area is outside Danish Maritime Authority AIS coverage.", "")
        return p

    def gfw():
        from app.ais.gfw_provider import GFWPresenceAISProvider
        return GFWPresenceAISProvider(progress=progress)

    def local():
        if not a.get("files"):
            raise AISDataError("No local AIS files configured (ais.files).", "")
        return LocalAISProvider([cfg.path(p) for p in a.files], a.get("column_map", "default"))

    table = {"dma": dma, "gfw": gfw, "local": local}
    if choice != "auto":
        return [(choice, table[choice])]
    order = []
    try:
        from app.ais.dma_provider import COVERAGE
        w, s, e, n = bbox
        if not (e < COVERAGE[0] or w > COVERAGE[2] or n < COVERAGE[1] or s > COVERAGE[3]):
            order.append(("dma", dma))
    except ImportError:
        pass
    if os.environ.get("GFW_API_TOKEN"):
        order.append(("gfw", gfw))
    if a.get("files"):
        order.append(("local", local))
    return order


def select_ais(cfg, bbox, start, end, context: dict | None = None, mode: str | None = None, progress=None):
    """Returns (dataframe, provider, info). info documents every attempt and why synthetic was (not) used."""
    mode = (mode or cfg.ais.get("mode", "auto")).lower()
    if mode not in MODES:
        raise AISDataError(f"Unknown AIS mode '{mode}'", f"Use one of {MODES}.")
    info = {"mode_requested": mode, "attempts": [], "source": None, "fallback_reason": None}
    if mode in ("auto", "real"):
        cands = _real_candidates(cfg, bbox, progress)
        if not cands:
            info["attempts"].append({"provider": "none", "outcome": "no real AIS provider available for this "
                                     "region (Danish coverage only for DMA; set GFW_API_TOKEN for Indian waters)"})
        for key, make in cands:
            try:
                prov = make()
                df = prov.get_tracks(bbox, start, end)
                if len(df):
                    info["attempts"].append({"provider": key, "outcome": f"{len(df)} positions"})
                    info["source"] = "real"
                    return df, prov, info
                info["attempts"].append({"provider": key, "outcome": "no AIS positions for this area/time"})
            except AISDataError as exc:
                info["attempts"].append({"provider": key, "outcome": exc.message})
        if mode == "real":
            info["source"] = "none"
            raise AISDataError("Real AIS is unavailable for this region/time: " +
                               "; ".join(f"{x['provider']}: {x['outcome']}" for x in info["attempts"]),
                               "Use ais mode 'auto' or 'synthetic' to demonstrate the algorithm with SYNTHETIC AIS.")
        if not cfg.ais.get("synthetic", {}).get("fallback", True):
            info["source"] = "none"
            raise AISDataError("Real AIS unavailable and synthetic fallback is disabled.", "Set ais.synthetic.fallback: true.")
        info["fallback_reason"] = "; ".join(f"{x['provider']}: {x['outcome']}" for x in info["attempts"])
    from app.ais.synthetic_provider import SyntheticAISProvider
    prov = SyntheticAISProvider(dict(cfg.ais.get("synthetic", {})), context, seed_key=(context or {}).get("seed_key", ""))
    if progress:
        progress("Generating SYNTHETIC AIS for the spill region (demonstration data, not real vessels)")
    df = prov.get_tracks(bbox, start, end)
    info["source"] = "synthetic"
    return df if len(df) else pd.DataFrame(columns=CANONICAL), prov, info
