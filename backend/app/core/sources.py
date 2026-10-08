"""External data sources: which need credentials, which are configured.

Credentials are read from the environment, optionally loaded from a project-root `.env`
file (see `.env.example`). Values are never returned by the API — only whether they are set.
Key-free sources are used automatically; keyed sources are optional upgrades.
"""
from __future__ import annotations

import os

from app.core.config import PROJECT_ROOT


def load_dotenv(path=None):
    p = path or PROJECT_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SOURCES = [
    {"id": "sentinel1_pc", "name": "Sentinel-1 GRD (Microsoft Planetary Computer)", "kind": "SAR imagery",
     "key_required": False, "env": [], "integrated": True, "coverage": "Global",
     "notes": "Anonymous STAC + SAS token; used automatically."},
    {"id": "open_meteo", "name": "Open-Meteo (marine currents + ERA5/forecast wind)", "kind": "Environmental forcing",
     "key_required": False, "env": [], "integrated": True, "coverage": "Global (~0.08° currents, 0.25° wind)",
     "notes": "Free tier is rate limited; results cached in data/forcing/cache."},
    {"id": "dma_ais", "name": "Danish Maritime Authority historical AIS", "kind": "AIS",
     "key_required": False, "env": [], "integrated": True, "coverage": "Danish waters, daily files from 2024-03",
     "notes": "~0.5 GB download per day, indexed once to a Parquet cache."},
    {"id": "gshhg", "name": "GSHHG full-resolution coastline (roaring-landmask)", "kind": "Land mask",
     "key_required": False, "env": [], "integrated": True, "coverage": "Global", "notes": "Bundled offline."},
    {"id": "cmems", "name": "Copernicus Marine (CMEMS) currents", "kind": "Environmental forcing (upgrade)",
     "key_required": True, "env": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
     "integrated": False, "coverage": "Global, higher resolution regional models",
     "notes": "Optional. Download NetCDF with the copernicusmarine toolbox and set environment.provider: netcdf."},
    {"id": "cdse", "name": "Copernicus Data Space Ecosystem", "kind": "SAR imagery (alternative)",
     "key_required": True, "env": ["CDSE_USERNAME", "CDSE_PASSWORD"], "integrated": False, "coverage": "Global",
     "notes": "Optional alternative to Planetary Computer (free account)."},
    {"id": "gfw", "name": "Global Fishing Watch 4Wings AIS vessel presence", "kind": "AIS (Indian waters)",
     "key_required": True, "env": ["GFW_API_TOKEN"], "integrated": True, "coverage": "Global incl. Indian EEZ",
     "notes": "Real AIS-derived presence: 1 position per vessel per hour, 0.01 deg cells, all vessel types. Free "
              "non-commercial token. Used automatically (ais.mode auto/real) when GFW_API_TOKEN is set."},
    {"id": "synthetic_ais", "name": "Synthetic AIS generator (Indian waters)", "kind": "AIS (fallback)",
     "key_required": False, "env": [], "integrated": True, "coverage": "Any region (presets for Indian waters)",
     "notes": "Used when no real AIS is available (ais.mode auto) or on request (synthetic). Always labelled "
              "SYNTHETIC; permitted by SIH26143 for demonstrating the algorithm."},
    {"id": "india_ais", "name": "Indian-waters historical AIS (INCOIS / DG Shipping NAIS / commercial)", "kind": "AIS",
     "key_required": True, "env": ["INDIA_AIS_API_KEY"], "integrated": False, "coverage": "Indian EEZ",
     "notes": "Institutional or paid access; implement AISProvider when available."},
]


def status() -> list[dict]:
    load_dotenv()
    out = []
    for s in SOURCES:
        configured = all(os.environ.get(e) for e in s["env"]) if s["env"] else True
        out.append({**s, "configured": configured,
                    "state": ("ACTIVE" if s["integrated"] and configured else
                              "READY_TO_INTEGRATE" if configured else "NOT_CONFIGURED")})
    return out
