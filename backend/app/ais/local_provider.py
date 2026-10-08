"""Local/offline AIS provider for CSV or Parquet files.

Column mappings support common open historical AIS exports:
  default         mmsi,timestamp,lat,lon,sog,cog,vessel_name,imo,vessel_type
  marinecadastre  MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName,IMO,VesselType   (NOAA, US waters)
  dma             MMSI,# Timestamp,Latitude,Longitude,SOG,COG,Name,IMO,Ship type (Danish Maritime Authority)
Provenance is read from a leading '# PROVENANCE: <label>' comment line if present.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.ais.provider import CANONICAL, AISProvider
from app.core.errors import AISDataError
from app.core.provenance import Provenance

COLUMN_MAPS = {
    "default": {c: c for c in CANONICAL},
    "marinecadastre": {"MMSI": "mmsi", "BaseDateTime": "timestamp", "LAT": "lat", "LON": "lon", "SOG": "sog",
                       "COG": "cog", "VesselName": "vessel_name", "IMO": "imo", "VesselType": "vessel_type"},
    "dma": {"MMSI": "mmsi", "# Timestamp": "timestamp", "Latitude": "lat", "Longitude": "lon", "SOG": "sog",
            "COG": "cog", "Name": "vessel_name", "IMO": "imo", "Ship type": "vessel_type"},
}


def _provenance_of(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        if first.startswith("# PROVENANCE:"):
            return first.split(":", 1)[1].strip().split()[0]
    return Provenance.LOCAL.value


class LocalAISProvider(AISProvider):
    name = "local_file"

    def __init__(self, files: list, column_map: str | dict = "default"):
        self.files = [Path(f) for f in files]
        missing = [str(f) for f in self.files if not f.exists()]
        if missing:
            raise AISDataError(f"AIS file(s) not found: {missing}",
                               "Place historical AIS CSV/Parquet in data/ais/ and list it under ais.files.")
        self.mapping = COLUMN_MAPS[column_map] if isinstance(column_map, str) else column_map
        provs = {_provenance_of(f) for f in self.files}
        self.provenance = provs.pop() if len(provs) == 1 else "MIXED"
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        frames = []
        for f in self.files:
            try:
                df = pd.read_parquet(f) if f.suffix.lower() == ".parquet" else pd.read_csv(f, comment="#", dtype=str)
            except Exception as exc:
                raise AISDataError(f"Malformed AIS file {f.name}: {exc}", "Check delimiter/encoding/columns.") from exc
            df = df.rename(columns=self.mapping)
            need = {"mmsi", "timestamp", "lat", "lon"}
            if not need.issubset(df.columns):
                raise AISDataError(f"{f.name} lacks required columns {sorted(need - set(df.columns))}",
                                   "Set ais.column_map (default | marinecadastre | dma) or provide a custom mapping.")
            for c in CANONICAL:
                if c not in df.columns:
                    df[c] = None
            frames.append(df[CANONICAL])
        df = pd.concat(frames, ignore_index=True)
        df["mmsi"] = df["mmsi"].astype(str).str.strip()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in ("lat", "lon", "sog", "cog"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        self._df = df
        return df

    def get_tracks(self, bbox, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        df = self._load()
        w, s, e, n = bbox
        sel = (df["lon"].between(w, e) & df["lat"].between(s, n) &
               (df["timestamp"] >= pd.Timestamp(start_time)) & (df["timestamp"] <= pd.Timestamp(end_time)))
        # keep rows with invalid values for vessels already selected so cleaning can report them
        mmsis = set(df.loc[sel, "mmsi"])
        tsel = (df["timestamp"].isna() | ((df["timestamp"] >= pd.Timestamp(start_time)) &
                                          (df["timestamp"] <= pd.Timestamp(end_time))))
        return df[df["mmsi"].isin(mmsis) & tsel].copy()

    def get_vessel_metadata(self, mmsi: str) -> dict:
        df = self._load()
        rows = df[df["mmsi"] == str(mmsi)]
        if rows.empty:
            return {}
        first = lambda c: next((v for v in rows[c] if isinstance(v, str) and v.strip() and v != "nan"), None)
        return {"mmsi": str(mmsi), "name": first("vessel_name"), "imo": first("imo"), "type_raw": first("vessel_type")}


def build_ais_provider(cfg, progress=None) -> AISProvider:
    a = cfg.ais
    if a.provider == "dma":
        from app.ais.dma_provider import DMAAISProvider
        return DMAAISProvider(cfg.path(a.get("cache_dir", "data/ais/cache")), a.get("thin_seconds", 60), progress)
    if a.provider == "gfw":
        from app.ais.gfw_provider import GFWPresenceAISProvider
        return GFWPresenceAISProvider(progress=progress)
    if a.provider == "synthetic":
        from app.ais.synthetic_provider import SyntheticAISProvider
        return SyntheticAISProvider(dict(a.get("synthetic", {})))
    if a.provider == "local":
        return LocalAISProvider([cfg.path(p) for p in a.files], a.get("column_map", "default"))
    raise AISDataError(f"Unknown ais.provider '{a.provider}'",
                       "Implement AISProvider for your API (see ais/provider.py) and register it here.")
