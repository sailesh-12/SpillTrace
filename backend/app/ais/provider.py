"""AIS provider abstraction. Any historical AIS source implements this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# Canonical AIS columns used everywhere downstream.
CANONICAL = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_name", "imo", "vessel_type"]


class AISProvider(ABC):
    name = "abstract"
    provenance = "UNAVAILABLE"

    @abstractmethod
    def get_tracks(self, bbox: list[float], start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Return raw AIS positions (CANONICAL columns) inside bbox [w,s,e,n] and time range."""

    @abstractmethod
    def get_vessel_metadata(self, mmsi: str) -> dict:
        """Return static vessel data (name, imo, type) for one MMSI, or {} if unknown."""

    def describe(self) -> dict:
        return {"provider": self.name, "provenance": self.provenance}
