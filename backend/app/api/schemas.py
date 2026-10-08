"""Pydantic request/response schemas (data-flow contracts, spec §31)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisCreated(BaseModel):
    analysis_id: str
    status: str = "QUEUED"


class AnalysisState(BaseModel):
    analysis_id: str
    status: str
    stage: str
    outcome: Optional[str] = None
    input: Optional[str] = None
    warnings: list[str] = []
    error: Optional[dict] = None
    events: list[dict] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SpillOutput(BaseModel):
    """Segmentation output contract."""
    spill_id: str
    timestamp: Optional[str]
    polygon: Optional[dict]
    centroid: Optional[dict]
    area_m2: Optional[float]
    confidence: float
    georef_status: str
    error: Optional[dict] = None


class BacktrackRequest(BaseModel):
    spill_id: str
    particles: Optional[int] = Field(None, ge=10, le=20000)
    members: Optional[int] = Field(None, ge=1, le=50)


class ForwardRequest(BaseModel):
    spill_id: str
    hours: Optional[float] = Field(None, gt=0, le=168)
    particles: Optional[int] = Field(None, ge=10, le=20000)


class AnalyzeRequest(BaseModel):
    spill_id: str
    particles: Optional[int] = Field(None, ge=10, le=20000)
    members: Optional[int] = Field(None, ge=1, le=50)


class DriftOutput(BaseModel):
    """Drift output contract."""
    spill_id: str
    release_window: dict
    source_region: dict
    probability_map: str          # URL of source_probability GeoJSON
    particles: str                # URL of particle frames
    uncertainties: list[str] = []


class CandidateOut(BaseModel):
    """Final candidate contract."""
    mmsi: str
    imo: Optional[str]
    vessel_name: Optional[str]
    vessel_type: str
    evidence_score: float
    feature_scores: dict[str, float]
    source_distance_km: Optional[float]
    time_difference_hours: Optional[float]
    evidence_summary: str
    uncertainties: list[str]
    evidence: dict[str, Any]
