"""Actionable pipeline errors.

Each error carries a machine-readable code, a human message, and a hint telling
the operator what to do next. The API returns these verbatim.
"""
from __future__ import annotations


class PipelineError(Exception):
    code = "PIPELINE_ERROR"

    def __init__(self, message: str, hint: str = "", stage: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.stage = stage

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint, "stage": self.stage}


class ModelError(PipelineError):
    code = "MODEL_ERROR"


class InputImageError(PipelineError):
    code = "INVALID_SAR_IMAGE"


class GeoreferenceUnavailable(PipelineError):
    code = "GEOREFERENCING_UNAVAILABLE"


class TimestampUnavailable(PipelineError):
    code = "TIMESTAMP_UNAVAILABLE"


class ForcingUnavailable(PipelineError):
    code = "ENVIRONMENTAL_FORCING_UNAVAILABLE"


class DriftError(PipelineError):
    code = "DRIFT_ERROR"


class AISDataError(PipelineError):
    code = "AIS_DATA_ERROR"
