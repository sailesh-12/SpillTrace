"""Data provenance labels.

Every dataset flowing through the pipeline is tagged so the report and UI can
distinguish LIVE/EXTERNAL data from DEMO/LOCAL data and from UNAVAILABLE inputs.
"""
from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    LIVE = "LIVE_EXTERNAL"              # fetched from an external operational service
    LOCAL = "LOCAL_FILE"                # real data supplied as a local file
    DEMO_SYNTHETIC = "SYNTHETIC_DEMO"   # generated for demonstration; not real-world
    DEMO_ASSUMED = "DEMO_ASSUMED"       # an operator/demo assumption (e.g. sidecar bbox)
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceKind(str, Enum):
    """Epistemic category of each statement in a report."""
    OBSERVED = "OBSERVED_FACT"
    MODEL = "MODEL_DERIVED"
    ASSUMPTION = "ASSUMPTION"
    INFERENCE = "CANDIDATE_INFERENCE"
