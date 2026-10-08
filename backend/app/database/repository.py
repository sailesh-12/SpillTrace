"""Persistence behind a repository interface.

Scientific modules never touch storage; only the orchestrator calls a
Repository. Large artifacts (masks, particle arrays, rasters, reports) always
live on the filesystem; the PostGIS repository additionally indexes analyses,
spills, source regions, vessels, candidates and evidence as rows with
geometries (schema in database/schema.sql) and stores artifact paths.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj) -> str:
    return json.dumps(obj, default=_json_default, allow_nan=False, indent=None)


def sanitize(obj):
    """Replace NaN/inf by None so output is strict JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):          # any numpy scalar (bool_, int64, float32, str_...)
        return sanitize(obj.item())
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


class FileRepository:
    """DEMO/OFFLINE persistence: one folder per analysis under outputs/analyses/<id>/."""

    kind = "file"

    def __init__(self, root: Path):
        self.root = Path(root) / "analyses"
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, analysis_id: str) -> Path:
        d = self.root / analysis_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exists(self, analysis_id: str) -> bool:
        return (self.root / analysis_id / "state.json").exists()

    def save_json(self, analysis_id: str, name: str, obj) -> Path:
        p = self.dir(analysis_id) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(dumps(sanitize(obj)), encoding="utf-8")
        for attempt in range(20):            # atomic replace; Windows may briefly lock a file being read
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            tmp.unlink(missing_ok=True)
            raise PermissionError(f"could not replace {p} (file locked)")
        return p

    def load_json(self, analysis_id: str, name: str):
        p = self.root / analysis_id / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found for analysis {analysis_id}")
        return json.loads(p.read_text(encoding="utf-8"))

    def artifact_path(self, analysis_id: str, name: str) -> Path:
        p = self.dir(analysis_id) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_state(self, analysis_id: str, state: dict):
        state = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
        self.save_json(analysis_id, "state.json", state)

    def load_state(self, analysis_id: str) -> dict:
        return self.load_json(analysis_id, "state.json")

    def list_analyses(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "state.json").exists():
                s = json.loads((d / "state.json").read_text(encoding="utf-8"))
                out.append({k: s.get(k) for k in ("analysis_id", "status", "stage", "input", "created_at", "updated_at")})
        return out

    # Hooks the PostGIS repository overrides; the file repository keeps everything in JSON.
    def index_results(self, analysis_id: str, results: dict):
        pass


class PostGISRepository(FileRepository):
    """Target architecture: filesystem artifacts + PostgreSQL/PostGIS index."""

    kind = "postgis"

    def __init__(self, root: Path, dsn: str):
        super().__init__(root)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg not installed: pip install 'psycopg[binary]'") from exc
        self.psycopg = psycopg
        self.dsn = dsn
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute((Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8"))

    def save_state(self, analysis_id: str, state: dict):
        super().save_state(analysis_id, state)
        with self.psycopg.connect(self.dsn) as conn:
            conn.execute(
                """INSERT INTO analyses (analysis_id, status, stage, input_path, artifact_dir, error, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s, now())
                   ON CONFLICT (analysis_id) DO UPDATE SET status=EXCLUDED.status, stage=EXCLUDED.stage,
                   error=EXCLUDED.error, updated_at=now()""",
                (analysis_id, state.get("status"), state.get("stage"), state.get("input"),
                 str(self.root / analysis_id), dumps(state.get("error")) if state.get("error") else None))

    def index_results(self, analysis_id: str, results: dict):
        spill = results.get("spill")
        with self.psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            if spill and spill.get("polygon"):
                cur.execute(
                    """INSERT INTO spills (spill_id, analysis_id, observed_at, area_m2, perimeter_m, confidence, geom, centroid)
                       VALUES (%s,%s,%s,%s,%s,%s, ST_SetSRID(ST_GeomFromGeoJSON(%s),4326),
                               ST_SetSRID(ST_MakePoint(%s,%s),4326))
                       ON CONFLICT (spill_id) DO UPDATE SET area_m2=EXCLUDED.area_m2, geom=EXCLUDED.geom""",
                    (spill["spill_id"], analysis_id, spill["timestamp"], spill["area_m2"], spill["perimeter_m"],
                     spill["detection_confidence"], json.dumps(spill["polygon"]),
                     spill["centroid"]["lon"], spill["centroid"]["lat"]))
            for level, reg in (results.get("source_regions") or {}).items():
                cur.execute(
                    """INSERT INTO source_regions (analysis_id, level, mass_fraction, area_km2, geom)
                       VALUES (%s,%s,%s,%s, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)))""",
                    (analysis_id, level, reg["mass_fraction"], reg["area_km2"], json.dumps(reg["geometry"])))
            for c in results.get("candidates") or []:
                v = c["vessel"]
                cur.execute("""INSERT INTO vessels (mmsi, imo, name, vessel_type) VALUES (%s,%s,%s,%s)
                               ON CONFLICT (mmsi) DO UPDATE SET name=EXCLUDED.name, vessel_type=EXCLUDED.vessel_type""",
                            (v["mmsi"], v.get("imo"), v.get("name"), v.get("type")))
                cur.execute("""INSERT INTO candidates (analysis_id, mmsi, rank, evidence_score, feature_scores,
                               evidence, uncertainties) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                            (analysis_id, v["mmsi"], c["rank"], c["score"], json.dumps(c["feature_scores"]),
                             json.dumps(sanitize(c["evidence"])), json.dumps(c["evidence"]["uncertainties"])))
            if results.get("report_path"):
                cur.execute("INSERT INTO reports (analysis_id, path, format) VALUES (%s,%s,%s)",
                            (analysis_id, results["report_path"], "html"))


def build_repository(cfg):
    root = cfg.path(cfg.storage.output_dir)
    if cfg.storage.backend == "postgis":
        return PostGISRepository(root, cfg.storage.postgis_dsn)
    return FileRepository(root)
