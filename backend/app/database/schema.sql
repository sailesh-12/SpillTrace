-- SIH26143 PostGIS schema (target persistent architecture).
-- Large artifacts (masks, particle arrays, rasters, reports) stay on disk; rows store paths.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id   TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    stage         TEXT,
    input_path    TEXT,
    artifact_dir  TEXT,
    error         JSONB,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id      SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    acquired_at   TIMESTAMPTZ,
    sensor        TEXT,
    georef_provenance TEXT,
    footprint     GEOMETRY(Polygon, 4326),
    image_path    TEXT
);

CREATE TABLE IF NOT EXISTS spills (
    spill_id      TEXT PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    observed_at   TIMESTAMPTZ,
    area_m2       DOUBLE PRECISION,
    perimeter_m   DOUBLE PRECISION,
    confidence    DOUBLE PRECISION,
    geom          GEOMETRY(Geometry, 4326),
    centroid      GEOMETRY(Point, 4326)
);
CREATE INDEX IF NOT EXISTS spills_geom_idx ON spills USING GIST (geom);

CREATE TABLE IF NOT EXISTS drift_runs (
    run_id        SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    direction     TEXT,
    model         TEXT,
    members       INT,
    particles     INT,
    forcing_provenance TEXT,
    particles_path TEXT,
    config        JSONB
);

CREATE TABLE IF NOT EXISTS source_regions (
    region_id     SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    level         TEXT,
    mass_fraction DOUBLE PRECISION,
    area_km2      DOUBLE PRECISION,
    geom          GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS source_regions_geom_idx ON source_regions USING GIST (geom);

CREATE TABLE IF NOT EXISTS vessels (
    mmsi          TEXT PRIMARY KEY,
    imo           TEXT,
    name          TEXT,
    vessel_type   TEXT
);

CREATE TABLE IF NOT EXISTS ais_positions (
    mmsi          TEXT,
    ts            TIMESTAMPTZ,
    geom          GEOMETRY(Point, 4326),
    sog           REAL,
    cog           REAL,
    PRIMARY KEY (mmsi, ts)
);
CREATE INDEX IF NOT EXISTS ais_positions_geom_idx ON ais_positions USING GIST (geom);
CREATE INDEX IF NOT EXISTS ais_positions_ts_idx ON ais_positions (ts);

CREATE TABLE IF NOT EXISTS ais_gaps (
    gap_id        SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    mmsi          TEXT,
    gap_start     TIMESTAMPTZ,
    gap_end       TIMESTAMPTZ,
    duration_min  REAL
);

CREATE TABLE IF NOT EXISTS sar_ship_detections (
    detection_id  SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    geom          GEOMETRY(Point, 4326),
    n_pixels      INT,
    matched_mmsi  TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    mmsi          TEXT REFERENCES vessels(mmsi),
    rank          INT,
    evidence_score DOUBLE PRECISION,       -- Evidence Correlation Score (NOT a probability of responsibility)
    feature_scores JSONB,
    evidence      JSONB,
    uncertainties JSONB
);

CREATE TABLE IF NOT EXISTS reports (
    report_id     SERIAL PRIMARY KEY,
    analysis_id   TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    path          TEXT,
    format        TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);
