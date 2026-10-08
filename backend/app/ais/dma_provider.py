"""Danish Maritime Authority historical AIS (public S3 bucket, no API key).

Daily zipped CSVs (~0.5 GB zip / 3 GB CSV per day) are downloaded once per UTC day,
streamed through pyarrow (vessel classes A/B only, needed columns only) into a
zstd Parquet cache (~0.25 GB/day), and then queried by bbox/time with predicate
push-down. Archive layout has changed over time, so several key patterns are tried.
Coverage: Danish waters and approaches (roughly 3–17 E, 53–59 N), 2024-03 onwards daily.
"""
from __future__ import annotations

import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.ais.provider import CANONICAL, AISProvider
from app.core.errors import AISDataError

BUCKET = "http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com/"
COVERAGE = (3.0, 53.0, 17.5, 59.5)     # approximate DMA receiver coverage (w, s, e, n)
COLS = ["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude", "SOG", "COG", "IMO", "Name", "Ship type"]


class DMAAISProvider(AISProvider):
    name = "Danish Maritime Authority (aisdata.ais.dk)"
    provenance = "LIVE_EXTERNAL"

    def __init__(self, cache_dir: Path, thin_seconds: int = 60, progress=None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.thin = thin_seconds
        self.progress = progress
        self._meta: dict[str, dict] = {}

    # --------------------------------------------------------------------------------------------
    def covers(self, bbox) -> bool:
        w, s, e, n = bbox
        return not (e < COVERAGE[0] or w > COVERAGE[2] or n < COVERAGE[1] or s > COVERAGE[3])

    def _say(self, msg):
        if self.progress:
            self.progress(msg)

    def _keys_for(self, day: datetime) -> list[str]:
        d = day.strftime("%Y-%m-%d")
        return [f"aisdk-{d}.zip", f"{day:%Y}/aisdk-{d}.zip"]

    def _download(self, day: datetime) -> Path:
        import httpx
        for key in self._keys_for(day):
            try:
                with httpx.Client(timeout=120) as c:
                    head = c.head(BUCKET + key)
                    if head.status_code != 200:
                        continue
                    size = int(head.headers.get("content-length", 0))
                    dst = self.cache / Path(key).name
                    self._say(f"Downloading DMA AIS {key} ({size / 1e6:.0f} MB)")
                    t0 = time.time()
                    with c.stream("GET", BUCKET + key) as r, open(dst.with_suffix(".part"), "wb") as f:
                        r.raise_for_status()
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                    dst.with_suffix(".part").replace(dst)
                    self._say(f"Downloaded {key} in {time.time() - t0:.0f}s")
                    return dst
            except httpx.HTTPError as exc:
                raise AISDataError(f"DMA AIS download failed for {key}: {exc}", "Check internet access to aisdata.ais.dk")
        raise AISDataError(f"No DMA AIS file published for {day:%Y-%m-%d}",
                           "DMA publishes daily files from March 2024 with a delay of a few days.")

    def _convert(self, zpath: Path, out: Path):
        import pyarrow as pa
        import pyarrow.compute as C
        import pyarrow.csv as pc
        import pyarrow.parquet as pq
        self._say(f"Indexing {zpath.name} into Parquet cache")
        types = {c: pa.string() for c in COLS}
        types.update({"Latitude": pa.float64(), "Longitude": pa.float64(), "SOG": pa.float32(), "COG": pa.float32()})
        z = zipfile.ZipFile(zpath)
        csv_info = next(i for i in z.infolist() if i.filename.endswith(".csv"))
        writer = None
        with z.open(csv_info) as fh:
            rd = pc.open_csv(fh, read_options=pc.ReadOptions(block_size=64 << 20),
                             convert_options=pc.ConvertOptions(include_columns=COLS, column_types=types))
            for b in rd:
                b = b.filter(C.is_in(b.column("Type of mobile"), value_set=pa.array(["Class A", "Class B"])))
                ts = C.strptime(b.column("# Timestamp"), format="%d/%m/%Y %H:%M:%S", unit="s")
                tbl = pa.table({"mmsi": b.column("MMSI"), "timestamp": ts, "lat": b.column("Latitude"),
                                "lon": b.column("Longitude"), "sog": b.column("SOG"), "cog": b.column("COG"),
                                "imo": b.column("IMO"), "vessel_name": b.column("Name"),
                                "vessel_type": b.column("Ship type")})
                if writer is None:
                    writer = pq.ParquetWriter(out.with_suffix(".part"), tbl.schema, compression="zstd")
                writer.write_table(tbl)
        writer.close()
        out.with_suffix(".part").replace(out)
        z.close()
        zpath.unlink(missing_ok=True)          # keep only the compact cache

    def ensure_day(self, day: datetime) -> Path:
        out = self.cache / f"dma_{day:%Y-%m-%d}.parquet"
        if not out.exists():
            self._convert(self._download(day), out)
        return out

    # --------------------------------------------------------------------------------------------
    def get_tracks(self, bbox, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        if not self.covers(bbox):
            raise AISDataError("No DMA AIS coverage for this region.",
                               "The DMA archive covers Danish waters only; configure another AISProvider elsewhere.")
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as C
        import pyarrow.parquet as pq
        w, s, e, n = bbox
        t0 = pa.scalar(int(start_time.timestamp()), type=pa.timestamp("s"))
        t1 = pa.scalar(int(end_time.timestamp()), type=pa.timestamp("s"))
        day = datetime(start_time.year, start_time.month, start_time.day, tzinfo=timezone.utc)
        days = []
        while day <= end_time:
            days.append(day)
            day += timedelta(days=1)
        # download + index missing days concurrently (sequential first-time fetch took ~50 s per day)
        from concurrent.futures import ThreadPoolExecutor
        missing = [d for d in days if not (self.cache / f"dma_{d:%Y-%m-%d}.parquet").exists()]
        if missing:
            with ThreadPoolExecutor(max_workers=min(4, len(missing))) as pool:
                list(pool.map(self.ensure_day, missing))
        day = days[0]
        tables = []
        while day <= end_time:
            p = self.ensure_day(day)
            tbl = pq.read_table(p, filters=[("lat", ">=", s), ("lat", "<=", n), ("lon", ">=", w), ("lon", "<=", e)])
            # everything filtered in Arrow BEFORE converting to pandas (10x fewer rows reach Python)
            ts = tbl.column("timestamp").cast(pa.timestamp("s"))
            keep = C.and_(C.and_(C.greater_equal(ts, t0), C.less_equal(ts, t1)),
                          C.match_substring_regex(tbl.column("mmsi"), r"^\d{9}$"))   # valid MMSI (drops AtoN/base)
            tbl = tbl.filter(keep)
            if self.thin and tbl.num_rows:
                # at most one report per `thin` seconds per vessel (first in each bin; no positions invented).
                # Done with integer keys in NumPy: sorting 10M Arrow-string rows in pandas took ~12 s.
                mm = C.cast(tbl.column("mmsi"), pa.int64()).to_numpy()
                sec = C.cast(C.cast(tbl.column("timestamp"), pa.timestamp("s")), pa.int64()).to_numpy()
                key = mm * 10_000_000 + (sec // self.thin) % 10_000_000
                order = np.lexsort((sec, key))
                _, first_idx = np.unique(key[order], return_index=True)
                tbl = tbl.take(pa.array(np.sort(order[first_idx])))
            tables.append(tbl)
            day += timedelta(days=1)
        if not tables:
            return pd.DataFrame(columns=CANONICAL)
        df = pa.concat_tables(tables).to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # DMA marks missing static fields as "Unknown"/"Undefined"; static data appear only in some messages,
        # so take each vessel's first valid value (vectorised groupby().first() skips nulls)
        static = ["imo", "vessel_name", "vessel_type"]
        for c in static:
            df[c] = df[c].astype(object).where(~df[c].isin(["Unknown", "Undefined", "", "nan"]) & df[c].notna(), None)
        first = df.groupby("mmsi", sort=False)[static].first()
        df = df.drop(columns=static).join(first, on="mmsi")
        self._meta.update({m: {"name": r.vessel_name, "imo": r.imo, "type_raw": r.vessel_type}
                           for m, r in first.iterrows()})
        return df[CANONICAL].reset_index(drop=True)

    def get_vessel_metadata(self, mmsi: str) -> dict:
        return {"mmsi": mmsi, **self._meta.get(str(mmsi), {})}

    def describe(self) -> dict:
        return {"provider": self.name, "provenance": self.provenance, "coverage_bbox": COVERAGE,
                "thinning_seconds": self.thin, "cache_dir": str(self.cache)}
