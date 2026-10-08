"""Fast windowed reads from remote Cloud-Optimised GeoTIFFs.

Measured against Planetary Computer blobs: ~2 s latency per HTTP request, ~3 MB/s per stream.
GDAL's defaults (16 KB chunks, sequential) needed 126 s for a 2048x2048 window. We:
  * set GDAL to fetch whole tiles per request and merge consecutive ranges,
  * read block-aligned sub-windows concurrently (one dataset handle per thread),
  * read from the COG's internal overview when a coarser resolution is requested.
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE",   # HTTP/1.1: HTTP/2 stalled under concurrency
                VSI_CACHE_SIZE="268435456", CPL_VSIL_CURL_CHUNK_SIZE="4194304",
                GDAL_INGESTED_BYTES_AT_OPEN="1048576", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                # without these a stalled TCP connection could block a tile read forever (observed in production)
                GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20",
                GDAL_HTTP_LOW_SPEED_TIME="30", GDAL_HTTP_LOW_SPEED_LIMIT="1024")


def read_window(href: str, row0: int, col0: int, rows: int, cols: int, factor: int = 1,
                workers: int = 12, progress=None, chunk_timeout_s: float = 120) -> tuple[np.ndarray, int]:
    """Read [row0:row0+rows, col0:col0+cols] (full-resolution pixel coords), decimated by `factor`.

    Returns (array, effective_factor). If the COG has an overview matching `factor` (2,4,8..) it is read
    directly (fewer bytes); otherwise full resolution is read and block-averaged.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            ovr = src.overviews(1)
            bh, bw = src.block_shapes[0]
    # Internal overviews were measured to be SLOWER on these blobs (160 s vs 44 s for a full AOI),
    # so we always read native tiles in parallel and block-average when a coarser grid is requested.
    level = None
    f_read = factor if level is not None else 1
    r0, c0 = row0 // f_read, col0 // f_read
    nr, nc = max(rows // f_read, 1), max(cols // f_read, 1)
    out = np.zeros((nr, nc), np.float32)
    # block-aligned chunks (2x2 blocks per request keeps request count low while parallel)
    ch, cw = bh, bw                                     # one block per task: better balancing of slow requests
    tasks = [(y, x) for y in range((r0 // ch) * ch, r0 + nr, ch) for x in range((c0 // cw) * cw, c0 + nc, cw)]
    local = threading.local()
    done = [0]
    lock = threading.Lock()

    def ds():
        if not hasattr(local, "src"):
            local.env = rasterio.Env(**GDAL_ENV)
            local.env.__enter__()
            local.src = rasterio.open(href, overview_level=level) if level is not None else rasterio.open(href)
        return local.src

    def job(yx):
        y, x = yx
        ys, xs = max(y, r0), max(x, c0)
        ye, xe = min(y + ch, r0 + nr), min(x + cw, c0 + nc)
        if ye <= ys or xe <= xs:
            return
        a = ds().read(1, window=Window(xs, ys, xe - xs, ye - ys)).astype(np.float32)
        out[ys - r0:ye - r0, xs - c0:xe - c0] = a
        with lock:
            done[0] += 1
            if progress and (done[0] % max(len(tasks) // 5, 1) == 0 or done[0] == len(tasks)):
                progress(f"Downloaded {done[0]}/{len(tasks)} image chunks")

    # Each chunk gets a deadline; stalled/failed chunks are retried once on a fresh handle, then the read
    # fails with a clear error instead of hanging the analysis worker.
    from concurrent.futures import TimeoutError as FTimeout, wait
    from app.core.errors import InputImageError
    pending = list(tasks)
    # 3 attempts; retries use fewer parallel connections and a longer deadline (Azure throttles bursts)
    for attempt, n_workers in enumerate((min(workers, max(len(tasks), 1)), 4, 2)):
        pool = ThreadPoolExecutor(max_workers=max(1, min(n_workers, len(pending))))
        try:
            futs = {pool.submit(job, t): t for t in pending}
            deadline = chunk_timeout_s * (attempt + 1) * max(1, len(pending) / n_workers)
            done_f, not_done = wait(futs, timeout=deadline)
            failed = [futs[f] for f in not_done] + [futs[f] for f in done_f if f.exception() is not None]
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if not failed:
            break
        pending = failed
        local.__dict__.clear()                          # force new dataset handles for the retry
        if progress:
            progress(f"{len(pending)} image chunk(s) slow — retrying with fewer connections (attempt {attempt + 2}/3)")
    else:
        raise InputImageError(f"Sentinel-1 download stalled: {len(pending)} image chunk(s) did not arrive after 3 "
                              "attempts.", "The Planetary Computer storage is slow right now: retry later, choose a "
                              "smaller area, or use a SYNTHETIC scene for demonstration.")
    if f_read == 1 and factor > 1:                      # no matching overview: block-average
        h, w = (nr // factor) * factor, (nc // factor) * factor
        out = out[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return out, factor
