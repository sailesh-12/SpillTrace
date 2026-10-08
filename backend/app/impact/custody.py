"""Tamper-evident evidence seal (chain of custody).

Every artifact of an analysis is hashed (SHA-256) into a manifest whose root hash covers all files; each new
seal records the previous root, forming a hash chain across re-analyses. `verify` recomputes the hashes and
reports modified, missing and new files. This supports MARPOL prosecutions where investigators must show that
the evidence package was not altered after generation. (It is integrity evidence, not a trusted timestamp; for
that, anchor the root hash with an RFC 3161 time-stamping authority.)"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "evidence_manifest.json"
EXCLUDE = {MANIFEST, "state.json"}           # state.json changes with every progress event


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name not in EXCLUDE and "forcing" not in p.parts)


def _root(entries: list[dict]) -> str:
    return hashlib.sha256("\n".join(f"{e['path']}:{e['sha256']}" for e in entries).encode()).hexdigest()


def seal(root: Path, analysis_id: str) -> dict:
    entries = [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p), "bytes": p.stat().st_size} for p in _files(root)]
    prev = None
    mp = root / MANIFEST
    if mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
            prev = {"root": old.get("root"), "sealed_at": old.get("sealed_at"), "version": old.get("version", 1)}
        except Exception:
            prev = None
    m = {"analysis_id": analysis_id, "algorithm": "SHA-256", "sealed_at": datetime.now(timezone.utc).isoformat(),
         "version": (prev or {}).get("version", 0) + 1, "previous": prev, "n_files": len(entries), "files": entries}
    m["root"] = _root(entries)
    m["chain"] = hashlib.sha256(f"{(prev or {}).get('root') or ''}|{m['root']}".encode()).hexdigest()
    mp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def verify(root: Path) -> dict:
    mp = root / MANIFEST
    if not mp.exists():
        return {"sealed": False, "ok": False, "message": "No evidence seal for this analysis yet."}
    m = json.loads(mp.read_text(encoding="utf-8"))
    recorded = {e["path"]: e["sha256"] for e in m["files"]}
    now = {p.relative_to(root).as_posix(): p for p in _files(root)}
    modified = [k for k, h in recorded.items() if k in now and _sha(now[k]) != h]
    missing = [k for k in recorded if k not in now]
    added = [k for k in now if k not in recorded]
    ok = not modified and not missing
    return {"sealed": True, "ok": ok, "root": m["root"], "chain": m.get("chain"), "version": m.get("version"),
            "sealed_at": m["sealed_at"], "n_files": m["n_files"], "modified": modified, "missing": missing, "added": added,
            "message": ("All sealed files match their recorded SHA-256 hashes." if ok else
                        f"{len(modified)} modified and {len(missing)} missing file(s) since sealing.")}
