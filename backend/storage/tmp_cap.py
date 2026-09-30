"""Cap the backend's scratch files on Cloud Run, where /tmp lives in the instance's memory
(docs/deploy-cloud-run.md; values justified in docs/container-measurements.md, "the /tmp cap").

Managed roots: the generated jobs (DW2_GENERATED_DIR, one folder per job) and the uploads / searched scenes
(DW2_UPLOADS_DIR, one folder per input id). Each direct child is an ENTRY; its "last used" time is the folder's
mtime, which touch() refreshes on every request that reads or writes it.

sweep() deletes, oldest first:
  1. entries idle longer than MAX_AGE_S, then
  2. more of the oldest entries while the managed total exceeds MAX_TOTAL_BYTES.
It NEVER deletes an entry that
  - is held by an in-flight request (in_use(), a reference count), or
  - was used in the last MIN_IDLE_S. This also covers responses that stream a file after the handler has
    returned (FileResponse), and the gaps between one input's requests (upload -> FABDEM -> generate).
So the total may briefly exceed the cap while everything is busy or fresh; it is bounded by the in-flight work.

maybe_sweep() is cheap and rate-limited; the writers call it before they write.
OFF unless DW2_TMP_CAP=1 (set by docker/Dockerfile only): the desktop app uses the same folder variables for its
per-user cache and keeps its files. Values: DW2_TMP_MAX_AGE_S / DW2_TMP_MAX_MB / DW2_TMP_MIN_IDLE_S.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

MAX_AGE_S = float(os.environ.get("DW2_TMP_MAX_AGE_S", 30 * 60))
MAX_TOTAL_BYTES = int(float(os.environ.get("DW2_TMP_MAX_MB", 64)) * 2**20)
MIN_IDLE_S = float(os.environ.get("DW2_TMP_MIN_IDLE_S", 120))
SWEEP_INTERVAL_S = 30.0

_lock = threading.Lock()  # guards _in_use and serialises sweeps
_in_use: dict[str, int] = {}
_last_sweep = 0.0


def enabled() -> bool:
    return os.environ.get("DW2_TMP_CAP") == "1" and MAX_TOTAL_BYTES > 0 and MAX_AGE_S > 0


def roots() -> list[Path]:
    out = []
    for var in ("DW2_GENERATED_DIR", "DW2_UPLOADS_DIR"):
        v = os.environ.get(var)
        if v:
            out.append(Path(v))
    return out


def _key(path: Path) -> str:
    return str(Path(path).resolve())


def touch(path: Path) -> None:
    """Mark an entry as just used (its folder mtime)."""
    try:
        os.utime(path)
    except OSError:
        pass


@contextmanager
def in_use(*paths: Path):
    """Hold entries for the duration of a request: sweep() will not delete them."""
    keys = [_key(p) for p in paths if p is not None]
    with _lock:
        for k in keys:
            _in_use[k] = _in_use.get(k, 0) + 1
    for p in paths:
        if p is not None:
            touch(p)
    try:
        yield
    finally:
        with _lock:
            for k in keys:
                n = _in_use.get(k, 0) - 1
                if n > 0:
                    _in_use[k] = n
                else:
                    _in_use.pop(k, None)
        for p in paths:
            if p is not None:
                touch(p)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def _entries() -> list[tuple[float, int, Path]]:
    out = []
    for root in roots():
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for p in children:
            try:
                out.append((p.stat().st_mtime, _size(p), p))
            except OSError:
                continue
    return out


def _remove(p: Path) -> bool:
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        log.warning("tmp cap: could not remove %s: %s", p, exc)
        return False


def sweep(now: float | None = None) -> dict:
    """One pass; returns what it did (for logs and tests)."""
    now = time.time() if now is None else now
    with _lock:
        entries = sorted(_entries())  # oldest first
        busy = set(_in_use)
        total = sum(s for _, s, _ in entries)
        removed, freed, kept_busy = 0, 0, 0

        def protected(mtime: float, p: Path) -> bool:
            return _key(p) in busy or now - mtime < MIN_IDLE_S

        survivors = []
        for mtime, size, p in entries:
            if now - mtime > MAX_AGE_S and not protected(mtime, p) and _remove(p):
                removed, freed, total = removed + 1, freed + size, total - size
            else:
                survivors.append((mtime, size, p))
        for mtime, size, p in survivors:  # still oldest first
            if total <= MAX_TOTAL_BYTES:
                break
            if protected(mtime, p):
                kept_busy += 1
                continue
            if _remove(p):
                removed, freed, total = removed + 1, freed + size, total - size
    res = {"removed": removed, "freed_bytes": freed, "total_bytes": total, "over_cap_but_protected": kept_busy}
    if removed:
        log.info("tmp cap: removed %d entries (%.1f MiB), %.1f MiB left", removed, freed / 2**20, total / 2**20)
    return res


def maybe_sweep() -> dict | None:
    """A sweep at most every SWEEP_INTERVAL_S (called by the writers before they write)."""
    global _last_sweep
    if not enabled():
        return None
    now = time.time()
    if now - _last_sweep < SWEEP_INTERVAL_S:
        return None
    _last_sweep = now
    try:
        return sweep(now)
    except Exception as exc:  # noqa: BLE001 — housekeeping must never fail a request
        log.warning("tmp cap: sweep failed: %s", exc)
        return None
