"""The /tmp cap (backend/storage/tmp_cap.py): age + total size, never an in-flight or freshly used entry."""

import os
import threading
import time

import pytest

from backend.storage import tmp_cap

MiB = 2**20


@pytest.fixture
def roots(tmp_path, monkeypatch):
    gen, up = tmp_path / "gen", tmp_path / "up"
    gen.mkdir()
    up.mkdir()
    monkeypatch.setenv("DW2_GENERATED_DIR", str(gen))
    monkeypatch.setenv("DW2_UPLOADS_DIR", str(up))
    monkeypatch.setenv("DW2_TMP_CAP", "1")
    monkeypatch.setattr(tmp_cap, "MAX_AGE_S", 1800.0)
    monkeypatch.setattr(tmp_cap, "MAX_TOTAL_BYTES", 10 * MiB)
    monkeypatch.setattr(tmp_cap, "MIN_IDLE_S", 120.0)
    monkeypatch.setattr(tmp_cap, "_last_sweep", 0.0)
    tmp_cap._in_use.clear()
    yield gen, up
    tmp_cap._in_use.clear()


def entry(root, name, mib, age_s, now):
    d = root / name
    d.mkdir()
    (d / "blob.bin").write_bytes(b"\0" * int(mib * MiB))
    os.utime(d, (now - age_s, now - age_s))
    return d


def test_old_entries_go_recent_ones_stay(roots):
    gen, up = roots
    now = time.time()
    old = entry(gen, "old", 1, 3600, now)
    recent = entry(up, "recent", 1, 60, now)
    mid = entry(gen, "mid", 1, 900, now)
    r = tmp_cap.sweep(now)
    assert not old.exists() and recent.exists() and mid.exists()
    assert r["removed"] == 1 and r["freed_bytes"] == MiB


def test_over_the_size_cap_the_oldest_idle_go_first(roots):
    gen, up = roots
    now = time.time()
    a = entry(gen, "a", 4, 1000, now)
    b = entry(up, "b", 4, 800, now)
    c = entry(gen, "c", 4, 600, now)  # 12 MiB > 10 MiB cap
    tmp_cap.sweep(now)
    assert not a.exists() and b.exists() and c.exists()


def test_never_deletes_an_in_flight_entry_even_old_and_over_cap(roots):
    gen, up = roots
    now = time.time()
    busy = entry(up, "busy", 8, 7200, now)
    other = entry(gen, "other", 8, 7000, now)
    with tmp_cap.in_use(busy):
        os.utime(busy, (now - 7200, now - 7200))  # pretend it's old (in_use touched it)
        r = tmp_cap.sweep(now)
        assert busy.exists()
        assert not other.exists()
        assert r["removed"] == 1
    assert "busy" not in str(tmp_cap._in_use)  # released


def test_freshly_used_entries_survive_size_pressure(roots):
    """e.g. a job whose assets are still streaming after its handler returned (FileResponse)."""
    gen, _ = roots
    now = time.time()
    fresh = entry(gen, "fresh", 20, 30, now)  # alone above the cap, but used 30 s ago
    r = tmp_cap.sweep(now)
    assert fresh.exists() and r["over_cap_but_protected"] == 1


def test_nested_holds_are_counted(roots):
    gen, _ = roots
    now = time.time()
    d = entry(gen, "job", 1, 5000, now)
    with tmp_cap.in_use(d):
        with tmp_cap.in_use(d):
            pass
        os.utime(d, (now - 5000, now - 5000))
        tmp_cap.sweep(now)
        assert d.exists()  # still held by the outer request


def test_off_unless_enabled_and_rate_limited(roots, monkeypatch):
    gen, _ = roots
    now = time.time()
    entry(gen, "old", 1, 9000, now)
    monkeypatch.delenv("DW2_TMP_CAP")
    assert tmp_cap.maybe_sweep() is None and (gen / "old").exists()  # the desktop app keeps its files
    monkeypatch.setenv("DW2_TMP_CAP", "1")
    assert tmp_cap.maybe_sweep()["removed"] == 1
    entry(gen, "old2", 1, 9000, now)
    assert tmp_cap.maybe_sweep() is None  # within SWEEP_INTERVAL_S


def test_concurrent_requests_keep_their_entries(roots):
    """Concurrency 4: requests holding entries while other threads sweep continuously."""
    gen, up = roots
    now = time.time()
    held = [entry(up, f"in{i}", 3, 7200, now) for i in range(4)]
    stop = threading.Event()
    errors = []

    def request(d):
        with tmp_cap.in_use(d):
            os.utime(d, (now - 7200, now - 7200))
            time.sleep(0.3)
            if not d.exists():
                errors.append(d.name)

    def sweeper():
        while not stop.is_set():
            tmp_cap.sweep()

    sw = [threading.Thread(target=sweeper) for _ in range(2)]
    rq = [threading.Thread(target=request, args=(d,)) for d in held]
    for t in sw + rq:
        t.start()
    for t in rq:
        t.join()
    stop.set()
    for t in sw:
        t.join()
    assert errors == []


def test_generation_holds_its_job_and_input(roots, monkeypatch):
    from backend.generation import pipeline

    gen, up = roots
    inp = up / "abc"
    inp.mkdir()
    seen = {}

    def fake(kind, item_id, preview, depth, job_id, out, **kw):
        seen["held"] = {tmp_cap._key(out), tmp_cap._key(inp)} <= set(tmp_cap._in_use)
        return {"job": job_id}

    monkeypatch.setattr(pipeline, "_generate", fake)
    pipeline.generate("input", "abc", b"", {}, input_dir=inp)
    assert seen["held"] is True
    assert tmp_cap._in_use == {}
