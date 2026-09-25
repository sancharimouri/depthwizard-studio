"""Run the frozen binary's self-test with a process-count guard; print one JSON line.

A respawn loop (missing multiprocessing.freeze_support) shows up as many distinct
processes running the same executable within the time limit."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

exe, img, geotiff, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
env = {**os.environ}
if "--skip-freeze-support" in sys.argv:
    env["DW2_SKIP_FREEZE_SUPPORT"] = "1"
for k in ("HF_TOKEN", "DAV2_INFERENCE_URL"):
    env.pop(k, None)
t0 = time.time()
p = subprocess.Popen([str(exe), "--selftest", img, str(Path(img).with_suffix(".npy")), geotiff],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
seen, limit = set(), 240
while p.poll() is None and time.time() - t0 < limit:
    for q in psutil.process_iter(["pid", "exe"]):
        try:
            if q.info["exe"] and Path(q.info["exe"]).resolve() == exe.resolve():
                seen.add(q.info["pid"])
        except (psutil.Error, OSError):
            pass
    time.sleep(0.2)
timed_out = p.poll() is None
if timed_out:
    for q in psutil.process_iter(["pid", "exe"]):
        try:
            if q.info["exe"] and Path(q.info["exe"]).resolve() == exe.resolve():
                q.kill()
        except (psutil.Error, OSError):
            pass
    p.kill()
out = p.stdout.read() if not timed_out else ""
lines = [l for l in out.splitlines() if l.startswith(("SELFTEST", "GEOTEST"))]
errors = [l for l in out.splitlines() if "Error" in l][-3:]
print(json.dumps({"check": label, "wall_s": round(time.time() - t0, 1), "timed_out": timed_out,
                  "distinct_processes": len(seen), "exit": p.returncode, "results": lines, "errors": errors}))
