#!/usr/bin/env python3
"""Backend container measurements (docs/container-measurements.md): size, start-up, memory, flows.

  size    IMAGE [--platform linux/amd64]  compressed size (pushed to a LOCAL registry at 127.0.0.1:5055,
          never a remote one), unpacked size (du inside the container), per-layer sizes (docker history),
          and the top 25 installed Python distributions by size
  start   IMAGE [--runs 5]                container start -> first 200 from /health (median), idle RSS
  flows   IMAGE [--flows png,geotiff,cdse,cdse_facts] one fresh container per flow; after it: peak RSS of the
          server process (VmHWM), cgroup memory.peak / memory.current, /tmp usage; outputs saved for parity.
          cdse_facts = the CDSE flow plus the live Facts + Scenario lookup for the scene's bbox (/api/facts)
  concurrent IMAGE                        png + geotiff + cdse_facts at once in ONE container, then its peaks
  trace   IMAGE                           runs all flows in one container under an import tracer and saves the
          modules actually loaded (Part 1 import audit)

Credentials are passed at run time only: build/slim/bench.env (HF_TOKEN, EARTHENGINE_PROJECT, CDSE_CLIENT_ID,
CDSE_CLIENT_SECRET, copied from the repo .env; gitignored) and ~/.config/earthengine mounted read-only.
Everything is written under build/slim/ (gitignored). Timings on this Mac (Apple Silicon, arm64 native under
Colima) are indicative only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/slim"
ENV_FILE = OUT / "bench.env"
INPUTS = OUT / "inputs"
EE_CREDS = Path.home() / ".config/earthengine"
PORT = 18080
REGISTRY = "127.0.0.1:5055"
CONTAINER_ENV = {"DW2_NO_DOTENV": "1", "DW2_UPLOADS_DIR": "/tmp/dw2/up", "DW2_CACHE_DIR": "/tmp/dw2/cache",
                 "DW2_GENERATED_DIR": "/tmp/dw2/gen", "PORT": "8080"}


def sh(*cmd, check=True, capture=True) -> str:
    r = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return r.stdout.strip() if capture else ""


# ----------------------------------------------------------------------------- setup
def ensure_env() -> None:
    """bench.env from the repo .env: only the four keys the flows need (values never printed)."""
    if ENV_FILE.exists():
        return
    keep = {"HF_TOKEN", "EARTHENGINE_PROJECT", "CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"}
    lines = [ln for ln in (ROOT / ".env").read_text().splitlines() if ln.split("=", 1)[0].strip() in keep]
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text("\n".join(ln.replace('"', "") for ln in lines) + "\n")
    ENV_FILE.chmod(0o600)


def ensure_inputs() -> dict:
    """The PNG (no georeference -> relative depth on a flat plane) and the GeoTIFF (georeferenced Sentinel-2, 10 m
    -> FABDEM + live GLO-30), copied read-only from the library into build/slim/inputs/."""
    from PIL import Image
    INPUTS.mkdir(parents=True, exist_ok=True)
    png, tif = INPUTS / "flow_png.png", INPUTS / "flow_geotiff.tif"
    if not png.exists():
        Image.open(ROOT / "data/library_v2_2026-09-29/previews/sentinel2-kota.jpg").convert("RGB").save(png)
    if not tif.exists():
        tif.write_bytes((ROOT / "data/sentinel2_benchmark/agricultural/bathinda/bathinda_RGB.tif").resolve().read_bytes())
    return {"png": png, "geotiff": tif}


def run_container(image: str, name: str, platform: str | None, ee_home: str, extra: list[str] = ()) -> None:
    cmd = ["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{PORT}:8080", "--env-file", str(ENV_FILE)]
    if platform:
        cmd += ["--platform", platform]
    for k, v in CONTAINER_ENV.items():
        cmd += ["-e", f"{k}={v}"]
    if EE_CREDS.is_dir():
        cmd += ["-v", f"{EE_CREDS}:{ee_home}/.config/earthengine:ro"]
    sh(*cmd, *extra, image)


def rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def wait_health(timeout=120.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError("no /health within timeout")


def mem(name: str) -> dict:
    """PID 1 (the server) RSS now and peak, the container cgroup's current/peak, and /tmp usage."""
    s = sh("docker", "exec", name, "sh", "-c",
           "grep -E 'VmRSS|VmHWM' /proc/1/status; echo cg_current $(cat /sys/fs/cgroup/memory.current);"
           " echo cg_peak $(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0); echo tmp $(du -sk /tmp | cut -f1)")
    out = {}
    for ln in s.splitlines():
        k, v = ln.replace(":", " ").split()[:2]
        out[k] = int(v)
    return {"rss_mib": out["VmRSS"] / 1024, "peak_rss_mib": out["VmHWM"] / 1024,
            "cgroup_current_mib": out["cg_current"] / 2**20, "cgroup_peak_mib": out["cg_peak"] / 2**20,
            "tmp_mib": out["tmp"] / 1024}


# ----------------------------------------------------------------------------- size
def cmd_size(image: str, platform: str) -> dict:
    tag = f"{REGISTRY}/dw2:{image.split(':')[-1]}"
    sh("docker", "tag", image, tag)
    sh("docker", "push", "-q", tag)
    ref = tag.split("/", 1)[1].split(":")
    accept = ("application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json,"
              "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json")
    m = httpx.get(f"http://{REGISTRY}/v2/{ref[0]}/manifests/{ref[1]}", headers={"Accept": accept}).json()
    if "manifests" in m:  # an index: pick this platform's manifest
        arch = platform.split("/")[1]
        d = next(x for x in m["manifests"] if x.get("platform", {}).get("architecture") == arch)
        m = httpx.get(f"http://{REGISTRY}/v2/{ref[0]}/manifests/{d['digest']}", headers={"Accept": accept}).json()
    compressed = sum(layer["size"] for layer in m["layers"])
    # per-layer: the image config's history (non-empty entries, in order) matched to the manifest's layer blobs
    cfg = httpx.get(f"http://{REGISTRY}/v2/{ref[0]}/blobs/{m['config']['digest']}").json()
    steps = [h.get("created_by", "") for h in cfg.get("history", []) if not h.get("empty_layer")]
    hist = [(f"{layer['size'] / 1e6:.1f} MB compressed", c) for layer, c in zip(m["layers"], steps)]
    unpacked = int(sh("docker", "run", "--rm", "--platform", platform, "--entrypoint", "sh", image, "-c",
                      "du -sxb / 2>/dev/null | cut -f1"))
    pkgs = json.loads(sh("docker", "run", "--rm", "--platform", platform, "--entrypoint", "python", image, "-c",
                         "import importlib.metadata as m, json, os\n"
                         "out = {}\n"
                         "for d in m.distributions():\n"
                         "    n = 0\n"
                         "    for f in d.files or []:\n"
                         "        p = d.locate_file(f)\n"
                         "        try: n += os.path.getsize(p)\n"
                         "        except OSError: pass\n"
                         "    out[d.metadata['Name']] = out.get(d.metadata['Name'], 0) + n\n"
                         "print(json.dumps(out))"))
    top = sorted(pkgs.items(), key=lambda kv: -kv[1])[:25]
    res = {"image": image, "platform": platform, "compressed_bytes": compressed, "unpacked_bytes": unpacked,
           "layers": [{"size": s, "created_by": c[:160]} for s, c in hist],
           "top25": [{"package": k, "bytes": v} for k, v in top], "all_packages": pkgs}
    return res


# ----------------------------------------------------------------------------- start
def cmd_start(image: str, platform: str | None, runs: int, ee_home: str) -> dict:
    out = []
    for i in range(runs):
        name = f"dw2bench-{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        run_container(image, name, platform, ee_home)
        try:
            wait_health()
            ready = time.perf_counter() - t0
            time.sleep(3)
            out.append({"start_to_health_s": ready, **mem(name)})
        finally:
            rm(name)
    return {"runs": out, "median_start_to_health_s": statistics.median(r["start_to_health_s"] for r in out),
            "median_idle_rss_mib": statistics.median(r["rss_mib"] for r in out),
            "median_idle_cgroup_mib": statistics.median(r["cgroup_current_mib"] for r in out)}


# ----------------------------------------------------------------------------- flows
def _generated(client: httpx.Client, job: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in job["assets"].items():
        (dest / Path(url).name).write_bytes(client.get(url).content)
    (dest / "response.json").write_text(json.dumps({k: v for k, v in job.items() if k != "depth"}, indent=1))
    (dest / "depth.json").write_text(json.dumps(job["depth"]))


def flow(kind: str, client: httpx.Client, inputs: dict, dest: Path) -> dict:
    t = time.perf_counter()
    if kind in ("png", "geotiff"):
        p = inputs[kind]
        up = client.post("/api/input/upload", files={"file": (p.name, p.read_bytes(), "image/png" if kind == "png" else "image/tiff")})
        up.raise_for_status()
        iid = up.json()["id"]
    else:  # cdse / cdse_facts
        s = client.post("/api/cdse/search", json={"lat": 30.21, "lon": 74.95, "aoi_km": 10, "date_from": "2025-11-01",
                                                  "date_to": "2025-12-31", "max_cloud": 10})
        s.raise_for_status()
        sj = s.json()
        sc = min(sj["scenes"], key=lambda x: x["cloud"])  # the AOI's clearest scene; the UI sends the AOI bbox
        up = client.post("/api/input/scene", json={"id": sc["id"], "date": sc["date"], "cloud": sc["cloud"], "bbox": sj["bbox"]})
        up.raise_for_status()
        iid = up.json()["id"]
    if kind in ("geotiff", "cdse", "cdse_facts"):  # what the UI does automatically for a > 2.4 m GeoTIFF and every CDSE scene
        f = client.post(f"/api/input/{iid}/fabdem")
        f.raise_for_status()
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fabdem_meta.json").write_text(json.dumps(f.json().get("dem"), indent=1))
    g = client.post(f"/api/generate/input/{iid}")
    g.raise_for_status()
    _generated(client, g.json(), dest)
    out = {"seconds": time.perf_counter() - t, "input_id": iid}
    if kind == "cdse_facts":  # what the UI asks when the Facts box or a Scenario card opens
        t2 = time.perf_counter()
        f = client.get("/api/facts", params={"bbox": ",".join(str(v) for v in sj["bbox"])})
        f.raise_for_status()
        fj = f.json()
        (dest / "facts.json").write_text(json.dumps(fj, indent=1))
        out.update(facts_seconds=time.perf_counter() - t2, facts_status=fj["status"],
                   facts_lines=len(fj["facts"]) + sum(len(v) for v in fj["scenario"].values()))
    return out


def cmd_flows(image: str, platform: str | None, kinds: list[str], ee_home: str, label: str) -> dict:
    inputs = ensure_inputs()
    res = {}
    for kind in kinds:
        name = f"dw2bench-{kind}-{uuid.uuid4().hex[:6]}"
        run_container(image, name, platform, ee_home)
        try:
            wait_health()
            time.sleep(2)
            idle = mem(name)
            with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=300) as c:
                info = flow(kind, c, inputs, OUT / "runs" / label / kind)
            res[kind] = {**info, "idle": idle, "after": mem(name)}
            print(kind, json.dumps(res[kind]), flush=True)
        finally:
            (OUT / "runs" / label).mkdir(parents=True, exist_ok=True)
            (OUT / "runs" / label / f"{kind}.log").write_text(sh("docker", "logs", name, check=False) or "")
            rm(name)
    return res


def cmd_concurrent(image: str, platform: str | None, ee_home: str, label: str) -> dict:
    """png + geotiff + cdse_facts at the same time in one container (Cloud Run concurrency: one instance)."""
    from concurrent.futures import ThreadPoolExecutor
    inputs = ensure_inputs()
    name = f"dw2bench-conc-{uuid.uuid4().hex[:6]}"
    run_container(image, name, platform, ee_home)
    try:
        wait_health()
        time.sleep(2)
        idle = mem(name)
        t = time.perf_counter()

        def one(kind):
            with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=300) as c:
                return kind, flow(kind, c, inputs, OUT / "runs" / label / kind)

        with ThreadPoolExecutor(3) as ex:
            results = dict(ex.map(one, ["png", "geotiff", "cdse_facts"]))
        res = {"seconds": time.perf_counter() - t, "idle": idle, "after": mem(name), "flows": results}
        print(json.dumps(res), flush=True)
        return res
    finally:
        (OUT / "runs" / label).mkdir(parents=True, exist_ok=True)
        (OUT / "runs" / label / "concurrent.log").write_text(sh("docker", "logs", name, check=False) or "")
        rm(name)


# ----------------------------------------------------------------------------- trace
TRACE_ENTRY = """
import atexit, json, os, sys
sys.path.insert(0, os.getcwd())  # the image's WORKDIR, where backend/ lives
atexit.register(lambda: json.dump(sorted(sys.modules), open('/bench/modules.json', 'w')))
import uvicorn
uvicorn.run('backend.main:app', host='0.0.0.0', port=8080)
"""


def cmd_trace(image: str, platform: str | None, ee_home: str, label: str) -> dict:
    inputs = ensure_inputs()
    d = OUT / "trace" / label
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o777)
    (d / "entry.py").write_text(TRACE_ENTRY)
    name = f"dw2trace-{uuid.uuid4().hex[:6]}"
    cmd = ["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{PORT}:8080", "--env-file", str(ENV_FILE),
           "-v", f"{d}:/bench", "--entrypoint", "python"]
    if platform:
        cmd += ["--platform", platform]
    for k, v in CONTAINER_ENV.items():
        cmd += ["-e", f"{k}={v}"]
    if EE_CREDS.is_dir():
        cmd += ["-v", f"{EE_CREDS}:{ee_home}/.config/earthengine:ro"]
    sh(*cmd, image, "/bench/entry.py")
    try:
        wait_health()
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=300) as c:
            for kind in ("png", "geotiff", "cdse"):
                flow(kind, c, inputs, d / "outputs" / kind)
        sh("docker", "stop", "-t", "20", name)
    finally:
        rm(name)
    mods = json.loads((d / "modules.json").read_text())
    return {"modules": len(mods), "top_level": sorted({m.split(".")[0] for m in mods})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["size", "start", "flows", "concurrent", "trace"])
    ap.add_argument("image")
    ap.add_argument("--platform")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--flows", default="png,geotiff,cdse")
    ap.add_argument("--ee-home", default="/root", help="the container user's HOME (EE credentials mount)")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    ensure_env()
    if a.cmd == "size":
        res = cmd_size(a.image, a.platform or "linux/amd64")
    elif a.cmd == "start":
        res = cmd_start(a.image, a.platform, a.runs, a.ee_home)
    elif a.cmd == "flows":
        res = cmd_flows(a.image, a.platform, a.flows.split(","), a.ee_home, a.label)
    elif a.cmd == "concurrent":
        res = cmd_concurrent(a.image, a.platform, a.ee_home, a.label)
    else:
        res = cmd_trace(a.image, a.platform, a.ee_home, a.label)
    out = OUT / "results" / f"{a.label}_{a.cmd}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("layers", "all_packages", "runs", "top_level")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
