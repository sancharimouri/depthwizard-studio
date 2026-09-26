"""Desktop-freeze TRIAL entry: the real backend (backend.main) + DAv2-Small run locally.

DAv2-Small on ONNX Runtime (bridge/dav2_server_onnx.py, same API as the torch bridge) is mounted at /local-dav2,
and DAV2_INFERENCE_URL points the backend's /api/depth routes at it. The model is
bundled as models/dav2_small.onnx.
  dw2-backend               serve on 127.0.0.1:8765
  dw2-backend --selftest IMG OUT.npy [GEOTIFF]  start, run one real image through
      /api/depth/relative (and optionally upload a GeoTIFF through /api/input/upload), exit
"""
import multiprocessing

# Must run before any heavy import: torch starts multiprocessing's resource tracker,
# which re-executes this frozen binary with "-c ..."; without this it respawns forever.
# (DW2_SKIP_FREEZE_SUPPORT=1 exists only so CI can check this call is still needed.)
if __name__ == "__main__" and __import__("os").environ.get("DW2_SKIP_FREEZE_SUPPORT") != "1":
    multiprocessing.freeze_support()

import os  # noqa: E402
import sys  # noqa: E402
import threading
import time
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
PORT = int(os.environ.get("DW2_PORT", "8765"))
# DAv2-Small runs on ONNX Runtime from the bundled models/dav2_small.onnx (no torch in
# the app; parity vs torch in docs/DESKTOP_APP.md). Nothing is written into the install
# folder: uploads and Hub downloads (library manifest, private previews) go to a
# writable per-user cache.


def _user_cache() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "DepthWizard"


USER_CACHE = Path(os.environ.get("DW2_USER_CACHE") or _user_cache())
os.environ.setdefault("DW2_UPLOADS_DIR", str(USER_CACHE / "uploads"))
os.environ.setdefault("DW2_GENERATED_DIR", str(USER_CACHE / "generated"))
os.environ["DW2_NO_DOTENV"] = "1"  # config comes only from the environment the shell passes
os.environ.setdefault("HF_HOME", str(USER_CACHE / "hf"))
os.environ["DW2_DAV2_ONNX"] = str(BASE / "models" / "dav2_small.onnx")
# The desktop shell passes DW2_LIBRARY_BUNDLE (the tiered tile library shipped with the app,
# desktop/tiles/); on-demand tiles download into the per-user cache.
os.environ.setdefault("DW2_LIBRARY", "bundle" if os.environ.get("DW2_LIBRARY_BUNDLE") else "local")
os.environ.setdefault("DW2_LIBRARY_USER", str(USER_CACHE / "library"))
os.environ["DAV2_INFERENCE_URL"] = f"http://127.0.0.1:{PORT}/local-dav2"

import uvicorn  # noqa: E402

import dav2_server_onnx as dav2_server  # noqa: E402  (bridge/dav2_server_onnx.py)
from backend.main import app  # noqa: E402

app.mount("/local-dav2", dav2_server.app)


def serve():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        import base64, zlib  # noqa: E401
        import httpx
        import numpy as np
        t0 = time.perf_counter()
        threading.Thread(target=serve, daemon=True).start()
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/local-dav2/health", timeout=2); break
            except httpx.HTTPError:
                time.sleep(0.2)
        ready = time.perf_counter() - t0
        img = Path(sys.argv[2]).read_bytes()
        t = time.perf_counter()
        r = httpx.post(f"http://127.0.0.1:{PORT}/api/depth/relative", files={"file": ("x.jpg", img, "image/jpeg")}, timeout=120)
        d = r.json()
        q = np.frombuffer(zlib.decompress(base64.b64decode(d["data_b64"])), "<u2").reshape(d["shape"])
        depth = d["min"] + q / 65535 * (d["max"] - d["min"])
        np.save(sys.argv[3] if len(sys.argv) > 3 else "selftest_depth.npy", depth)
        if len(sys.argv) > 4:  # a georeferenced GeoTIFF: exercises GDAL/PROJ data (proj.db, GDAL_DATA)
            g = httpx.post(f"http://127.0.0.1:{PORT}/api/input/upload",
                           files={"file": (Path(sys.argv[4]).name, Path(sys.argv[4]).read_bytes(), "image/tiff")}, timeout=120)
            gj = g.json()
            print(f"GEOTEST HTTP {g.status_code} crs={gj.get('crs')} gsd_m={gj.get('gsd_m')} "
                  f"tier={(gj.get('routing') or {}).get('tier')} detail={gj.get('detail')}")
        print(f"SELFTEST HTTP {r.status_code} host={d['host']} device={d['device']} shape={d['shape']} "
              f"infer={d['infer_s']}s request={time.perf_counter() - t:.2f}s startup={ready:.1f}s "
              f"range={depth.min():.3f}..{depth.max():.3f}")
    else:
        serve()
