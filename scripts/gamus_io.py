"""Streaming fetch of single GAMUS HF tiles (earthflow/GAMUS), with retries. Own code (2026-09-23).
Reads HF_TOKEN from the environment or .env (for rate limits only; the dataset is public)."""
import json, os, tempfile, time, urllib.request
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_API = "https://huggingface.co/api/datasets/earthflow/GAMUS"
URL = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"


def _token():
    t = os.environ.get("HF_TOKEN")
    if not t and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"').strip("'")
    return t


_TOK = _token()


def list_files():
    return sorted(s["rfilename"] for s in json.loads(urllib.request.urlopen(REPO_API).read())["siblings"])


def get(rel: str, tries: int = 6) -> np.ndarray:
    for k in range(tries):
        fd, p = tempfile.mkstemp(suffix=".h5"); os.close(fd)
        try:
            req = urllib.request.Request(URL.format(rel), headers={"Authorization": f"Bearer {_TOK}"} if _TOK else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
            with h5py.File(p) as f:
                return f["image"][...]
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
        finally:
            os.remove(p)


def tile_paths(img_rel: str):
    """images/<split>/<CITY>_<id>_{RGB|IMG}.h5 -> (heights path, classes path)."""
    stem = img_rel.split("/", 1)[1].rsplit("_", 1)[0]
    return f"heights/{stem}_AGL.h5", f"classes/{stem}_CLS.h5"
