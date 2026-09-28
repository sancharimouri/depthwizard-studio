#!/usr/bin/env python3
"""QA for the DFC2019 curated packs (08-dfc2019-terrain-packs, Prompt 5): every DFC2019 library item
generated through the running backend's web path (POST /api/generate/library/<id>, remote mode, pack
from the private HF dataset). Checks, per tile:
  - has_elevation, the curated-pack provenance, the local-frame CRS label, the note;
  - terrain.json is a 341 x 341 mesh with real relief (not the old flat zero plane);
  - elevation.png is not a single colour;
  - no accuracy/error wording or numbers anywhere in meta.
Usage: python scripts/dfc2019_packs_qa.py [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/dfc2019/terrain_packs/qa.json"
ACCURACY = re.compile(r"MAE|RMSE|error|accura|±|\bcorrelation\b|uncertain", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.api, timeout=300)
    ids = sorted(p.name[:-8] for p in (ROOT / "data/dfc2019/raw/Truth/Track1-Truth").glob("*_AGL.tif"))
    res, fails = {}, []
    for i, t in enumerate(ids):
        iid = f"dfc2019-{t}"
        t0 = time.time()
        r = c.post(f"/api/generate/library/{iid}")
        if r.status_code != 200:
            fails.append((t, f"HTTP {r.status_code} {r.text[:120]}"))
            print(t, "FAIL", fails[-1][1], flush=True)
            continue
        body = r.json()
        m = body["meta"]
        terr = c.get(body["assets"]["terrain"]).json()
        h = np.asarray(terr["heights"], np.float32)
        elev = np.asarray(Image.open(io.BytesIO(c.get(body["assets"]["elevation"]).content)).convert("RGB"))
        checks = {
            "has_elevation": m["has_elevation"] is True,
            "curated_pack": m.get("how") == "curated elevation pack",
            "local_frame": m.get("crs") == "local frame (no georeference)",
            "note": bool(m.get("note")),
            "mesh_341": (terr["width"], terr["height"]) == (341, 341),
            "relief": terr["elevationMax"] - terr["elevationMin"] > 1.0 and float(h.std()) > 0.01,
            "elevation_png_not_flat": float(elev.reshape(-1, 3).std(0).max()) > 5,
            "no_accuracy_wording": not ACCURACY.search(json.dumps({k: v for k, v in m.items() if k != "depth"})),
        }
        ok = all(checks.values())
        res[t] = {"ok": ok, "checks": checks, "terrain_source": m.get("terrain_source"),
                  "surface_range_m": m.get("surface_range_m"), "relief_m": round(terr["elevationMax"] - terr["elevationMin"], 1),
                  "job": m["job"], "sec": round(time.time() - t0, 1)}
        if not ok:
            fails.append((t, [k for k, v in checks.items() if not v]))
        print(f"[{i + 1}/{len(ids)}] {t}: {'ok' if ok else 'FAIL ' + str(fails[-1][1])}  relief {res[t]['relief_m']} m  "
              f"{m.get('terrain_source')[:40]}  ({res[t]['sec']}s)", flush=True)
    OUT.write_text(json.dumps({"api": a.api, "tiles": res, "fails": fails}, indent=1))
    print(f"{sum(v['ok'] for v in res.values())}/{len(ids)} ok; fails: {fails}")
    return 0 if not fails and len(res) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
