#!/usr/bin/env python3
"""Maxar VHR display packs (library_v2, 2026-09-29): a cosmetic DISPLAY band for the 3D mesh only.

Reads each Maxar pack from data/library/dem/vhr-<crop>.tif (read-only) and writes a 3-band copy:
  band 1 TERRAIN  unchanged (FABDEM bare earth)          -> statistics, readouts, elevation layer
  band 2 SURFACE  unchanged (FABDEM + Method 6 AGL)      -> statistics, readouts, Measure
  band 3 DISPLAY  cosmetic mesh shape (backend/generation/pipeline.py uses it for terrain.json's
                  optional `display` field; frontend/src/terrain.js extrudes it when present)

  obj     = max(0, SURFACE - local_ground), local_ground = the Pth percentile (default 10) over a moving
            W x W m window; obj < T m -> 0 (noise). --ground agl (default): the percentile is taken on the
            above-ground part SURFACE - TERRAIN and added back to TERRAIN, so a hillside's own slope is not
            an "object". --ground surface: the percentile of SURFACE itself (the first recipe; on these
            Himalayan crops it marked 77-99 % of pixels as objects, because a 30-40 m window on a steep slope
            has its 10th percentile far below the centre pixel).
  DISPLAY = mean(TERRAIN) + f * (TERRAIN - mean(TERRAIN)) + k * obj
            f = ground flatness (0 flat, 1 real slope), k = object boost

Presets (--preset NAME writes <out>/dem_maxar/<NAME>/; the three below are the 2026-09-29 comparison):
  subtle  f=0.15 k=3  W=40 T=1.5
  medium  f=0.05 k=6  W=40 T=1.5
  strong  f=0.0  k=10 W=30 T=1.0
Explicit --f/--k/--W/--T override a preset's values (and need --preset for the folder name).

  --activate NAME  copies <out>/dem_maxar/NAME/*.tif over <out>/dem/vhr-*.tif (the packs the app reads)
                   and records the choice in <out>/dem_maxar/ACTIVE. Only files inside <out> change.

Usage:
  python scripts/build_maxar_display_packs.py --out data/library_v2_2026-09-29 --preset subtle
  python scripts/build_maxar_display_packs.py --out data/library_v2_2026-09-29 --all-presets
  python scripts/build_maxar_display_packs.py --out data/library_v2_2026-09-29 --activate medium
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import percentile_filter, zoom

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/library/dem"
PRESETS = {
    "subtle": dict(f=0.15, k=3.0, W=40.0, T=1.5),
    "medium": dict(f=0.05, k=6.0, W=40.0, T=1.5),
    "strong": dict(f=0.0, k=10.0, W=30.0, T=1.0),
}
GROUND = "agl"  # set by --ground
COARSE = 4  # the ground percentile runs on a 4x block-mean grid (~4.9 m cells), then bilinear back: seconds, not minutes


def local_ground(surface: np.ndarray, res_m: float, window_m: float, pct: float) -> np.ndarray:
    h, w = surface.shape
    c = surface[: h // COARSE * COARSE, : w // COARSE * COARSE].reshape(h // COARSE, COARSE, w // COARSE, COARSE).mean((1, 3))
    size = max(3, int(round(window_m / (res_m * COARSE))) | 1)
    g = percentile_filter(c, pct, size=size, mode="nearest")
    return zoom(g, (h / g.shape[0], w / g.shape[1]), order=1, mode="nearest")[:h, :w]


def display_band(terrain, surface, res_m, f, k, W, T, pct=10.0, ground="agl"):
    if ground == "agl":
        agl = surface - terrain
        obj = np.maximum(0.0, agl - local_ground(agl, res_m, W, pct))
    else:
        obj = np.maximum(0.0, surface - local_ground(surface, res_m, W, pct))
    obj[obj < T] = 0.0
    mu = float(np.nanmean(terrain))
    return (mu + f * (terrain - mu) + k * obj).astype(np.float32), obj


def build(out: Path, name: str, p: dict) -> dict:
    dst = out / "dem_maxar" / name
    dst.mkdir(parents=True, exist_ok=True)
    report = {"params": {**p, "ground": GROUND}, "tiles": {}}
    for src in sorted(SRC.glob("vhr-*.tif")):
        t0 = time.time()
        with rasterio.open(src) as r:
            terrain, surface = r.read(1).astype(np.float32), r.read(2).astype(np.float32)
            prof, tags = r.profile.copy(), r.tags()
            res = abs(r.transform.a)
        disp, obj = display_band(terrain, surface, res, **p, ground=GROUND)
        prof.update(count=3, dtype="float32", compress="deflate", predictor=3, zlevel=9)
        with rasterio.open(dst / src.name, "w", **prof) as w:
            w.write(terrain, 1)
            w.write(surface, 2)
            w.write(disp, 3)
            for i, d in enumerate(("TERRAIN", "SURFACE", "DISPLAY"), 1):
                w.set_band_description(i, d)
            w.update_tags(**tags, DISPLAY_SOURCE=(f"cosmetic display surface (preset {name}: f={p['f']} k={p['k']} "
                                                  f"W={p['W']} m T={p['T']} m ground={GROUND}); mesh shape only"),
                          DISPLAY_NOTE="3D shape is a cosmetic display surface; statistics show real elevations.")
        report["tiles"][src.stem] = {"obj_px_frac": round(float((obj > 0).mean()), 4), "obj_p99_m": round(float(np.percentile(obj, 99)), 2),
                                     "surface_range_m": round(float(np.nanmax(surface) - np.nanmin(surface)), 1),
                                     "display_range_m": round(float(disp.max() - disp.min()), 1), "s": round(time.time() - t0, 2)}
        print(name, src.stem, report["tiles"][src.stem], flush=True)
    (dst / "report.json").write_text(json.dumps(report, indent=1))
    return report


def activate(out: Path, name: str) -> None:
    src = out / "dem_maxar" / name
    files = sorted(src.glob("vhr-*.tif"))
    assert files, f"no packs in {src}: build the preset first"
    for f in files:
        shutil.copyfile(f, out / "dem" / f.name)
    (out / "dem_maxar" / "ACTIVE").write_text(name + "\n")
    print(f"active Maxar preset: {name} ({len(files)} packs -> {out / 'dem'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset")
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--activate")
    ap.add_argument("--ground", choices=["agl", "surface"], default="agl")
    for a in ("f", "k", "W", "T"):
        ap.add_argument(f"--{a}", type=float)
    a = ap.parse_args()
    global GROUND
    GROUND = a.ground
    out = a.out.resolve()
    assert out != (ROOT / "data/library").resolve(), "never write into data/library"
    if a.all_presets:
        for n, p in PRESETS.items():
            build(out, n, p)
    elif a.preset:
        p = dict(PRESETS.get(a.preset, {}))
        p.update({k: getattr(a, k) for k in ("f", "k", "W", "T") if getattr(a, k) is not None})
        assert set(p) == {"f", "k", "W", "T"}, f"custom preset {a.preset!r} needs --f --k --W --T"
        build(out, a.preset, p)
    if a.activate:
        activate(out, a.activate)


if __name__ == "__main__":
    main()
