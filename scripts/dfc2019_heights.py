#!/usr/bin/env python3
"""Method 6 held-out heights for the 50 DFC2019 tiles (08-dfc2019-terrain-packs, Prompt 4). INFERENCE ONLY.

For each tile, each seed-43 height-balanced fold model q runs on the WHOLE 1024 px tile with
context, through scripts/vhr_dsm_pipeline.py's seam-safe tiling (tiled_predict_margin, margin
192, reflect/mirror padding), and only its held-out quadrant q is kept (fold q never trained on
quadrant q; quadrant_bounds as in evaluate_method4.py). Quadrants are feathered over +-FEATHER px
across the two quadrant borders. There, a neighbouring fold's prediction contributes to pixels that
fold trained on: 2*FEATHER px bands, about 6 % of the tile. This is disclosed, not hidden.

  predict   -> data/dfc2019/terrain_packs/heights/<tile>.npz  (agl, 1024^2 float32, >= 0)
  seams     quadrant-border seam ratio (vhr_dsm_pipeline.seam_ratio, 2 px bands) per tile
  cap       height cap + margin from the held-out predictions vs AGL, by the rule pre-registered
            in docs/method-audit/08-dfc2019-terrain-packs/heights.md
  parity    one tile: this script vs a direct vhr_dsm_pipeline.tiled_predict_margin call
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data/dfc2019/terrain_packs/heights"
RGB_DIR = ROOT / "data/dfc2019/raw/RGB/Track1-RGB"
AGL_DIR = ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
MARGIN, FEATHER, HALF = 192, 16, 512


def read_rgb(tid):
    import rasterio
    with rasterio.open(RGB_DIR / f"{tid}_RGB.tif") as r:
        return r.read([1, 2, 3])


def read_agl(tid):
    import rasterio
    with rasterio.open(AGL_DIR / f"{tid}_AGL.tif") as r:
        return r.read(1).astype(np.float32)


def quadrant_weights(n=1024, half=HALF, feather=FEATHER):
    """w[q] (n x n): 1 inside quadrant q, linear ramp over +-feather across each border, sums to 1."""
    x = np.arange(n) + 0.5
    lo = np.clip((half + feather - x) / (2 * feather), 0, 1)   # weight of the 'first half' (rows/cols < half)
    hi = 1 - lo
    rows = {0: lo, 1: hi}
    cols = {0: lo, 1: hi}
    # quadrant q: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right (evaluate_method4.quadrant_bounds)
    return [np.outer(rows[q // 2], cols[q % 2]).astype(np.float32) for q in range(4)]


def predict_tile(models, rgb, device):
    import vhr_dsm_pipeline as vp
    preds = [vp.tiled_predict_margin(m, rgb, device, MARGIN)[0] for m in models]
    ws = quadrant_weights(rgb.shape[1])
    hard = np.zeros(rgb.shape[1:], np.float32)
    for q, p in enumerate(preds):
        r0, c0 = (q // 2) * HALF, (q % 2) * HALF
        hard[r0:r0 + HALF, c0:c0 + HALF] = p[r0:r0 + HALF, c0:c0 + HALF]
    soft = sum(w * p for w, p in zip(ws, preds))
    return soft.astype(np.float32), hard, preds


def quad_seam(a):
    import vhr_dsm_pipeline as vp
    return vp.seam_ratio(a, np.isfinite(a), [HALF])


def load_folds(device):
    import vhr_dsm_pipeline as vp
    folds, _full = vp.load_models(device)   # the full-data model is loaded by the helper but never used here
    return folds


def cmd_predict(a):
    from evaluate_method6_finetune_twinhead import get_device
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    models = load_folds(device)
    tiles = sorted(p.name[:-8] for p in RGB_DIR.glob("*_RGB.tif"))
    if a.tiles:
        tiles = [t for t in tiles if t in a.tiles]
    seams = {}
    sp = OUT / "seams.json"
    if sp.exists():
        seams = json.loads(sp.read_text())
    t0 = time.time()
    todo = [t for t in tiles if not (OUT / f"{t}.npz").exists()]
    for i, tid in enumerate(todo):
        soft, hard, _ = predict_tile(models, read_rgb(tid), device)
        agl = np.clip(soft, 0, None)
        np.savez(OUT / f"{tid}.npz", agl=agl)
        seams[tid] = {"seam_feathered": round(quad_seam(agl), 3), "seam_hard": round(quad_seam(np.clip(hard, 0, None)), 3)}
        sp.write_text(json.dumps(seams, indent=1))
        el = time.time() - t0
        print(f"{tid}: seam {seams[tid]['seam_feathered']:.2f} (hard {seams[tid]['seam_hard']:.2f})  "
              f"max {agl.max():.1f} m  [{i + 1}/{len(todo)}, {el / (i + 1):.0f}s/tile, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min]",
              flush=True)


def cmd_cap(a):
    """The pre-registered rule (heights.md). Pooled over all 50 tiles, valid AGL only (finite,
    no sentinel); small negative AGL counts as 0."""
    P, Y = [], []
    for p in sorted(OUT.glob("*_*_*.npz")):
        tid = p.stem
        y = read_agl(tid)
        m = np.isfinite(y) & (y > -100) & (y < 1000)
        P.append(np.load(p)["agl"][m][::a.stride])
        Y.append(np.clip(y[m], 0, None)[::a.stride])
    P, Y = np.concatenate(P), np.concatenate(Y)
    edges = np.arange(0, 62, 2.0)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Y >= lo) & (Y < hi)
        if m.sum() < a.min_px:
            continue
        rows.append({"bin": [lo, hi], "n": int(m.sum()), "agl_median": round(float(np.median(Y[m])), 2),
                     "pred_median": round(float(np.median(P[m])), 2),
                     "shortfall": round(float(np.median(Y[m]) - np.median(P[m])), 2)})
    M = a.margin
    cap = None
    for i, r in enumerate(rows):
        if r["bin"][0] < 4:
            continue
        if all(rr["shortfall"] > M for rr in rows[i:]):
            cap = r["bin"][0]
            break
    # plateau check (reported): slope of median prediction vs median AGL over the bins at/above the cap
    plateau = None
    if cap is not None:
        hi_rows = [r for r in rows if r["bin"][0] >= cap]
        if len(hi_rows) >= 2:
            x = np.array([r["agl_median"] for r in hi_rows]); y = np.array([r["pred_median"] for r in hi_rows])
            plateau = round(float(np.polyfit(x, y, 1)[0]), 3)
    res = {"rule": f"cap = lowest 2 m AGL-bin lower edge >= 4 m from which EVERY populated bin (>= {a.min_px} px) has "
                   f"median(AGL) - median(pred) > margin; margin = {M} m (committed)",
           "margin_m": M, "cap_m": cap, "slope_above_cap": plateau, "pixels": int(len(Y)), "stride": a.stride, "bins": rows}
    (OUT / "cap.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"  AGL {r['bin'][0]:4.0f}-{r['bin'][1]:<4.0f} n={r['n']:>8d}  median AGL {r['agl_median']:6.2f}  pred {r['pred_median']:6.2f}  short {r['shortfall']:+6.2f}")
    print(f"cap {cap} m, margin {M} m, slope above cap {plateau}")


def cmd_parity(a):
    """This script's per-fold prediction vs a direct call of the VHR pipeline's function, fresh
    model instances; plus the composite recomputed from the direct calls."""
    import torch
    import vhr_dsm_pipeline as vp
    from evaluate_method6_finetune_twinhead import get_device
    device = get_device()
    rgb = read_rgb(a.tile)
    soft, _, preds = predict_tile(load_folds(device), rgb, device)
    folds2, _ = vp.load_models(device)
    direct = [vp.tiled_predict_margin(m, rgb, device, MARGIN)[0] for m in folds2]
    d_fold = max(float(np.abs(x - y).max()) for x, y in zip(preds, direct))
    ws = quadrant_weights(rgb.shape[1])
    soft2 = sum(w * p for w, p in zip(ws, direct))
    saved = np.load(OUT / f"{a.tile}.npz")["agl"] if (OUT / f"{a.tile}.npz").exists() else None
    out = {"tile": a.tile, "device": str(device), "max_abs_diff_per_fold_m": d_fold,
           "max_abs_diff_composite_m": float(np.abs(soft - soft2).max()),
           "max_abs_diff_vs_saved_pack_input_m": float(np.abs(np.clip(soft2, 0, None) - saved).max()) if saved is not None else None,
           "torch": torch.__version__}
    (OUT / "parity.json").write_text(json.dumps(out, indent=1))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--tiles", nargs="*")
    p = sub.add_parser("cap"); p.add_argument("--margin", type=float, required=True); p.add_argument("--stride", type=int, default=7); p.add_argument("--min-px", type=int, default=2000)
    p = sub.add_parser("parity"); p.add_argument("--tile", default="JAX_004_006")
    a = ap.parse_args()
    {"predict": cmd_predict, "cap": cmd_cap, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
