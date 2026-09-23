#!/usr/bin/env python3
"""Part B (2026-09-23): Method 6 zero-shot on the GAMUS HF test split, streamed tile by tile.
Pre-registration: docs/method-audit/07-gamus-generalization/log.md ("Part B pre-registration").

Per test tile (1024^2): four 512^2 quadrants.
  - m6_s{42,43,44}: mean of that seed's 4 DFC2019 fold models (metric AGL, no calibration =
    Method 6's DFC2019 protocol); each quadrant reflect-padded to 518, as in training/eval.
  - oracle: frozen DAv2 (the demo engine, backend/depth/depth_engine.py -- DAv2-Large, the same
    engine that produced the DFC2019 oracle's depth maps), OLS AGL = a*d + b fitted on the tile's
    other 3 quadrants, scored on the held-out one (scripts/evaluate_prior_spatial_cv.py definition).
  - m6_s*_ols: descriptive only, the same 3-quadrant OLS applied to each seed ensemble's output.
  - dav2_raw: frozen DAv2 normalised depth, correlations only.
Per (tile, quadrant, predictor): MAE, RMSE, Pearson, Spearman, variance ratio, bias; plus
per-class and per-height-bin sums (n, sum|e|, sum e^2, sum e, sum p, sum y) for pixel-pooled
breakdowns, and per-predictor pooled moments. Leakage confirmation: non-flat 32x32 RGB block hashes
vs. the 50 DFC2019 JAX tiles.
Output (append-only, resumable): data/gamus_eval/zeroshot_tiles.jsonl
  nohup .venv/bin/python scripts/gamus_zeroshot_eval.py > data/gamus_eval/zeroshot.log 2>&1 &
"""
from __future__ import annotations

import hashlib
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from gamus_io import get, list_files, tile_paths  # noqa: E402
from evaluate_method6_gsd_film_height_balanced import TwinHeadDav2GSD  # noqa: E402
from evaluate_method6_finetune_twinhead import (IMAGENET_MEAN, IMAGENET_STD, PAD_TO, get_device,  # noqa: E402
                                                 load_tile_rgb_agl, pad_to, tile_ids)

OUT = ROOT / "data/gamus_eval/zeroshot_tiles.jsonl"
EXP = ROOT / "data/dfc2019/experiments"
SEED_DIRS = {"m6_s42": EXP / "method6_height_balanced_seed42_ckpt",
             "m6_s43": EXP / "method6_height_balanced_seed43",
             "m6_s44": EXP / "method6_height_balanced_seed44"}
CLASSES = {0: "unlabelled", 1: "ground", 2: "low_veg", 3: "building", 4: "water", 5: "road", 6: "tree"}
BINS = [0, 2, 5, 10, 20, 30, 50, np.inf]
QUAD = [(0, 0), (0, 512), (512, 0), (512, 512)]


def block_hashes(rgb_hwc: np.ndarray) -> set:
    out = set()
    for r in range(0, rgb_hwc.shape[0] - 31, 32):
        for c in range(0, rgb_hwc.shape[1] - 31, 32):
            b = rgb_hwc[r:r + 32, c:c + 32]
            if b.std() > 2.0:  # skip flat blocks (nodata, water glare)
                out.add(hashlib.blake2b(np.ascontiguousarray(b).tobytes(), digest_size=12).digest())
    return out


def metrics(y, p):
    e = p - y
    vy = float(np.var(y))
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "pearson": float(pearsonr(y, p).statistic) if vy > 0 and np.var(p) > 0 else None,
            "spearman": float(spearmanr(y, p).statistic) if vy > 0 and np.var(p) > 0 else None,
            "var_ratio": float(np.var(p) / vy) if vy > 0 else None, "bias": float(np.mean(e)), "n": int(len(y))}


def sums(y, p, keys, labels):
    e = p - y
    out = {}
    for k, lab in zip(keys, labels):
        s = k
        if s.any():
            out[lab] = [int(s.sum()), float(np.abs(e[s]).sum()), float((e[s] ** 2).sum()), float(e[s].sum()),
                        float(p[s].sum()), float(y[s].sum())]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", default=list(SEED_DIRS))
    args = ap.parse_args()
    out_path = args.out
    dev = get_device()
    models = {}
    for tag, d in ((t, SEED_DIRS[t]) for t in args.seeds):
        ms = []
        for f in range(4):
            ck = torch.load(d / f"fold{f}.pt", map_location="cpu", weights_only=False)
            m = TwinHeadDav2GSD(height_scale=ck["height_scale"])
            m.load_state_dict(ck["state_dict"]); ms.append(m.to(dev).eval())
        models[tag] = ms
    from backend.depth.depth_engine import _DEVICE, _MODEL, _PROCESSOR, infer_with  # DAv2-Large, frozen

    print("hashing DFC2019 JAX blocks ...", flush=True)
    dfc_hash = set()
    for t in tile_ids():
        rgb, _, _ = load_tile_rgb_agl(t)
        dfc_hash |= block_hashes(np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0))
    print(f"{len(dfc_hash)} DFC2019 blocks", flush=True)

    imgs = [f for f in list_files() if f.startswith("images/test/")]
    done = set()
    if out_path.exists():
        done = {json.loads(l)["tile"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [f for f in imgs if f not in done][: args.limit]
    print(f"test tiles {len(imgs)}, done {len(done)}, todo {len(todo)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q: queue.Queue = queue.Queue(maxsize=24)

    def fetch(rel):
        h, c = tile_paths(rel)
        return rel, get(rel), get(h), get(c)

    def producer():
        with ThreadPoolExecutor(12) as ex:
            for item in ex.map(fetch, todo):
                q.put(item)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, n = time.time(), 0
    fout = out_path.open("a")
    while (item := q.get()) is not None:
        rel, rgb, agl, cls = item
        agl = agl.astype(np.float32)
        valid = np.isfinite(agl) & (agl >= 0)
        rec = {"tile": rel, "city": rel.split("/")[-1].split("_")[0], "valid_frac": float(valid.mean()),
               "neg_frac": float((np.isfinite(agl) & (agl < 0)).mean()),
               "shared_dfc_blocks": len(block_hashes(rgb) & dfc_hash)}
        if valid.sum() < 400:
            rec["skipped"] = "too few valid px"; fout.write(json.dumps(rec) + "\n"); fout.flush(); continue

        x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        batch = torch.stack([pad_to(x[:, r:r + 512, c:c + 512], PAD_TO) for r, c in QUAD]).float().to(dev)
        full = {}
        with torch.no_grad():
            for tag, ms in models.items():
                mu = torch.stack([m(batch)[0][:, 0, :512, :512] for m in ms]).mean(0).cpu().numpy()
                P = np.zeros((1024, 1024), np.float32)
                for i, (r, c) in enumerate(QUAD):
                    P[r:r + 512, c:c + 512] = mu[i]
                full[tag] = P
        d = infer_with(Image.fromarray(rgb, "RGB"), _PROCESSOR, _MODEL, _DEVICE)

        qmask = np.zeros((4, 1024, 1024), bool)
        for i, (r, c) in enumerate(QUAD):
            qmask[i, r:r + 512, c:c + 512] = True
        rec["quads"] = []
        pooled = {}
        for i in range(4):
            te = valid & qmask[i]
            tr = valid & ~qmask[i]
            if te.sum() < 100 or tr.sum() < 100:
                continue
            y = agl[te].astype(np.float64)
            preds = {}
            for tag, P in full.items():
                preds[tag] = P[te].astype(np.float64)
            A = np.vstack([d[tr], np.ones(tr.sum())]).T
            preds["oracle"] = np.polyval(np.linalg.lstsq(A, agl[tr], rcond=None)[0], d[te]).astype(np.float64)
            for tag, P in full.items():
                A2 = np.vstack([P[tr], np.ones(tr.sum())]).T
                preds[tag + "_ols"] = np.polyval(np.linalg.lstsq(A2, agl[tr], rcond=None)[0], P[te]).astype(np.float64)
            qrec = {"q": i, "m": {k: metrics(y, p) for k, p in preds.items()}}
            dv = d[te].astype(np.float64)
            qrec["m"]["dav2_raw"] = {"pearson": float(pearsonr(y, dv).statistic) if np.var(y) > 0 and np.var(dv) > 0 else None,
                                     "spearman": float(spearmanr(y, dv).statistic) if np.var(y) > 0 and np.var(dv) > 0 else None,
                                     "n": int(len(y))}
            c = cls[te]
            ckeys = [c == k for k in CLASSES]
            bkeys = [(y >= lo) & (y < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]
            blabs = [f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])]
            qrec["cls"] = {k: sums(y, p, ckeys, list(CLASSES.values())) for k, p in preds.items() if not k.endswith("_ols")}
            qrec["bin"] = {k: sums(y, p, bkeys, blabs) for k, p in preds.items() if not k.endswith("_ols")}
            # tree pixels only, by height bin (canopy-ceiling question)
            tk = [(c == 6) & b for b in bkeys]
            qrec["tree_bin"] = {k: sums(y, p, tk, blabs) for k, p in preds.items() if not k.endswith("_ols")}
            rec["quads"].append(qrec)
            for k, p in preds.items():
                s = pooled.setdefault(k, np.zeros(5))
                s += [len(y), p.sum(), (p ** 2).sum(), y.sum(), (y ** 2).sum()]
        rec["pooled_moments"] = {k: v.tolist() for k, v in pooled.items()}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"{n}/{len(todo)} tiles, {el/n:.2f} s/tile, ETA {(len(todo)-n)*el/n/3600:.2f} h", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
