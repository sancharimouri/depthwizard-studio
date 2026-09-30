#!/usr/bin/env python3
"""Fair re-test (docs/method6-checkpoint-audit.md §7.8, pre-registered 2026-10-01): the retrained Method 6 full
model vs the four seed-42 fold models on the 07 Part B GAMUS test tiles, against GAMUS LiDAR nDSM.

The protocol is Part B's, reused by import from scripts/gamus_zeroshot_eval.py: the same tile list
(data/gamus_eval/zeroshot_tiles_merged.jsonl, same order), the same streaming fetch, four 512^2 quadrants each
ImageNet-normalised and reflect-padded to 518, the model's mu scored directly (no calibration), valid = finite &
AGL >= 0, tiles with < 400 valid px skipped, quadrants with < 100 valid px skipped, and Part B's metrics().

Predictors per quadrant: full, f0..f3 (seed-42 folds, individually), ens (mean of f0..f3's mu; must reproduce Part
B's m6_s42), plus Part B's saved oracle and m6_s42 metrics copied in for context and the parity check.
Output (append-only, resumable): build/gamus_fulltest/tiles.jsonl
  nohup .venv/bin/python -u scripts/gamus_fulltest_eval.py > build/gamus_fulltest/run.log 2>&1 &
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from gamus_io import get, tile_paths  # noqa: E402
from gamus_zeroshot_eval import QUAD, block_hashes, metrics  # noqa: E402  (Part B's own code)
from evaluate_method6_gsd_film_height_balanced import TwinHeadDav2GSD  # noqa: E402
from evaluate_method6_finetune_twinhead import (IMAGENET_MEAN, IMAGENET_STD, PAD_TO, get_device,  # noqa: E402
                                                 load_tile_rgb_agl, pad_to, tile_ids)

EXP = ROOT / "data/dfc2019/experiments"
FULL = EXP / "method6_full_checkpoint/method6_full_dfc2019_hb_seed42.pt"
FOLDS = [EXP / "method6_height_balanced_seed42_ckpt" / f"fold{q}.pt" for q in range(4)]
PARTB = ROOT / "data/gamus_eval/zeroshot_tiles_merged.jsonl"
OUT = ROOT / "build/gamus_fulltest/tiles.jsonl"


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TwinHeadDav2GSD(height_scale=ck["height_scale"])
    m.load_state_dict(ck.get("state_dict") or ck.get("model"), strict=True)
    return m.to(dev).eval()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dev = get_device()
    models = {"full": load(FULL, dev), **{f"f{q}": load(p, dev) for q, p in enumerate(FOLDS)}}

    partb = {}
    for line in PARTB.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            partb[r["tile"]] = r
    tiles = list(partb)  # Part B's tile list, in its order
    print(f"Part B tiles: {len(tiles)}", flush=True)

    print("hashing DFC2019 JAX blocks ...", flush=True)
    dfc_hash = set()
    for t in tile_ids():
        rgb, _, _ = load_tile_rgb_agl(t)
        dfc_hash |= block_hashes(np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["tile"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [t for t in tiles if t not in done][: args.limit]
    print(f"done {len(done)}, todo {len(todo)}", flush=True)

    q: queue.Queue = queue.Queue(maxsize=24)

    def fetch(rel):
        h, c = tile_paths(rel)
        return rel, get(rel), get(h)

    def producer():
        with ThreadPoolExecutor(12) as ex:
            for item in ex.map(fetch, todo):
                q.put(item)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, n = time.time(), 0
    fout = args.out.open("a")
    while (item := q.get()) is not None:
        rel, rgb, agl = item
        agl = agl.astype(np.float32)
        valid = np.isfinite(agl) & (agl >= 0)
        rec = {"tile": rel, "city": rel.split("/")[-1].split("_")[0], "valid_frac": float(valid.mean()),
               "shared_dfc_blocks": len(block_hashes(rgb) & dfc_hash)}
        if valid.sum() < 400:
            rec["skipped"] = "too few valid px"; fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
        x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        batch = torch.stack([pad_to(x[:, r:r + 512, c:c + 512], PAD_TO) for r, c in QUAD]).float().to(dev)
        full = {}
        with torch.no_grad():
            mus = {k: m(batch)[0][:, 0, :512, :512].cpu().numpy() for k, m in models.items()}
        mus["ens"] = np.mean([mus[f"f{k}"] for k in range(4)], axis=0)
        for k, mu in mus.items():
            P = np.zeros((1024, 1024), np.float32)
            for i, (r, c) in enumerate(QUAD):
                P[r:r + 512, c:c + 512] = mu[i]
            full[k] = P
        pb_quads = {qq["q"]: qq["m"] for qq in partb[rel].get("quads", [])}
        qmask = np.zeros((4, 1024, 1024), bool)
        for i, (r, c) in enumerate(QUAD):
            qmask[i, r:r + 512, c:c + 512] = True
        rec["quads"], pooled = [], {}
        for i in range(4):
            te, tr = valid & qmask[i], valid & ~qmask[i]
            if te.sum() < 100 or tr.sum() < 100:  # Part B's quadrant rule (tr is needed by its oracle)
                continue
            y = agl[te].astype(np.float64)
            m = {k: metrics(y, P[te].astype(np.float64)) for k, P in full.items()}
            for k in ("oracle", "m6_s42"):
                if k in pb_quads.get(i, {}):
                    m[f"partB_{k}"] = pb_quads[i][k]
            rec["quads"].append({"q": i, "m": m})
            for k, P in full.items():
                p = P[te].astype(np.float64)
                s = pooled.setdefault(k, np.zeros(5))
                s += [len(y), p.sum(), (p ** 2).sum(), y.sum(), (y ** 2).sum()]
        rec["pooled_moments"] = {k: v.tolist() for k, v in pooled.items()}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"{n}/{len(todo)} tiles, {el/n:.2f} s/tile, ETA {(len(todo)-n)*el/n/3600:.2f} h, queue {q.qsize()}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
