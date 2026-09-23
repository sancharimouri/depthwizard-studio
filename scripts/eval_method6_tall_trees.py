#!/usr/bin/env python3
"""
Held-out DFC2019 prediction-vs-true-height check for a set of Method 6 fold checkpoints
(2026-09-23). Each fold{q}.pt predicts only its own held-out quadrant q of the 50 benchmark
tiles (the same quadrant split it was trained without). Reports median / p90 prediction per
true-height bin for DFC2019 buildings (ASPRS 6) and high vegetation (5), and the
pre-registered retrain criterion (ii): median prediction on high-vegetation pixels with
true AGL in [20, 30) m.

    python scripts/eval_method6_tall_trees.py <ckpt_dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_method4 as m4  # noqa: E402
import evaluate_method6_finetune_twinhead as m6  # noqa: E402
import evaluate_method6_gsd_film_height_balanced as hb  # noqa: E402

BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (20, 30), (30, 50), (50, 200)]


def main(ckpt_dir: Path, out: Path):
    dev = m6.get_device()
    folds = []
    for q in range(4):
        ck = torch.load(ckpt_dir / f"fold{q}.pt", map_location="cpu", weights_only=False)
        m = hb.TwinHeadDav2GSD(height_scale=ck["height_scale"], init_sigma_m=5.0, log_var_max=7.0,
                               log_var_min=-8.0, enable_gsd_film=False)
        m.load_state_dict(ck["state_dict"])
        folds.append(m.to(dev).eval())
    P, T, C = [], [], []
    for tid in m6.tile_ids():
        rgb, agl, valid = m6.load_tile_rgb_agl(tid)
        with rasterio.open(m6.TRUTH_DIR / f"{tid}_CLS.tif") as src:
            cls = src.read(1)
        h, w = agl.shape
        for q in range(4):
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
            x = torch.from_numpy(rgb[:, r0:r1, c0:c1] / 255.0)
            x = ((x - m6.IMAGENET_MEAN[0]) / m6.IMAGENET_STD[0]).float()
            with torch.no_grad():
                mu, _ = folds[q](m6.pad_to(x, m6.PAD_TO)[None].to(dev))
            mu = mu[0, 0, : r1 - r0, : c1 - c0].float().cpu().numpy()
            v = valid[r0:r1, c0:c1]
            P.append(mu[v]); T.append(agl[r0:r1, c0:c1][v]); C.append(cls[r0:r1, c0:c1][v])
    P, T, C = map(np.concatenate, (P, T, C))
    res = {"ckpt_dir": str(ckpt_dir),
           "pred_all": {f"p{p}": float(np.percentile(P, p)) for p in (50, 95, 99, 99.9)} | {"max": float(P.max())},
           "truth_all": {f"p{p}": float(np.percentile(T, p)) for p in (50, 95, 99, 99.9)}, "bins": {}}
    for name, m in [("building", C == 6), ("highveg", C == 5)]:
        for lo, hi in BINS:
            s = m & (T >= lo) & (T < hi)
            if s.sum() < 100:
                continue
            res["bins"][f"{name}_{lo}-{hi}"] = dict(n=int(s.sum()), truth_med=float(np.median(T[s])),
                                                    pred_med=float(np.median(P[s])),
                                                    pred_p90=float(np.percentile(P[s], 90)))
    res["criterion_ii_highveg_20_30_pred_median"] = res["bins"]["highveg_20-30"]["pred_med"]
    out.write_text(json.dumps(res, indent=1))
    for k, b in res["bins"].items():
        print(f"{k:18s} n={b['n']:>9,d} truth_med {b['truth_med']:6.1f} pred_med {b['pred_med']:6.1f} pred_p90 {b['pred_p90']:6.1f}")
    print("criterion (ii) value:", res["criterion_ii_highveg_20_30_pred_median"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
