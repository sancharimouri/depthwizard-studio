#!/usr/bin/env python3
"""
Part B2 (2026-09-23): Method 6 resolution sweep, INFERENCE ONLY, using the per-fold
checkpoints saved by the C3 seed run (default seed 43; the original run saved none).
Protocol pre-registered in docs/method-audit/05-rdah-net-fusion/summary.md §13:
for each fold's 50 held-out 512^2 quadrants, RGB is block-averaged by f in {1,2,4,8},
bilinearly upsampled back to 512^2, then fed through Method 6's normal preprocessing
(/255, ImageNet norm, pad_to(PAD_TO)). The prediction is block-averaged to the factor
grid and compared with the NaN-aware block-averaged AGL (>=50% valid).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_method4 as m4  # noqa: E402
import evaluate_method6_gsd_film_height_balanced as hb  # noqa: E402
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, PAD_TO, get_device, load_tile_rgb_agl, pad_to, tile_ids,
)
from rdah_resolution_sweep import block_mean  # noqa: E402

FACTORS = (1, 2, 4, 8)
GSD_1X_M = 0.3


def main(seed: int = 43):
    ck_dir = ROOT / f"data/dfc2019/experiments/method6_height_balanced_seed{seed}"
    out = ROOT / "data/dfc2019/experiments/method6_resolution_sweep"
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tids = tile_ids()
    cache = {t: load_tile_rgb_agl(t) for t in tids}
    recs = []
    for q in range(4):
        ck = torch.load(ck_dir / f"fold{q}.pt", map_location="cpu", weights_only=False)
        model = hb.TwinHeadDav2GSD(height_scale=ck["height_scale"], init_sigma_m=5.0, log_var_max=7.0,
                                   log_var_min=-8.0, enable_gsd_film=False).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for tid in tids:
            rgb, agl, valid = cache[tid]
            r0, r1, c0, c1 = m4.quadrant_bounds(*agl.shape, q)
            rgb_c, agl_c, val_c = rgb[:, r0:r1, c0:c1].astype(np.float64), agl[r0:r1, c0:c1], valid[r0:r1, c0:c1]
            n = rgb_c.shape[1]
            for f in FACTORS:
                small = block_mean(rgb_c.transpose(1, 2, 0), f)[0].transpose(2, 0, 1)
                x = torch.from_numpy(small).float()[None]
                if f > 1:
                    x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
                x = (x[0] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                with torch.no_grad():
                    mu, _ = model(pad_to(x, PAD_TO)[None].to(device))
                pred = mu[0, 0, :n, :n].cpu().numpy().astype(np.float64)
                p_f = block_mean(pred, f)[0]
                y_f, v_f = block_mean(np.where(val_c, agl_c, np.nan), f, val_c)
                p, y = p_f[v_f], y_f[v_f]
                ok = np.isfinite(p) & np.isfinite(y)
                p, y = p[ok], y[ok]
                if len(p) < 10 or np.std(p) == 0:
                    continue
                recs.append({"fold": q, "tile": tid, "factor": f, "gsd_m": GSD_1X_M * f,
                             "pearson": float(stats.pearsonr(p, y)[0]), "spearman": float(stats.spearmanr(p, y)[0]),
                             "suff": {"n": int(len(p)), "sp": float(p.sum()), "sy": float(y.sum()), "spp": float((p * p).sum()),
                                      "syy": float((y * y).sum()), "spy": float((p * y).sum())}})
        print(f"fold {q} done", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    (out / f"per_quadrant_seed{seed}.json").write_text(json.dumps(recs, indent=1) + "\n")
    print("records", len(recs))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 43)
