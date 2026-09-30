#!/usr/bin/env python3
"""Train one canonical Method 6 checkpoint on the FULL DFC2019 50-tile
benchmark (all 4 quadrants of all 50 tiles, no held-out fold) and save it.

Why no holdout: the original evaluate_method6_finetune_twinhead.py run
(docs/method-audit/06-full-finetune-twin-head/) never saved checkpoints --
it only needed aggregate fold metrics. This script exists for a different
purpose: producing one "unchanged, DFC2019-trained" Method 6 model to run
inference with on real out-of-domain VHR imagery (a domain-transfer sanity
check, not a DFC2019 accuracy claim), so there is nothing to hold out --
using all the DFC2019 supervision available is the right choice here.

Architecture/training exactly as validated (TwinHeadDav2, gaussian_nll,
masked_huber warmup, LR warmup+decay) -- imported unchanged from
evaluate_method6_finetune_twinhead.py. Only the fold/eval loop is removed.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    TwinHeadDav2, gaussian_nll, masked_huber, pad_to, seed_everything, get_device,
    IMAGENET_MEAN, IMAGENET_STD, QuadrantDataset, tile_ids, load_tile_rgb_agl, PAD_TO,
)

OUT_DIR = PROJECT_ROOT / "data/dfc2019/experiments/method6_full_checkpoint"
EPOCHS = 12
BATCH = 2
WARMUP_STEPS = 40


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything()
    device = get_device()
    print(f"device={device}")

    tids = tile_ids()
    print(f"tiles: {len(tids)}")
    cache = {tid: load_tile_rgb_agl(tid) for tid in tids}

    samples, scale_vals = [], []
    for tid in tids:
        rgb, agl, valid = cache[tid]
        h, w = agl.shape
        for q in range(4):
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
            agl_c, valid_c = agl[r0:r1, c0:c1], valid[r0:r1, c0:c1]
            samples.append((rgb[:, r0:r1, c0:c1], agl_c, valid_c))
            if valid_c.any():
                scale_vals.append(agl_c[valid_c])

    height_scale = max(float(np.percentile(np.concatenate(scale_vals), 95)), 1.0)
    print(f"{len(samples)} quadrant samples (no holdout), height_scale(p95)={height_scale:.2f}m")

    model = TwinHeadDav2(height_scale=height_scale, init_sigma_m=5.0).to(device)
    opt = torch.optim.AdamW(model.param_groups(5e-6, 5e-6 * 50, 0.01))

    ds = QuadrantDataset(samples)
    ld = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    steps_per_epoch = max(1, len(ds) // BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warm = max(10, int(total_steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.05, (total_steps - s) / max(1, total_steps - warm)))

    model.train()
    gstep = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        running = []
        for rgb_t, agl_t, valid_t in ld:
            rgb_t, agl_t, valid_t = rgb_t.to(device), agl_t.to(device), valid_t.to(device)
            mu, log_var = model(rgb_t)
            if gstep < WARMUP_STEPS:
                loss = masked_huber(mu[:, 0], agl_t, valid_t)
            else:
                loss = gaussian_nll(mu[:, 0], log_var[:, 0], agl_t, valid_t)
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {gstep}, aborting")
                return
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(float(loss.detach().cpu()))
            gstep += 1
        print(f"epoch {epoch:02d}/{EPOCHS}: loss={np.mean(running):.4f} ({time.time()-t0:.0f}s)")

    # 2026-10-01: never overwrite an existing checkpoint (this legacy script trains the PRE-height-balanced recipe;
    # the adopted-recipe full model comes from evaluate_method6_gsd_film_height_balanced.py --train-on-all)
    assert not (OUT_DIR / "method6_full_dfc2019.pt").exists(), "refusing to overwrite method6_full_dfc2019.pt"
    ckpt = {"model": model.state_dict(), "height_scale": height_scale,
            "config": {"init_sigma_m": 5.0, "log_var_max": 7.0, "log_var_min": -8.0}}
    torch.save(ckpt, OUT_DIR / "method6_full_dfc2019.pt")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "n_tiles": len(tids), "n_samples": len(samples), "epochs": EPOCHS,
        "height_scale": height_scale, "total_time_sec": time.time() - t0,
    }, indent=2))
    print(f"saved {OUT_DIR/'method6_full_dfc2019.pt'}")


if __name__ == "__main__":
    main()
