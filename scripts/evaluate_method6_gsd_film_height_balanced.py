#!/usr/bin/env python3
"""
Method 6 + devendrakushwah80's GSD-FiLM conditioning and height-balanced
loss/sampling -- an ablation on top of the existing Method 6 baseline
(scripts/evaluate_method6_finetune_twinhead.py), NOT a Sentinel-2
experiment (that door is closed per the three convergent CNN failures
documented in docs/method-audit/sentinel2/sign-flip-detector.md).

Both additions are read directly from
external/DepthWizard-SIH26175/src/{models/m3_net.py,training/{m3_losses.py,
m3_dataset.py,m3_trainer.py}} (devendrakushwah80/DepthWizard-SIH26175),
never executed or imported -- independent reimplementation adapted onto
this project's own Method 6 architecture, data, and 4-fold spatial-
quadrant holdout (quadrant_bounds imported from evaluate_method4.py, same
as the existing Method 6 script, so fold geometry stays bit-identical).

1. GSD-FiLM conditioning (their GSDFiLMBlock, src/models/m3_net.py):
   a small MLP(2 -> 64 -> 2*C) producing per-channel (gamma, beta),
   zero-initialized so it's identity at init, applied as
   `(1+gamma)*feat + beta` at two points in the network. Their model
   conditions on a real per-tile gsd_vec = [gsd_m, gsd_known] because
   their M3 pipeline trains across genuinely varying sensor resolutions.
   DFC2019 (this project's Method 6 benchmark) uses one fixed sensor
   product for all 50 tiles -- gsd_vec is therefore a CONSTANT [1.0, 1.0]
   for every sample here, not a per-sample conditioning signal. Ported
   faithfully anyway, at the closest architectural analogues to their two
   injection points (their "global transformer bottleneck" -> our neck
   output `feat` before conv1; their "decoder stage 2" -> our `x` after
   conv2/activation1, right before the mu/log_var heads), and the
   constant-input caveat is reported honestly rather than silently
   patched over with fabricated per-tile GSD variation.

2. Height-balanced loss/sampling (their CappedHeightWeightedLoss +
   height_aware_sampling, src/training/m3_losses.py +
   src/training/m3_dataset.py):
   - Loss: `weight = clip(1 + alpha*target, 1, max_weight)` (alpha=0.08,
     max_weight=4.0, their values) applied to a masked-Huber term, added
     to the existing primary loss with lambda_height_weight=0.35 (their
     default) -- an auxiliary term, same additive-composite-loss pattern
     as their M3CompositeLoss, not a replacement for the existing
     Huber-warmup/Gaussian-NLL primary loss.
   - Sampling: their version anchors 65% of training CROPS on tall
     (>=10m) or canopy (4-25m) pixels within a big tile, because their
     dataset yields sub-tile crops. Method 6 has no cropping step at all
     -- one training example IS a whole 512x512 quadrant (see the
     existing script's docstring) -- so "which pixel to crop around" has
     no analogue here. Adapted to the coarser granularity this
     architecture actually has: a WeightedRandomSampler over whole
     training quadrants, weight = 1 + 3*frac_tall + 2*frac_canopy (same
     10m/4-25m thresholds as their code), oversampling quadrants that
     contain more of the underrepresented tall/canopy regime rather than
     anchoring crops within them. This is a real adaptation of the
     technique to a different granularity, not the literal same
     algorithm -- flagged as such rather than claimed as a 1:1 port.

Everything else (backbone, twin head, height_scale-via-training-p95,
two-LR-group AdamW, LR warmup+decay, Huber-warmup-then-NLL schedule,
padding scheme, evaluation protocol, 4-fold spatial-quadrant holdout) is
reused UNCHANGED by importing directly from
evaluate_method6_finetune_twinhead.py.

Usage
-----
    python scripts/evaluate_method6_gsd_film_height_balanced.py \
        --outdir data/dfc2019/experiments/method6_gsd_film --tag m6_gsdfilm \
        --enable-gsd-film --epochs 12

    python scripts/evaluate_method6_gsd_film_height_balanced.py \
        --outdir data/dfc2019/experiments/method6_height_balanced --tag m6_heightbal \
        --enable-height-balanced --epochs 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    SEED, MODEL_ID, PATCH_SIZE, PAD_TO, IMAGENET_MEAN, IMAGENET_STD,
    seed_everything, get_device, gaussian_nll, masked_huber,
    tile_ids, load_tile_rgb_agl, pad_to, QuadrantDataset,
    compute_metrics, agg,
)

TALL_THRESH_M = 10.0        # devendrakushwah80's tall-structure threshold
CANOPY_LO_M, CANOPY_HI_M = 4.0, 25.0  # their canopy/vegetation band
HEIGHT_LOSS_ALPHA = 0.08    # their CappedHeightWeightedLoss alpha
HEIGHT_LOSS_MAX_WEIGHT = 4.0
LAMBDA_HEIGHT_WEIGHT = 0.35  # their M3CompositeLoss default


# ============================================================================
# 1. GSD-FiLM conditioning -- ported from src/models/m3_net.py::GSDFiLMBlock
# ============================================================================

class GSDFiLMBlock(nn.Module):
    """Direct port of devendrakushwah80's GSDFiLMBlock (src/models/m3_net.py)."""

    def __init__(self, in_features: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, 64), nn.GELU(), nn.Linear(64, channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, gsd_vec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        params = self.mlp(gsd_vec)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1, 1) + 1.0
        beta = beta.view(B, C, 1, 1)
        return gamma * feat + beta


class TwinHeadDav2GSD(nn.Module):
    """evaluate_method6_finetune_twinhead.TwinHeadDav2 + optional GSD-FiLM,
    at the two architectural analogues of their injection points (see the
    module docstring)."""

    def __init__(self, model_id: str = MODEL_ID, height_scale: float = 30.0,
                 init_sigma_m: float = 5.0, log_var_max: float = 7.0, log_var_min: float = -8.0,
                 enable_gsd_film: bool = False):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        base = AutoModelForDepthEstimation.from_pretrained(model_id)
        cfg = base.config

        self.backbone = base.backbone
        self.neck = base.neck
        self.patch_size = int(getattr(cfg, "patch_size", PATCH_SIZE))
        self.head_in_index = int(getattr(cfg, "head_in_index", -1))
        self.height_scale = float(height_scale)
        self.log_var_max = float(log_var_max)
        self.log_var_min = float(log_var_min)
        self.enable_gsd_film = enable_gsd_film

        self.conv1 = base.head.conv1
        self.conv2 = base.head.conv2
        self.activation1 = base.head.activation1
        self.conv_mu = base.head.conv3

        hidden = self.conv_mu.in_channels
        self.conv_log_var = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.conv_log_var.weight)
        nn.init.constant_(self.conv_log_var.bias,
                          2.0 * math.log(max(init_sigma_m, 1e-6) / self.height_scale))
        del base

        if self.enable_gsd_film:
            feat_channels = self.conv1.in_channels
            dec_channels = self.conv_mu.in_channels
            self.film_bottleneck = GSDFiLMBlock(in_features=2, channels=feat_channels)
            self.film_dec = GSDFiLMBlock(in_features=2, channels=dec_channels)
        else:
            self.film_bottleneck = None
            self.film_dec = None

    def forward(self, pixel_values: torch.Tensor):
        _, _, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        ph, pw = H // self.patch_size, W // self.patch_size

        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False
        )
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]

        if self.enable_gsd_film:
            # DFC2019 (this benchmark) is one fixed sensor product -- gsd_vec is a
            # constant [1.0, 1.0] (gsd_known=1.0 always known) for every sample,
            # not real per-tile variation. See module docstring.
            B = pixel_values.shape[0]
            gsd_vec = torch.tensor([1.0, 1.0], device=pixel_values.device).unsqueeze(0).expand(B, -1)
            feat = self.film_bottleneck(feat, gsd_vec)

        x = self.conv1(feat)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))

        if self.enable_gsd_film:
            x = self.film_dec(x, gsd_vec)

        mu = self.conv_mu(x) * self.height_scale
        log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        return mu, log_var

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        head_modules = [self.neck, self.conv1, self.conv2, self.conv_mu, self.conv_log_var]
        if self.enable_gsd_film:
            head_modules += [self.film_bottleneck, self.film_dec]
        head = nn.ModuleList(head_modules)
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": [p for p in head.parameters() if p.requires_grad],
             "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
        ]


# ============================================================================
# 2. Height-balanced loss -- ported from src/training/m3_losses.py::CappedHeightWeightedLoss
# ============================================================================

def capped_height_weighted_huber(mu, target, mask, alpha=HEIGHT_LOSS_ALPHA, max_weight=HEIGHT_LOSS_MAX_WEIGHT, beta=1.0):
    """Direct port of CappedHeightWeightedLoss (src/training/m3_losses.py),
    select-then-compute (see gaussian_nll's docstring in the base Method 6
    script for why this ordering matters on this project's own benchmark --
    JAX_004_016 has real NaN in its raw AGL raster)."""
    mu_v, target_v = mu[mask], target[mask]
    diff = torch.abs(mu_v - target_v)
    loss_base = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    weights = torch.clamp(1.0 + alpha * target_v, min=1.0, max=max_weight)
    return (loss_base * weights).mean()


# ============================================================================
# 2b. Height-balanced sampling -- adapted from height_aware_sampling in
#     src/training/m3_dataset.py (crop-anchoring -> whole-quadrant reweighting,
#     see module docstring for why the granularity had to change)
# ============================================================================

def quadrant_sample_weight(agl_c: np.ndarray, valid_c: np.ndarray) -> float:
    valid_agl = agl_c[valid_c]
    if valid_agl.size == 0:
        return 1.0
    frac_tall = float(np.mean(valid_agl >= TALL_THRESH_M))
    frac_canopy = float(np.mean((valid_agl >= CANOPY_LO_M) & (valid_agl <= CANOPY_HI_M)))
    return 1.0 + 3.0 * frac_tall + 2.0 * frac_canopy


# ============================================================================
# Main
# ============================================================================

BASE_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"  # the DAv2-Small revision every adopted run used


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--init-sigma", type=float, default=5.0)
    ap.add_argument("--log-var-max", type=float, default=7.0)
    ap.add_argument("--log-var-min", type=float, default=-8.0)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tiles-limit", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed (default = the original run's SEED, so omitting it reproduces that run)")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="save each fold's final model state_dict to <outdir>/fold<k>.pt (eval-only use)")
    ap.add_argument("--enable-gsd-film", action="store_true")
    ap.add_argument("--enable-height-balanced", action="store_true",
                    help="height-weighted loss term + weighted quadrant sampling")
    # Added 2026-09-23 (Method 6 + GAMUS-DC retrain). Off by default: omitting both reproduces
    # every earlier run exactly.
    ap.add_argument("--extra-train-npz", type=Path, default=None,
                    help="npz with rgb uint8 (N,3,512,512) + agl (N,512,512); appended to EVERY fold's "
                         "training set (never evaluated on). valid = finite & agl >= 0")
    ap.add_argument("--gamus-test-npz", type=Path, default=None,
                    help="npz with rgb (M,3,1024,1024), agl, cls; each fold model is also scored on it "
                         "(descriptive only)")
    # Added 2026-10-01 (owner decision: one production model). The ONLY change is the data split: one run
    # that trains on all 4 quadrants of all 50 tiles (no held-out quadrant, so no evaluation) and saves the
    # final weights under a new name. Loss, sampler, schedule, warm-up, epochs, height_scale rule, seed
    # handling and inputs are the same code path as every fold. Off by default.
    ap.add_argument("--train-on-all", action="store_true",
                    help="train on ALL quadrants of all tiles (no fold split); implies --save-checkpoints")
    ap.add_argument("--all-checkpoint-name", default=None,
                    help="file name for --train-on-all (default method6_full_dfc2019_hb_seed<seed>.pt)")
    return ap


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """The effective training config (the head LR defaults to 50x the backbone LR)."""
    args.lr_head = args.lr_head if args.lr_head is not None else args.lr_backbone * 50
    return args


def main():
    args = resolve_args(build_parser().parse_args())

    assert args.outdir.name not in ("method4", "method4_v2"), "refusing to overwrite an existing method's outputs"
    args.outdir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = get_device()
    print(f"[{args.tag}] device={device} gsd_film={args.enable_gsd_film} height_balanced={args.enable_height_balanced}")

    tids = tile_ids()
    if args.tiles_limit:
        tids = tids[: args.tiles_limit]
    print(f"[{args.tag}] tiles: {len(tids)}")

    t_load = time.time()
    cache = {tid: load_tile_rgb_agl(tid) for tid in tids}
    print(f"[{args.tag}] loaded in {time.time()-t_load:.1f}s")

    extra = []
    if args.extra_train_npz:
        z = np.load(args.extra_train_npz)
        for rgb_e, agl_e in zip(z["rgb"], z["agl"]):
            agl_e = agl_e.astype(np.float32)
            extra.append((rgb_e.astype(np.float32), agl_e, np.isfinite(agl_e) & (agl_e >= 0)))
        print(f"[{args.tag}] extra training quadrants: {len(extra)} from {args.extra_train_npz}")
    gtest = np.load(args.gamus_test_npz) if args.gamus_test_npz else None

    if args.train_on_all:
        assert not args.extra_train_npz and not args.gamus_test_npz, "--train-on-all is DFC2019-only"
        from transformers import AutoConfig
        rev = AutoConfig.from_pretrained(MODEL_ID)._commit_hash  # the snapshot this run actually loads
        assert rev == BASE_REVISION, f"DAv2-Small revision {rev} != adopted {BASE_REVISION}"
        print(f"[{args.tag}] train-on-all: base revision {rev}")
        args.folds = [None]  # one pass, nothing held out
        args.save_checkpoints = True
        args.all_checkpoint_name = args.all_checkpoint_name or f"method6_full_dfc2019_hb_seed{args.seed}.pt"
        assert not (args.outdir / args.all_checkpoint_name).exists(), "never overwrite an existing checkpoint"

    fold_results = []
    t_start_all = time.time()

    for held_out_q in args.folds:
        t_fold = time.time()
        print(f"\n{'='*72}\n[{args.tag}] FOLD {held_out_q}\n{'='*72}")

        train_samples, train_agl_for_scale, train_weights = [], [], []
        eval_samples = []

        for tid in tids:
            rgb, agl, valid = cache[tid]
            h, w = agl.shape
            for q in range(4):
                r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
                rgb_c = rgb[:, r0:r1, c0:c1]
                agl_c = agl[r0:r1, c0:c1]
                valid_c = valid[r0:r1, c0:c1]
                if held_out_q is not None and q == held_out_q:
                    eval_samples.append((tid, rgb_c, agl_c, valid_c))
                else:
                    train_samples.append((rgb_c, agl_c, valid_c))
                    if valid_c.any():
                        train_agl_for_scale.append(agl_c[valid_c])
                    if args.enable_height_balanced:
                        train_weights.append(quadrant_sample_weight(agl_c, valid_c))

        for rgb_e, agl_e, valid_e in extra:
            train_samples.append((rgb_e, agl_e, valid_e))
            if valid_e.any():
                train_agl_for_scale.append(agl_e[valid_e])
            if args.enable_height_balanced:
                train_weights.append(quadrant_sample_weight(agl_e, valid_e))

        height_scale = float(np.percentile(np.concatenate(train_agl_for_scale), 95))
        height_scale = max(height_scale, 1.0)
        print(f"[{args.tag}] fold{held_out_q}: {len(train_samples)} train quadrants, "
              f"{len(eval_samples)} eval quadrants, height_scale(train p95)={height_scale:.2f}m")

        model = TwinHeadDav2GSD(height_scale=height_scale, init_sigma_m=args.init_sigma,
                                log_var_max=args.log_var_max, log_var_min=args.log_var_min,
                                enable_gsd_film=args.enable_gsd_film).to(device)
        opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head, args.weight_decay))

        train_ds = QuadrantDataset(train_samples)
        if args.enable_height_balanced:
            sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds), replacement=True)
            train_ld = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
            print(f"[{args.tag}] fold{held_out_q}: height-balanced sampler, weight range "
                  f"[{min(train_weights):.2f}, {max(train_weights):.2f}], mean={np.mean(train_weights):.2f}")
        else:
            train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

        steps_per_epoch = max(1, len(train_ds) // args.batch)
        total_steps = steps_per_epoch * args.epochs
        warm = max(10, int(total_steps * 0.05))

        def lr_lambda(step):
            if step < warm:
                return step / warm
            return max(0.05, (total_steps - step) / max(1, total_steps - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        model.train()
        gstep = 0
        for epoch in range(1, args.epochs + 1):
            running = []
            for rgb_t, agl_t, valid_t in train_ld:
                rgb_t, agl_t, valid_t = rgb_t.to(device), agl_t.to(device), valid_t.to(device)
                mu, log_var = model(rgb_t)
                if gstep < args.warmup_steps:
                    loss = masked_huber(mu[:, 0], agl_t, valid_t)
                else:
                    loss = gaussian_nll(mu[:, 0], log_var[:, 0], agl_t, valid_t)
                if args.enable_height_balanced:
                    loss = loss + LAMBDA_HEIGHT_WEIGHT * capped_height_weighted_huber(mu[:, 0], agl_t, valid_t)
                if not torch.isfinite(loss):
                    print(f"[{args.tag}] fold{held_out_q}: non-finite loss at step {gstep}, aborting fold")
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                running.append(float(loss.detach().cpu()))
                gstep += 1
            if not running or not math.isfinite(running[-1]):
                break
            if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs:
                print(f"[{args.tag}] fold{held_out_q} epoch {epoch:02d}/{args.epochs}: "
                      f"loss={np.mean(running):.4f}  ({time.time()-t_fold:.0f}s elapsed)")
            if args.max_minutes and (time.time() - t_fold) > args.max_minutes * 60:
                print(f"[{args.tag}] fold{held_out_q}: wall-clock budget reached, stopping epochs")
                break

        fold_agg, fold_diag, per_tile = {}, {}, []
        if eval_samples:  # empty with --train-on-all
            model.eval()
            per_tile = []
            pooled_y, pooled_p = [], []
            with torch.no_grad():
                for tid, rgb_c, agl_c, valid_c in eval_samples:
                    rgb_t = torch.from_numpy(rgb_c / 255.0)
                    rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    rgb_t = pad_to(rgb_t, PAD_TO)[None].to(device)
                    mu, _ = model(rgb_t)
                    mu = mu[0, 0, : agl_c.shape[0], : agl_c.shape[1]].cpu().numpy()

                    yv = agl_c[valid_c]
                    pv = mu[valid_c]
                    m = compute_metrics(yv, pv)
                    m["tile"] = tid
                    per_tile.append(m)
                    ok = np.isfinite(yv) & np.isfinite(pv)
                    pooled_y.append(yv[ok].astype(np.float64)); pooled_p.append(pv[ok].astype(np.float64))

            fold_agg = agg(per_tile)
            # Evaluation-only diagnostics (added 2026-09-23 for C1; do not affect training):
            # pooled-within-fold variance ratio var(pred)/var(gt), OLS slope pred~gt, bias.
            Y, P = np.concatenate(pooled_y), np.concatenate(pooled_p)
            fold_diag = {"pooled_variance_ratio": float(np.var(P) / np.var(Y)),
                         "pooled_ols_slope": float(np.polyfit(Y, P, 1)[0]),
                         "pooled_bias_m": float(np.mean(P - Y)), "n_pixels": int(len(Y))}
            print(f"[{args.tag}] fold{held_out_q} diagnostics={fold_diag}")
            if gtest is not None:
                g_tiles, gy, gp, gc = [], [], [], []
                with torch.no_grad():
                    for rgb_g, agl_g, cls_g in zip(gtest["rgb"], gtest["agl"], gtest["cls"]):
                        pred = np.zeros(agl_g.shape, np.float32)
                        for r0 in (0, 512):
                            for c0 in (0, 512):
                                x = torch.from_numpy(rgb_g[:, r0:r0 + 512, c0:c0 + 512].astype(np.float32) / 255.0)
                                x = ((x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]).float()
                                mu, _ = model(pad_to(x, PAD_TO)[None].to(device))
                                pred[r0:r0 + 512, c0:c0 + 512] = mu[0, 0, :512, :512].float().cpu().numpy()
                        v = np.isfinite(agl_g) & (agl_g >= 0)
                        if v.sum() < 100:
                            continue
                        g_tiles.append(compute_metrics(agl_g[v], pred[v]))
                        gy.append(agl_g[v]); gp.append(pred[v]); gc.append(cls_g[v])
                gy, gp, gc = map(np.concatenate, (gy, gp, gc))
                tree = gc == 6  # GAMUS: 6 = tree, 3 = building
                gdiag = {"tiles": agg([m | {"tile": str(i)} for i, m in enumerate(g_tiles)])}
                for lo, hi in [(10, 20), (20, 30), (30, 50)]:
                    s = tree & (gy >= lo) & (gy < hi)
                    if s.sum() >= 100:
                        gdiag[f"tree_{lo}-{hi}"] = {"n": int(s.sum()), "truth_med": float(np.median(gy[s])),
                                                   "pred_med": float(np.median(gp[s]))}
                fold_diag["gamus_dc_test"] = gdiag
                print(f"[{args.tag}] fold{held_out_q} GAMUS-DC test (descriptive): {gdiag}")
        if args.save_checkpoints:
            name = args.all_checkpoint_name if held_out_q is None else f"fold{held_out_q}.pt"
            torch.save({"state_dict": model.state_dict(), "height_scale": height_scale, "seed": args.seed,
                        "fold": held_out_q}, args.outdir / name)
        fold_time = time.time() - t_fold
        print(f"[{args.tag}] fold{held_out_q} DONE in {fold_time:.1f}s. metrics={fold_agg}")

        fold_results.append({
            "fold": held_out_q, "held_out_quadrant": held_out_q,
            "method6_variant": fold_agg, "height_scale": height_scale,
            "fold_time_sec": fold_time, "tiles": per_tile, "diagnostics": fold_diag,
        })

        partial = {"tag": args.tag, "config": vars(args) | {"outdir": str(args.outdir)},
                   "folds_completed": [f["fold"] for f in fold_results], "folds": fold_results}
        (args.outdir / f"{args.tag}_results.json").write_text(json.dumps(partial, indent=2, default=str))

        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()

    scored = [f for f in fold_results if f["method6_variant"]]  # train-on-all has no held-out quadrant to score
    final = agg([f["method6_variant"] | {"tile": "fold_mean"} for f in scored]) if scored else {}
    output = {
        "tag": args.tag,
        "config": {"seed": args.seed, "epochs": args.epochs, "batch": args.batch, "lr_backbone": args.lr_backbone,
                   "lr_head": args.lr_head, "weight_decay": args.weight_decay,
                   "enable_gsd_film": args.enable_gsd_film, "enable_height_balanced": args.enable_height_balanced,
                   "extra_train_npz": str(args.extra_train_npz) if args.extra_train_npz else None,
                   "warmup_steps": args.warmup_steps, "train_on_all": args.train_on_all,
                   "checkpoint": args.all_checkpoint_name},
        "overall": final,
        "total_time_sec": time.time() - t_start_all,
        "folds": fold_results,
    }
    out_path = args.outdir / f"{args.tag}_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    print(f"\n{'='*78}\n[{args.tag}] FINAL RESULT\n{'='*78}")
    print("Overall (mean of fold means):", output["overall"])
    print(f"Total time: {output['total_time_sec']:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
