#!/usr/bin/env python3
"""
Method 6 — Full DAv2-Small fine-tune, twin (mean, log-variance) head.

Adapted from the design in zaidnansari2011/sih2026-depthwizard
(depthwizard/model.py, train.py, docs/evaluation-protocol.md — read directly,
nothing from that repo executed or imported; this is an independent
reimplementation of the core idea). That repo full-fine-tunes DA-V2-Small
into a Kendall & Gal-style twin head (mean + log-variance) predicting AGL in
metres directly, and reports (their own dataset/split, protocol 2, 80 val
tiles): zero-shot+global-affine baseline RMSE 9.308/MAE 4.528, zero-shot+
oracle-affine RMSE 7.285/MAE 3.147, their fine-tuned run02 RMSE 6.454/
MAE 1.840 — i.e. full fine-tuning beat even an oracle per-tile affine fit.

What is adapted here vs. simplified
------------------------------------
Adapted (the load-bearing part of the method):
  - Full backbone fine-tune (not frozen-feature scale modulation like our
    own Method 4).
  - Twin head: reuse the pretrained conv1/conv2/conv3(->mu) from the DA-V2
    depth head, add a fresh 1x1 conv_log_var alongside it (shared trunk,
    same as the original).
  - Output scaled to metres via a per-fold height_scale (this fold's
    training-quadrant AGL p95 — never the held-out quadrant's).
  - Two learning rates: backbone low, head 50x higher.
  - Gaussian NLL loss (heteroscedastic regression), masked to valid pixels.

Simplified away (auxiliary engineering from their repo, not the core
method-under-test, and unnecessary to answer "does full fine-tuning beat an
oracle affine fit on OUR split"):
  - No gradient-matching loss term, no beta-NLL reweighting, no MSE warmup
    phase, no binned/head-tail-cut height head, no thermal governor
    (irrelevant on Apple Silicon), no mid-run NaN/rail tripwires (we do keep
    a plain non-finite-loss abort). If this experiment looks promising, those
    are the natural next additions -- see gaps-and-fixes note in the summary.

Data / split (OURS, not theirs)
--------------------------------
Same 50-tile DFC2019 benchmark and same 4-fold spatial-quadrant holdout as
Method 4 (scripts/evaluate_method4.py): for each fold, one quadrant is held
out from every tile; the model trains on the union of the other 3 quadrants
across all 50 tiles, and is scored only on the held-out quadrant. Ground
truth is AGL (`*_AGL.tif`), the same quantity DFC2019's Method 4 baseline and
zaidnansari2011 both predict, so the quantity matches even though the split
protocol differs from theirs (fold-quadrant vs. their region/city holdout).
quadrant_bounds() is imported directly from evaluate_method4.py so the fold
geometry is bit-identical to Method 4's.

Each 512x512 quadrant is reflect-padded to 518x518 (37 * patch_size 14) so
the ViT patch embedding sees a size it can factor exactly, then the
prediction is cropped back to the original 512x512 before scoring -- no
resizing/interpolation of the input, and no loss of quadrant coverage.

Usage
-----
    python scripts/evaluate_method6_finetune_twinhead.py --outdir data/dfc2019/experiments/method6_finetune_twinhead --tag m6 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402 -- reuse quadrant_bounds only (bit-identical fold geometry)

SEED = 42
RGB_DIR = PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB"
TRUTH_DIR = PROJECT_ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
TILE_LIST_SOURCE = PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth"  # defines the 50-tile set

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0, 24.8M params
PATCH_SIZE = 14
PAD_TO = 518  # 37 * 14; smallest multiple of 14 >= 512

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Model: DA-V2-Small backbone + neck + twin (mean, log-var) head
# ============================================================================

class TwinHeadDav2(nn.Module):
    """DA-V2-Small backbone/neck + pretrained mean head + fresh log-variance head.

    Independent reimplementation of the twin-head idea in
    zaidnansari2011/sih2026-depthwizard/depthwizard/model.py (read, not
    imported or executed).
    """

    def __init__(self, model_id: str = MODEL_ID, height_scale: float = 30.0,
                 init_sigma_m: float = 5.0, log_var_max: float = 7.0, log_var_min: float = -8.0,
                 init_mu: str = "pretrained", init_mu_m: float = 0.0):
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

        self.conv1 = base.head.conv1
        self.conv2 = base.head.conv2
        self.activation1 = base.head.activation1
        self.conv_mu = base.head.conv3  # pretrained 32 -> 1, reused as the mean readout

        # "pretrained" reproduces DA-V2's own disparity-scale readout, which is fine when
        # height_scale is small (DFC2019 AGL, ~O(30m)). "constant" re-inits conv_mu to emit
        # a spatially-uniform init_mu_m everywhere -- needed when height_scale is large and
        # variable across tiles (e.g. absolute SRTM elevation across India, -112m to +4125m):
        # otherwise the pretrained readout's O(1)-normalized output times a large height_scale
        # produces a huge initial residual on every pixel, which is exactly the mechanism the
        # source repo (zaidnansari2011/model.py) documents killing its own run05 (a 666m
        # residual railing log_var at init). Same fix, applied for the same reason.
        if init_mu == "constant":
            nn.init.zeros_(self.conv_mu.weight)
            nn.init.constant_(self.conv_mu.bias, float(init_mu_m) / float(height_scale))
        elif init_mu != "pretrained":
            raise ValueError(f"init_mu must be 'pretrained' or 'constant', got {init_mu!r}")

        hidden = self.conv_mu.in_channels
        self.conv_log_var = nn.Conv2d(hidden, 1, kernel_size=1)
        # Start as a spatially-uniform sigma = init_sigma_m; only the data should
        # make it spatially varying (same reasoning as the source repo).
        nn.init.zeros_(self.conv_log_var.weight)
        nn.init.constant_(self.conv_log_var.bias,
                          2.0 * math.log(max(init_sigma_m, 1e-6) / self.height_scale))
        del base

    def forward(self, pixel_values: torch.Tensor):
        _, _, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"{H}x{W} not a multiple of patch size {self.patch_size}"
        ph, pw = H // self.patch_size, W // self.patch_size

        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False
        )
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]

        x = self.conv1(feat)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))

        mu = self.conv_mu(x) * self.height_scale
        log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        return mu, log_var

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        head = nn.ModuleList([self.neck, self.conv1, self.conv2, self.conv_mu, self.conv_log_var])
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": [p for p in head.parameters() if p.requires_grad],
             "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
        ]


def gaussian_nll(mu, log_var, target, mask):
    """Select-then-compute, not compute-then-select.

    Root cause of a real training-crash found while smoke-testing: at least one
    DFC2019 tile (JAX_004_016) has genuine NaN values in its raw AGL raster at
    invalid pixels (not just out-of-range sentinels). Computing the NLL formula
    elementwise over the FULL tensor and masking afterwards (`per_pixel[mask]`)
    still corrupts the gradient, because autograd's backward for the earlier
    elementwise ops (e.g. `mu - target`) computes a local derivative at every
    position -- including the NaN ones -- and the outer chain rule multiplies
    that by the *upstream* gradient, which is exactly 0 at masked-out positions
    post-selection: `0 * NaN = NaN`, not 0. The masked positions poison the
    whole batch's gradient. Selecting the valid pixels first (as masked_huber
    already did) means the NaN values never enter any tensor the loss touches.
    """
    mu_v, log_var_v, target_v = mu[mask], log_var[mask], target[mask]
    var_inv = torch.exp(-log_var_v)
    per_pixel = 0.5 * (var_inv * (mu_v - target_v) ** 2 + log_var_v)
    return per_pixel.mean()


def masked_huber(mu, target, mask):
    return F.smooth_l1_loss(mu[mask], target[mask], beta=1.0)


# ============================================================================
# Data
# ============================================================================

def tile_ids() -> list[str]:
    return sorted(p.stem.removesuffix("_depth") for p in TILE_LIST_SOURCE.glob("*_depth.npy"))


def load_tile_rgb_agl(tile_id: str):
    with rasterio.open(RGB_DIR / f"{tile_id}_RGB.tif") as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
    with rasterio.open(TRUTH_DIR / f"{tile_id}_AGL.tif") as src:
        agl = src.read(1).astype(np.float32)  # (H, W)
    valid = np.isfinite(agl) & (agl >= 0)
    return rgb, agl, valid


def pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Reflect-pad the last two dims up to `size` (pads bottom/right only)."""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = size - h, size - w
    assert ph >= 0 and pw >= 0
    return F.pad(x, (0, pw, 0, ph), mode="reflect")


class QuadrantDataset(Dataset):
    """One sample = one (tile, quadrant) pair, full quadrant, padded to PAD_TO."""

    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.samples = samples  # list of (rgb_crop[3,512,512] float32 0-255, agl_crop[512,512], valid[512,512])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, agl, valid = self.samples[idx]
        rgb_t = torch.from_numpy(rgb / 255.0)  # (3, 512, 512)
        rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        rgb_t = pad_to(rgb_t, PAD_TO)
        # nan_to_num as defense in depth: at least one DFC2019 tile (JAX_004_016) has
        # real NaN in its raw AGL raster at invalid pixels. gaussian_nll's mask-then-
        # compute ordering already prevents this from corrupting gradients (see its
        # docstring), but sanitizing here too means no future caller can reintroduce
        # the same bug by computing on agl_t before masking.
        agl_t = torch.nan_to_num(pad_to(torch.from_numpy(agl)[None], PAD_TO)[0], nan=0.0)
        valid_t = pad_to(torch.from_numpy(valid.astype(np.float32))[None], PAD_TO)[0] > 0.5
        return rgb_t, agl_t, valid_t


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(np.float64), p[m].astype(np.float64)
    return {
        "mae_m": float(np.mean(np.abs(y - p))),
        "rmse_m": float(np.sqrt(np.mean((y - p) ** 2))),
        "pearson": float(pearsonr(y, p).statistic) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 1 else float("nan"),
        "n_pixels": int(len(y)),
    }


def agg(per_tile: list[dict]) -> dict:
    return {
        "mae_m": float(np.mean([r["mae_m"] for r in per_tile])),
        "rmse_m": float(np.mean([r["rmse_m"] for r in per_tile])),
        "pearson": float(np.mean([r["pearson"] for r in per_tile])),
        "spearman": float(np.mean([r["spearman"] for r in per_tile])),
        "n_tiles": len(per_tile),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=None, help="default: 50x backbone")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--init-sigma", type=float, default=5.0)
    ap.add_argument("--log-var-max", type=float, default=7.0)
    ap.add_argument("--log-var-min", type=float, default=-8.0)
    ap.add_argument("--warmup-steps", type=int, default=30,
                    help="plain masked-Huber steps (log_var head not trained) before "
                         "switching to Gaussian NLL. Smoke-testing found the NLL loss "
                         "goes non-finite by step ~3 without this -- the same failure "
                         "mode zaidnansari2011's own docs attribute to their run05 and "
                         "built --warmup-mse to prevent. Confirmed reproducible on both "
                         "CPU and MPS, so this is a genuine optimization instability, "
                         "not a backend bug.")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tiles-limit", type=int, default=None, help="debug: limit number of tiles")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock budget per fold; 0 = none")
    args = ap.parse_args()
    args.lr_head = args.lr_head if args.lr_head is not None else args.lr_backbone * 50

    assert args.outdir.name not in ("method4", "method4_v2"), "refusing to overwrite an existing method's outputs"
    args.outdir.mkdir(parents=True, exist_ok=True)

    seed_everything()
    device = get_device()
    print(f"[{args.tag}] device={device}")

    tids = tile_ids()
    if args.tiles_limit:
        tids = tids[: args.tiles_limit]
    print(f"[{args.tag}] tiles: {len(tids)}")

    print(f"[{args.tag}] loading {len(tids)} tiles into memory ...")
    t_load = time.time()
    cache = {tid: load_tile_rgb_agl(tid) for tid in tids}
    print(f"[{args.tag}] loaded in {time.time()-t_load:.1f}s")

    fold_results = []
    t_start_all = time.time()

    for held_out_q in args.folds:
        t_fold = time.time()
        print(f"\n{'='*72}\n[{args.tag}] FOLD {held_out_q}\n{'='*72}")

        train_samples, train_agl_for_scale = [], []
        eval_samples = []  # (tile_id, rgb_crop, agl_crop, valid_crop)

        for tid in tids:
            rgb, agl, valid = cache[tid]
            h, w = agl.shape
            for q in range(4):
                r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
                rgb_c = rgb[:, r0:r1, c0:c1]
                agl_c = agl[r0:r1, c0:c1]
                valid_c = valid[r0:r1, c0:c1]
                if q == held_out_q:
                    eval_samples.append((tid, rgb_c, agl_c, valid_c))
                else:
                    train_samples.append((rgb_c, agl_c, valid_c))
                    if valid_c.any():
                        train_agl_for_scale.append(agl_c[valid_c])

        height_scale = float(np.percentile(np.concatenate(train_agl_for_scale), 95))
        height_scale = max(height_scale, 1.0)
        print(f"[{args.tag}] fold{held_out_q}: {len(train_samples)} train quadrants, "
              f"{len(eval_samples)} eval quadrants, height_scale(train p95)={height_scale:.2f}m")

        model = TwinHeadDav2(height_scale=height_scale, init_sigma_m=args.init_sigma,
                             log_var_max=args.log_var_max, log_var_min=args.log_var_min).to(device)
        opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head, args.weight_decay))

        train_ds = QuadrantDataset(train_samples)
        train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

        # LR warmup + linear decay (ported from train.py's lr_lambda_factory). Smoke-testing
        # found this is load-bearing, not auxiliary: AdamW normalizes each parameter's step
        # by its own running gradient RMS, so grad-norm clipping alone does NOT bound the
        # actual parameter movement per step -- mu oscillated between -19m and +24m step to
        # step under a fixed lr_head=2.5e-4 with clipping alone, still diverging to NaN by
        # step ~16. An LR ramp from 0 fixes it because it bounds the STEP SIZE directly,
        # which clipping the gradient direction does not.
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

        # ---------------------------------------------------------------- evaluate
        model.eval()
        per_tile = []
        all_y, all_p = [], []
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
                all_y.append(yv)
                all_p.append(pv)

        fold_agg = agg(per_tile)
        fold_time = time.time() - t_fold
        print(f"[{args.tag}] fold{held_out_q} DONE in {fold_time:.1f}s. metrics={fold_agg}")

        fold_results.append({
            "fold": held_out_q, "held_out_quadrant": held_out_q,
            "method6": fold_agg, "height_scale": height_scale,
            "fold_time_sec": fold_time, "tiles": per_tile,
        })

        partial = {"tag": args.tag, "config": vars(args) | {"outdir": str(args.outdir)},
                   "folds_completed": [f["fold"] for f in fold_results], "folds": fold_results}
        (args.outdir / f"{args.tag}_results.json").write_text(
            json.dumps(partial, indent=2, default=str))

        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()

    final = agg([f["method6"] | {"tile": "fold_mean"} for f in fold_results]) if fold_results else {}
    # agg() above expects per-tile metric dicts; fold_results' "method6" blocks are
    # already fold-level means, so this is mean-of-fold-means, matching Method 4's
    # own final_agg() convention exactly.
    output = {
        "tag": args.tag,
        "config": {"epochs": args.epochs, "batch": args.batch, "lr_backbone": args.lr_backbone,
                   "lr_head": args.lr_head, "weight_decay": args.weight_decay,
                   "init_sigma": args.init_sigma, "log_var_max": args.log_var_max},
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
