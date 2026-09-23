#!/usr/bin/env python3
"""
Method 4 v2 — configurable retraining for the Phase 1 / Phase 2 ablations.

Built on top of scripts/evaluate_method4.py (same ScaleModulationNet
architecture, same 4-fold spatial-quadrant holdout protocol, same
quadrant_bounds/predict_quadrant machinery, imported directly rather than
reimplemented) with these additions, each individually toggleable:

  --patch-mode {sparse,dense}
      sparse = original behaviour (N random 64x64 patches/quadrant/tile).
      dense  = near-full-coverage: a stepped grid of 64x64 patches tiling
               each training quadrant (512x512 -> up to 8x8=64 patches),
               filtered by the same MIN_VALID_FRACTION threshold.

  --epochs N
  --smoothness-weight W        (0 disables the scale-smoothness term)

  --extra-channel building     adds Method 3's existing DFC building-
                                probability .npy maps
                                (data/dfc2019/experiments/semantic/building/)
                                as a 5th input channel. No new model.

  --ground-plane-weight W --ground-plane-threshold T
      For training pixels with true AGL < T, adds
      W * mean(relu(pred - T)^2) to the loss -- an asymmetric penalty that
      only fires when the model predicts meaningfully above the ground
      plane where the truth says it shouldn't. Uses only existing AGL
      ground truth.

  --rank-weight W --rank-pairs-per-patch N
      Pairwise ranking loss: for N random pixel pairs sampled per patch
      (from valid pixels), penalizes disagreement between predicted order
      and true-AGL order via a margin hinge on the *difference*:
          loss = mean(relu(margin - sign(y_i - y_j) * (pred_i - pred_j)))
      i.e. a soft pairwise ranking / margin loss (no external deps).

  --sid-bins K (0 = disabled, default)
      Ordinal height-discretization constraint, adapted from weakIM2H
      (Chen, Shi, Zhu 2025, arXiv:2506.02534 eq. 7) into the *existing*
      rank_pair_loss machinery above -- not a separate classification head
      and not a full reimplementation of that paper (it also has a
      separate balanced soft-height sampling loss that is NOT ported here).
      Bin edges use the paper's exact SID formula:
          t_k = exp(log(h_min) + (log(h_max/h_min) / K) * k),  k = 0..K
      widening at greater heights rather than uniform bins. h_min/h_max
      are each tile's own AGL range (fit on that fold's training quadrants
      only, same train/test separation already used for rgb/depth/building
      stats). The paper does not specify how it handles h_min at or near 0
      (a real case here -- many ground pixels sit at AGL=0) -- --sid-eps
      (default 0.1 m) floors h_min for the log computation only, chosen to
      sit below this project's own established AGL noise floor (median
      true AGL 0.055 m across the 50-tile benchmark) so it doesn't distort
      any real bin boundary. When enabled, rank_pair_loss's target_sign is
      derived from each pair's *bin* difference instead of raw height
      difference -- pairs falling in the same bin are dropped (analogous
      to the existing exact-tie drop), pairs in different bins keep the
      same margin-ranking mechanics unchanged.

Every configuration reports, on the held-out quadrants across all 50 tiles:
  - MAE/RMSE/Pearson/Spearman (this run's own metrics vs. truth)
  - predicted-vs-true variance ratio and OLS calibration slope
    (regression-to-the-mean diagnostic from the original audit)

Outputs go to --outdir (must be a NEW folder under
data/dfc2019/experiments/ -- never data/dfc2019/experiments/method4/).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402 -- reuse quadrant_bounds, load_tile, Tile, fit_stats, compute_metrics

SEED = 42
PATCH = 64
BATCH_SIZE = 64
LR = 2e-4
WEIGHT_DECAY = 1e-4
MIN_VALID_FRACTION = 0.85

BUILDING_DIR = PROJECT_ROOT / "data/dfc2019/experiments/semantic/building"


def seed_everything(seed: int = SEED) -> None:
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
# Dense patch sampling (Phase 1(a))
# ============================================================================

def dense_patch_origins(valid: np.ndarray, bounds, patch: int):
    """Stepped grid tiling of a quadrant into patch x patch origins,
    filtered by MIN_VALID_FRACTION. Near-full coverage instead of random
    sparse sampling."""
    r0, r1, c0, c1 = bounds
    origins = []
    rr = r0
    while rr + patch <= r1:
        cc = c0
        while cc + patch <= c1:
            vm = valid[rr:rr + patch, cc:cc + patch]
            if vm.mean() >= MIN_VALID_FRACTION:
                origins.append((rr, cc))
            cc += patch
        rr += patch
    return origins


# ============================================================================
# Model (5th channel optional)
# ============================================================================

class ScaleModulationNetV2(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.scale_head = nn.Conv2d(64, 1, 1)
        self.residual_head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x, raw_depth):
        z = self.features(x)
        scale = torch.exp(math.log(4.0) * torch.tanh(self.scale_head(z)))
        residual = 50.0 * torch.tanh(self.residual_head(z))
        pred = raw_depth * scale + residual
        return pred, scale, residual


# ============================================================================
# Losses
# ============================================================================

def huber_masked(pred, target, mask):
    return F.smooth_l1_loss(pred[mask], target[mask], beta=1.0)


def smoothness_loss(scale):
    dx = torch.abs(scale[:, :, :, 1:] - scale[:, :, :, :-1]).mean()
    dy = torch.abs(scale[:, :, 1:, :] - scale[:, :, :-1, :]).mean()
    return dx + dy


def ground_plane_loss(pred, target, mask, threshold):
    """Asymmetric: penalize pred > threshold wherever true AGL < threshold."""
    ground_mask = mask & (target < threshold)
    if ground_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    over = F.relu(pred[ground_mask] - threshold)
    return (over ** 2).mean()


def sid_bin_index(height: torch.Tensor, h_min: float, h_max: float, k_bins: int, eps: float) -> torch.Tensor:
    """
    Spacing-Increasing Discretization bin index, per weakIM2H (Chen, Shi,
    Zhu 2025, arXiv:2506.02534 eq. 7): bin edges
        t_k = exp(log(h_min) + (log(h_max/h_min) / K) * k),  k = 0..K
    widen at greater heights. Inverting for a given height h (h_min <= h <=
    h_max by construction, since h_min/h_max are this tile's own min/max):
        bin = floor(K * log(h / h_min) / log(h_max / h_min))
    h_min is floored at `eps` for the log only (the paper does not specify
    this; see the CLI help for --sid-bins for the reasoning) -- height
    values themselves are clamped to the same floor so a height of exactly
    h_min (very common: many AGL pixels sit at 0) maps cleanly to bin 0
    instead of log(0).
    """
    h_min_c = max(float(h_min), eps)
    h_max_c = max(float(h_max), h_min_c + eps)
    height_c = torch.clamp(height, min=h_min_c)
    log_ratio = math.log(h_max_c / h_min_c)
    t = (torch.log(height_c / h_min_c) / log_ratio) * k_bins
    return torch.clamp(t.floor(), min=0, max=k_bins - 1)


def rank_pair_loss(pred, target, mask, pairs_per_sample, margin, hmin=None, hmax=None, sid_bins=0, sid_eps=0.1):
    """
    Pairwise ranking loss via torch.nn.functional.margin_ranking_loss.

    Per sample in the batch: randomly sample `pairs_per_sample` (i, j) pixel
    pairs from that sample's valid-pixel mask (NOT all O(N^2) pairs -- a
    fixed-size random sample, so this stays O(K) per sample). For each pair,
    target_sign = sign(true_agl[i] - true_agl[j]) by default, OR --
    when sid_bins > 0 -- sign(sid_bin(true_agl[i]) - sid_bin(true_agl[j]))
    using that sample's own tile height range (hmin/hmax, one scalar per
    batch element). Pairs that tie (equal height, or equal SID bin) are
    dropped before the loss, since MarginRankingLoss requires target in
    {-1, +1}. The gather/pair-construction is per-sample (a Python loop over
    the batch, since each sample has a different-sized valid-pixel set to
    index into), but the actual loss computation for all K pairs in a
    sample is one fully vectorized F.margin_ranking_loss call -- no
    Python-level loop over individual pairs.

    pred, target, mask: (B, 1, H, W). hmin, hmax: (B,) or None.
    """
    B = pred.shape[0]
    losses = []
    for b in range(B):
        m = mask[b, 0]
        idx = torch.nonzero(m, as_tuple=False)
        n = idx.shape[0]
        if n < 2:
            continue
        k = min(pairs_per_sample, n * (n - 1) // 2)
        if k <= 0:
            continue
        i1 = torch.randint(0, n, (k,), device=pred.device)
        i2 = torch.randint(0, n, (k,), device=pred.device)
        keep = i1 != i2
        i1, i2 = i1[keep], i2[keep]
        if i1.numel() == 0:
            continue
        r1, c1 = idx[i1, 0], idx[i1, 1]
        r2, c2 = idx[i2, 0], idx[i2, 1]
        y1 = target[b, 0, r1, c1]
        y2 = target[b, 0, r2, c2]
        p1 = pred[b, 0, r1, c1]
        p2 = pred[b, 0, r2, c2]

        if sid_bins > 0:
            bin1 = sid_bin_index(y1, hmin[b], hmax[b], sid_bins, sid_eps)
            bin2 = sid_bin_index(y2, hmin[b], hmax[b], sid_bins, sid_eps)
            target_sign = torch.sign(bin1 - bin2)
        else:
            target_sign = torch.sign(y1 - y2)
        tie_mask = target_sign != 0
        if tie_mask.sum() == 0:
            continue
        p1, p2, target_sign = p1[tie_mask], p2[tie_mask], target_sign[tie_mask]

        losses.append(F.margin_ranking_loss(p1, p2, target_sign, margin=margin))
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


# ============================================================================
# Data
# ============================================================================

class PatchDatasetV2(Dataset):
    def __init__(self, samples, rgb_stats, depth_stats, building_stats=None):
        self.samples = samples
        self.rgb_mean, self.rgb_std = rgb_stats
        self.depth_mean, self.depth_std = depth_stats
        self.building_stats = building_stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.building_stats is not None:
            rgb, depth, truth, building, hmin, hmax = item
        else:
            rgb, depth, truth, hmin, hmax = item

        rgb_n = rgb / 255.0
        rgb_n = (rgb_n - self.rgb_mean) / self.rgb_std
        depth_n = (depth - self.depth_mean) / self.depth_std

        channels = [rgb_n, depth_n[..., None]]
        if self.building_stats is not None:
            b_mean, b_std = self.building_stats
            building_n = (building - b_mean) / b_std
            channels.append(building_n[..., None])

        x = np.concatenate(channels, axis=-1)
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)).astype(np.float32))
        y = torch.from_numpy(truth[None].astype(np.float32))
        mask = torch.isfinite(y) & (y >= 0)
        y = torch.nan_to_num(y, nan=0.0)
        return x, y, mask, torch.tensor(hmin, dtype=torch.float32), torch.tensor(hmax, dtype=torch.float32)


def fit_building_stats(train_tiles_building):
    s, sq, n = 0.0, 0.0, 0
    for b, v in train_tiles_building:
        bv = b[v]
        s += float(bv.sum())
        sq += float((bv.astype(np.float64) ** 2).sum())
        n += len(bv)
    mean = s / n
    var = max(sq / n - mean ** 2, 1e-6)
    return float(mean), float(math.sqrt(var))


def predict_quadrant_v2(model, rgb, depth, building, bounds, rgb_stats, depth_stats, building_stats, device):
    r0, r1, c0, c1 = bounds
    rgb_mean, rgb_std = rgb_stats
    depth_mean, depth_std = depth_stats

    out = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)

    for rr in range(r0, r1, PATCH):
        for cc in range(c0, c1, PATCH):
            re = min(rr + PATCH, r1)
            ce = min(cc + PATCH, c1)
            ph, pw = re - rr, ce - cc

            rgb_p = rgb[rr:re, cc:ce]
            d_p = depth[rr:re, cc:ce]

            rgb_pad = np.zeros((PATCH, PATCH, 3), dtype=np.float32)
            d_pad = np.zeros((PATCH, PATCH), dtype=np.float32)
            rgb_pad[:ph, :pw] = rgb_p
            d_pad[:ph, :pw] = d_p

            rgb_n = rgb_pad / 255.0
            rgb_n = (rgb_n - rgb_mean) / rgb_std
            depth_n = (d_pad - depth_mean) / depth_std

            channels = [rgb_n, depth_n[..., None]]
            if building_stats is not None:
                b_pad = np.zeros((PATCH, PATCH), dtype=np.float32)
                b_p = building[rr:re, cc:ce]
                b_pad[:ph, :pw] = b_p
                b_mean, b_std = building_stats
                b_n = (b_pad - b_mean) / b_std
                channels.append(b_n[..., None])

            x = np.concatenate(channels, axis=-1)
            x = torch.from_numpy(np.transpose(x, (2, 0, 1))[None].astype(np.float32)).to(device)
            raw_depth = torch.from_numpy(d_pad[None, None].astype(np.float32)).to(device)

            with torch.no_grad():
                pred, _, _ = model(x, raw_depth)

            pred = pred[0, 0].detach().cpu().numpy()
            out[rr - r0:re - r0, cc - c0:ce - c0] = pred[:ph, :pw]

    return out


# ============================================================================
# Diagnostics
# ============================================================================

def compute_metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(np.float64)
    p = p[mask].astype(np.float64)
    return {
        "mae_m": float(np.mean(np.abs(y - p))),
        "rmse_m": float(np.sqrt(np.mean((y - p) ** 2))),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "n_pixels": int(len(y)),
    }


def compute_calibration(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(np.float64)
    p = p[mask].astype(np.float64)
    res = linregress(y, p)
    return {
        "true_std": float(y.std()),
        "pred_std": float(p.std()),
        "variance_ratio": float((p.std() ** 2) / (y.std() ** 2)),
        "ols_slope": float(res.slope),
        "ols_intercept": float(res.intercept),
        "true_mean": float(y.mean()),
        "pred_mean": float(p.mean()),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-dir", type=Path, default=PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth")
    parser.add_argument("--rgb-dir", type=Path, default=PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB")
    parser.add_argument("--truth-dir", type=Path, default=PROJECT_ROOT / "data/dfc2019/raw/Truth/Track1-Truth")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--patch-mode", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--sparse-patches-per-quadrant", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)

    parser.add_argument("--extra-channel", choices=["none", "building"], default="none")

    parser.add_argument("--ground-plane-weight", type=float, default=0.0)
    parser.add_argument("--ground-plane-threshold", type=float, default=0.5)

    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-pairs-per-patch", type=int, default=2000)
    parser.add_argument("--rank-margin", type=float, default=0.25)

    parser.add_argument("--sid-bins", type=int, default=0, help="0=disabled (raw-height rank target). See module docstring.")
    parser.add_argument("--sid-eps", type=float, default=0.1, help="h_min floor for the SID log computation, meters.")

    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--tiles-limit", type=int, default=None, help="debug: limit number of tiles")

    args = parser.parse_args()

    assert "method4/" != str(args.outdir).rstrip("/").split("data/dfc2019/experiments/")[-1] + "/", \
        "Refusing to write into the original method4/ output directory."
    assert args.outdir.name != "method4", "Refusing to write into the original method4/ output directory."

    seed_everything()
    device = get_device()
    print(f"[{args.tag}] Device: {device}")
    print(f"[{args.tag}] patch_mode={args.patch_mode} epochs={args.epochs} smoothness_weight={args.smoothness_weight} "
          f"extra_channel={args.extra_channel} ground_plane_weight={args.ground_plane_weight} "
          f"(threshold={args.ground_plane_threshold}) rank_weight={args.rank_weight} "
          f"sid_bins={args.sid_bins} sid_eps={args.sid_eps}")

    use_building = args.extra_channel == "building"
    in_channels = 5 if use_building else 4

    depth_files = sorted(args.depth_dir.glob("*_depth.npy"))
    tiles = []
    for dp in depth_files:
        tile_id = dp.stem.removesuffix("_depth")
        rp = args.rgb_dir / f"{tile_id}_RGB.tif"
        tp = args.truth_dir / f"{tile_id}_AGL.tif"
        building_path = BUILDING_DIR / f"{tile_id}.npy"
        if rp.exists() and tp.exists() and (not use_building or building_path.exists()):
            tiles.append(m4.Tile(tile_id, dp, rp, tp))
    if args.tiles_limit:
        tiles = tiles[: args.tiles_limit]

    print(f"[{args.tag}] Tiles: {len(tiles)}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    cached = []
    building_cache = []
    for tile in tiles:
        rgb, depth, truth, valid = m4.load_tile(tile)
        cached.append((rgb, depth, truth, valid))
        if use_building:
            b = np.load(BUILDING_DIR / f"{tile.tile_id}.npy").astype(np.float32)
            building_cache.append(b)
        else:
            building_cache.append(None)

    fold_results = []
    rng = np.random.default_rng(SEED)
    t_start_all = time.time()

    for held_out_q in args.folds:
        t_fold_start = time.time()
        print(f"\n{'=' * 72}\n[{args.tag}] FOLD {held_out_q} — held-out quadrant {held_out_q}\n{'=' * 72}")

        train_stats_subset = []
        for data in cached:
            rgb, depth, truth, valid = data
            h, w = truth.shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            train_valid = valid.copy()
            train_valid[r0:r1, c0:c1] = False
            train_stats_subset.append((rgb, depth, truth, train_valid))

        rgb_stats, depth_stats = m4.fit_stats(train_stats_subset)

        building_stats = None
        if use_building:
            b_subset = []
            for (rgb, depth, truth, valid), b in zip(cached, building_cache):
                h, w = truth.shape
                r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
                train_valid = valid.copy()
                train_valid[r0:r1, c0:c1] = False
                b_subset.append((b, train_valid))
            building_stats = fit_building_stats(b_subset)

        # Per-tile AGL min/max for the SID ordinal constraint, fit on this
        # fold's training quadrants only -- same train/test separation as
        # rgb_stats/depth_stats/building_stats above, so held-out-quadrant
        # height range never leaks into the bin edges used during training.
        tile_height_range = {}
        for tile, (rgb, depth, truth, train_valid) in zip(tiles, train_stats_subset):
            valid_truth = truth[train_valid]
            if valid_truth.size > 0:
                tile_height_range[tile.tile_id] = (float(valid_truth.min()), float(valid_truth.max()))
            else:
                tile_height_range[tile.tile_id] = (0.0, 1.0)

        print(f"[{args.tag}] rgb_stats={rgb_stats[0]} depth_stats={depth_stats} building_stats={building_stats}")

        # Build training samples.
        samples = []
        for tile, (rgb, depth, truth, valid), b in zip(tiles, cached, building_cache):
            h, w = truth.shape
            hmin, hmax = tile_height_range[tile.tile_id]
            for q in range(4):
                if q == held_out_q:
                    continue
                bounds = m4.quadrant_bounds(h, w, q)
                if args.patch_mode == "dense":
                    origins = dense_patch_origins(valid, bounds, PATCH)
                else:
                    origins = m4.sample_patch_origins(valid, bounds, args.sparse_patches_per_quadrant, PATCH, rng)
                for rr, cc in origins:
                    r = rgb[rr:rr + PATCH, cc:cc + PATCH].copy()
                    d = depth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    y = truth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    v = valid[rr:rr + PATCH, cc:cc + PATCH]
                    d[~v] = 0.0
                    y[~v] = np.nan
                    if use_building:
                        bb = b[rr:rr + PATCH, cc:cc + PATCH].copy()
                        samples.append((r, d, y, bb, hmin, hmax))
                    else:
                        samples.append((r, d, y, hmin, hmax))

        print(f"[{args.tag}] Training patches: {len(samples)}")

        dataset = PatchDatasetV2(samples, rgb_stats, depth_stats, building_stats)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                             pin_memory=(device.type == "cuda"))

        model = ScaleModulationNetV2(in_channels=in_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        model.train()
        for epoch in range(1, args.epochs + 1):
            running = []
            for batch in loader:
                x, y, mask, hmin, hmax = batch
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                hmin = hmin.to(device)
                hmax = hmax.to(device)
                raw_depth = x[:, 3:4] * depth_stats[1] + depth_stats[0]

                optimizer.zero_grad(set_to_none=True)
                pred, scale, _ = model(x, raw_depth)

                loss = huber_masked(pred, y, mask)
                if args.smoothness_weight > 0:
                    loss = loss + args.smoothness_weight * smoothness_loss(scale)
                if args.ground_plane_weight > 0:
                    loss = loss + args.ground_plane_weight * ground_plane_loss(pred, y, mask, args.ground_plane_threshold)
                if args.rank_weight > 0:
                    loss = loss + args.rank_weight * rank_pair_loss(
                        pred, y, mask, args.rank_pairs_per_patch, args.rank_margin,
                        hmin=hmin, hmax=hmax, sid_bins=args.sid_bins, sid_eps=args.sid_eps,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running.append(float(loss.detach().cpu()))

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"[{args.tag}] fold{held_out_q} epoch {epoch:03d}/{args.epochs}: loss={np.mean(running):.5f}")

        model_path = args.outdir / f"{args.tag}_fold{held_out_q}.pt"
        torch.save(model.state_dict(), model_path)

        # Evaluate on held-out quadrant, all 50 tiles.
        fold_tile_results = []
        all_true = []
        all_pred = []

        model.eval()
        for tile, data, b in zip(tiles, cached, building_cache):
            rgb, depth, truth, valid = data
            h, w = truth.shape
            bounds = m4.quadrant_bounds(h, w, held_out_q)
            r0, r1, c0, c1 = bounds

            pred_q = predict_quadrant_v2(model, rgb, depth, b, bounds, rgb_stats, depth_stats, building_stats, device)

            y_test = truth[r0:r1, c0:c1]
            v_test = valid[r0:r1, c0:c1]

            yv = y_test[v_test]
            pv = pred_q[v_test]

            fold_tile_results.append({"tile": tile.tile_id, "method4v2": compute_metrics(yv, pv)})
            all_true.append(yv)
            all_pred.append(pv)

        y_all = np.concatenate(all_true)
        p_all = np.concatenate(all_pred)
        calib = compute_calibration(y_all, p_all)

        def agg(records, key):
            vals = [r[key] for r in records]
            return {
                "mae_m": float(np.mean([x["mae_m"] for x in vals])),
                "rmse_m": float(np.mean([x["rmse_m"] for x in vals])),
                "pearson": float(np.mean([x["pearson"] for x in vals])),
                "spearman": float(np.mean([x["spearman"] for x in vals])),
                "n_tiles": len(vals),
            }

        fold_agg = agg(fold_tile_results, "method4v2")
        fold_time = time.time() - t_fold_start
        print(f"[{args.tag}] fold{held_out_q} DONE in {fold_time:.1f}s. metrics={fold_agg} calib={calib}")

        fold_results.append({
            "fold": held_out_q,
            "held_out_quadrant": held_out_q,
            "method4v2": fold_agg,
            "calibration": calib,
            "model_path": str(model_path),
            "n_training_patches": len(samples),
            "fold_time_sec": fold_time,
            "tiles": fold_tile_results,
        })

        # Incremental save after every fold, so partial progress is never lost.
        partial = {
            "tag": args.tag,
            "args": vars(args) | {"depth_dir": str(args.depth_dir), "rgb_dir": str(args.rgb_dir),
                                    "truth_dir": str(args.truth_dir), "outdir": str(args.outdir)},
            "folds_completed": [f["fold"] for f in fold_results],
            "folds": fold_results,
        }
        (args.outdir / f"{args.tag}_results.json").write_text(json.dumps(partial, indent=2, default=str))

    # Final aggregate across completed folds.
    def final_agg(key):
        blocks = [f[key] for f in fold_results]
        return {
            "mae_m": float(np.mean([b["mae_m"] for b in blocks])),
            "rmse_m": float(np.mean([b["rmse_m"] for b in blocks])),
            "pearson": float(np.mean([b["pearson"] for b in blocks])),
            "spearman": float(np.mean([b["spearman"] for b in blocks])),
            "n_folds": len(blocks),
        }

    def final_calib():
        vr = np.mean([f["calibration"]["variance_ratio"] for f in fold_results])
        sl = np.mean([f["calibration"]["ols_slope"] for f in fold_results])
        return {"variance_ratio_mean": float(vr), "ols_slope_mean": float(sl)}

    output = {
        "tag": args.tag,
        "config": {
            "patch_mode": args.patch_mode,
            "epochs": args.epochs,
            "smoothness_weight": args.smoothness_weight,
            "extra_channel": args.extra_channel,
            "ground_plane_weight": args.ground_plane_weight,
            "ground_plane_threshold": args.ground_plane_threshold,
            "rank_weight": args.rank_weight,
            "rank_pairs_per_patch": args.rank_pairs_per_patch,
            "rank_margin": args.rank_margin,
            "sid_bins": args.sid_bins,
            "sid_eps": args.sid_eps,
        },
        "overall": final_agg("method4v2"),
        "overall_calibration": final_calib(),
        "total_time_sec": time.time() - t_start_all,
        "folds": fold_results,
    }

    out_path = args.outdir / f"{args.tag}_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    print(f"\n{'=' * 78}\n[{args.tag}] FINAL RESULT\n{'=' * 78}")
    print("Overall:", output["overall"])
    print("Calibration:", output["overall_calibration"])
    print(f"Total time: {output['total_time_sec']:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
