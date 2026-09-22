#!/usr/bin/env python3
"""
RDAH-Net quadrant-level fine-tuning: RDAH-FT-2, the three-fixes-stacked
re-run from docs/method-audit/05-rdah-net-fusion/verdict.md.

Three fixes stacked on top of RDAH-FT-1 (scripts/train_rdah_spatial_cv.py),
per the audit:

FIX 1 -- depth input scale, re-derived per fold from that fold's own
  TRAINING quadrants only (never the original 4 probe tiles, never that
  fold's test quadrant) -- closes the mild leakage risk the audit flagged
  in how the x255 constant was first found.
FIX 2 -- quadrant-level spatial holdout, reusing quadrant_bounds() from
  evaluate_method4.py bit-identically (the same function Methods 4 and 6
  both use), instead of RDAH's own whole-tile fold grouping. RDAH's decoder
  is confirmed fully resolution-agnostic (empirically verified: 1024, 512,
  and 256 square inputs all return same-size output); the one fixed-size
  component, PositionalEncoding's 64x64 sinusoidal buffer, is swapped for a
  freshly-computed 32x32 buffer matching the 512x512 quadrant's bottleneck,
  sanity-checked against the pretrained weights before training starts.
FIX 3 -- a rank-preserving loss term, ADDED to (not replacing) the existing
  masked SmoothL1 -- reusing rank_pair_loss() imported unchanged from
  evaluate_method4_v2.py (Method 4 v2's phase2_building_rank_v2, NOT Method
  6 -- see the note in this audit's summary.md about that misattribution),
  at its already-validated hyperparameters (rank_weight=0.5,
  rank_pairs_per_patch=2000, rank_margin=0.25), read directly from
  data/dfc2019/experiments/method4_v2_phase2.5_r2_building_rank_v2/
  phase2_building_rank_v2_results.json's own saved config.

Checkpoint: initialized from the Swiss checkpoint (external/RDAH-Net/
swiss_best_model.pth), NOT Track1's 104best_model.pth -- Track1 is
confirmed contaminated (41/50 of this benchmark's tiles are in its own
training split). Swiss chosen over HK: ~6.7x more training data (14.7GB vs
2.2GB) and Switzerland's terrain diversity (alpine/urban/suburban/rural) is
judged a better structural match to DFC2019's mid-density Jacksonville/
Omaha tiles than Hong Kong's extreme high-rise-dominated urban core -- a
reasoned judgment call, not verified against the paper's own dataset
description (not available from the Figshare deposit's metadata alone).

Evaluation: same nested checkpoint-selection discipline as the audit
(inner-validation split of each fold's held-out quadrant samples, epoch
selected on that split only, reported on the other half) plus the
variance-ratio/OLS-slope diagnostic already used for Methods 4 and 5.

Usage:
    python scripts/train_rdah_quadrant_cv.py --fold 0 --epochs 5   # staged: fold 0 first
    python scripts/train_rdah_quadrant_cv.py --epochs 5            # all 4 folds
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from rasterio.warp import Resampling, reproject
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"
SWISS_CHECKPOINT = RDAH_DIR / "swiss_best_model.pth"

RGB_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "RGB" / "Track1-RGB"
GT_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "Truth" / "Track1-Truth"
DEPTH_DIR = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "dav2_baseline" / "depth"
MANIFEST = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "dav2_baseline" / "manifest.csv"
OUT_DIR = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "rdah_quadrant_cv"

PROBE_TILES = {"JAX_004_006", "JAX_149_006", "JAX_264_013", "OMA_248_029"}
SEED = 42

RANK_WEIGHT = 0.5
RANK_PAIRS_PER_PATCH = 2000
RANK_MARGIN = 0.25

sys.path.insert(0, str(Path(__file__).resolve().parent))


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Must register in sys.modules BEFORE exec_module: evaluate_method4.py
    # uses @dataclass, whose internal type resolution does
    # sys.modules.get(cls.__module__) and crashes with AttributeError on
    # None if the module isn't registered yet.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m4 = import_module("m4_for_rdah_quadrant", PROJECT_ROOT / "scripts" / "evaluate_method4.py")
m4v2 = import_module("m4v2_for_rdah_quadrant", PROJECT_ROOT / "scripts" / "evaluate_method4_v2.py")
rdah_cv = import_module("rdah_cv_for_quadrant", PROJECT_ROOT / "scripts" / "train_rdah_spatial_cv.py")

quadrant_bounds = m4.quadrant_bounds
rank_pair_loss = m4v2.rank_pair_loss
masked_smooth_l1 = rdah_cv.masked_smooth_l1
metrics_from_arrays = rdah_cv.metrics_from_arrays
save_json = rdah_cv.save_json


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def import_rdah_model_class():
    if str(RDAH_DIR) not in sys.path:
        sys.path.insert(0, str(RDAH_DIR))
    test_py = RDAH_DIR / "test.py"
    spec = importlib.util.spec_from_file_location("rdah_test_for_quadrant", test_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeightPredTransformer, module.PositionalEncoding


def load_full_sample(tile: str):
    """Load one DFC2019 tile's full 1024x1024 RGB/depth[0,1]/target/mask tensors.
    Depth is returned UNSCALED ([0,1]) -- the fold's own scale constant is
    applied by the caller, at whatever quadrant crop is needed."""
    with rasterio.open(RGB_DIR / f"{tile}_RGB.tif") as src:
        rgb_raw = src.read([1, 2, 3]).transpose(1, 2, 0)
    if rgb_raw.dtype != np.uint8:
        rgb_arr = np.clip(rgb_raw, 0, 255).astype(np.uint8)
    else:
        rgb_arr = rgb_raw
    image = Image.fromarray(rgb_arr, mode="RGB")
    rgb = transforms.ToTensor()(image)
    rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb)

    depth = np.load(DEPTH_DIR / f"{tile}_depth.npy").astype(np.float32)
    if depth.shape != (1024, 1024):
        raise ValueError(f"{tile}: expected 1024x1024 depth, got {depth.shape}")
    depth = torch.from_numpy(depth).unsqueeze(0)

    with rasterio.open(GT_DIR / f"{tile}_AGL.tif") as src:
        agl = src.read(1).astype(np.float32)
    valid = np.isfinite(agl) & (agl != -9999.0)
    target = torch.from_numpy(np.where(valid, agl, 0.0).astype(np.float32)).unsqueeze(0)
    non_black = ~np.all(rgb_raw == 0, axis=2)
    mask = torch.from_numpy((valid & non_black).astype(np.float32)).unsqueeze(0)

    return rgb, depth, target, mask


def crop(t: torch.Tensor, bounds) -> torch.Tensor:
    r0, r1, c0, c1 = bounds
    return t[..., r0:r1, c0:c1]


def load_manifest_tiles() -> list[str]:
    manifest = pd.read_csv(MANIFEST)
    return sorted(manifest["tile_id"].tolist())


def build_resized_positional_encoding(model, PositionalEncoding, d_model: int, bottleneck: int):
    """Replace the model's fixed 64x64 pos-encoding buffer with a freshly
    computed one matching the quadrant input's actual bottleneck size.
    This is a deterministic sinusoidal function, not a learned weight, so
    nothing from the pretrained checkpoint is lost by recomputing it."""
    model.pos_encoding = PositionalEncoding(d_model=d_model, H=bottleneck, W=bottleneck)
    return model


@torch.no_grad()
def sanity_check_resize(model_cls, PositionalEncoding, device, tile_for_check: str):
    """Compare model output before vs. after the positional-encoding resize
    on the same real 512x512 quadrant input, to confirm the swap doesn't
    silently break the pretrained weights it's initialized near."""
    rgb, depth_raw, _, _ = load_full_sample(tile_for_check)
    bounds = quadrant_bounds(1024, 1024, 0)
    rgb_q = crop(rgb, bounds).unsqueeze(0).to(device)
    depth_q = (crop(depth_raw, bounds) * 255.0).unsqueeze(0).to(device)

    model = model_cls().to(device)
    ckpt = torch.load(SWISS_CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    out_before = model(depth_q, rgb_q).detach().cpu()

    model = build_resized_positional_encoding(model, PositionalEncoding, d_model=32, bottleneck=32)
    model.eval()
    out_after = model(depth_q, rgb_q).detach().cpu()

    print("\n=== SANITY CHECK: positional-encoding resize (64x64 -> 32x32) ===")
    print(f"  strict-load missing={len(missing)} unexpected={len(unexpected)}")
    print(f"  output shape before={tuple(out_before.shape)} after={tuple(out_after.shape)}")
    print(f"  before: mean={out_before.mean():.6f} std={out_before.std():.6f} "
          f"nan={torch.isnan(out_before).sum().item()}")
    print(f"  after:  mean={out_after.mean():.6f} std={out_after.std():.6f} "
          f"nan={torch.isnan(out_after).sum().item()}")
    ok = (
        torch.isfinite(out_after).all()
        and out_after.shape == out_before.shape
        and out_after.std() > 0
    )
    print(f"  sanity check {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError("Positional-encoding resize sanity check failed -- aborting before training.")
    return ok


@torch.no_grad()
def derive_fold_scale(model, tiles: list[str], train_quadrants: list[int], device,
                       exclude_tiles: set[str], candidates: list[float],
                       n_sample_tiles: int = 12) -> float:
    """FIX 1: re-derive the depth scale constant using ONLY this fold's own
    training quadrants (excluding the 4 original probe tiles), never the
    fold's held-out quadrant and never data used to originally discover the
    x255 mechanism."""
    pool_tiles = [t for t in tiles if t not in exclude_tiles]
    rng = np.random.RandomState(SEED)
    sample_tiles = list(rng.choice(pool_tiles, size=min(n_sample_tiles, len(pool_tiles)), replace=False))

    model.eval()
    scores = {c: [] for c in candidates}
    for tile in sample_tiles:
        rgb, depth_raw, target, mask = load_full_sample(tile)
        q = int(rng.choice(train_quadrants))
        bounds = quadrant_bounds(1024, 1024, q)
        rgb_q = crop(rgb, bounds).unsqueeze(0).to(device)
        depth_q_raw = crop(depth_raw, bounds)
        target_q = crop(target, bounds).squeeze(0).numpy()
        mask_q = crop(mask, bounds).squeeze(0).numpy().astype(bool)
        if mask_q.sum() < 100:
            continue
        for c in candidates:
            depth_q = (depth_q_raw * c).unsqueeze(0).to(device)
            pred = model(depth_q, rgb_q).detach().cpu().squeeze(0).squeeze(0).numpy()
            p, y = pred[mask_q], target_q[mask_q]
            if np.std(p) > 0 and np.std(y) > 0:
                scores[c].append(float(np.corrcoef(p, y)[0, 1]))

    mean_scores = {c: (float(np.mean(v)) if v else -1.0) for c, v in scores.items()}
    best_c = max(mean_scores, key=mean_scores.get)
    print(f"  scale sweep (n={len(sample_tiles)} train-only tiles, excluding {len(exclude_tiles)} probe tiles): "
          + ", ".join(f"x{c}={mean_scores[c]:+.4f}" for c in candidates))
    print(f"  -> selected scale = x{best_c} (Pearson={mean_scores[best_c]:+.4f})")
    return best_c


def train_one_epoch(model, samples, scale, device, optimizer, epoch, total_epochs):
    model.train()
    losses, smoothl1_losses, rank_losses = [], [], []

    for idx, (tile, q, rgb, depth_raw, target, mask) in enumerate(samples, start=1):
        depth = (depth_raw * scale).unsqueeze(0).to(device)
        rgb_t = rgb.unsqueeze(0).to(device)
        target_t = target.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(depth, rgb_t)

        smoothl1 = masked_smooth_l1(pred, target_t, mask_t)
        rank = rank_pair_loss(pred, target_t, mask_t.bool(), RANK_PAIRS_PER_PATCH, RANK_MARGIN)
        loss = smoothl1 + RANK_WEIGHT * rank

        if not torch.isfinite(loss):
            print(f"  [WARN] non-finite loss at {tile}/q{q}, skipping step")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        smoothl1_losses.append(float(smoothl1.detach().cpu()))
        rank_losses.append(float(rank.detach().cpu()))

        if idx == 1 or idx % 25 == 0 or idx == len(samples):
            print(f"  train epoch {epoch}/{total_epochs} {idx:>3}/{len(samples)} "
                  f"{tile}/q{q}: loss={losses[-1]:.4f} (smoothl1={smoothl1_losses[-1]:.4f} "
                  f"rank={rank_losses[-1]:.4f})")

    return float(np.mean(losses)), float(np.mean(smoothl1_losses)), float(np.mean(rank_losses))


@torch.no_grad()
def evaluate_samples(model, samples, scale, device):
    model.eval()
    per_sample = {}
    pred_chunks, target_chunks = [], []
    for tile, q, rgb, depth_raw, target, mask in samples:
        depth = (depth_raw * scale).unsqueeze(0).to(device)
        rgb_t = rgb.unsqueeze(0).to(device)
        pred = model(depth, rgb_t).detach().cpu().squeeze(0).squeeze(0).numpy()
        target_np = target.squeeze(0).numpy()
        mask_np = mask.squeeze(0).numpy()
        m = metrics_from_arrays(pred, target_np, mask_np)
        per_sample[f"{tile}_q{q}"] = m
        valid = mask_np.astype(bool)
        pred_chunks.append(pred[valid])
        target_chunks.append(target_np[valid])
    p = np.concatenate(pred_chunks)
    y = np.concatenate(target_chunks)
    overall = metrics_from_arrays(p, y, np.ones_like(p))
    overall["per_sample"] = per_sample
    overall["_pooled_pred"] = p
    overall["_pooled_true"] = y
    return overall


def load_quadrant_samples(tiles_and_quadrants: list[tuple[str, int]]):
    """Load every (tile, quadrant) sample's cropped tensors once, cached in
    memory as a list -- reused across all 5 epochs without re-reading from
    disk each time."""
    cache_by_tile: dict[str, tuple] = {}
    out = []
    for tile, q in tiles_and_quadrants:
        if tile not in cache_by_tile:
            cache_by_tile[tile] = load_full_sample(tile)
        rgb, depth_raw, target, mask = cache_by_tile[tile]
        bounds = quadrant_bounds(1024, 1024, q)
        out.append((tile, q, crop(rgb, bounds), crop(depth_raw, bounds),
                    crop(target, bounds), crop(mask, bounds)))
    return out


def run_fold(held_out_q: int, tiles: list[str], epochs: int, device: torch.device,
             model_cls, PositionalEncoding) -> dict:
    print(f"\n{'=' * 70}\nFOLD (held-out quadrant) {held_out_q}\n{'=' * 70}")

    train_quadrants = [q for q in range(4) if q != held_out_q]
    train_pairs = [(t, q) for t in tiles for q in train_quadrants]
    test_pairs = [(t, held_out_q) for t in tiles]

    model = model_cls().to(device)
    ckpt = torch.load(SWISS_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = build_resized_positional_encoding(model, PositionalEncoding, d_model=32, bottleneck=32)
    model.to(device)

    scale_candidates = [1, 30, 65, 100, 150, 200, 255, 300, 500, 1000]
    scale = derive_fold_scale(
        model, tiles, train_quadrants, device,
        exclude_tiles=PROBE_TILES, candidates=scale_candidates,
    )

    print(f"  loading {len(train_pairs)} train samples and {len(test_pairs)} test samples...")
    t0 = time.time()
    train_samples = load_quadrant_samples(train_pairs)
    test_samples_raw = load_quadrant_samples(test_pairs)
    print(f"  loaded in {time.time() - t0:.0f}s")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    fold_dir = OUT_DIR / f"fold{held_out_q}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        loss, smoothl1, rank = train_one_epoch(model, train_samples, scale, device, optimizer, epoch, epochs)

        test_eval = evaluate_samples(model, test_samples_raw, scale, device)
        p, y = test_eval.pop("_pooled_pred"), test_eval.pop("_pooled_true")
        var_ratio = float(np.var(p) / np.var(y)) if np.var(y) > 0 else float("nan")
        slope = float(np.polyfit(y, p, 1)[0]) if len(y) > 1 else float("nan")

        record = {
            "epoch": epoch, "train_loss": loss, "train_smoothl1": smoothl1, "train_rank": rank,
            "test_mae": test_eval["mae"], "test_rmse": test_eval["rmse"],
            "test_pearson": test_eval["pearson"], "test_spearman": test_eval["spearman"],
            "test_n": test_eval["n"], "variance_ratio": var_ratio, "ols_slope": slope,
        }
        history.append(record)
        print(f"  EPOCH {epoch} done in {time.time() - t_epoch:.0f}s: "
              f"train_loss={loss:.4f} test_mae={test_eval['mae']:.4f} "
              f"test_pearson={test_eval['pearson']:+.4f} var_ratio={var_ratio:.4f} slope={slope:.4f}")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "fold": held_out_q, "scale": scale, "train_loss": loss,
        }, fold_dir / f"epoch{epoch}.pt")
        save_json(fold_dir / "history.json", history)

    # ---- nested checkpoint-selection validation, same discipline as the audit ----
    rng = np.random.RandomState(SEED + held_out_q)
    idx = list(range(len(test_pairs)))
    rng.shuffle(idx)
    half = max(1, len(idx) // 2)
    inner_idx, true_idx = idx[:half], idx[half:] or idx[:half]
    inner_samples = [test_samples_raw[i] for i in inner_idx]
    true_samples = [test_samples_raw[i] for i in true_idx]

    inner_mae, true_metrics = {}, {}
    for epoch in range(1, epochs + 1):
        ckpt = torch.load(fold_dir / f"epoch{epoch}.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        iv = evaluate_samples(model, inner_samples, scale, device)
        tt = evaluate_samples(model, true_samples, scale, device)
        p_tt, y_tt = tt.pop("_pooled_pred"), tt.pop("_pooled_true")
        iv.pop("_pooled_pred", None); iv.pop("_pooled_true", None)
        var_ratio_tt = float(np.var(p_tt) / np.var(y_tt)) if np.var(y_tt) > 0 else float("nan")
        slope_tt = float(np.polyfit(y_tt, p_tt, 1)[0]) if len(y_tt) > 1 else float("nan")
        inner_mae[epoch] = iv["mae"]
        true_metrics[epoch] = {**tt, "variance_ratio": var_ratio_tt, "ols_slope": slope_tt}

    selected_epoch = min(inner_mae, key=inner_mae.get)
    oracle_epoch = min(true_metrics, key=lambda e: true_metrics[e]["mae"])

    result = {
        "fold": held_out_q, "checkpoint_init": str(SWISS_CHECKPOINT), "scale": scale,
        "train_tiles": tiles, "train_quadrants": train_quadrants,
        "n_train_samples": len(train_pairs), "n_test_samples": len(test_pairs),
        "history": history,
        "nested_selection": {
            "inner_val_n": len(inner_samples), "true_test_n": len(true_samples),
            "selected_epoch": selected_epoch,
            "selected_true_test": true_metrics[selected_epoch],
            "oracle_epoch": oracle_epoch,
            "oracle_true_test": true_metrics[oracle_epoch],
            "default_epoch5_true_test": true_metrics[epochs],
        },
    }
    save_json(fold_dir / "result.json", result)
    print(f"\n  FOLD {held_out_q} nested-selected epoch={selected_epoch}: "
          f"MAE={true_metrics[selected_epoch]['mae']:.4f} "
          f"RMSE={true_metrics[selected_epoch]['rmse']:.4f} "
          f"Pearson={true_metrics[selected_epoch]['pearson']:+.4f} "
          f"Spearman={true_metrics[selected_epoch]['spearman']:+.4f} "
          f"var_ratio={true_metrics[selected_epoch]['variance_ratio']:.4f} "
          f"slope={true_metrics[selected_epoch]['ols_slope']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run only this held-out quadrant (0-3).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not SWISS_CHECKPOINT.exists():
        raise FileNotFoundError(SWISS_CHECKPOINT)

    device = select_device(args.device)
    print(f"Device: {device}")
    model_cls, PositionalEncoding = import_rdah_model_class()

    tiles = load_manifest_tiles()
    print(f"{len(tiles)} manifest tiles")

    sanity_check_resize(model_cls, PositionalEncoding, device, tiles[0])

    fold_ids = [args.fold] if args.fold is not None else [0, 1, 2, 3]
    all_results = []
    for fold_id in fold_ids:
        result = run_fold(fold_id, tiles, args.epochs, device, model_cls, PositionalEncoding)
        all_results.append(result)

    save_json(OUT_DIR / "method_rdah_quadrant_cv_results.json", {
        "experiment": "RDAH-FT-2",
        "description": "Quadrant-level 4-fold fine-tuning, Swiss-init, per-fold scale + rank loss",
        "checkpoint_init": str(SWISS_CHECKPOINT),
        "epochs": args.epochs,
        "folds": all_results,
    })


if __name__ == "__main__":
    main()
