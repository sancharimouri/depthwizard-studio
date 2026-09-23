#!/usr/bin/env python3
"""
Method 6 on Sentinel-2 / SRTM — staged test of whether full DAv2-Small
fine-tuning replicates Method 4's interpolation-memorization failure, or
avoids it.

Background (established, not re-derived here): Method 4 (frozen-feature
scale modulation, scripts/run_method4_sentinel2.py) trained a small CNN to
correct a per-tile linear-OLS baseline toward SRTM on Sentinel-2, at the
DEM's native 10m *reprojected* grid, and produced the textbook
interpolation-memorization signature: it won big on the DEM held-out check
(RMSE 75.97m vs. linear's 117.87m) but LOST on the independent ICESat-2
photon check (69.79m vs. linear's 53.40m) -- it had learned to reproduce
SRTM's own bilinear-interpolation artifacts, not real elevation signal.
A follow-up at SRTM's true native ~30m grid (scripts/run_method4_sentinel2_
native30m.py, removing the interpolation-memorization opportunity) produced
a DIFFERENT failure signature: it lost on BOTH checks (DEM win count
24/100, ICESat-2 win count 16/100) -- plain underperformance, not
memorization, plausibly because native resolution cuts training patches
from ~1000-1100/fold to ~300/fold.

This script ports Method 6 (full DAv2-Small backbone fine-tune, twin
mean/log-variance head -- see evaluate_method6_finetune_twinhead.py, which
this imports from directly) onto the SAME 10m-grid setup that produced the
memorization signature (data/sentinel2_benchmark/method4_port/), predicting
absolute (ellipsoidal) elevation DIRECTLY from RGB -- not a residual on top
of the linear baseline, since Method 6 has no frozen depth-correction step
to be a residual "on top of". The comparison target throughout is the SAME
per-tile linear-OLS baseline Method 4 was compared against
("the existing linear-calibration baseline").

Staging (per instructions -- do not skip):
  1. Run fold 0 ALONE. Score both the DEM held-out check and the
     independent ICESat-2 check, per tile.
  2. STOP CONDITION: if more than half of fold 0's 25 tiles show the CNN
     losing to the linear baseline on the ICESat-2 check (the check that
     actually matters -- DEM wins can be memorization, per Method 4's own
     lesson), stop and report which failure signature it is:
       - interpolation-memorization: wins DEM, loses ICESat-2 (like the
         original 10m run)
       - plain underperformance: loses both (like the native30m retry)
  3. Only if fold 0 clears the bar, continue folds 1-3.

Usage:
    python scripts/evaluate_method6_sentinel2.py --fold0-only     # stage 1
    python scripts/evaluate_method6_sentinel2.py --folds 1 2 3    # stage 3, if stage 1 cleared
"""
from __future__ import annotations

import argparse
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
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402 -- quadrant_bounds only
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    TwinHeadDav2, gaussian_nll, masked_huber, pad_to, seed_everything, get_device,
    IMAGENET_MEAN, IMAGENET_STD,
)

MANIFEST = PROJECT_ROOT / "data/sentinel2_benchmark/manifest.csv"
PORT_DIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_port"
PHOTON_DIR = PROJECT_ROOT / "data/icesat2_photons"
VERDICTS_CSV = PROJECT_ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUTDIR = PROJECT_ROOT / "data/sentinel2_benchmark/method6_sentinel2_results"

PATCH_SIZE = 14


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def load_tile(tile_id: str, rgb_path: Path):
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
        transform, crs = src.transform, src.crs
    with rasterio.open(PORT_DIR / f"{tile_id}_linear_baseline.tif") as src:
        linear = src.read(1).astype(np.float32)
    with rasterio.open(PORT_DIR / f"{tile_id}_residual_truth.tif") as src:
        residual_norm = src.read(1).astype(np.float32)
    return rgb, linear, residual_norm, transform, crs


def photon_pixel_rc(tile_id: str, transform, crs, h: int, w: int):
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(photons["lon"].values, photons["lat"].values)
    inv = ~transform
    cols, rows = inv * (x, y)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    in_bounds = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    return photons[in_bounds].reset_index(drop=True), rows[in_bounds], cols[in_bounds]


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(np.float64), p[m].astype(np.float64)
    return {"mae_m": float(np.mean(np.abs(y - p))), "rmse_m": float(np.sqrt(np.mean((y - p) ** 2))),
            "pearson": float(pearsonr(y, p).statistic) if len(y) > 1 else float("nan"),
            "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--fold0-only", action="store_true", help="alias for --folds 0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--log-var-max", type=float, default=7.0)
    ap.add_argument("--log-var-min", type=float, default=-8.0)
    ap.add_argument("--tiles-limit", type=int, default=None)
    args = ap.parse_args()
    if args.fold0_only:
        args.folds = [0]
    args.lr_head = args.lr_head if args.lr_head is not None else args.lr_backbone * 50

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seed_everything()
    device = get_device()
    print(f"device={device}")

    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    scale_factors = pd.read_csv(PORT_DIR / "scale_factors.csv").set_index("tile_id")["scale_factor"]
    if args.tiles_limit:
        accepted = accepted[: args.tiles_limit]
    print(f"tiles: {len(accepted)}")

    cache = {}
    for tid in accepted:
        rgb_path = PROJECT_ROOT / manifest.loc[tid, "rgb_path"]
        rgb, linear, residual_norm, transform, crs = load_tile(tid, rgb_path)
        sf = float(scale_factors[tid])
        true_abs = linear + residual_norm * sf
        valid = np.isfinite(true_abs) & np.isfinite(linear)
        cache[tid] = dict(rgb=rgb, linear=linear, true_abs=true_abs, valid=valid,
                          transform=transform, crs=crs)
    h0, w0 = cache[accepted[0]]["true_abs"].shape
    print(f"tile grid: {h0}x{w0}")
    qh, qw = h0 // 2, w0 // 2
    pad_h = math.ceil(qh / PATCH_SIZE) * PATCH_SIZE
    pad_w = math.ceil(qw / PATCH_SIZE) * PATCH_SIZE
    print(f"quadrant {qh}x{qw}, padded to {pad_h}x{pad_w}")

    fold_results = []
    for held_out_q in args.folds:
        t_fold = time.time()
        print(f"\n{'='*72}\nFOLD {held_out_q}\n{'='*72}")

        train_samples, train_abs_vals = [], []
        eval_tiles = []  # (tid, rgb_q, true_abs_q, linear_q, valid_q, transform, crs)
        for tid in accepted:
            d = cache[tid]
            h, w = d["true_abs"].shape
            for q in range(4):
                r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
                rgb_c = d["rgb"][:, r0:r1, c0:c1]
                abs_c = d["true_abs"][r0:r1, c0:c1]
                valid_c = d["valid"][r0:r1, c0:c1]
                if q == held_out_q:
                    eval_tiles.append((tid, rgb_c, abs_c, d["linear"][r0:r1, c0:c1], valid_c,
                                       d["transform"], d["crs"]))
                else:
                    train_samples.append((rgb_c, abs_c, valid_c))
                    if valid_c.any():
                        train_abs_vals.append(abs_c[valid_c])

        pooled = np.concatenate(train_abs_vals)
        init_mu_m = float(pooled.mean())
        height_scale = max(float(pooled.std()), 1.0)
        print(f"fold{held_out_q}: {len(train_samples)} train quadrants, {len(eval_tiles)} eval tiles, "
              f"init_mu_m={init_mu_m:.1f}m height_scale(std)={height_scale:.1f}m")

        model = TwinHeadDav2(height_scale=height_scale, init_sigma_m=height_scale * 0.3,
                             log_var_max=args.log_var_max, log_var_min=args.log_var_min,
                             init_mu="constant", init_mu_m=init_mu_m).to(device)
        opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head, args.weight_decay))

        def make_ds(samples):
            class DS(torch.utils.data.Dataset):
                def __len__(self_): return len(samples)
                def __getitem__(self_, i):
                    rgb, abs_c, valid_c = samples[i]
                    rgb_t = torch.from_numpy(rgb / 255.0)
                    rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    rgb_t = pad_to(rgb_t, pad_h if pad_h == pad_w else max(pad_h, pad_w))
                    abs_t = torch.nan_to_num(pad_to(torch.from_numpy(abs_c)[None], pad_h)[0], nan=0.0)
                    valid_t = pad_to(torch.from_numpy(valid_c.astype(np.float32))[None], pad_h)[0] > 0.5
                    return rgb_t, abs_t, valid_t
            return DS()

        # pad_h == pad_w here since the tile grid is square; assert to catch a future
        # non-square benchmark tile rather than silently mis-padding one axis.
        assert pad_h == pad_w, f"non-square padded quadrant {pad_h}x{pad_w} not supported"
        train_ld = DataLoader(make_ds(train_samples), batch_size=args.batch, shuffle=True, num_workers=0)

        steps_per_epoch = max(1, len(train_samples) // args.batch)
        total_steps = steps_per_epoch * args.epochs
        warm = max(10, int(total_steps * 0.05))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else max(0.05, (total_steps - s) / max(1, total_steps - warm)))

        model.train()
        gstep = 0
        for epoch in range(1, args.epochs + 1):
            running = []
            for rgb_t, abs_t, valid_t in train_ld:
                rgb_t, abs_t, valid_t = rgb_t.to(device), abs_t.to(device), valid_t.to(device)
                mu, log_var = model(rgb_t)
                if gstep < args.warmup_steps:
                    loss = masked_huber(mu[:, 0], abs_t, valid_t)
                else:
                    loss = gaussian_nll(mu[:, 0], log_var[:, 0], abs_t, valid_t)
                if not torch.isfinite(loss):
                    print(f"fold{held_out_q}: non-finite loss at step {gstep}, aborting fold")
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
                print(f"fold{held_out_q} epoch {epoch:02d}/{args.epochs}: loss={np.mean(running):.4f} "
                      f"({time.time()-t_fold:.0f}s)")

        # ---------------------------------------------------------------- evaluate
        model.eval()
        fold_tile_results = []
        with torch.no_grad():
            for tid, rgb_c, abs_c, linear_c, valid_c, transform, crs in eval_tiles:
                rgb_t = torch.from_numpy(rgb_c / 255.0)
                rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                rgb_t = pad_to(rgb_t, pad_h)[None].to(device)
                mu, _ = model(rgb_t)
                cnn_abs = mu[0, 0, : abs_c.shape[0], : abs_c.shape[1]].cpu().numpy()

                dem_true, dem_cnn, dem_linear = abs_c[valid_c], cnn_abs[valid_c], linear_c[valid_c]

                h, w = abs_c.shape
                photons, prows, pcols = photon_pixel_rc(tid, transform, crs, h, w)
                ice_n = len(photons)
                if ice_n > 0:
                    true_h = photons["height"].values
                    cnn_h = cnn_abs[prows, pcols]
                    lin_h = linear_c[prows, pcols]
                else:
                    true_h = cnn_h = lin_h = np.array([])

                fold_tile_results.append({
                    "tile": tid,
                    "dem_cnn_rmse_m": rmse(dem_cnn, dem_true),
                    "dem_linear_rmse_m": rmse(dem_linear, dem_true),
                    "dem_cnn_wins": bool(rmse(dem_cnn, dem_true) < rmse(dem_linear, dem_true)),
                    "icesat2_n": ice_n,
                    "icesat2_cnn_rmse_m": rmse(cnn_h, true_h) if ice_n > 0 else None,
                    "icesat2_linear_rmse_m": rmse(lin_h, true_h) if ice_n > 0 else None,
                    "icesat2_cnn_wins": bool(rmse(cnn_h, true_h) < rmse(lin_h, true_h)) if ice_n > 0 else None,
                })

        dem_wins = sum(t["dem_cnn_wins"] for t in fold_tile_results)
        ice_scored = [t for t in fold_tile_results if t["icesat2_cnn_wins"] is not None]
        ice_wins = sum(t["icesat2_cnn_wins"] for t in ice_scored)
        n_tiles = len(fold_tile_results)

        fold_summary = {
            "fold": held_out_q,
            "dem_win_count": f"{dem_wins}/{n_tiles}",
            "icesat2_win_count": f"{ice_wins}/{len(ice_scored)}",
            "overall_dem_cnn_rmse_m": float(np.mean([t["dem_cnn_rmse_m"] for t in fold_tile_results])),
            "overall_dem_linear_rmse_m": float(np.mean([t["dem_linear_rmse_m"] for t in fold_tile_results])),
            "overall_icesat2_cnn_rmse_m": float(np.mean([t["icesat2_cnn_rmse_m"] for t in ice_scored])),
            "overall_icesat2_linear_rmse_m": float(np.mean([t["icesat2_linear_rmse_m"] for t in ice_scored])),
            "init_mu_m": init_mu_m, "height_scale": height_scale,
            "fold_time_sec": time.time() - t_fold,
            "tiles": fold_tile_results,
        }
        fold_results.append(fold_summary)
        print(f"\nFOLD {held_out_q} SUMMARY:")
        for k, v in fold_summary.items():
            if k != "tiles":
                print(f"  {k}: {v}")

        (OUTDIR / "method6_sentinel2_results.json").write_text(
            json.dumps({"folds_completed": [f["fold"] for f in fold_results], "folds": fold_results},
                      indent=2, default=str))

        # ---------------------------------------------------------------- staging gate
        if held_out_q == 0:
            n_ice = len(ice_scored)
            losses = n_ice - ice_wins
            print(f"\n{'='*72}\nSTAGE 1 GATE (fold 0): ICESat-2 wins {ice_wins}/{n_ice}, losses {losses}/{n_ice}\n{'='*72}")
            if losses > n_ice / 2:
                signature = ("interpolation-memorization (wins DEM, loses ICESat-2)"
                            if dem_wins > n_tiles / 2 else
                            "plain underperformance (loses both DEM and ICESat-2)")
                print(f"STOP CONDITION MET: more than half of fold 0's tiles lose to the linear "
                      f"baseline on ICESat-2 ({losses}/{n_ice}).")
                print(f"Failure signature: {signature}")
                print(f"DEM win count: {dem_wins}/{n_tiles}")
                (OUTDIR / "STOPPED_at_fold0.json").write_text(json.dumps({
                    "stopped": True, "icesat2_losses": losses, "icesat2_n": n_ice,
                    "dem_win_count": f"{dem_wins}/{n_tiles}", "signature": signature,
                }, indent=2))
                return
            else:
                print(f"Fold 0 CLEARS the bar ({ice_wins}/{n_ice} ICESat-2 wins). Proceeding.")

        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\n{'='*78}\nALL REQUESTED FOLDS DONE\n{'='*78}")


if __name__ == "__main__":
    main()
