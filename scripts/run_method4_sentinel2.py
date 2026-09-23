"""Port of phase2_building_rank_v2 (Method 4 v2) to Sentinel-2/SRTM, on
the 25 sign-flip-detector-accepted tiles. Architecture and losses are
imported UNCHANGED from evaluate_method4_v2.py (ScaleModulationNetV2,
huber_masked, smoothness_loss, rank_pair_loss) and evaluate_method4.py
(quadrant_bounds, fit_stats) -- this script only adapts the data-loading
layer to this project's file layout (different directory structure than
DFC2019's flat depth-dir/rgb-dir/truth-dir convention) and adds the
independent ICESat-2 evaluation this domain has that DFC2019 didn't.

Domain adaptations (see scripts/prepare_method4_sentinel2_data.py's
docstring for the full reasoning, confirmed with the user before
training):
  - Truth = per-tile-normalized linear-fit residual (SRTM-ellipsoidal
    minus the per-tile OLS baseline, divided by that tile's own max-abs
    value to fit the architecture's native +-50 output range), not raw
    elevation and not literal height-above-tile-minimum.
  - Building channel = ESA WorldCover built-up fraction (native 10m),
    not DFC2019's airborne-imagery ONNX model.
  - Same exact hyperparameters as the phase2_building_rank_v2 config:
    dense patches, 60 epochs, smoothness_weight=0.01, rank_weight=0.5,
    rank_pairs_per_patch=2000, rank_margin=0.25, ground_plane_weight=0.

Evaluates, per fold's held-out quadrant:
  1. Against the DEM (held-out, corrected-SRTM) -- same as DFC2019's own
     within-tile check.
  2. Independently against real ICESat-2 ground-height photons falling
     within that held-out quadrant -- the stronger, independent check
     used everywhere else in this project.
  3. The linear-only baseline on the exact same held-out pixels/photons,
     for a fair CNN-vs-linear comparison (the DFC2019 framing: does the
     learned correction beat the per-tile-OLS baseline).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402
import evaluate_method4_v2 as m4v2  # noqa: E402
from lib_backbone_correlation import sample_depth_at_photons  # noqa: E402

MANIFEST = PROJECT_ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = PROJECT_ROOT / "data/sentinel2_benchmark/dav2_depth"
PORT_DIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_port"
PHOTON_DIR = PROJECT_ROOT / "data/icesat2_photons"
VERDICTS_CSV = PROJECT_ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUTDIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_sentinel2_results"

EPOCHS = int(os.environ.get("M4S2_EPOCHS", 60))  # env override for smoke-testing only; real run uses 60 (unchanged config)
FOLDS = [int(x) for x in os.environ.get("M4S2_FOLDS", "0,1,2,3").split(",")]
SMOOTHNESS_WEIGHT = 0.01
RANK_WEIGHT = 0.5
RANK_PAIRS_PER_PATCH = 2000
RANK_MARGIN = 0.25
PATCH = m4v2.PATCH

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def load_building(tile_id):
    with rasterio.open(PORT_DIR / f"{tile_id}_building.tif") as src:
        return src.read(1).astype(np.float32)


def photon_pixel_rc(tile_id, rgb_path):
    """Row/col of every real ICESat-2 photon for this tile, in the tile's
    own pixel grid -- same affine-invert convention as
    lib_backbone_correlation.sample_depth_at_photons, exposed here so we
    can filter photons by which quadrant they fall in."""
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
    with rasterio.open(rgb_path) as src:
        crs, transform, h, w = src.crs, src.transform, src.height, src.width
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(photons["lon"].values, photons["lat"].values)
    inv = ~transform
    cols, rows = inv * (x, y)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    in_bounds = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    return photons[in_bounds].reset_index(drop=True), rows[in_bounds], cols[in_bounds]


def main():
    m4v2.seed_everything()
    device = m4v2.get_device()
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    scale_factors = pd.read_csv(PORT_DIR / "scale_factors.csv").set_index("tile_id")["scale_factor"]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    tiles, cached, building_cache = [], [], []
    linear_baselines, srtm_ellipsoidals, n_geoids = {}, {}, {}
    for tile_id in accepted:
        m = manifest.loc[tile_id]
        rgb_path = PROJECT_ROOT / m["rgb_path"]
        depth_path = DAV2_DIR / f"{tile_id}_depth.npy"
        truth_path = PORT_DIR / f"{tile_id}_residual_truth.tif"
        tile = m4.Tile(tile_id, depth_path, rgb_path, truth_path)
        tiles.append(tile)

        rgb, depth, truth, valid = m4.load_tile(tile)
        cached.append((rgb, depth, truth, valid))
        building_cache.append(load_building(tile_id))

        with rasterio.open(PORT_DIR / f"{tile_id}_linear_baseline.tif") as src:
            linear_baselines[tile_id] = src.read(1).astype(np.float32)
        n_geoids[tile_id] = n_egm96(m["lon"], m["lat"])

    print(f"Tiles: {len(tiles)}")

    fold_results = []
    t0 = time.time()

    for held_out_q in FOLDS:
        t_fold = time.time()
        print(f"\n{'='*72}\nFOLD {held_out_q}\n{'='*72}")

        train_stats_subset, b_subset = [], []
        for (rgb, depth, truth, valid), b in zip(cached, building_cache):
            h, w = truth.shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            train_valid = valid.copy()
            train_valid[r0:r1, c0:c1] = False
            train_stats_subset.append((rgb, depth, truth, train_valid))
            b_subset.append((b, train_valid))

        rgb_stats, depth_stats = m4.fit_stats(train_stats_subset)
        building_stats = m4v2.fit_building_stats(b_subset)
        print(f"rgb_stats={rgb_stats[0]} depth_stats={depth_stats} building_stats={building_stats}")

        samples = []
        for (rgb, depth, truth, valid), b in zip(cached, building_cache):
            h, w = truth.shape
            for q in range(4):
                if q == held_out_q:
                    continue
                bounds = m4.quadrant_bounds(h, w, q)
                origins = m4v2.dense_patch_origins(valid, bounds, PATCH)
                for rr, cc in origins:
                    r = rgb[rr:rr + PATCH, cc:cc + PATCH].copy()
                    d = depth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    y = truth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    v = valid[rr:rr + PATCH, cc:cc + PATCH]
                    d[~v] = 0.0
                    y[~v] = np.nan
                    bb = b[rr:rr + PATCH, cc:cc + PATCH].copy()
                    samples.append((r, d, y, bb, 0.0, 1.0))  # hmin/hmax unused (sid_bins=0)

        print(f"Training patches: {len(samples)}")
        dataset = m4v2.PatchDatasetV2(samples, rgb_stats, depth_stats, building_stats)
        loader = torch.utils.data.DataLoader(dataset, batch_size=m4v2.BATCH_SIZE, shuffle=True, num_workers=0)

        model = m4v2.ScaleModulationNetV2(in_channels=5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=m4v2.LR, weight_decay=m4v2.WEIGHT_DECAY)

        model.train()
        for epoch in range(1, EPOCHS + 1):
            running = []
            for x, y, mask, hmin, hmax in loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                hmin, hmax = hmin.to(device), hmax.to(device)
                raw_depth = x[:, 3:4] * depth_stats[1] + depth_stats[0]

                optimizer.zero_grad(set_to_none=True)
                pred, scale, _ = model(x, raw_depth)
                loss = m4v2.huber_masked(pred, y, mask)
                loss = loss + SMOOTHNESS_WEIGHT * m4v2.smoothness_loss(scale)
                loss = loss + RANK_WEIGHT * m4v2.rank_pair_loss(pred, y, mask, RANK_PAIRS_PER_PATCH, RANK_MARGIN)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running.append(float(loss.detach().cpu()))
            if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
                print(f"  epoch {epoch:03d}/{EPOCHS}: loss={np.mean(running):.5f}")

        model_path = OUTDIR / f"method4_sentinel2_fold{held_out_q}.pt"
        torch.save(model.state_dict(), model_path)
        model.eval()

        # ---- Evaluation on the held-out quadrant, all 25 tiles ----
        fold_tile_results = []
        dem_true_all, dem_cnn_all, dem_linear_all = [], [], []
        ice_true_all, ice_cnn_all, ice_linear_all = [], [], []

        for tile, (rgb, depth, truth, valid), b in zip(tiles, cached, building_cache):
            tile_id = tile.tile_id
            h, w = truth.shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            scale_factor = float(scale_factors[tile_id])
            n = n_geoids[tile_id]

            pred_q_norm = m4v2.predict_quadrant_v2(model, rgb, depth, b, (r0, r1, c0, c1), rgb_stats, depth_stats, building_stats, device)
            pred_residual_m = pred_q_norm * scale_factor
            linear_q = linear_baselines[tile_id][r0:r1, c0:c1]
            cnn_abs_pred = linear_q + pred_residual_m

            true_residual_norm_q = truth[r0:r1, c0:c1]
            true_abs_q = linear_q + true_residual_norm_q * scale_factor  # = corrected SRTM in this quadrant
            v_test = valid[r0:r1, c0:c1]

            dem_true = true_abs_q[v_test]
            dem_cnn = cnn_abs_pred[v_test]
            dem_linear = linear_q[v_test]
            dem_true_all.append(dem_true)
            dem_cnn_all.append(dem_cnn)
            dem_linear_all.append(dem_linear)

            # ICESat-2 photons falling within this held-out quadrant
            photons, prows, pcols = photon_pixel_rc(tile_id, PROJECT_ROOT / manifest.loc[tile_id, "rgb_path"])
            in_q = (prows >= r0) & (prows < r1) & (pcols >= c0) & (pcols < c1)
            if in_q.sum() > 0:
                pr, pc = prows[in_q] - r0, pcols[in_q] - c0
                true_h = photons["height"].values[in_q]
                cnn_h = cnn_abs_pred[pr, pc]
                lin_h = linear_q[pr, pc]
                ice_true_all.append(true_h)
                ice_cnn_all.append(cnn_h)
                ice_linear_all.append(lin_h)
                ice_n = int(in_q.sum())
            else:
                ice_n = 0

            fold_tile_results.append({
                "tile": tile_id,
                "dem_cnn_rmse_m": rmse(dem_cnn, dem_true),
                "dem_linear_rmse_m": rmse(dem_linear, dem_true),
                "icesat2_n": ice_n,
                "icesat2_cnn_rmse_m": rmse(cnn_h, true_h) if ice_n > 0 else None,
                "icesat2_linear_rmse_m": rmse(lin_h, true_h) if ice_n > 0 else None,
            })

        dem_true_all = np.concatenate(dem_true_all)
        dem_cnn_all = np.concatenate(dem_cnn_all)
        dem_linear_all = np.concatenate(dem_linear_all)
        ice_true_all = np.concatenate(ice_true_all)
        ice_cnn_all = np.concatenate(ice_cnn_all)
        ice_linear_all = np.concatenate(ice_linear_all)

        fold_summary = {
            "fold": held_out_q,
            "dem_cnn_rmse_m": rmse(dem_cnn_all, dem_true_all),
            "dem_linear_rmse_m": rmse(dem_linear_all, dem_true_all),
            "dem_cnn_pearson": float(pearsonr(dem_cnn_all, dem_true_all).statistic),
            "dem_linear_pearson": float(pearsonr(dem_linear_all, dem_true_all).statistic),
            "icesat2_cnn_rmse_m": rmse(ice_cnn_all, ice_true_all),
            "icesat2_linear_rmse_m": rmse(ice_linear_all, ice_true_all),
            "icesat2_cnn_pearson": float(pearsonr(ice_cnn_all, ice_true_all).statistic),
            "icesat2_linear_pearson": float(pearsonr(ice_linear_all, ice_true_all).statistic),
            "n_icesat2_photons": len(ice_true_all),
            "tiles": fold_tile_results,
            "fold_time_sec": time.time() - t_fold,
        }
        fold_results.append(fold_summary)
        print(f"\nFOLD {held_out_q} SUMMARY:")
        for k, v in fold_summary.items():
            if k != "tiles":
                print(f"  {k}: {v}")

        (OUTDIR / "method4_sentinel2_results.json").write_text(
            json.dumps({"folds_completed": [f["fold"] for f in fold_results], "folds": fold_results}, indent=2, default=str)
        )

    output = {
        "config": {
            "epochs": EPOCHS, "smoothness_weight": SMOOTHNESS_WEIGHT, "rank_weight": RANK_WEIGHT,
            "rank_pairs_per_patch": RANK_PAIRS_PER_PATCH, "rank_margin": RANK_MARGIN,
            "extra_channel": "building (WorldCover, ported)", "target": "per-tile-normalized linear-fit residual",
        },
        "overall_dem_cnn_rmse_m": float(np.mean([f["dem_cnn_rmse_m"] for f in fold_results])),
        "overall_dem_linear_rmse_m": float(np.mean([f["dem_linear_rmse_m"] for f in fold_results])),
        "overall_icesat2_cnn_rmse_m": float(np.mean([f["icesat2_cnn_rmse_m"] for f in fold_results])),
        "overall_icesat2_linear_rmse_m": float(np.mean([f["icesat2_linear_rmse_m"] for f in fold_results])),
        "total_time_sec": time.time() - t0,
        "folds": fold_results,
    }
    (OUTDIR / "method4_sentinel2_results.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"\n{'='*78}\nFINAL RESULT\n{'='*78}")
    for k, v in output.items():
        if k != "folds":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
