"""Follow-up to run_method4_sentinel2.py: same architecture/loss, same
fold structure, same win-count evaluation methodology -- but trained on
data/sentinel2_benchmark/method4_port_native30m/ (RGB/depth/building all
downsampled to SRTM's real native ~30m resolution, truth reprojected from
SRTM's own native grid onto a same-resolution 30m UTM grid instead of the
original 10m grid). This directly tests whether removing the
bilinear-interpolation-memorization opportunity diagnosed in the 10m-grid
run closes the DEM-check/ICESat-2-check gap found there.

Same exact hyperparameters as before (unchanged): dense patches, 60
epochs, smoothness_weight=0.01, rank_weight=0.5, rank_pairs_per_patch=2000,
rank_margin=0.25, building channel, ground_plane_weight=0.

Note (reported transparently, not routed around): at 30m the tile grid
shrinks from 1000x1000 to ~333x333, so with PATCH=64 unchanged, each
166x166 training quadrant yields only ~4 dense patches (vs ~49 at 10m) --
training patch count per fold drops from ~1000-1100 to ~300. This is a
real, honest consequence of testing at native resolution with an
otherwise-unchanged pipeline, not a bug to fix.
"""
from __future__ import annotations

import json
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

MANIFEST = PROJECT_ROOT / "data/sentinel2_benchmark/manifest.csv"
PORT_DIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_port_native30m"
PHOTON_DIR = PROJECT_ROOT / "data/icesat2_photons"
VERDICTS_CSV = PROJECT_ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUTDIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_sentinel2_native30m_results"

EPOCHS = int(os.environ.get("M4S2_EPOCHS", 60))
FOLDS = [int(x) for x in os.environ.get("M4S2_FOLDS", "0,1,2,3").split(",")]
SMOOTHNESS_WEIGHT = 0.01
RANK_WEIGHT = 0.5
RANK_PAIRS_PER_PATCH = 2000
RANK_MARGIN = 0.25
PATCH = m4v2.PATCH


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def load_tile_native(tile_id):
    with rasterio.open(PORT_DIR / f"{tile_id}_RGB.tif") as src:
        rgb = np.moveaxis(src.read([1, 2, 3]).astype(np.float32), 0, -1)
        transform, crs = src.transform, src.crs
    with rasterio.open(PORT_DIR / f"{tile_id}_depth.tif") as src:
        depth = src.read(1).astype(np.float32)
    with rasterio.open(PORT_DIR / f"{tile_id}_residual_truth.tif") as src:
        truth = src.read(1).astype(np.float32)
    with rasterio.open(PORT_DIR / f"{tile_id}_building.tif") as src:
        building = src.read(1).astype(np.float32)
    with rasterio.open(PORT_DIR / f"{tile_id}_linear_baseline.tif") as src:
        linear_baseline = src.read(1).astype(np.float32)
    valid = np.isfinite(truth)
    return rgb, depth, truth, building, linear_baseline, valid, transform, crs


def photon_pixel_rc_native(tile_id, transform, crs, h, w):
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
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

    tiles_data = {}
    for tile_id in accepted:
        rgb, depth, truth, building, linear_baseline, valid, transform, crs = load_tile_native(tile_id)
        tiles_data[tile_id] = dict(rgb=rgb, depth=depth, truth=truth, building=building,
                                    linear_baseline=linear_baseline, valid=valid, transform=transform, crs=crs)
    print(f"Tiles: {len(tiles_data)}, grid shape: {tiles_data[accepted[0]]['truth'].shape}")

    fold_results = []
    t0 = time.time()

    for held_out_q in FOLDS:
        t_fold = time.time()
        print(f"\n{'='*72}\nFOLD {held_out_q}\n{'='*72}")

        train_stats_subset, b_subset = [], []
        for tile_id in accepted:
            d = tiles_data[tile_id]
            h, w = d["truth"].shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            train_valid = d["valid"].copy()
            train_valid[r0:r1, c0:c1] = False
            train_stats_subset.append((d["rgb"], d["depth"], d["truth"], train_valid))
            b_subset.append((d["building"], train_valid))

        rgb_stats, depth_stats = m4.fit_stats(train_stats_subset)
        building_stats = m4v2.fit_building_stats(b_subset)
        print(f"rgb_stats={rgb_stats[0]} depth_stats={depth_stats} building_stats={building_stats}")

        samples = []
        for tile_id in accepted:
            d = tiles_data[tile_id]
            h, w = d["truth"].shape
            for q in range(4):
                if q == held_out_q:
                    continue
                bounds = m4.quadrant_bounds(h, w, q)
                origins = m4v2.dense_patch_origins(d["valid"], bounds, PATCH)
                for rr, cc in origins:
                    r = d["rgb"][rr:rr + PATCH, cc:cc + PATCH].copy()
                    dep = d["depth"][rr:rr + PATCH, cc:cc + PATCH].copy()
                    y = d["truth"][rr:rr + PATCH, cc:cc + PATCH].copy()
                    v = d["valid"][rr:rr + PATCH, cc:cc + PATCH]
                    dep[~v] = 0.0
                    y[~v] = np.nan
                    bb = d["building"][rr:rr + PATCH, cc:cc + PATCH].copy()
                    samples.append((r, dep, y, bb, 0.0, 1.0))

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

        model_path = OUTDIR / f"method4_native30m_fold{held_out_q}.pt"
        torch.save(model.state_dict(), model_path)
        model.eval()

        fold_tile_results = []
        dem_true_all, dem_cnn_all, dem_linear_all = [], [], []
        ice_true_all, ice_cnn_all, ice_linear_all = [], [], []

        for tile_id in accepted:
            d = tiles_data[tile_id]
            h, w = d["truth"].shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            scale_factor = float(scale_factors[tile_id])

            pred_q_norm = m4v2.predict_quadrant_v2(model, d["rgb"], d["depth"], d["building"], (r0, r1, c0, c1),
                                                     rgb_stats, depth_stats, building_stats, device)
            pred_residual_m = pred_q_norm * scale_factor
            linear_q = d["linear_baseline"][r0:r1, c0:c1]
            cnn_abs_pred = linear_q + pred_residual_m

            true_residual_norm_q = d["truth"][r0:r1, c0:c1]
            true_abs_q = linear_q + true_residual_norm_q * scale_factor
            v_test = d["valid"][r0:r1, c0:c1]

            dem_true, dem_cnn, dem_linear = true_abs_q[v_test], cnn_abs_pred[v_test], linear_q[v_test]
            dem_true_all.append(dem_true); dem_cnn_all.append(dem_cnn); dem_linear_all.append(dem_linear)

            photons, prows, pcols = photon_pixel_rc_native(tile_id, d["transform"], d["crs"], h, w)
            in_q = (prows >= r0) & (prows < r1) & (pcols >= c0) & (pcols < c1)
            ice_n = int(in_q.sum())
            if ice_n > 0:
                pr, pc = prows[in_q] - r0, pcols[in_q] - c0
                true_h = photons["height"].values[in_q]
                cnn_h = cnn_abs_pred[pr, pc]
                lin_h = linear_q[pr, pc]
                ice_true_all.append(true_h); ice_cnn_all.append(cnn_h); ice_linear_all.append(lin_h)

            fold_tile_results.append({
                "tile": tile_id,
                "dem_cnn_rmse_m": rmse(dem_cnn, dem_true),
                "dem_linear_rmse_m": rmse(dem_linear, dem_true),
                "icesat2_n": ice_n,
                "icesat2_cnn_rmse_m": rmse(cnn_h, true_h) if ice_n > 0 else None,
                "icesat2_linear_rmse_m": rmse(lin_h, true_h) if ice_n > 0 else None,
            })

        dem_true_all, dem_cnn_all, dem_linear_all = map(np.concatenate, (dem_true_all, dem_cnn_all, dem_linear_all))
        ice_true_all, ice_cnn_all, ice_linear_all = map(np.concatenate, (ice_true_all, ice_cnn_all, ice_linear_all))

        fold_summary = {
            "fold": held_out_q,
            "dem_cnn_rmse_m": rmse(dem_cnn_all, dem_true_all),
            "dem_linear_rmse_m": rmse(dem_linear_all, dem_true_all),
            "icesat2_cnn_rmse_m": rmse(ice_cnn_all, ice_true_all),
            "icesat2_linear_rmse_m": rmse(ice_linear_all, ice_true_all),
            "n_icesat2_photons": len(ice_true_all),
            "n_training_patches": len(samples),
            "tiles": fold_tile_results,
            "fold_time_sec": time.time() - t_fold,
        }
        fold_results.append(fold_summary)
        print(f"\nFOLD {held_out_q} SUMMARY:")
        for k, v in fold_summary.items():
            if k != "tiles":
                print(f"  {k}: {v}")

        (OUTDIR / "method4_native30m_results.json").write_text(
            json.dumps({"folds_completed": [f["fold"] for f in fold_results], "folds": fold_results}, indent=2, default=str)
        )

    wins_dem = sum(t["dem_cnn_rmse_m"] < t["dem_linear_rmse_m"] for f in fold_results for t in f["tiles"])
    wins_ice = sum(t["icesat2_cnn_rmse_m"] is not None and t["icesat2_cnn_rmse_m"] < t["icesat2_linear_rmse_m"]
                   for f in fold_results for t in f["tiles"])
    total = sum(len(f["tiles"]) for f in fold_results)

    output = {
        "config": {"epochs": EPOCHS, "resolution_m": 30, "patch": PATCH},
        "dem_win_count": f"{wins_dem}/{total}",
        "icesat2_win_count": f"{wins_ice}/{total}",
        "overall_dem_cnn_rmse_m": float(np.mean([f["dem_cnn_rmse_m"] for f in fold_results])),
        "overall_dem_linear_rmse_m": float(np.mean([f["dem_linear_rmse_m"] for f in fold_results])),
        "overall_icesat2_cnn_rmse_m": float(np.mean([f["icesat2_cnn_rmse_m"] for f in fold_results])),
        "overall_icesat2_linear_rmse_m": float(np.mean([f["icesat2_linear_rmse_m"] for f in fold_results])),
        "total_time_sec": time.time() - t0,
        "folds": fold_results,
    }
    (OUTDIR / "method4_native30m_results.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"\n{'='*78}\nFINAL RESULT\n{'='*78}")
    for k, v in output.items():
        if k != "folds":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
