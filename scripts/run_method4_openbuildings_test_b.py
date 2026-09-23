"""Test B — Method 4's EXACT architecture (phase2_building_rank_v2 /
ScaleModulationNetV2), retrained with Google Open Buildings 2.5D Temporal
as the training target on building pixels instead of SRTM, on the 8 urban
Sentinel-2 tiles only (data/sentinel2_benchmark/method4_port_openbuildings/,
built by scripts/prepare_method4_openbuildings_data.py).

Direct structural mirror of scripts/run_method4_sentinel2.py -- same
architecture/loss imports (evaluate_method4_v2.ScaleModulationNetV2,
huber_masked, smoothness_loss, rank_pair_loss, dense_patch_origins,
PatchDatasetV2, predict_quadrant_v2; evaluate_method4.quadrant_bounds,
fit_stats), same hyperparameters (dense patches, smoothness_weight=0.01,
rank_weight=0.5, rank_pairs_per_patch=2000, rank_margin=0.25, building
channel, ground_plane_weight=0), same fold/DEM-check/ICESat-2-check
structure. Only the tile list (8 urban, not 25), the port directory, and
the stop-condition comparator change.

Stop condition (staged, per instructions): after fold 0, compare each
tile's ICESat-2 CNN RMSE against that tile's OWN best pre-existing
baseline (whichever of {linear SRTM calibration, existing Method 4/SRTM
result} scored better on ICESat-2 for that tile in the original 25-tile
run) -- not a single shared number, since the two prior baselines trade
off which is better tile by tile. If more than 4 of 8 tiles lose, stop and
report the failure signature (memorization: wins the Open-Buildings-target
check but loses ICESat-2, vs plain underperformance: loses both).
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

MANIFEST = PROJECT_ROOT / "data/sentinel2_benchmark/manifest.csv"
PORT_DIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_port_openbuildings"
PHOTON_DIR = PROJECT_ROOT / "data/icesat2_photons"
OUTDIR = PROJECT_ROOT / "data/sentinel2_benchmark/method4_openbuildings_testb_results"

URBAN_TILES = ["bengaluru", "chennai", "delhi", "hyderabad", "jaipur", "kochi_city", "mumbai", "pune"]

# Per-tile best pre-existing baseline on ICESat-2, fold 0 (computed from the
# already-completed 25-tile method4_sentinel2_results.json): min of
# {icesat2_linear_rmse_m, icesat2_cnn_rmse_m (existing Method4/SRTM)}.
BEST_EXISTING_BASELINE_FOLD0 = {
    "bengaluru": 16.36, "chennai": 5.46, "delhi": 11.46, "jaipur": 16.61,
    "kochi_city": 1.58, "mumbai": 11.2,       # all favor linear
    "hyderabad": 21.03, "pune": 38.68,        # favor existing Method4/SRTM
}

EPOCHS = int(os.environ.get("M4OB_EPOCHS", 60))
FOLDS = [int(x) for x in os.environ.get("M4OB_FOLDS", "0").split(",")]
SMOOTHNESS_WEIGHT = 0.01
RANK_WEIGHT = 0.5
RANK_PAIRS_PER_PATCH = 2000
RANK_MARGIN = 0.25
PATCH = m4v2.PATCH


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def load_building(tile_id: str) -> np.ndarray:
    with rasterio.open(PORT_DIR / f"{tile_id}_building.tif") as src:
        return src.read(1).astype(np.float32)


def photon_pixel_rc(tile_id: str, rgb_path: Path):
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
    scale_factors = pd.read_csv(PORT_DIR / "scale_factors.csv").set_index("tile_id")["scale_factor"]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    tiles, cached, building_cache = [], [], []
    linear_baselines = {}
    for tile_id in URBAN_TILES:
        m = manifest.loc[tile_id]
        rgb_path = PROJECT_ROOT / m["rgb_path"]
        depth_path = PROJECT_ROOT / "data/sentinel2_benchmark/dav2_depth" / f"{tile_id}_depth.npy"
        truth_path = PORT_DIR / f"{tile_id}_residual_truth.tif"
        tile = m4.Tile(tile_id, depth_path, rgb_path, truth_path)
        tiles.append(tile)

        rgb, depth, truth, valid = m4.load_tile(tile)
        cached.append((rgb, depth, truth, valid))
        building_cache.append(load_building(tile_id))

        with rasterio.open(PORT_DIR / f"{tile_id}_linear_baseline.tif") as src:
            linear_baselines[tile_id] = src.read(1).astype(np.float32)

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
                    samples.append((r, d, y, bb, 0.0, 1.0))

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

        model_path = OUTDIR / f"method4_ob_testb_fold{held_out_q}.pt"
        torch.save(model.state_dict(), model_path)
        model.eval()

        fold_tile_results = []
        dem_true_all, dem_cnn_all, dem_linear_all = [], [], []
        ice_true_all, ice_cnn_all, ice_linear_all = [], [], []

        for tile, (rgb, depth, truth, valid), b in zip(tiles, cached, building_cache):
            tile_id = tile.tile_id
            h, w = truth.shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)
            scale_factor = float(scale_factors[tile_id])

            pred_q_norm = m4v2.predict_quadrant_v2(model, rgb, depth, b, (r0, r1, c0, c1),
                                                     rgb_stats, depth_stats, building_stats, device)
            pred_residual_m = pred_q_norm * scale_factor
            linear_q = linear_baselines[tile_id][r0:r1, c0:c1]
            cnn_abs_pred = linear_q + pred_residual_m

            true_residual_norm_q = truth[r0:r1, c0:c1]
            true_abs_q = linear_q + true_residual_norm_q * scale_factor  # this run's own OB-informed target
            v_test = valid[r0:r1, c0:c1]

            dem_true, dem_cnn, dem_linear = true_abs_q[v_test], cnn_abs_pred[v_test], linear_q[v_test]
            dem_true_all.append(dem_true); dem_cnn_all.append(dem_cnn); dem_linear_all.append(dem_linear)

            photons, prows, pcols = photon_pixel_rc(tile_id, PROJECT_ROOT / manifest.loc[tile_id, "rgb_path"])
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

        (OUTDIR / "method4_ob_testb_results.json").write_text(
            json.dumps({"folds_completed": [f["fold"] for f in fold_results], "folds": fold_results}, indent=2, default=str))

        if held_out_q == 0:
            losses, wins, detail = 0, 0, []
            for t in fold_tile_results:
                baseline = BEST_EXISTING_BASELINE_FOLD0[t["tile"]]
                cnn = t["icesat2_cnn_rmse_m"]
                win = cnn is not None and cnn < baseline
                wins += int(win)
                losses += int(not win)
                detail.append({"tile": t["tile"], "cnn_rmse": cnn, "best_existing_baseline": baseline, "cnn_wins": win})
            n = len(fold_tile_results)
            dem_wins = sum(t["dem_cnn_rmse_m"] < t["dem_linear_rmse_m"] for t in fold_tile_results)
            print(f"\n{'='*72}\nTEST B STAGE 1 GATE (fold 0): wins {wins}/{n}, losses {losses}/{n} "
                  f"(vs. each tile's own best pre-existing baseline)\n{'='*72}")
            for d in detail:
                print(f"  {d['tile']:12s} cnn={d['cnn_rmse']:.2f}m  best_existing={d['best_existing_baseline']:.2f}m  "
                      f"{'WIN' if d['cnn_wins'] else 'loss'}")
            (OUTDIR / "stage1_gate.json").write_text(json.dumps({
                "wins": wins, "losses": losses, "n": n, "dem_win_count": f"{dem_wins}/{n}", "detail": detail,
            }, indent=2))
            if losses > n / 2:
                signature = ("memorization (wins the Open-Buildings-target check, loses ICESat-2)"
                            if dem_wins > n / 2 else
                            "plain underperformance (loses both checks)")
                print(f"\nSTOP CONDITION MET: {losses}/{n} tiles lose to their own best pre-existing baseline.")
                print(f"Failure signature: {signature}")
                print(f"DEM-equivalent (Open-Buildings-target) win count: {dem_wins}/{n}")
                return
            else:
                print(f"\nFold 0 CLEARS the bar ({wins}/{n} wins). Proceeding to folds 1-3.")

    wins_dem = sum(t["dem_cnn_rmse_m"] < t["dem_linear_rmse_m"] for f in fold_results for t in f["tiles"])
    total = sum(len(f["tiles"]) for f in fold_results)
    output = {
        "config": {"epochs": EPOCHS, "tiles": URBAN_TILES},
        "dem_win_count": f"{wins_dem}/{total}",
        "overall_dem_cnn_rmse_m": float(np.mean([f["dem_cnn_rmse_m"] for f in fold_results])),
        "overall_dem_linear_rmse_m": float(np.mean([f["dem_linear_rmse_m"] for f in fold_results])),
        "overall_icesat2_cnn_rmse_m": float(np.mean([f["icesat2_cnn_rmse_m"] for f in fold_results])),
        "overall_icesat2_linear_rmse_m": float(np.mean([f["icesat2_linear_rmse_m"] for f in fold_results])),
        "total_time_sec": time.time() - t0,
        "folds": fold_results,
    }
    (OUTDIR / "method4_ob_testb_results.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"\n{'='*78}\nFINAL RESULT (all requested folds)\n{'='*78}")
    for k, v in output.items():
        if k != "folds":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
