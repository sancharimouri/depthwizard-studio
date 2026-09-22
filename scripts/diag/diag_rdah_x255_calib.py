import sys
from pathlib import Path
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import stats

from run_rdah_probe import load_rdah_module, load_rgb, preprocess_rgb, CHECKPOINT

PROJECT_ROOT = Path(".").resolve()
manifest = pd.read_csv("data/dfc2019/experiments/dav2_baseline/manifest.csv").set_index("tile_id")
TILES = ["JAX_004_006", "JAX_149_006", "JAX_264_013", "OMA_248_029"]
SCALE = 255
SEED = 42

module = load_rdah_module()
model = module.HeightPredTransformer()
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))

all_mae, all_rmse, all_pearson = [], [], []
for tile in TILES:
    rgb_path = PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB" / f"{tile}_RGB.tif"
    depth_path = PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth" / f"{tile}_depth.npy"
    agl_path = PROJECT_ROOT / manifest.loc[tile, "agl_path"]

    rgb = load_rgb(rgb_path)
    depth = np.load(depth_path).astype(np.float32) * SCALE
    with rasterio.open(agl_path) as src:
        agl = src.read(1).astype(np.float32)
    valid = np.isfinite(agl) & (agl != -9999.0)
    rgb_t = preprocess_rgb(rgb)
    depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(depth_t, rgb_t).detach().cpu().float().squeeze().numpy()

    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    pred_flat, agl_flat = pred.ravel(), agl.ravel()
    lr = stats.linregress(pred_flat[train_idx], agl_flat[train_idx])
    a, b = lr.slope, lr.intercept
    calibrated = a * pred_flat[test_idx] + b

    m = mae(calibrated, agl_flat[test_idx])
    r = rmse(calibrated, agl_flat[test_idx])
    p, _ = stats.pearsonr(pred_flat[test_idx], agl_flat[test_idx])
    sp, _ = stats.spearmanr(pred_flat[test_idx], agl_flat[test_idx])
    print(f"{tile:15s} held-out(50%) affine-calibrated: MAE={m:.4f} RMSE={r:.4f} Pearson={p:+.4f} Spearman={sp:+.4f}  (slope={a:.4f} intercept={b:.4f})")
    all_mae.append(m); all_rmse.append(r); all_pearson.append(p)

print(f"\nMean across 4 tiles: MAE={np.mean(all_mae):.4f} RMSE={np.mean(all_rmse):.4f} Pearson={np.mean(all_pearson):.4f}")
print("\nFor reference, this project's already-established DFC2019 baselines:")
print("  Method 2 (20-anchor global calib): MAE 3.13m RMSE 4.48m Pearson 0.526")
print("  oracle per-tile-OLS baseline:      MAE 3.39m RMSE 4.58m (per CLAUDE.md/Method 6 verdict)")
print("  Method 6 (current best):            MAE 1.98m RMSE 3.49m Pearson (fold-avg, see verdict.md)")
