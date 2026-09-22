import sys
from pathlib import Path
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import stats

from run_rdah_probe import load_rdah_module, load_rgb, preprocess_rgb, preprocess_depth, CHECKPOINT

PROJECT_ROOT = Path(".").resolve()
manifest = pd.read_csv("data/dfc2019/experiments/dav2_baseline/manifest.csv").set_index("tile_id")

TILES = ["JAX_004_006", "JAX_149_006", "JAX_264_013", "OMA_248_029"]

module = load_rdah_module()
model = module.HeightPredTransformer()
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

# "global" AGL mean/std across all 50 of our benchmark tiles (candidate fixed-constant denorm)
global_mean = manifest["agl_mean"].mean()
global_std = manifest["agl_std"].mean()
print(f"Global (avg-of-tile) AGL mean/std candidate: mean={global_mean:.4f} std={global_std:.4f}\n")

for tile in TILES:
    rgb_path = PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB" / f"{tile}_RGB.tif"
    depth_path = PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth" / f"{tile}_depth.npy"
    agl_path = PROJECT_ROOT / manifest.loc[tile, "agl_path"]

    rgb = load_rgb(rgb_path)
    depth = np.load(depth_path).astype(np.float32)
    with rasterio.open(agl_path) as src:
        agl = src.read(1).astype(np.float32)
    valid = np.isfinite(agl) & (agl != -9999.0)

    rgb_t = preprocess_rgb(rgb)
    depth_t = preprocess_depth(depth)
    with torch.no_grad():
        pred = model(depth_t, rgb_t).detach().cpu().float().squeeze().numpy()

    raw_pearson, _ = stats.pearsonr(pred[valid], agl[valid])
    raw_spearman, _ = stats.spearmanr(pred[valid], agl[valid])

    tile_mean, tile_std = manifest.loc[tile, "agl_mean"], manifest.loc[tile, "agl_std"]
    denorm_oracle = pred * tile_std + tile_mean
    denorm_global = pred * global_std + global_mean

    oracle_pearson, _ = stats.pearsonr(denorm_oracle[valid], agl[valid])
    oracle_mae = np.mean(np.abs(denorm_oracle[valid] - agl[valid]))
    global_pearson, _ = stats.pearsonr(denorm_global[valid], agl[valid])
    global_mae = np.mean(np.abs(denorm_global[valid] - agl[valid]))
    raw_mae = np.mean(np.abs(pred[valid] - agl[valid]))

    print(f"=== {tile} ===")
    print(f"  true AGL: mean={agl[valid].mean():.4f} std={agl[valid].std():.4f} range=[{agl[valid].min():.2f},{agl[valid].max():.2f}]")
    print(f"  raw pred: mean={pred.mean():.4f} std={pred.std():.4f}")
    print(f"  RAW pred vs AGL:            Pearson={raw_pearson:.4f} Spearman={raw_spearman:.4f} MAE={raw_mae:.4f}")
    print(f"  denorm(oracle tile mean/std): pred*{tile_std:.3f}+{tile_mean:.3f} -> Pearson={oracle_pearson:.4f} MAE={oracle_mae:.4f}")
    print(f"  denorm(global mean/std):      pred*{global_std:.3f}+{global_mean:.3f} -> Pearson={global_pearson:.4f} MAE={global_mae:.4f}")
    print()
