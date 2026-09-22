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
SCALES = [1, 10, 30, 50, 65, 100, 150, 200, 255, 300, 500, 1000]

module = load_rdah_module()
model = module.HeightPredTransformer()
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

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

    print(f"=== {tile} ===")
    for scale in SCALES:
        depth_scaled = depth * scale
        depth_t = torch.from_numpy(depth_scaled.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(depth_t, rgb_t).detach().cpu().float().squeeze().numpy()
        pearson, _ = stats.pearsonr(pred[valid], agl[valid])
        spearman, _ = stats.spearmanr(pred[valid], agl[valid])
        print(f"  x{scale:<6} pred mean={pred.mean():10.4f} std={pred.std():10.4f}  Pearson={pearson:+.4f} Spearman={spearman:+.4f}")
    print()
