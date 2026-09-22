import sys
from pathlib import Path
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import stats

from run_rdah_probe import load_rdah_module, load_rgb, preprocess_rgb, preprocess_depth

PROJECT_ROOT = Path(".").resolve()
manifest = pd.read_csv("data/dfc2019/experiments/dav2_baseline/manifest.csv").set_index("tile_id")
TILES = ["JAX_004_006", "JAX_149_006", "JAX_264_013", "OMA_248_029"]

CKPTS = {
    "Track1 (104best, used throughout)": PROJECT_ROOT / "external/RDAH-Net/104best_model.pth",
    "HK": Path("/tmp/rdah_checkpoints/hk_best_model.pth"),
    "Swiss": Path("/tmp/rdah_checkpoints/swiss_best_model.pth"),
}

module = load_rdah_module()

for ckpt_name, ckpt_path in CKPTS.items():
    print(f"\n{'='*70}\n{ckpt_name}: {ckpt_path}\n{'='*70}")
    model = module.HeightPredTransformer()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else "NOT A DICT"
    print("checkpoint top-level keys:", keys)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if isinstance(checkpoint, dict):
        print(f"epoch={checkpoint.get('epoch')} loss={checkpoint.get('loss')}")
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
        depth_t = preprocess_depth(depth)
        with torch.no_grad():
            pred = model(depth_t, rgb_t).detach().cpu().float().squeeze().numpy()

        pearson, _ = stats.pearsonr(pred[valid], agl[valid])
        spearman, _ = stats.spearmanr(pred[valid], agl[valid])
        print(f"  {tile:15s} pred mean={pred.mean():9.5f} std={pred.std():9.5f}  Pearson={pearson:+.4f} Spearman={spearman:+.4f}")
