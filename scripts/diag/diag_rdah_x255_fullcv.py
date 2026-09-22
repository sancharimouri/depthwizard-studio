import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import stats

sys.path.insert(0, "scripts")
from run_rdah_probe import load_rdah_module, load_rgb, preprocess_rgb, CHECKPOINT

# reuse train_rdah_spatial_cv.py's exact fold-construction logic
spec = importlib.util.spec_from_file_location("rdah_cv", "scripts/train_rdah_spatial_cv.py")
cv_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_mod)

PROJECT_ROOT = Path(".").resolve()
manifest = pd.read_csv("data/dfc2019/experiments/dav2_baseline/manifest.csv").set_index("tile_id")
tiles = sorted(manifest.index.tolist())
SCALE = 255
SEED = 42
SUBSAMPLE_PER_TILE = 5000  # match Method 4's baseline-pooling convention

folds = cv_mod.make_spatial_folds(tiles)
for k, v in folds.items():
    print(f"fold {k}: {len(v)} tiles")

module = load_rdah_module()
model = module.HeightPredTransformer()
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))

# Run zero-shot inference (scaled input) once per tile, cache flattened valid pred/true.
cache = {}
rng_global = np.random.RandomState(SEED)
for i, tile in enumerate(tiles, 1):
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

    pred_valid = pred[valid]
    agl_valid = agl[valid]
    cache[tile] = (pred_valid, agl_valid)
    if i % 10 == 0 or i == len(tiles):
        print(f"  inferred {i}/{len(tiles)} tiles")

# Per-fold: pool subsampled TRAIN pixels -> fit one global OLS affine -> eval on FULL test-fold pixels
fold_results = []
for fold_id in range(4):
    test_tiles = folds[fold_id]
    train_tiles = [t for f in range(4) if f != fold_id for t in folds[f]]

    train_pred, train_true = [], []
    for t in train_tiles:
        p, y = cache[t]
        n = len(p)
        if n > SUBSAMPLE_PER_TILE:
            idx = rng_global.choice(n, SUBSAMPLE_PER_TILE, replace=False)
            p, y = p[idx], y[idx]
        train_pred.append(p); train_true.append(y)
    train_pred = np.concatenate(train_pred)
    train_true = np.concatenate(train_true)

    lr = stats.linregress(train_pred, train_true)
    a, b = lr.slope, lr.intercept

    test_pred, test_true = [], []
    for t in test_tiles:
        p, y = cache[t]
        test_pred.append(p); test_true.append(y)
    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)
    calibrated = a * test_pred + b

    m = mae(calibrated, test_true)
    r = rmse(calibrated, test_true)
    pear, _ = stats.pearsonr(test_pred, test_true)
    sp, _ = stats.spearmanr(test_pred, test_true)
    print(f"\nFOLD {fold_id}: train={len(train_tiles)} tiles / test={len(test_tiles)} tiles "
          f"(pooled global affine slope={a:.4f} intercept={b:.4f})")
    print(f"  MAE={m:.4f} RMSE={r:.4f} Pearson={pear:+.4f} Spearman={sp:+.4f}  n_test_px={len(test_true)}")
    fold_results.append({"fold": fold_id, "mae": m, "rmse": r, "pearson": pear, "spearman": sp, "n": len(test_true)})

df = pd.DataFrame(fold_results)
print("\n" + "=" * 70)
print("EQUAL-WEIGHT FOLD-AVERAGE (matches PDF's own averaging convention):")
print(df[["mae", "rmse", "pearson", "spearman"]].mean())

# pooled/pixel-weighted too, for completeness
total_n = df["n"].sum()
weighted = (df[["mae","rmse","pearson","spearman"]].multiply(df["n"], axis=0)).sum() / total_n
print("\nPIXEL-WEIGHTED AVERAGE:")
print(weighted)

df.to_csv("/private/tmp/claude-501/-Users-anweshasaha-projects-DepthWizard2/fa0035ab-aff5-4ffe-9847-a473b6cae7a5/scratchpad/rdah_x255_zeroshot_fold_results.csv", index=False)
