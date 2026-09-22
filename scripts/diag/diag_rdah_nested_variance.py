import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("rdah_cv", "scripts/train_rdah_spatial_cv.py")
cv_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_mod)

PROJECT_ROOT = Path(".").resolve()
OUT_DIR = PROJECT_ROOT / "data/dfc2019/experiments/rdah"
device = torch.device("cpu")
HeightPredTransformer, _ = cv_mod.import_rdah_model_class()

SEED = 42
rng = np.random.RandomState(SEED)

# ============ ITEM 8: variance-compression check, fold0 epoch3 vs epoch5 ============
print("=" * 70)
print("ITEM 8: variance-compression check (fold0, epoch3 vs epoch5)")
print("=" * 70)

fold0_result = json.load(open(OUT_DIR / "fold0" / "result.json"))
test_tiles_f0 = fold0_result["test_tiles"]

def load_epoch_model(fold, epoch):
    model = HeightPredTransformer()
    ckpt = torch.load(OUT_DIR / f"fold{fold}" / f"epoch{epoch}.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    return model

def pooled_pred_true(model, tiles):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for t in tiles:
            depth, rgb, target, mask = cv_mod.load_sample(t)
            depth = depth.unsqueeze(0).to(device)
            rgb = rgb.unsqueeze(0).to(device)
            pred = model(depth, rgb).detach().cpu()
            if tuple(pred.shape[-2:]) != (1024, 1024):
                pred = torch.nn.functional.interpolate(pred, size=(1024, 1024), mode="bilinear", align_corners=False)
            pred_np = pred.squeeze(0).squeeze(0).numpy()
            target_np = target.numpy().squeeze(0)
            mask_np = mask.numpy().squeeze(0).astype(bool)
            preds.append(pred_np[mask_np])
            trues.append(target_np[mask_np])
    return np.concatenate(preds), np.concatenate(trues)

for epoch in [3, 5]:
    model = load_epoch_model(0, epoch)
    p, y = pooled_pred_true(model, test_tiles_f0)
    var_ratio = np.var(p) / np.var(y)
    slope = np.polyfit(y, p, 1)[0]  # slope of pred ~ true (Method4 convention: slope of prediction vs true)
    print(f"epoch{epoch}: pred mean={p.mean():.4f} std={p.std():.4f} | true mean={y.mean():.4f} std={y.std():.4f}")
    print(f"          variance ratio (pred/true) = {var_ratio:.4f} | OLS slope(pred~true) = {slope:.4f}")
    print(f"          p1-p99 pred range = {np.percentile(p,99)-np.percentile(p,1):.4f}  true range = {np.percentile(y,99)-np.percentile(y,1):.4f}")
    print()

# ============ ITEM 7: nested checkpoint-selection validation, all 4 folds ============
print("=" * 70)
print("ITEM 7: nested validation (inner-val epoch selection vs true held-out test)")
print("=" * 70)

for fold in range(4):
    result = json.load(open(OUT_DIR / f"fold{fold}" / "result.json"))
    test_tiles = result["test_tiles"]
    n = len(test_tiles)
    idx = list(range(n))
    rng2 = np.random.RandomState(SEED + fold)
    rng2.shuffle(idx)
    half = max(1, n // 2)
    inner_val_tiles = [test_tiles[i] for i in idx[:half]]
    true_test_tiles = [test_tiles[i] for i in idx[half:]]
    if not true_test_tiles:  # guard for very small folds
        true_test_tiles = inner_val_tiles

    print(f"\n--- FOLD {fold}: {n} test tiles -> inner_val={len(inner_val_tiles)}, true_test={len(true_test_tiles)} ---")

    inner_mae = {}
    true_mae = {}
    for epoch in range(1, 6):
        model = load_epoch_model(fold, epoch)
        p_iv, y_iv = pooled_pred_true(model, inner_val_tiles)
        mae_iv = float(np.mean(np.abs(p_iv - y_iv)))
        p_tt, y_tt = pooled_pred_true(model, true_test_tiles)
        mae_tt = float(np.mean(np.abs(p_tt - y_tt)))
        pearson_tt = float(np.corrcoef(p_tt, y_tt)[0, 1])
        inner_mae[epoch] = mae_iv
        true_mae[epoch] = (mae_tt, pearson_tt)
        print(f"  epoch{epoch}: inner_val MAE={mae_iv:.4f} | true_test MAE={mae_tt:.4f} Pearson={pearson_tt:+.4f}")

    selected_epoch = min(inner_mae, key=inner_mae.get)
    oracle_epoch = min(true_mae, key=lambda e: true_mae[e][0])
    default_epoch = 5

    print(f"  -> NESTED-SELECTED epoch (by inner_val MAE): {selected_epoch}  "
          f"-> true_test MAE={true_mae[selected_epoch][0]:.4f} Pearson={true_mae[selected_epoch][1]:+.4f}")
    print(f"  -> ORACLE-BEST epoch (peeked at true_test):   {oracle_epoch}  "
          f"-> true_test MAE={true_mae[oracle_epoch][0]:.4f} Pearson={true_mae[oracle_epoch][1]:+.4f}")
    print(f"  -> DEFAULT epoch5 (PDF's reported number):    5  "
          f"-> true_test MAE={true_mae[default_epoch][0]:.4f} Pearson={true_mae[default_epoch][1]:+.4f}")
