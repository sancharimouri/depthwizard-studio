import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "scripts")
import importlib.util

def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

rdah_q = import_module("rdah_q_diag", Path("scripts/train_rdah_quadrant_cv.py"))

device = torch.device("cpu")
model_cls, PositionalEncoding = rdah_q.import_rdah_model_class()
tiles = rdah_q.load_manifest_tiles()

results = json.load(open("data/dfc2019/experiments/rdah_quadrant_cv/method_rdah_quadrant_cv_results.json")) \
    if Path("data/dfc2019/experiments/rdah_quadrant_cv/method_rdah_quadrant_cv_results.json").exists() else None

for held_out_q in range(4):
    fold_result = json.load(open(f"data/dfc2019/experiments/rdah_quadrant_cv/fold{held_out_q}/result.json"))
    scale = fold_result["scale"]
    test_pairs = [(t, held_out_q) for t in tiles]

    rng = np.random.RandomState(rdah_q.SEED + held_out_q)
    idx = list(range(len(test_pairs)))
    rng.shuffle(idx)
    half = max(1, len(idx) // 2)
    true_idx = idx[half:] or idx[:half]
    true_pairs = [test_pairs[i] for i in true_idx]

    true_samples = rdah_q.load_quadrant_samples(true_pairs)

    model = model_cls().to(device)
    ckpt = torch.load(rdah_q.SWISS_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = rdah_q.build_resized_positional_encoding(model, PositionalEncoding, d_model=32, bottleneck=32)
    model.to(device)

    ev = rdah_q.evaluate_samples(model, true_samples, scale, device)
    p, y = ev.pop("_pooled_pred"), ev.pop("_pooled_true")
    var_ratio = float(np.var(p) / np.var(y)) if np.var(y) > 0 else float("nan")
    slope = float(np.polyfit(y, p, 1)[0]) if len(y) > 1 else float("nan")
    print(f"fold{held_out_q} (scale=x{scale}) SWISS ZERO-SHOT on true-test half (n_samples={len(true_pairs)}): "
          f"MAE={ev['mae']:.4f} RMSE={ev['rmse']:.4f} Pearson={ev['pearson']:+.4f} Spearman={ev['spearman']:+.4f} "
          f"var_ratio={var_ratio:.4f} slope={slope:.4f}")
