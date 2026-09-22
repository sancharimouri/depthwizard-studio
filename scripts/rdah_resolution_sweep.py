#!/usr/bin/env python3
"""
Phase 2b (2026-09-23): RDAH-Net resolution sweep on DFC2019, Swiss checkpoint,
zero-shot. INFERENCE ONLY. Pre-registered rule:
docs/method-audit/sentinel2/sign-flip-detector.md, 2026-09-23 (continued),
"Phase 2 -- PRE-REGISTRATION".

Same model path as scripts/diag/diag_swiss_zeroshot_per_fold.py
(train_rdah_quadrant_cv.load_full_sample / import_rdah_model_class /
build_resized_positional_encoding, Swiss checkpoint, depth x255).
8 whole tiles (2 per make_spatial_folds fold), block-averaged by 1/2/4/8.
Primary depth: DAv2 re-run on the downsampled RGB (backend/depth/depth_engine).
Secondary: block-averaged 1x cached depth.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import stats
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rq = _imp("rq_sweep", ROOT / "scripts/train_rdah_quadrant_cv.py")
rs = _imp("rs_sweep", ROOT / "scripts/train_rdah_spatial_cv.py")
from rdah_sentinel2_zeroshot import periodic_score  # noqa: E402
from backend.depth import depth_engine  # noqa: E402

FACTORS = (1, 2, 4, 8)
SCALE = 255.0
GSD_1X_M = 0.3  # DFC2019 Track1 RGB (WorldView-3); the file's own transform carries no georeferencing
OUT = ROOT / "data/dfc2019/experiments/rdah_zeroshot/resolution_sweep"


def block_mean(a: np.ndarray, f: int, valid: np.ndarray | None = None, min_frac=0.5):
    if f == 1:
        return a.copy(), (valid.copy() if valid is not None else None)
    h, w = a.shape[:2]
    if a.ndim == 3:
        return a.reshape(h // f, f, w // f, f, a.shape[2]).mean(axis=(1, 3)), None
    v = valid.astype(np.float64) if valid is not None else np.ones_like(a, dtype=np.float64)
    num = np.where(v > 0, a, 0.0).reshape(h // f, f, w // f, f).sum(axis=(1, 3))
    den = v.reshape(h // f, f, w // f, f).sum(axis=(1, 3))
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out, den / (f * f) >= min_frac


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles_all = rq.load_manifest_tiles()
    folds = rs.make_spatial_folds(tiles_all)
    tiles = [t for k in range(4) for t in sorted(folds[k])[:2]]
    device = rq.select_device("auto")
    model_cls, PE = rq.import_rdah_model_class()
    ckpt = torch.load(rq.SWISS_CHECKPOINT, map_location="cpu")

    records, pooled = [], {v: {f: ([], []) for f in FACTORS} for v in ("dav2_rerun", "depth_blockavg")}
    for tile in tiles:
        with rasterio.open(rq.RGB_DIR / f"{tile}_RGB.tif") as s:
            rgb_raw = s.read([1, 2, 3]).transpose(1, 2, 0)
        rgb_u8 = np.clip(rgb_raw, 0, 255).astype(np.uint8) if rgb_raw.dtype != np.uint8 else rgb_raw
        depth_1x = np.load(rq.DEPTH_DIR / f"{tile}_depth.npy").astype(np.float32)
        with rasterio.open(rq.GT_DIR / f"{tile}_AGL.tif") as s:
            agl = s.read(1).astype(np.float32)
        valid = np.isfinite(agl) & (agl != -9999.0) & ~np.all(rgb_raw == 0, axis=2)

        for f in FACTORS:
            n = 1024 // f
            rgb_f = block_mean(rgb_u8.astype(np.float64), f)[0]
            rgb_f8 = np.clip(np.round(rgb_f), 0, 255).astype(np.uint8)
            agl_f, val_f = block_mean(np.where(valid, agl, np.nan), f, valid)
            dav2_rerun = depth_engine.run_inference(Image.fromarray(rgb_f8, mode="RGB"))
            depth_ba = block_mean(depth_1x, f)[0].astype(np.float32)
            if f == 1:
                repro = float(np.abs(dav2_rerun - depth_1x).max())

            model = model_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model = rq.build_resized_positional_encoding(model, PE, d_model=32, bottleneck=n // 16).to(device).eval()
            rgb_t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                transforms.ToTensor()(Image.fromarray(rgb_f8, mode="RGB"))).unsqueeze(0).to(device)
            for vname, dep in (("dav2_rerun", dav2_rerun), ("depth_blockavg", depth_ba)):
                dep_t = torch.from_numpy(dep * SCALE).float()[None, None].to(device)
                with torch.no_grad():
                    pred = model(dep_t, rgb_t).squeeze().cpu().numpy().astype(np.float64)
                p, y = pred[val_f], agl_f[val_f]
                cb_periods = [str(q) for q in (2, 4, 8, 16) if q <= n // 2]
                fft = periodic_score(pred)
                rec = {"tile": tile, "factor": f, "px": n, "gsd_m": GSD_1X_M * f, "depth": vname,
                       "n_valid": int(val_f.sum()),
                       "pearson": float(stats.pearsonr(p, y)[0]), "spearman": float(stats.spearmanr(p, y)[0]),
                       "variance_ratio": float(np.var(p) / np.var(y)),
                       "pred_mean": float(p.mean()), "pred_std": float(p.std()),
                       "cb": float(max(fft[q]["peak_to_background"] for q in cb_periods)),
                       "fft": {q: fft[q]["peak_to_background"] for q in fft if int(q) <= n // 2}}
                if f == 1 and vname == "dav2_rerun":
                    rec["dav2_rerun_vs_cached_maxabs"] = repro
                records.append(rec)
                pooled[vname][f][0].append(p)
                pooled[vname][f][1].append(y)
                print(f"{tile} x{f} ({n}px) {vname:14s} r={rec['pearson']:+.3f} rho={rec['spearman']:+.3f} "
                      f"vr={rec['variance_ratio']:.3f} CB={rec['cb']:.3g}", flush=True)

    summary = {"tiles": tiles, "gsd_1x_m_assumed": GSD_1X_M, "scale": SCALE, "by_variant": {}}
    for vname in pooled:
        rows = {}
        for f in FACTORS:
            P = np.concatenate(pooled[vname][f][0]); Y = np.concatenate(pooled[vname][f][1])
            recs = [r for r in records if r["depth"] == vname and r["factor"] == f]
            rows[str(f)] = {"px": 1024 // f, "gsd_m": GSD_1X_M * f,
                            "pooled_pearson": float(stats.pearsonr(P, Y)[0]),
                            "pooled_spearman": float(stats.spearmanr(P, Y)[0]),
                            "pooled_variance_ratio": float(np.var(P) / np.var(Y)),
                            "mean_tile_pearson": float(np.mean([r["pearson"] for r in recs])),
                            "median_cb": float(np.median([r["cb"] for r in recs]))}
        r1, r8 = rows["1"], rows["8"]
        rows["decision"] = {"pearson_8x_over_1x": r8["pooled_pearson"] / r1["pooled_pearson"],
                            "cb_8x_over_1x": r8["median_cb"] / r1["median_cb"],
                            "close_pearson": bool(r8["pooled_pearson"] < 0.5 * r1["pooled_pearson"]),
                            "close_cb": bool(r8["median_cb"] > 10 * r1["median_cb"])}
        summary["by_variant"][vname] = rows
    summary["dav2_rerun_1x_vs_cached_maxabs"] = max(r.get("dav2_rerun_vs_cached_maxabs", 0) for r in records)
    (OUT / "per_tile.json").write_text(json.dumps(records, indent=1) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
