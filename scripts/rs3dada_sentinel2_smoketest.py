#!/usr/bin/env python3
"""
RS3DAda genuine zero-shot smoke test on real Sentinel-2 India tiles
(docs/method-audit/stage0-gates/rs3dada-audit.md).

Unlike scripts/rs3dada_mps_smoketest.py (which ran on JAX_004_006, a
DFC2019 tile confirmed to be in RS3DAda's own training split — see that
audit doc's contamination section), this IS a genuine zero-shot test:
Sentinel-2 India was never in SynRS3D's training domains (SynRS3D's own
README lists its sources; India Sentinel-2 at 10m GSD is not among them).
No contamination caveat needed here.

This is a few-tiles smoke test only, not a benchmark: one forward pass per
region, no metrics against ground truth (none exists for these regions'
"absolute elevation" -- that's exactly the capability this project's own
CLAUDE.md notes was never established), just shape/finite/range checks and
a distribution comparison against what RS3DAda's own training domain
implies.

Key thing being checked: SynRS3D's training GSD is 0.05-1 m/pixel (its own
README). These Sentinel-2 tiles are 10 m/pixel -- 10x to 200x coarser than
anything the model was trained on. At 10m GSD, individual buildings
(typically 5-30m footprints) collapse to 1-3 pixels, roads disappear
entirely, and the fine texture RS3DAda's height head relies on to infer
structure is simply not present in the input. The likely failure mode is a
near-flat, low-variance, or otherwise degenerate output -- not because the
checkpoint is broken, but because the input has been effectively
low-pass-filtered relative to what the model expects. That's the specific
thing this script's stats are checking for, not just "does it run."

Reuses the same model class, checkpoint, and preprocessing as the DFC2019
smoke test -- no changes to the model or inference code, only the input.
"""
import sys, time
sys.path.insert(0, "/Users/anweshasaha/projects/DepthWizard2/external/SynRS3D")

import numpy as np
import torch
import rasterio
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from models.dpt import DPT_DINOv2

REPO_ROOT = "/Users/anweshasaha/projects/DepthWizard2"
CKPT_PATH = f"{REPO_ROOT}/external/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth"

# The 4 production Sentinel-2 regions -- all now have complete asset sets
# (real 10m/3-band/EPSG:32645-or-32646 GeoTIFFs), so this covers hilly
# (Darjeeling), urban (Kolkata), agricultural/flat (Bardhaman), and
# coastal/deltaic (Sundarbans) in one pass -- "a few tiles", not all four
# treated as a benchmark.
REGIONS = {
    "darjeeling": f"{REPO_ROOT}/data/sentinel2/darjeeling/Darjeeling_RGB.tif",
    "kolkata": f"{REPO_ROOT}/data/sentinel2/kolkata/Kolkata_RGB.tif",
    "bardhaman": f"{REPO_ROOT}/data/sentinel2/bardhaman/Bardhaman_RGB.tif",
    "sundarbans": f"{REPO_ROOT}/data/sentinel2/sundarbans/Sundarbans_RGB.tif",
}

PATCH_SIZE = 1022  # divisible by 14 (ViT patch size), matches the DFC2019 smoke test

# RS3DAda-on-DFC2019 output stats from the original (contaminated) smoke
# test, for a side-by-side reference point only -- not a ground truth.
DFC2019_REFERENCE = {
    "mean": 2.75, "p50": 1.25, "p99": 11.42, "min": -1.03,
}


def build_model(device: torch.device) -> torch.nn.Module:
    head_configs = [
        {"name": "regression", "nclass": 1},
        {"name": "segmentation", "nclass": 8},
    ]
    print("Building model (torch.hub.load facebookresearch/dinov2)...")
    t0 = time.time()
    model = DPT_DINOv2(encoder="vitl", head_configs=head_configs, pretrained=False)
    print(f"Model built in {time.time() - t0:.1f}s")

    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    model.to(device)
    return model


def run_one(model: torch.nn.Module, device: torch.device, transform: Compose, region: str, rgb_path: str) -> dict:
    print(f"\n{'=' * 78}\n{region.upper()}\n{'=' * 78}")

    with rasterio.open(rgb_path) as src:
        img = src.read([1, 2, 3])  # (3, H, W)
        crs = src.crs
        res = src.res
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)  # (H, W, 3)
    print(f"Loaded RGB: {img.shape} {img.dtype}  min/max: {img.min():.1f}/{img.max():.1f}  crs={crs}  res={res}")

    h, w = img.shape[:2]
    # ViT patch size is 14 -- input H/W must be a multiple of it. Darjeeling's
    # tile (1004x1118) is smaller than the fixed 1022 crop used for the other
    # three regions in one dimension, so use the largest multiple-of-14 patch
    # that actually fits this tile instead of skipping it. Not identical crop
    # sizes across regions, but this is a per-tile smoke test, not a
    # controlled-comparison benchmark -- noted in the per-region patch size
    # printed below.
    patch = min(PATCH_SIZE, (min(h, w) // 14) * 14)

    # Center crop, not top-left, so the test patch isn't dominated by a
    # single edge of the AOI for tiles with strong spatial structure
    # (e.g. Darjeeling's ridge running through the tile).
    r0 = (h - patch) // 2
    c0 = (w - patch) // 2
    crop = img[r0:r0 + patch, c0:c0 + patch, :].copy()
    print(f"Center crop: {crop.shape} (patch={patch}, PATCH_SIZE default={PATCH_SIZE})")

    x = transform(image=crop)["image"].unsqueeze(0).float().to(device)

    t1 = time.time()
    with torch.no_grad():
        outputs = model(x)
    elapsed = time.time() - t1

    pred = outputs.get("regression", None)
    result = {"region": region, "skipped": False, "elapsed_sec": elapsed, "patch": patch}
    if pred is None:
        print("No 'regression' key in outputs -- inference wiring issue, not an input-domain issue.")
        result["no_regression_output"] = True
        return result

    pred_np = pred.detach().cpu().numpy().squeeze()
    finite = np.isfinite(pred_np)
    stats = {
        "shape": pred_np.shape,
        "dtype": str(pred_np.dtype),
        "all_finite": bool(finite.all()),
        "nan_count": int(np.isnan(pred_np).sum()),
        "inf_count": int(np.isinf(pred_np).sum()),
        "min": float(pred_np.min()),
        "max": float(pred_np.max()),
        "mean": float(pred_np.mean()),
        "std": float(pred_np.std()),
        "p1": float(np.percentile(pred_np, 1)),
        "p50": float(np.percentile(pred_np, 50)),
        "p99": float(np.percentile(pred_np, 99)),
    }
    result["stats"] = stats

    print(f"Forward pass in {elapsed:.1f}s")
    print(f"Pred shape: {stats['shape']}  dtype: {stats['dtype']}")
    print(f"Finite: {stats['all_finite']}  NaN: {stats['nan_count']}  Inf: {stats['inf_count']}")
    print(f"min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"p1={stats['p1']:.4f} p50={stats['p50']:.4f} p99={stats['p99']:.4f}")

    return result


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    model = build_model(device)

    transform = Compose([
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
        ToTensorV2(),
    ])

    results = []
    for region, path in REGIONS.items():
        results.append(run_one(model, device, transform, region, path))

    print(f"\n{'=' * 78}\nSUMMARY -- RS3DAda zero-shot on Sentinel-2 India (10m GSD)\n{'=' * 78}")
    print(f"RS3DAda training GSD: 0.05-1 m/pixel (SynRS3D README). Sentinel-2: 10 m/pixel "
          f"-- 10x to 200x coarser than anything in the training distribution.\n")

    print(f"{'region':<12} {'finite':>7} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'p50':>8}")
    for r in results:
        if r.get("skipped") or r.get("no_regression_output"):
            print(f"{r['region']:<12} SKIPPED/NO-OUTPUT")
            continue
        s = r["stats"]
        print(f"{r['region']:<12} {str(s['all_finite']):>7} {s['min']:>8.3f} {s['max']:>8.3f} "
              f"{s['mean']:>8.3f} {s['std']:>8.3f} {s['p50']:>8.3f}")

    print(f"\nFor reference, RS3DAda-on-DFC2019 (contaminated, ~0.3m GSD, from the "
          f"earlier audit): mean={DFC2019_REFERENCE['mean']} p50={DFC2019_REFERENCE['p50']} "
          f"p99={DFC2019_REFERENCE['p99']} min={DFC2019_REFERENCE['min']}")

    stds = [r["stats"]["std"] for r in results if r.get("stats")]
    if stds:
        low_variance = [r["region"] for r in results if r.get("stats") and r["stats"]["std"] < 0.5]
        print(f"\nStd-dev across regions: {[round(s, 3) for s in stds]}")
        if low_variance:
            print(f"LOW-VARIANCE (std<0.5m) regions -- consistent with the expected GSD-mismatch "
                  f"failure mode (near-flat output when input structure is coarser than training "
                  f"domain): {low_variance}")
        else:
            print("No region shows near-flat output -- the model is producing spatially-varying "
                  "predictions despite the GSD mismatch (does not by itself mean the values are "
                  "meaningful heights -- no ground truth exists here to check against).")


if __name__ == "__main__":
    main()
