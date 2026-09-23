#!/usr/bin/env python3
"""
RS3DAda feasibility smoke test (docs/method-audit/stage0-gates/rs3dada-audit.md).

The official external/SynRS3D/infer_height.py hardcodes .cuda() in several
places (model.cuda(), input.cuda() in the patch loop) and would fail as-is
on a non-CUDA machine. Rather than patching that file in place, this is a
minimal (~60-line) standalone script that imports the same DPT_DINOv2 model
class and the same preprocessing (Normalize + ToTensorV2) but uses
device-agnostic .to(device) with MPS/CPU fallback, and reads the input tile
via rasterio instead of gdal (gdal isn't installed in this venv and wasn't
needed for a single-tile shape/range check).

Loads the real, hash-verified RS3DAda_vitl_DPT_height.pth checkpoint and
runs one forward pass on a 1022x1022 crop of JAX_004_006 -- one of this
project's own 50 benchmark tiles, which is a Target-Domain training tile in
RS3DAda's own split files (see the audit doc; this is NOT a zero-shot test).
"""
import sys, time
sys.path.insert(0, "/Users/anweshasaha/projects/DepthWizard2/external/SynRS3D")

import numpy as np
import torch
import rasterio
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

t0 = time.time()

from models.dpt import DPT_DINOv2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

head_configs = [
    {'name': 'regression', 'nclass': 1},
    {'name': 'segmentation', 'nclass': 8},
]

print("Building model (torch.hub.load facebookresearch/dinov2)...")
model = DPT_DINOv2(encoder='vitl', head_configs=head_configs, pretrained=False)
print(f"Model built in {time.time()-t0:.1f}s")

ckpt_path = "/Users/anweshasaha/projects/DepthWizard2/external/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing[:10]:", missing[:10])
if unexpected:
    print("  unexpected[:10]:", unexpected[:10])

model.eval()
model.to(device)
print(f"Model on {device}")

# Read JAX_004_006 RGB via rasterio (avoids gdal dependency).
rgb_path = "/Users/anweshasaha/projects/DepthWizard2/data/dfc2019/raw/RGB/Track1-RGB/JAX_004_006_RGB.tif"
with rasterio.open(rgb_path) as src:
    img = src.read([1, 2, 3])  # (3, H, W)
img = np.transpose(img, (1, 2, 0)).astype(np.float32)  # (H, W, 3)
print("Loaded RGB:", img.shape, img.dtype, "min/max:", img.min(), img.max())

patch_size = 1022  # divisible by 14, matches infer_height.py default
crop = img[:patch_size, :patch_size, :].copy()
print("Crop:", crop.shape)

transform = Compose([
    Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
    ToTensorV2(),
])
x = transform(image=crop)['image'].unsqueeze(0).float().to(device)
print("Input tensor:", x.shape, x.dtype, x.device)

t1 = time.time()
with torch.no_grad():
    outputs = model(x)
print(f"Forward pass in {time.time()-t1:.1f}s")

print("Output keys:", list(outputs.keys()))
pred = outputs.get('regression', None)
print("regression output type:", type(pred))
if pred is not None:
    pred_np = pred.detach().cpu().numpy().squeeze()
    print("Pred shape:", pred_np.shape, "dtype:", pred_np.dtype)
    print("Finite:", np.isfinite(pred_np).all(), "  NaN count:", np.isnan(pred_np).sum(), "  Inf count:", np.isinf(pred_np).sum())
    print(f"min={pred_np.min():.4f} max={pred_np.max():.4f} mean={pred_np.mean():.4f} std={pred_np.std():.4f}")
    print(f"p1={np.percentile(pred_np,1):.4f} p50={np.percentile(pred_np,50):.4f} p99={np.percentile(pred_np,99):.4f}")

print(f"\nTotal script time: {time.time()-t0:.1f}s")
