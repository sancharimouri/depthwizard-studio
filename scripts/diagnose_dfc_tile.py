#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TILE_ID = sys.argv[1] if len(sys.argv) > 1 else "JAX_022_009"

RGB_PATH = (
    PROJECT_ROOT
    / "data/dfc2019/raw/RGB/Track1-RGB"
    / f"{TILE_ID}_RGB.tif"
)

AGL_PATH = (
    PROJECT_ROOT
    / "data/dfc2019/raw/Truth/Track1-Truth"
    / f"{TILE_ID}_AGL.tif"
)

DEPTH_PATH = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/dav2_baseline/depth"
    / f"{TILE_ID}_depth.npy"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/dav2_baseline/diagnostics"
    / TILE_ID
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def corr(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)

    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)

    p = float(pearsonr(x, y).statistic)
    s = float(spearmanr(x, y).statistic)

    return p, s, len(x)


def print_stats(name: str, values: np.ndarray):
    values = values[np.isfinite(values)]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"count   : {values.size:,}")
    print(f"min     : {values.min():.6f}")
    print(f"p01     : {np.percentile(values, 1):.6f}")
    print(f"p05     : {np.percentile(values, 5):.6f}")
    print(f"median  : {np.median(values):.6f}")
    print(f"p95     : {np.percentile(values, 95):.6f}")
    print(f"p99     : {np.percentile(values, 99):.6f}")
    print(f"max     : {values.max():.6f}")
    print(f"mean    : {values.mean():.6f}")
    print(f"std     : {values.std():.6f}")


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

if not RGB_PATH.exists():
    raise FileNotFoundError(RGB_PATH)

if not AGL_PATH.exists():
    raise FileNotFoundError(AGL_PATH)

if not DEPTH_PATH.exists():
    raise FileNotFoundError(DEPTH_PATH)

with rasterio.open(RGB_PATH) as src:
    rgb = src.read().astype(np.float32)

# rasterio gives (bands, H, W)
rgb = np.transpose(rgb, (1, 2, 0))

# Normalize only for display.
rgb_display = rgb / 255.0
rgb_display = np.clip(rgb_display, 0, 1)

agl = rasterio.open(AGL_PATH).read(1).astype(np.float32)
dav2 = np.load(DEPTH_PATH).astype(np.float32)

if dav2.shape != agl.shape:
    raise ValueError(
        f"Shape mismatch: DAv2={dav2.shape}, AGL={agl.shape}"
    )

print("=" * 70)
print(f"DFC2019 TILE DIAGNOSTIC: {TILE_ID}")
print("=" * 70)

print(f"RGB shape : {rgb.shape}")
print(f"AGL shape : {agl.shape}")
print(f"DAv2 shape: {dav2.shape}")

print_stats("AGL", agl)
print_stats("DAv2", dav2)

# ---------------------------------------------------------------------
# Basic pixel statistics
# ---------------------------------------------------------------------

finite = np.isfinite(agl) & np.isfinite(dav2)

print("\nVALIDITY")
print("--------")
print(f"finite pixels : {finite.sum():,}")
print(f"total pixels  : {finite.size:,}")
print(f"valid %       : {100 * finite.mean():.2f}%")
print(f"AGL < 0 %     : {100 * np.mean(agl < 0):.2f}%")
print(f"AGL > 0 %     : {100 * np.mean(agl > 0):.2f}%")
print(f"AGL > 1m %    : {100 * np.mean(agl > 1):.2f}%")
print(f"AGL > 2m %    : {100 * np.mean(agl > 2):.2f}%")

# ---------------------------------------------------------------------
# Correlations under different masks
# ---------------------------------------------------------------------

masks = {
    "all_finite": finite,
    "agl_gt_0m": finite & (agl > 0),
    "agl_gt_1m": finite & (agl > 1),
    "agl_gt_2m": finite & (agl > 2),
    "agl_gt_5m": finite & (agl > 5),
}

rows = []

for name, mask in masks.items():
    p, s, n = corr(dav2[mask], agl[mask])

    rows.append({
        "condition": name,
        "pixels": n,
        "pixel_pct": 100 * n / finite.size,
        "pearson": p,
        "spearman": s,
    })

    print(
        f"{name:12s} | "
        f"pixels={n:8,d} | "
        f"Pearson={p: .6f} | "
        f"Spearman={s: .6f}"
    )

# ---------------------------------------------------------------------
# Light smoothing diagnostic
#
# This is NOT our official benchmark.
# It only tells us whether pixel-scale noise is affecting the score.
# ---------------------------------------------------------------------

dav2_blur = gaussian_filter(dav2, sigma=2)
agl_blur = gaussian_filter(agl, sigma=2)

p_blur, s_blur, n_blur = corr(
    dav2_blur[finite],
    agl_blur[finite],
)

print("\nSMOOTHED DIAGNOSTIC")
print("-------------------")
print(f"Gaussian sigma=2")
print(f"Pearson : {p_blur:.6f}")
print(f"Spearman: {s_blur:.6f}")

rows.append({
    "condition": "gaussian_sigma2_both",
    "pixels": n_blur,
    "pixel_pct": 100 * n_blur / finite.size,
    "pearson": p_blur,
    "spearman": s_blur,
})

# Save correlation table
corr_df = pd.DataFrame(rows)
corr_df.to_csv(OUT_DIR / "correlation_diagnostics.csv", index=False)

# ---------------------------------------------------------------------
# AGL display stretch
#
# Use robust percentiles so floating-point heights are visually useful.
# ---------------------------------------------------------------------

agl_low = np.percentile(agl[np.isfinite(agl)], 1)
agl_high = np.percentile(agl[np.isfinite(agl)], 99)

agl_display = np.clip(
    (agl - agl_low) / (agl_high - agl_low),
    0,
    1,
)

dav2_display = np.clip(dav2, 0, 1)

# ---------------------------------------------------------------------
# Three-panel visual comparison
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb_display)
axes[0].set_title("RGB")
axes[0].axis("off")

axes[1].imshow(dav2_display, cmap="gray")
axes[1].set_title("DAv2 relative depth")
axes[1].axis("off")

axes[2].imshow(agl_display, cmap="gray")
axes[2].set_title(
    f"AGL (1–99% stretch)\n"
    f"range {agl_low:.2f}–{agl_high:.2f} m"
)
axes[2].axis("off")

plt.tight_layout()
fig.savefig(
    OUT_DIR / "rgb_dav2_agl_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# ---------------------------------------------------------------------
# Scatter plot
#
# Randomly sample to keep the plot readable.
# ---------------------------------------------------------------------

rng = np.random.default_rng(42)

x = dav2[finite].astype(np.float32)
y = agl[finite].astype(np.float32)

sample_n = min(100_000, len(x))
indices = rng.choice(len(x), size=sample_n, replace=False)

xs = x[indices]
ys = y[indices]

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    xs,
    ys,
    s=2,
    alpha=0.15,
)

ax.set_xlabel("DAv2 relative depth")
ax.set_ylabel("AGL (m)")
ax.set_title(
    f"{TILE_ID}: DAv2 vs AGL\n"
    f"Pearson={corr(dav2[finite], agl[finite])[0]:.3f}, "
    f"Spearman={corr(dav2[finite], agl[finite])[1]:.3f}"
)

ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(
    OUT_DIR / "dav2_vs_agl_scatter.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

print("\nOUTPUTS")
print("-------")
print(OUT_DIR / "correlation_diagnostics.csv")
print(OUT_DIR / "rgb_dav2_agl_comparison.png")
print(OUT_DIR / "dav2_vs_agl_scatter.png")
