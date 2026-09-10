from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "elevation"
    / "darjeeling"
    / "Darjeeling_OpenTopography_DSM.tif"
)

OUTPUT = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "darjeeling"
    / "elevation.png"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


import rasterio

with rasterio.open(INPUT) as src:
    elevation = src.read(1).astype(np.float32)


low = float(elevation.min())
high = float(elevation.max())

normalized = (
    elevation - low
) / (
    high - low
)

normalized = np.clip(normalized, 0, 1)


# Terrain-style colour ramp:
# low = blue/cyan
# middle = green/yellow
# high = orange/white

stops = np.array([
    [0.00, 0.08, 0.18, 0.55],
    [0.25, 0.00, 0.55, 0.75],
    [0.50, 0.05, 0.65, 0.35],
    [0.70, 0.75, 0.80, 0.10],
    [0.85, 0.75, 0.35, 0.05],
    [1.00, 0.95, 0.95, 0.95],
])


rgb = np.zeros(
    (*normalized.shape, 3),
    dtype=np.float32
)

for channel in range(3):
    rgb[..., channel] = np.interp(
        normalized,
        stops[:, 0],
        stops[:, channel + 1]
    )


rgb = (
    np.clip(rgb, 0, 1) * 255
).astype(np.uint8)


Image.fromarray(
    rgb,
    mode="RGB"
).save(
    OUTPUT,
    optimize=True
)


print("Created:", OUTPUT)
print(f"Elevation range: {low:.2f}–{high:.2f} m")
print("Size:", elevation.shape[1], "x", elevation.shape[0])
