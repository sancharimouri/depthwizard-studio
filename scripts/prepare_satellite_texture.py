import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

REGION = sys.argv[1] if len(sys.argv) > 1 else "darjeeling"

INPUT = (
    ROOT
    / "data"
    / "sentinel2"
    / REGION
    / f"{REGION.capitalize()}_RGB.tif"
)

OUTPUT_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / REGION
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT = OUTPUT_DIR / "satellite.png"


with rasterio.open(INPUT) as src:

    rgb = src.read(
        [1, 2, 3]
    ).astype(np.float32)


# ------------------------------------------------------------
# Normalize each RGB band for display.
# Use percentile stretching so very bright/dark pixels
# don't destroy the overall contrast.
# ------------------------------------------------------------

for i in range(3):

    band = rgb[i]

    low = np.percentile(
        band,
        2
    )

    high = np.percentile(
        band,
        98
    )

    band = (
        (band - low)
        / (high - low)
    )

    band = np.clip(
        band,
        0,
        1
    )

    rgb[i] = band


# CHW -> HWC
rgb = np.transpose(
    rgb,
    (1, 2, 0)
)


rgb = (
    rgb * 255
).astype(np.uint8)


Image.fromarray(
    rgb,
    mode="RGB"
).save(
    OUTPUT,
    optimize=True
)


print("Created:")
print(OUTPUT)

print(
    "Size:",
    rgb.shape[1],
    "x",
    rgb.shape[0]
)

print(
    "File size:",
    OUTPUT.stat().st_size / 1024 / 1024,
    "MB"
)
