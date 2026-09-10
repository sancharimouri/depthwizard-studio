from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "diagnostics"
    / "darjeeling"
    / "Darjeeling_RGB_depth.npy"
)

OUTPUT_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "darjeeling"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT = OUTPUT_DIR / "relative_depth.png"


depth = np.load(INPUT).astype(np.float32)

print("Depth shape:", depth.shape)
print("Depth range:", depth.min(), depth.max())


# ------------------------------------------------------------
# Normalize for visualization
# ------------------------------------------------------------

low = np.percentile(depth, 2)
high = np.percentile(depth, 98)

depth = (
    depth - low
) / (
    high - low
)

depth = np.clip(
    depth,
    0.0,
    1.0
)


# ------------------------------------------------------------
# Convert to 8-bit grayscale
# ------------------------------------------------------------

image = (
    depth * 255
).astype(np.uint8)


Image.fromarray(
    image,
    mode="L"
).save(
    OUTPUT,
    optimize=True
)


print("Created:", OUTPUT)
print(
    "Output size:",
    OUTPUT.stat().st_size / 1024 / 1024,
    "MB"
)
