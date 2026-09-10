from pathlib import Path
import json

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]

DSM_PATH = (
    ROOT
    / "data"
    / "elevation"
    / "darjeeling"
    / "Darjeeling_OpenTopography_DSM.tif"
)

OUTPUT_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "darjeeling"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with rasterio.open(DSM_PATH) as src:
    elevation = src.read(1).astype(np.float32)

    bounds = {
        "west": float(src.bounds.left),
        "south": float(src.bounds.bottom),
        "east": float(src.bounds.right),
        "north": float(src.bounds.top),
    }

    width = src.width
    height = src.height


if not np.isfinite(elevation).all():
    raise ValueError("DSM contains NaN/Inf values.")

minimum = float(elevation.min())
maximum = float(elevation.max())

normalized = (elevation - minimum) / (maximum - minimum)

terrain = {
    "width": width,
    "height": height,
    "bounds": bounds,
    "elevationMin": minimum,
    "elevationMax": maximum,
    "heights": normalized.flatten().tolist(),
}

output_path = OUTPUT_DIR / "terrain.json"

with output_path.open("w", encoding="utf-8") as f:
    json.dump(terrain, f, separators=(",", ":"))

print("Created:", output_path)
print("Grid:", width, "x", height)
print("Elevation:", minimum, "to", maximum, "m")
print("Bounds:", bounds)
print("Output size:", output_path.stat().st_size / 1024 / 1024, "MB")
