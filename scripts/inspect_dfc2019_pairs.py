#!/usr/bin/env python3

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RGB_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "RGB"
TRUTH_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "Truth"


def tile_id_from_rgb(path: Path) -> str | None:
    """
    JAX_004_006_RGB.tif -> JAX_004_006
    OMA_001_003_RGB.tif -> OMA_001_003
    """
    match = re.match(r"(.+)_RGB$", path.stem)
    return match.group(1) if match else None


def tile_id_from_agl(path: Path) -> str | None:
    """
    JAX_004_006_AGL.tif -> JAX_004_006
    """
    match = re.match(r"(.+)_AGL$", path.stem)
    return match.group(1) if match else None


def main() -> int:
    if not RGB_DIR.exists():
        print(f"ERROR: RGB directory not found: {RGB_DIR}")
        return 1

    if not TRUTH_DIR.exists():
        print(f"ERROR: Truth directory not found: {TRUTH_DIR}")
        return 1

    rgb_files = sorted(RGB_DIR.rglob("*_RGB.tif"))
    agl_files = sorted(TRUTH_DIR.rglob("*_AGL.tif"))

    print(f"RGB files found : {len(rgb_files)}")
    print(f"AGL files found : {len(agl_files)}")
    print()

    rgb_map = {}
    agl_map = {}

    for path in rgb_files:
        tile_id = tile_id_from_rgb(path)
        if tile_id:
            rgb_map[tile_id] = path

    for path in agl_files:
        tile_id = tile_id_from_agl(path)
        if tile_id:
            agl_map[tile_id] = path

    rgb_ids = set(rgb_map)
    agl_ids = set(agl_map)

    matched = sorted(rgb_ids & agl_ids)
    missing_agl = sorted(rgb_ids - agl_ids)
    missing_rgb = sorted(agl_ids - rgb_ids)

    print(f"Matched pairs   : {len(matched)}")
    print(f"Missing AGL     : {len(missing_agl)}")
    print(f"Missing RGB     : {len(missing_rgb)}")
    print()

    if missing_agl:
        print("RGB files with no matching AGL:")
        for tile_id in missing_agl:
            print(f"  {tile_id}")

    if missing_rgb:
        print("\nAGL files with no matching RGB:")
        for tile_id in missing_rgb:
            print(f"  {tile_id}")

    print("\nSample matched pairs:")
    for tile_id in matched[:10]:
        print(f"  {tile_id}")
        print(f"    RGB: {rgb_map[tile_id]}")
        print(f"    AGL: {agl_map[tile_id]}")

    # Useful city-level count
    jax = sum(tile_id.startswith("JAX_") for tile_id in matched)
    oma = sum(tile_id.startswith("OMA_") for tile_id in matched)

    print("\nMatched by city:")
    print(f"  JAX: {jax}")
    print(f"  OMA: {oma}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
