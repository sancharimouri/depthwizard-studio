#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DFC_DIR = PROJECT_ROOT / "data" / "dfc2019"
BASELINE_DIR = DFC_DIR / "experiments" / "dav2_baseline"

FULL_RGB_DIR = DFC_DIR / "raw" / "RGB"
FULL_TRUTH_DIR = DFC_DIR / "raw" / "Truth"

BENCHMARK_MANIFEST = BASELINE_DIR / "manifest.csv"
OUTPUT = BASELINE_DIR / "calibration_manifest.csv"

TARGET_TOTAL = 200
SEED = 42


def tile_id_from_rgb(path):
    return path.stem.removesuffix("_RGB")


def tile_id_from_agl(path):
    return path.stem.removesuffix("_AGL")


def main():
    # --------------------------------------------------------------
    # Read the fixed 50-tile benchmark.
    # --------------------------------------------------------------
    benchmark = pd.read_csv(BENCHMARK_MANIFEST)
    benchmark_ids = set(benchmark["tile_id"].astype(str))

    # --------------------------------------------------------------
    # Build all available RGB/AGL pairs.
    # --------------------------------------------------------------
    rgb_files = sorted(FULL_RGB_DIR.rglob("*_RGB.tif"))
    agl_files = sorted(FULL_TRUTH_DIR.rglob("*_AGL.tif"))

    rgb_map = {
        tile_id_from_rgb(p): p
        for p in rgb_files
    }

    agl_map = {
        tile_id_from_agl(p): p
        for p in agl_files
    }

    matched_ids = sorted(set(rgb_map) & set(agl_map))

    rows = []

    # We can reuse the already-computed AGL statistics from the
    # benchmark-building manifest only for benchmark tiles, so for
    # calibration tiles we calculate the P95 directly.
    #
    # To keep this script lightweight, we use the AGL files only to
    # calculate p95 and max.
    import numpy as np
    import rasterio

    for tile_id in matched_ids:
        if tile_id in benchmark_ids:
            continue

        agl_path = agl_map[tile_id]

        with rasterio.open(agl_path) as src:
            agl = src.read(1).astype(np.float32)

        valid = agl[np.isfinite(agl)]

        if valid.size == 0:
            continue

        city = tile_id.split("_")[0]

        rows.append(
            {
                "tile_id": tile_id,
                "city": city,
                "rgb_path": str(rgb_map[tile_id].relative_to(PROJECT_ROOT)),
                "agl_path": str(agl_path.relative_to(PROJECT_ROOT)),
                "agl_p95": float(np.percentile(valid, 95)),
                "agl_max": float(np.max(valid)),
            }
        )

    df = pd.DataFrame(rows)

    print(f"Available non-benchmark pairs: {len(df)}")

    rng = random.Random(SEED)

    selected = []

    # --------------------------------------------------------------
    # Balanced 100 JAX + 100 OMA, with AGL stratification.
    # --------------------------------------------------------------
    for city in ["JAX", "OMA"]:
        city_df = df[df["city"] == city].copy()

        target = TARGET_TOTAL // 2

        if len(city_df) < target:
            raise RuntimeError(
                f"Not enough {city} calibration tiles: "
                f"{len(city_df)}"
            )

        city_df["stratum"] = pd.qcut(
            city_df["agl_p95"],
            q=5,
            labels=False,
            duplicates="drop",
        )

        selected_city_ids = []

        # Select roughly equal numbers from each p95 stratum.
        strata = sorted(city_df["stratum"].dropna().unique())

        while len(selected_city_ids) < target:
            progress = False

            for stratum in strata:
                if len(selected_city_ids) >= target:
                    break

                candidates = city_df[
                    (city_df["stratum"] == stratum)
                    & (~city_df["tile_id"].isin(selected_city_ids))
                ]

                if candidates.empty:
                    continue

                tile_id = rng.choice(candidates["tile_id"].tolist())
                selected_city_ids.append(tile_id)
                progress = True

            if not progress:
                break

        selected.append(
            city_df[city_df["tile_id"].isin(selected_city_ids)]
            .drop(columns=["stratum"])
        )

    calibration = (
        pd.concat(selected, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    assert len(calibration) == TARGET_TOTAL
    assert not benchmark_ids.intersection(
        calibration["tile_id"].astype(str)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(OUTPUT, index=False)

    print("\nCalibration set:")
    print(f"Total: {len(calibration)}")
    print(calibration["city"].value_counts())

    print("\nAGL p95:")
    print(calibration["agl_p95"].describe())

    print("\nWritten:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
