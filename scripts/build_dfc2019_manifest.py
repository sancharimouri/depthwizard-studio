#!/usr/bin/env python3

from pathlib import Path
import re
import random

import numpy as np
import pandas as pd
import rasterio


RGB_DIR = Path("data/dfc2019/raw/RGB")
TRUTH_DIR = Path("data/dfc2019/raw/Truth")
OUTPUT = Path("data/dfc2019/experiments/dav2_baseline/manifest.csv")

TARGET_TOTAL = 50
SEED = 42


def tile_id_from_rgb(path: Path) -> str | None:
    match = re.match(r"(.+)_RGB$", path.stem)
    return match.group(1) if match else None


def tile_id_from_agl(path: Path) -> str | None:
    match = re.match(r"(.+)_AGL$", path.stem)
    return match.group(1) if match else None


def agl_stats(path: Path) -> dict:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        nodata = src.nodata

        if nodata is not None:
            valid = data[data != nodata]
        else:
            valid = data[np.isfinite(data)]

    valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        raise ValueError(f"No valid AGL values: {path}")

    return {
        "width": int(src.width),
        "height": int(src.height),
        "agl_min": float(np.min(valid)),
        "agl_max": float(np.max(valid)),
        "agl_mean": float(np.mean(valid)),
        "agl_std": float(np.std(valid)),
        "agl_median": float(np.median(valid)),
        "agl_p95": float(np.percentile(valid, 95)),
        "agl_p99": float(np.percentile(valid, 99)),
        "agl_nonzero_pct": float(np.mean(valid > 0) * 100.0),
    }


def main():
    rgb_files = sorted(RGB_DIR.rglob("*_RGB.tif"))
    agl_files = sorted(TRUTH_DIR.rglob("*_AGL.tif"))

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

    matched_ids = sorted(set(rgb_map) & set(agl_map))

    if not matched_ids:
        raise RuntimeError("No RGB/AGL pairs found.")

    print(f"Matched pairs available: {len(matched_ids)}")
    print("Computing AGL statistics...\n")

    rows = []

    for i, tile_id in enumerate(matched_ids, start=1):
        rgb_path = rgb_map[tile_id]
        agl_path = agl_map[tile_id]

        stats = agl_stats(agl_path)

        city = tile_id.split("_")[0]

        row = {
            "tile_id": tile_id,
            "city": city,
            "rgb_path": str(rgb_path),
            "agl_path": str(agl_path),
            **stats,
        }

        rows.append(row)

        if i % 50 == 0:
            print(f"Processed {i}/{len(matched_ids)}")

    df = pd.DataFrame(rows)

    print("\n=== DATASET AGL SUMMARY ===")
    print(
        df[
            [
                "agl_min",
                "agl_p95",
                "agl_max",
                "agl_nonzero_pct",
            ]
        ].describe()
    )

    print("\n=== CITY COUNTS ===")
    print(df["city"].value_counts())

    # ------------------------------------------------------------------
    # Select 50 tiles.
    #
    # We aim for equal JAX/OMA representation and stratify by p95 height
    # so we don't accidentally choose only flat scenes.
    # ------------------------------------------------------------------

    rng = random.Random(SEED)

    city_groups = []

    for city in ["JAX", "OMA"]:
        city_df = df[df["city"] == city].copy()

        if city_df.empty:
            continue

        target = TARGET_TOTAL // 2

        # If one city has fewer than target tiles, take everything
        target = min(target, len(city_df))

        # Create 5 vertical-content strata using p95.
        city_df["height_stratum"] = pd.qcut(
            city_df["agl_p95"],
            q=min(5, len(city_df)),
            labels=False,
            duplicates="drop",
        )

        selected_parts = []

        strata = sorted(city_df["height_stratum"].dropna().unique())

        # Round-robin selection across strata.
        while sum(len(x) for x in selected_parts) < target:
            made_progress = False

            for stratum in strata:
                if sum(len(x) for x in selected_parts) >= target:
                    break

                candidates = city_df[
                    (city_df["height_stratum"] == stratum)
                    & (~city_df["tile_id"].isin(
                        pd.concat(selected_parts)["tile_id"]
                        if selected_parts
                        else []
                    ))
                ]

                if candidates.empty:
                    continue

                idx = rng.choice(candidates.index.tolist())
                selected_parts.append(city_df.loc[[idx]])
                made_progress = True

            if not made_progress:
                break

        if selected_parts:
            selected = pd.concat(selected_parts)
            city_groups.append(selected)

    selected_df = pd.concat(city_groups, ignore_index=True)

    # In the unlikely case we selected fewer than 50 because of uneven
    # city availability, fill from the remaining highest-diversity pool.
    if len(selected_df) < min(TARGET_TOTAL, len(df)):
        remaining = df[~df["tile_id"].isin(selected_df["tile_id"])].copy()

        needed = min(TARGET_TOTAL, len(df)) - len(selected_df)

        remaining = remaining.sort_values(
            ["agl_p95", "agl_nonzero_pct"],
            ascending=[False, False],
        )

        selected_df = pd.concat(
            [selected_df, remaining.head(needed)],
            ignore_index=True,
        )

    # Always include our already-tested tile if it exists.
    known_tile = "JAX_004_006"

    if known_tile in set(df["tile_id"]):
        if known_tile not in set(selected_df["tile_id"]):
            # Replace the last selected tile.
            selected_df = selected_df.iloc[:-1].copy()
            selected_df = pd.concat(
                [selected_df, df[df["tile_id"] == known_tile]],
                ignore_index=True,
            )

    selected_df = selected_df.drop(
        columns=["height_stratum"],
        errors="ignore",
    )

    selected_df = selected_df.sort_values("tile_id").reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(OUTPUT, index=False)

    print("\n=== SELECTED BENCHMARK ===")
    print(f"Tiles selected: {len(selected_df)}")

    print("\nBy city:")
    print(selected_df["city"].value_counts())

    print("\nSelected AGL p95 distribution:")
    print(selected_df["agl_p95"].describe())

    print("\nSelected AGL max distribution:")
    print(selected_df["agl_max"].describe())

    print("\nSelected tiles:")
    print(selected_df["tile_id"].to_string(index=False))

    print(f"\nManifest written to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
