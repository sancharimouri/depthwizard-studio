#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.depth.depth_engine import run_geotiff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen DAv2 on a manifest of RGB images."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to CSV containing tile_id and rgb_path.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where *_depth.npy files will be stored.",
    )

    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def find_generated_depth(result: dict) -> Path:
    value = result.get("depth_npy")

    if not value:
        raise RuntimeError(
            f"run_geotiff() returned no depth_npy: {result}"
        )

    path = Path(value)

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(
            f"Expected DAv2 output does not exist: {path}"
        )

    return path


def main():
    args = parse_args()

    manifest_path = resolve_project_path(args.manifest)
    output_dir = resolve_project_path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    required = {"tile_id", "rgb_path"}

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    print("=" * 70)
    print("DAV2 BATCH RUN")
    print("=" * 70)

    print(f"Manifest : {manifest_path}")
    print(f"Tiles    : {len(manifest)}")
    print(f"Output   : {output_dir}")
    print()

    successes = 0
    skipped = 0
    errors = 0

    runtimes = []

    # Temporary directory for the existing run_geotiff() outputs.
    temp_dir = output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, row in manifest.iterrows():
        tile_id = str(row["tile_id"])

        rgb_path = resolve_project_path(str(row["rgb_path"]))

        final_depth = output_dir / f"{tile_id}_depth.npy"

        print(
            f"[{i + 1}/{len(manifest)}] {tile_id}"
        )

        # -------------------------------------------------------------
        # Resume-safe behavior
        # -------------------------------------------------------------

        if final_depth.exists():
            print("  Already exists — skipping.")
            skipped += 1
            print()
            continue

        if not rgb_path.exists():
            print(f"  ERROR: RGB not found: {rgb_path}")
            errors += 1
            print()
            continue

        try:
            start = time.perf_counter()

            result = run_geotiff(
                rgb_path,
                temp_dir,
            )

            elapsed = time.perf_counter() - start

            generated = find_generated_depth(result)

            depth = np.load(generated)

            np.save(final_depth, depth)

            successes += 1
            runtimes.append(elapsed)

            print(
                f"  Done: {elapsed:.2f} s"
            )

        except Exception as exc:
            errors += 1
            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

        print()

    print("=" * 70)
    print("DAV2 BATCH COMPLETE")
    print("=" * 70)

    print(f"Total tiles : {len(manifest)}")
    print(f"New runs    : {successes}")
    print(f"Skipped     : {skipped}")
    print(f"Errors      : {errors}")

    if runtimes:
        print(f"Mean runtime: {np.mean(runtimes):.3f} s")
        print(f"Median      : {np.median(runtimes):.3f} s")

    print(f"\nDepth directory:\n{output_dir}")


if __name__ == "__main__":
    main()
