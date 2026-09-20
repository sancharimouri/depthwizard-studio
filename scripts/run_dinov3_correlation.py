#!/usr/bin/env python3
"""Step 1.2 correlation: DINOv3 SAT493M+CHMv2 vs ICESat-2 ground height."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib_backbone_correlation import pooled_correlation, run_backbone_correlation  # noqa: E402

DEPTH_DIR = PROJECT_ROOT / "data" / "sentinel2_benchmark" / "dinov3_depth"


def filename_fn(tile_id: str) -> Path:
    return DEPTH_DIR / f"{tile_id}_depth.npy"


def main() -> None:
    print("=" * 70)
    print("DINOv3 (SAT493M + CHMv2) vs ICESat-2 — per-tile correlation")
    print("=" * 70)
    per_tile = run_backbone_correlation(DEPTH_DIR, filename_fn)

    out_csv = DEPTH_DIR / "dinov3_correlation_per_tile.csv"
    per_tile.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\n" + "=" * 70)
    print("POOLED")
    print("=" * 70)
    pooled = pooled_correlation(DEPTH_DIR, filename_fn)
    print(f"Pooled: n={pooled['pooled']['n']} pearson={pooled['pooled']['pearson']:+.4f} "
          f"spearman={pooled['pooled']['spearman']:+.4f}")
    for cat, r in pooled["by_category"].items():
        print(f"  {cat:14s} n={r['n']:7d} pearson={r['pearson']:+.4f} spearman={r['spearman']:+.4f}")


if __name__ == "__main__":
    main()
