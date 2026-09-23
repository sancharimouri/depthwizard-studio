#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np


RESULTS = Path(
    "data/dfc2019/experiments/method4/"
    "method4_spatial_cv_results.json"
)


def main():
    data = json.loads(RESULTS.read_text())

    folds = data["folds"]

    print("=" * 72)
    print("METHOD 4 FOLD DIAGNOSTIC")
    print("=" * 72)

    print(f"\nTotal folds: {len(folds)}")

    baseline = []
    method4 = []

    for f in folds:
        b = f["baseline"]
        m = f["method4"]

        baseline.append(b)
        method4.append(m)

        print(
            f"{f['tile']:18s} {f['fold']:12s} | "
            f"MAE {b['mae_m']:.3f} -> {m['mae_m']:.3f} | "
            f"RMSE {b['rmse_m']:.3f} -> {m['rmse_m']:.3f} | "
            f"Pearson {b['pearson']:.3f} -> {m['pearson']:.3f} | "
            f"Spearman {b['spearman']:.3f} -> {m['spearman']:.3f}"
        )

    print("\n" + "=" * 72)
    print("METHOD 4 IMPROVEMENT COUNTS")
    print("=" * 72)

    for metric in ["mae_m", "rmse_m", "pearson", "spearman"]:
        improved = 0

        for b, m in zip(baseline, method4):
            if metric in ("mae_m", "rmse_m"):
                improved += m[metric] < b[metric]
            else:
                improved += m[metric] > b[metric]

        print(
            f"{metric:10s}: "
            f"{improved}/{len(folds)} folds improved"
        )

    print("\n" + "=" * 72)
    print("WORST CORRELATION DROPS")
    print("=" * 72)

    drops = []

    for f in folds:
        b = f["baseline"]
        m = f["method4"]

        drops.append(
            (
                m["pearson"] - b["pearson"],
                m["spearman"] - b["spearman"],
                f["tile"],
                f["fold"],
            )
        )

    drops.sort()

    for pearson_delta, spearman_delta, tile, fold in drops[:10]:
        print(
            f"{tile:18s} {fold:12s} | "
            f"Pearson Δ={pearson_delta:+.4f} | "
            f"Spearman Δ={spearman_delta:+.4f}"
        )


if __name__ == "__main__":
    main()
