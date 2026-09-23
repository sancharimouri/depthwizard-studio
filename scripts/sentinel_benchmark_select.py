"""Rank candidates within each category by ICESat-2 land-classified ground-
photon yield and keep the top 6-8 (or fewer, if too few clear the minimum
threshold). Writes the selected subset as a CSV for the downloader script.

Threshold choice: candidates need at least MIN_GND_PH_COUNT=5000 classified
ground-photon returns across their 10x10km footprint. That's a low bar
(a few dozen ICESat-2 land segments with real ground hits) chosen because
ICESat-2's ground tracks are narrow and widely spaced, so most small AOIs
get comparatively sparse coverage regardless of terrain quality — the goal
is filtering out near-zero-coverage misses, not demanding dense coverage.
"""

import csv
import sys
from collections import defaultdict

MIN_GND_PH_COUNT = 5000
TARGET_MIN, TARGET_MAX = 6, 8


def main(coverage_csv: str, out_csv: str) -> None:
    by_category = defaultdict(list)
    with open(coverage_csv) as f:
        for row in csv.DictReader(f):
            row["gnd_ph_count"] = int(row["gnd_ph_count"])
            row["segments"] = int(row["segments"])
            by_category[row["category"]].append(row)

    selected = []
    report_lines = []
    for category, rows in by_category.items():
        rows.sort(key=lambda r: r["gnd_ph_count"], reverse=True)
        passing = [r for r in rows if r["gnd_ph_count"] >= MIN_GND_PH_COUNT]
        keep = passing[:TARGET_MAX]
        selected.extend(keep)
        report_lines.append(
            f"{category}: {len(keep)}/{len(rows)} candidates cleared "
            f"gnd_ph_count>={MIN_GND_PH_COUNT} (tried {len(rows)}, kept top {len(keep)})"
        )
        for r in rows:
            marker = "KEEP" if r in keep else "drop"
            report_lines.append(
                f"    [{marker}] {r['slug']:16s} gnd_ph_count={r['gnd_ph_count']:6d} segments={r['segments']:5d}"
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "slug", "label", "lat", "lon", "gnd_ph_count", "segments"])
        writer.writeheader()
        for r in selected:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print("\n".join(report_lines))
    print(f"\nSelected {len(selected)} tiles total -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
