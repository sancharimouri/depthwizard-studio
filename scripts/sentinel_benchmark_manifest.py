"""Combine ICESat-2 coverage results + CDSE download results into the final
data/sentinel2_benchmark/manifest.csv, one row per successfully acquired
tile: tile ID, category, lat/lon bounds, ICESat-2 photon count, date
acquired, cloud %.
"""

import csv
import sys


def main(coverage_csv: str, download_results_csv: str, out_csv: str) -> None:
    coverage = {}
    with open(coverage_csv) as f:
        for row in csv.DictReader(f):
            coverage[row["slug"]] = row

    rows = []
    with open(download_results_csv) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            cov = coverage.get(row["slug"], {})
            rows.append(
                {
                    "tile_id": row["slug"],
                    "category": row["category"],
                    "label": row["label"],
                    "lat": cov.get("lat", ""),
                    "lon": cov.get("lon", ""),
                    "bbox_utm_epsg": row["epsg"],
                    "bbox_utm": row["bbox_utm"],
                    "icesat2_gnd_ph_count": cov.get("gnd_ph_count", ""),
                    "icesat2_segments": cov.get("segments", ""),
                    "date_acquired": row["date"],
                    "cloud_pct": row["cloud"],
                    "resolution_m": 10,
                    "rgb_path": row["path"],
                }
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "category",
                "label",
                "lat",
                "lon",
                "bbox_utm_epsg",
                "bbox_utm",
                "icesat2_gnd_ph_count",
                "icesat2_segments",
                "date_acquired",
                "cloud_pct",
                "resolution_m",
                "rgb_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
