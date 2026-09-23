"""Fetch a 512x512 true-color preview PNG for each of the 32 final tiles
(kept-as-is originals use the existing manifest lat/lon/date; remediated
tiles use their new lat/lon/date from remediation_results.csv), for visual
review before any manifest update.
"""
import csv
import math
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from backend.cdse import client as cdse  # noqa: E402

OUT_DIR = "data/sentinel2_benchmark/audit_thumbnails"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = {r["tile_id"]: r for r in csv.DictReader(open("data/sentinel2_benchmark/manifest.csv"))}
    remediated = {r["tile_id"]: r for r in csv.DictReader(open("data/sentinel2_benchmark/remediation_results.csv"))}

    for i, (tile_id, row) in enumerate(sorted(manifest.items()), 1):
        if tile_id in remediated:
            r = remediated[tile_id]
            lat, lon, date = float(r["lat"]), float(r["lon"]), r["date"]
        else:
            lat, lon, date = float(row["lat"]), float(row["lon"]), row["date_acquired"]

        half_lat = 5 / 111.0
        half_lon = 5 / (111.0 * max(0.1, abs(math.cos(math.radians(lat)))))
        bbox = [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]

        try:
            png = cdse.fetch_true_color_png(bbox, date, width=512, height=512)
            out_path = f"{OUT_DIR}/{tile_id}.png"
            with open(out_path, "wb") as f:
                f.write(png)
            print(f"[{i}/32] {tile_id:16s} -> {out_path} ({'REPLACED' if tile_id in remediated else 'original'})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/32] {tile_id:16s} ERROR: {exc}", flush=True)


if __name__ == "__main__":
    main()
