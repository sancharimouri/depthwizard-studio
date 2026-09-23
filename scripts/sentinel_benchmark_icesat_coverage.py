"""Score ICESat-2 ATL08 land-photon coverage for each benchmark candidate.

For every candidate location, builds a ~10x10km WGS84 polygon and runs
sliderule's atl08p (ATL03+ATL08 PhoREAL processing) over 2019-01-01..now,
land surface type only. The yield metric is the sum of gnd_ph_count
(classified ground-photon returns) across all returned land segments —
this is what "usable, land-classified photon returns" means here: photons
ATL08 already classified as ground hits within a segment, not raw
unclassified ATL03 photons and not counting canopy-only segments.

No separate cloud filter is applied: ATL08's photon classification and
segment QA already suppress atmosphere/cloud contamination upstream, so
gnd_ph_count naturally accumulates only where usable ground signal was
detected — a segment over a persistently cloudy area yields low counts
without needing a manual filter to say so.

Writes one CSV row per candidate to the given output path.
"""

import csv
import sys
import time

from sliderule import icesat2, sliderule

from sentinel_benchmark_candidates import CANDIDATES

AAA_KM = 10.0  # target tile footprint


def bbox_poly(lat: float, lon: float, km: float) -> list[dict]:
    half_lat = (km / 2) / 111.0
    half_lon = (km / 2) / (111.0 * max(0.1, abs(__import__("math").cos(__import__("math").radians(lat)))))
    return [
        {"lon": lon - half_lon, "lat": lat - half_lat},
        {"lon": lon + half_lon, "lat": lat - half_lat},
        {"lon": lon + half_lon, "lat": lat + half_lat},
        {"lon": lon - half_lon, "lat": lat + half_lat},
        {"lon": lon - half_lon, "lat": lat - half_lat},
    ]


def score_candidate(lat: float, lon: float) -> dict:
    poly = bbox_poly(lat, lon, AAA_KM)
    parms = {
        "poly": poly,
        "t0": "2019-01-01T00:00:00Z",
        "t1": "2025-12-31T23:59:59Z",
        "srt": icesat2.SRT_LAND,
        "len": 100,
        "res": 100,
        # Without an explicit "phoreal" block, PhoREAL's classified-photon
        # fields (gnd_ph_count, veg_ph_count, landcover) come back as raw
        # uint32/uint8 fill sentinels (near their max value) instead of real
        # counts — confirmed by inspecting the field ranges directly.
        "phoreal": {
            "binsize": 1.0,
            "geoloc": "center",
            "use_abs_h": False,
            "send_waveform": False,
            "above_classifier": False,
        },
    }
    try:
        gdf = icesat2.atl08p(parms)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return {"segments": 0, "gnd_ph_count": 0, "ph_count": 0, "error": str(exc)}

    if gdf is None or len(gdf) == 0:
        return {"segments": 0, "gnd_ph_count": 0, "ph_count": 0, "error": ""}

    # landcover == 255 is ATL08's fill value for "no valid land classification
    # in this segment" — excluded so the yield only counts genuinely
    # land-classified segments, per the task's "land-classified" requirement.
    land = gdf[gdf["landcover"] != 255]

    return {
        "segments": int(len(land)),
        "gnd_ph_count": int(land["gnd_ph_count"].sum()),
        "ph_count": int(land["ph_count"].sum()),
        "error": "",
    }


def main(out_path: str) -> None:
    sliderule.init("slideruleearth.io", verbose=False)

    rows = []
    total = sum(len(v) for v in CANDIDATES.values())
    done = 0
    for category, candidates in CANDIDATES.items():
        for slug, label, lat, lon in candidates:
            done += 1
            t0 = time.time()
            result = score_candidate(lat, lon)
            elapsed = time.time() - t0
            rows.append(
                {
                    "category": category,
                    "slug": slug,
                    "label": label,
                    "lat": lat,
                    "lon": lon,
                    **result,
                }
            )
            print(
                f"[{done}/{total}] {category:14s} {slug:15s} "
                f"segments={result['segments']:5d} gnd_ph_count={result['gnd_ph_count']:8d} "
                f"({elapsed:.1f}s)"
                + (f"  ERROR: {result['error']}" if result["error"] else ""),
                flush=True,
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "slug", "label", "lat", "lon", "segments", "gnd_ph_count", "ph_count", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sentinel2_benchmark/icesat2_coverage.csv")
