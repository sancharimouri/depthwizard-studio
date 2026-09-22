#!/usr/bin/env python3
"""
Phase 3a (2026-09-23): ICESat-2 ATL08-style 20 m PhoREAL segments (sliderule
atl08p) for the 25 accepted Sentinel-2 benchmark tiles + Darjeeling.
Surface (canopy-top) reference for detail-source scoring. Data acquisition only.

Same call as scripts/sentinel_benchmark_icesat_coverage.py (atl08p, SRT_LAND,
2019-01-01..2025-12-31, identical "phoreal" block), except:
  len/res 100 -> 20 (the requested segment length), and
  ats 20 (sliderule default) -> 5, cnt -> 5: the default minimum along-track
  photon spread of 20 m cannot be met by 20 m segments and dropped ~99% of
  them (dehradun: 73 segments with defaults vs 7,185 with ats=5).
use_abs_h=False, so h_*_canopy are heights ABOVE h_te_median;
surface_h = h_te_median + h_max_canopy (WGS84 ellipsoidal, like h_te_median).
Writes data/icesat2_segments20m/<tile>.csv (new dir; ground-photon CSVs untouched).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import rasterio
from sliderule import icesat2, sliderule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import query_icesat2_photons as q  # noqa: E402

OUT = ROOT / "data/icesat2_segments20m"
KEEP = ["h_te_median", "h_max_canopy", "h_canopy", "h_mean_canopy", "h_min_canopy", "canopy_openness",
        "gnd_ph_count", "veg_ph_count", "ph_count", "landcover", "snowcover", "solar_elevation",
        "rgt", "cycle", "spot", "gt", "segment_id"]


def tiles():
    m = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv").set_index("tile_id")
    v = pd.read_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv").set_index("tile_id")
    for t in v[~v["flagged"]].index:
        yield t, q.utm_bbox_to_wgs84_poly(ast.literal_eval(m.loc[t, "bbox_utm"]), int(m.loc[t, "bbox_utm_epsg"]))
    with rasterio.open(ROOT / "data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif") as s:
        b = s.bounds
        yield "darjeeling", q.utm_bbox_to_wgs84_poly([b.left, b.bottom, b.right, b.top], s.crs.to_epsg())


def main():
    sliderule.init("slideruleearth.io", verbose=False)
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for t, poly in tiles():
        if only and t not in only:
            continue
        path = OUT / f"{t}.csv"
        if path.exists():
            print(f"{t}: exists, skip", flush=True)
            continue
        parms = {"poly": poly, "t0": "2019-01-01T00:00:00Z", "t1": "2025-12-31T23:59:59Z",
                 "srt": icesat2.SRT_LAND, "len": 20, "res": 20, "ats": 5.0, "cnt": 5,
                 "phoreal": {"binsize": 1.0, "geoloc": "center", "use_abs_h": False,
                             "send_waveform": False, "above_classifier": False}}
        g = icesat2.atl08p(parms)
        if g is None or len(g) == 0:
            df = pd.DataFrame(columns=["lat", "lon", "surface_h"] + KEEP)
        else:
            df = pd.DataFrame({"lat": g.geometry.y.values, "lon": g.geometry.x.values,
                               **{k: g[k].values for k in KEEP if k in g.columns}})
            df["surface_h"] = df["h_te_median"] + df["h_max_canopy"]
        df.to_csv(path, index=False)
        print(f"{t}: {len(df)} segments, gnd>0: {(df['gnd_ph_count'] > 0).sum() if len(df) else 0}", flush=True)


if __name__ == "__main__":
    main()
