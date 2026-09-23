#!/usr/bin/env python3
"""Part D.3 (2026-09-23, descriptive only; pre-registered in 07-gamus-generalization/log.md).
Composed Sikkim/Darjeeling VHR DSMs (data/vhr_dsm/<crop>_margin192/dsm.tif = FABDEM + max(Method 6
seed-43 AGL, 0), EGM2008) vs. sparse spaceborne lidar surface heights inside each crop:
  - ICESat-2 20 m PhoREAL segments (sliderule; same parameters as fetch_icesat2_segments20m.py),
    surface = h_te_median + h_max_canopy (WGS84 ellipsoid);
  - GEDI L2A (fetch_gedi_l2a.fetch; quality-filtered), surface = elev_lowestmode + rh98 (ellipsoid).
    (The pre-registration said elev_highestreturn; the existing fetcher exports lowestmode + rh98,
    which is the same canopy-top surface up to the rh98-vs-rh100 difference. Logged as a deviation.)
Each point is converted to EGM2008 per point (PROJ_NETWORK=ON, |N| asserted > 1 m).
Raster value per point = p98 of the raster within the footprint radius (GEDI 12.5 m, ICESat-2 10 m),
the same operator for the composed DSM, raw FABDEM and raw GLO-30.
Writes data/vhr_dsm/_diagnostics/sparse_lidar_dsm_check.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ["PROJ_NETWORK"] = "ON"
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CROPS = ["c_town", "c_river", "c_terraces", "a_forest", "a_valley", "b_glacier"]
OUT = ROOT / "data/vhr_dsm/_diagnostics/sparse_lidar_dsm_check.json"


def sample_p98(arr, T, x, y, radius_m):
    res = abs(T.a)
    cols, rows = ~T * (x, y)
    k = int(np.ceil(radius_m / res))
    out = []
    for r0, c0 in zip(rows, cols):
        rr, cc = np.mgrid[int(r0) - k:int(r0) + k + 1, int(c0) - k:int(c0) + k + 1]
        m = (np.hypot(rr - r0, cc - c0) * res <= radius_m) & (rr >= 0) & (rr < arr.shape[0]) & (cc >= 0) & (cc < arr.shape[1])
        v = arr[rr[m], cc[m]] if m.sum() > 0.8 * np.pi * (radius_m / res) ** 2 else np.array([])
        v = v[np.isfinite(v)]
        out.append(float(np.percentile(v, 98)) if v.size else np.nan)
    return np.array(out)


def stats(pred, ref):
    ok = np.isfinite(pred) & np.isfinite(ref)
    e = pred[ok] - ref[ok]
    if ok.sum() == 0:
        return {"n": 0}
    return {"n": int(ok.sum()), "bias": float(e.mean()), "mae": float(np.abs(e).mean()),
            "medae": float(np.median(np.abs(e))), "rmse": float(np.sqrt((e ** 2).mean())),
            # post-hoc (2026-09-23, after seeing gross cloud/false-return outliers): robust summaries
            "median_bias": float(np.median(e)), "n_abs_gt_50m": int((np.abs(e) > 50).sum())}


def main():
    import ee
    import fetch_gedi_l2a as fg
    from sliderule import icesat2, sliderule
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    sliderule.init("slideruleearth.io", verbose=False)
    to_egm = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
    res = {}
    for crop in CROPS:
        d = ROOT / f"data/vhr_dsm/{crop}_margin192"
        rasters = {}
        for name, f in (("composed", "dsm.tif"), ("fabdem", "dtm_fabdem.tif"), ("glo30", "glo30.tif")):
            with rasterio.open(d / f) as s:
                rasters[name] = s.read(1).astype(np.float64); T, crs, b = s.transform, s.crs, s.bounds
        tl = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tl.transform([b.left, b.right, b.right, b.left], [b.bottom, b.bottom, b.top, b.top])
        poly = [{"lon": a, "lat": c} for a, c in zip(lon, lat)] + [{"lon": lon[0], "lat": lat[0]}]
        to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        out = {}
        # GEDI
        g = fg.fetch(poly)
        g = g[g["rh98"] <= 80] if len(g) else g
        refs = {}
        if len(g):
            _, _, H = to_egm.transform(g["lon"].values, g["lat"].values, (g["elev_lowestmode"] + g["rh98"]).values)
            N = (g["elev_lowestmode"] + g["rh98"]).values - H
            assert np.all(np.abs(N) > 1), "geoid N ~ 0: PROJ grid not used"
            refs["gedi"] = (g["lon"].values, g["lat"].values, H, 12.5)
        # ICESat-2 20 m
        parms = {"poly": poly, "t0": "2019-01-01T00:00:00Z", "t1": "2025-12-31T23:59:59Z",
                 "srt": icesat2.SRT_LAND, "len": 20, "res": 20, "ats": 5.0, "cnt": 5,
                 "phoreal": {"binsize": 1.0, "geoloc": "center", "use_abs_h": False,
                             "send_waveform": False, "above_classifier": False}}
        try:
            a = icesat2.atl08p(parms)
        except Exception as e:  # noqa: BLE001
            a, out["icesat2_error"] = None, f"{type(e).__name__}: {e}"
        if a is not None and len(a):
            s = (a["h_te_median"] + a["h_max_canopy"]).values
            ok = np.isfinite(s)
            x, y = a.geometry.x.values[ok], a.geometry.y.values[ok]
            _, _, H = to_egm.transform(x, y, s[ok])
            assert np.all(np.abs(s[ok] - H) > 1), "geoid N ~ 0"
            refs["icesat2_20m"] = (x, y, H, 10.0)
        for rname, (lo, la, H, rad) in refs.items():
            X, Y = to_crs.transform(lo, la)
            rr = {}
            for name, arr in rasters.items():
                rr[name] = stats(sample_p98(arr, T, np.asarray(X), np.asarray(Y), rad), H)
            out[rname] = rr
        res[crop] = out
        print(crop, json.dumps(out), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
