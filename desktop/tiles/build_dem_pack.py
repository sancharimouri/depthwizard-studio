"""Per-item elevation packs for real generation (docs/DESKTOP_APP.md, "Real generation").

  python desktop/tiles/build_dem_pack.py OUT_DIR [ITEM_ID ...]   # writes OUT_DIR/<item id>.tif

One small 2-band float32 GeoTIFF per georeferenced library item, on that item's own
footprint and CRS (NaN = no data), with the sources in its tags:
  band 1  TERRAIN  bare-earth DEM
  band 2  SURFACE  surface model (DSM)
Sources (all real, produced by this project's earlier pipelines; nothing is estimated here):
  Sentinel-2  terrain = surface = calibrated FABDEM (2026-09-29): FABDEM (data/sentinel2_benchmark/
              fabdem/<t>_fabdem.npy, the tile's 10 m grid) through a per-tile OLS fit a + b*FABDEM
              against ICESat-2 ground (terrain_rf_residual/samples.parquet). Copernicus GLO-30
              (copernicus_dem_raw/<t>_dem.tif) is only a cross-check in the calibration report, never
              drawn. Output ~30 m, the DEMs' native resolution.
              Darjeeling (the demo scene, scripts/library_catalog.py EXTRA_S2; no ICESat-2 samples, so
              still terrain = FABDEM, surface = GLO-30): FABDEM from
              data/library/extra/darjeeling_fabdem.npy (Earth Engine, tile grid); GLO-30 from
              data/library/extra/darjeeling_glo30.tif (backend/dem/glo30.py on the tile footprint;
              the older data/elevation/darjeeling crop leaves 1.4% of the tile's edges empty).
Remaining small gaps (NaN) are filled from neighbouring cells (rasterio fillnodata): the
viewer drops NaN cells to the base, which draws walls. No-op for packs without gaps.
  Maxar VHR   terrain = FABDEM, surface = the VHR pipeline's DSM = FABDEM + Method 6
              above-ground height (data/vhr_dsm/<crop>_margin192/, scripts/vhr_dsm_pipeline.py).
              Output 512x512 (~1.2 m).
DFC2019 tiles have no georeference, so there is no DEM for them.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
items = json.loads((ROOT / "data/library/manifest.json").read_text())["items"]


def regrid(src, src_transform, src_crs, dst_shape, dst_transform, dst_crs, method):
    dst = np.full(dst_shape, np.nan, np.float32)
    reproject(np.asarray(src, np.float32), dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs, resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def write(path, terrain, surface, transform, crs, tags):
    with rasterio.open(path, "w", driver="GTiff", width=terrain.shape[1], height=terrain.shape[0], count=2,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate", predictor=3, zlevel=9) as w:
        w.write(terrain, 1)
        w.write(surface, 2)
        w.set_band_description(1, "TERRAIN")
        w.set_band_description(2, "SURFACE")
        w.update_tags(**tags)


import pandas as pd  # noqa: E402

S2_SAMPLES = pd.read_parquet(ROOT / "data/sentinel2_benchmark/terrain_rf_residual/samples.parquet",
                             columns=["tile_id", "row", "col", "rgt", "h_ref", "fabdem"])
S2_REPORT = {}
n = 0
only = set(sys.argv[2:])  # optional item ids: rebuild just these packs
for it in items:
    if only and it["id"] not in only:
        continue
    iid, col = it["id"], it["collection"]
    rgb = Path(os.path.realpath(ROOT / it["file"]))
    if col == "sentinel2":
        t = it["tile_id"]
        with rasterio.open(rgb) as r:
            crs, bounds, tile_tf, tile_shape = r.crs, r.bounds, r.transform, r.shape
        size = 334                                            # 10 km / 334 = 29.9 m
        tf = from_bounds(*bounds, size, size)
        extra = t == "darjeeling"
        fab_path = ("data/library/extra/darjeeling_fabdem.npy" if extra
                    else f"data/sentinel2_benchmark/fabdem/{t}_fabdem.npy")
        glo_path = ("data/library/extra/darjeeling_glo30.tif" if extra
                    else f"data/sentinel2_benchmark/copernicus_dem_raw/{t}_dem.tif")
        fab = np.load(ROOT / fab_path).astype(np.float32)
        assert fab.shape == tile_shape, (iid, fab.shape, tile_shape)
        with rasterio.open(ROOT / glo_path) as g:
            glo = g.read(1).astype(np.float32)
            if g.nodata is not None:
                glo[glo == g.nodata] = np.nan
            glo_tf, glo_crs = g.transform, g.crs
        if extra:  # no ICESat-2 samples for the demo scene: previous recipe (FABDEM terrain, GLO-30 surface)
            terrain = regrid(fab, tile_tf, crs, (size, size), tf, crs, Resampling.average)
            surface = regrid(glo, glo_tf, glo_crs, (size, size), tf, crs, Resampling.bilinear)
            tags = {"TERRAIN_SOURCE": "FABDEM v1-2 (bare earth, 30 m)", "SURFACE_SOURCE": "Copernicus GLO-30 DSM (30 m)"}
        else:
            # Calibrated FABDEM (2026-09-29): per-tile OLS h_ICESat-2 = a + b*FABDEM on the benchmark's
            # ICESat-2 ground samples (terrain_rf_residual/samples.parquet, EGM2008 orthometric), applied
            # at 10 m, then averaged. Both bands = calibrated FABDEM; GLO-30 is a cross-check only.
            s = S2_SAMPLES[S2_SAMPLES.tile_id == t]
            b, a = np.polyfit(s.fabdem.values, s.h_ref.values, 1)
            cal = (a + b * fab).astype(np.float32)
            terrain = regrid(cal, tile_tf, crs, (size, size), tf, crs, Resampling.average)
            surface = terrain.copy()
            glo10 = regrid(glo, glo_tf, glo_crs, tile_shape, tile_tf, crs, Resampling.bilinear)
            glo_pack = regrid(glo, glo_tf, glo_crs, (size, size), tf, crs, Resampling.bilinear)
            r_, c_ = s.row.values, s.col.values
            rmse = lambda e: float(np.sqrt(np.nanmean(np.square(e))))  # noqa: E731
            d = glo_pack - terrain
            S2_REPORT[iid] = {"a": float(a), "b": float(b), "n": len(s), "n_rgt": int(s.rgt.nunique()),
                              "fab_range_sampled_m": [float(s.fabdem.min()), float(s.fabdem.max())],
                              "fab_range_tile_m": [float(np.nanmin(fab)), float(np.nanmax(fab))],
                              "rmse_vs_icesat2_insample": {"raw_fabdem": rmse(s.fabdem.values - s.h_ref.values),
                                                           "calibrated_fabdem": rmse(cal[r_, c_] - s.h_ref.values),
                                                           "glo30": rmse(glo10[r_, c_] - s.h_ref.values)},
                              "glo30_minus_calibrated_on_pack": {"median": float(np.nanmedian(d)),
                                                                 "p05": float(np.nanpercentile(d, 5)),
                                                                 "p95": float(np.nanpercentile(d, 95))}}
            tags = {"TERRAIN_SOURCE": "FABDEM v1-2 calibrated to ICESat-2 ground (per-tile linear fit)",
                    "SURFACE_SOURCE": "FABDEM v1-2 calibrated to ICESat-2 ground (per-tile linear fit)",
                    "CALIBRATION": f"h = {a:.4f} + {b:.6f} * FABDEM (OLS on {len(s)} ICESat-2 ground samples)"}
    elif col == "vhr":
        crop = Path(it["file"]).stem
        d = ROOT / f"data/vhr_dsm/{crop}_margin192"
        with rasterio.open(d / "dtm_fabdem.tif") as a, rasterio.open(d / "dsm.tif") as b:
            crs, bounds, src_tf = a.crs, a.bounds, a.transform
            size = 512
            tf = from_bounds(*bounds, size, size)
            terrain = regrid(a.read(1), src_tf, crs, (size, size), tf, crs, Resampling.average)
            surface = regrid(b.read(1), b.transform, crs, (size, size), tf, crs, Resampling.average)
        tags = {"TERRAIN_SOURCE": "FABDEM v1-2 (bare earth, 30 m, bicubic onto the crop)",
                "SURFACE_SOURCE": "FABDEM + Method 6 above-ground height (research model; under-states canopy above ~18-23 m)"}
    else:
        continue
    for name, arr in (("terrain", terrain), ("surface", surface)):
        assert np.isfinite(arr).mean() > 0.95, f"{iid}: {name} only {np.isfinite(arr).mean():.0%} valid"
        if not np.isfinite(arr).all():
            arr[:] = fillnodata(arr, mask=np.isfinite(arr).astype(np.uint8), max_search_distance=100)
    write(out / f"{iid}.tif", terrain, surface, tf, crs, tags)
    n += 1
    print(f"{iid:28s} {terrain.shape}  terrain {np.nanmin(terrain):7.1f}..{np.nanmax(terrain):7.1f} m"
          f"  surface {np.nanmin(surface):7.1f}..{np.nanmax(surface):7.1f} m")
if S2_REPORT:
    rep = ROOT / "data/sentinel2_benchmark/pack_calibration.json"
    rep.write_text(json.dumps(S2_REPORT, indent=1))
    print(f"calibration report -> {rep}")
total = sum(p.stat().st_size for p in out.glob("*.tif")) / 1e6
print(f"{n} packs, {total:.1f} MB -> {out}")
