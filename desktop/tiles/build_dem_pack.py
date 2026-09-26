"""Per-item elevation packs for real generation (docs/DESKTOP_APP.md, "Real generation").

  python desktop/tiles/build_dem_pack.py OUT_DIR [ITEM_ID ...]   # writes OUT_DIR/<item id>.tif

One small 2-band float32 GeoTIFF per georeferenced library item, on that item's own
footprint and CRS (NaN = no data), with the sources in its tags:
  band 1  TERRAIN  bare-earth DEM
  band 2  SURFACE  surface model (DSM)
Sources (all real, produced by this project's earlier pipelines; nothing is estimated here):
  Sentinel-2  terrain = FABDEM (data/sentinel2_benchmark/fabdem/<t>_fabdem.npy, already on the
              tile's 10 m grid); surface = Copernicus GLO-30 (copernicus_dem_raw/<t>_dem.tif,
              EPSG:4326, reprojected here). Output ~30 m, the DEMs' native resolution.
              Darjeeling (the demo scene, scripts/library_catalog.py EXTRA_S2): FABDEM from
              data/library/extra/darjeeling_fabdem.npy (Earth Engine, tile grid); GLO-30 from
              data/elevation/darjeeling/Darjeeling_Copernicus_GLO30_DSM_cropped.tif (N26+N27 mosaic).
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
        glo_path = ("data/elevation/darjeeling/Darjeeling_Copernicus_GLO30_DSM_cropped.tif" if extra
                    else f"data/sentinel2_benchmark/copernicus_dem_raw/{t}_dem.tif")
        fab = np.load(ROOT / fab_path).astype(np.float32)
        assert fab.shape == tile_shape, (iid, fab.shape, tile_shape)
        terrain = regrid(fab, tile_tf, crs, (size, size), tf, crs, Resampling.average)
        with rasterio.open(ROOT / glo_path) as g:
            glo = g.read(1).astype(np.float32)
            if g.nodata is not None:
                glo[glo == g.nodata] = np.nan
            surface = regrid(glo, g.transform, g.crs, (size, size), tf, crs, Resampling.bilinear)
        tags = {"TERRAIN_SOURCE": "FABDEM v1-2 (bare earth, 30 m)", "SURFACE_SOURCE": "Copernicus GLO-30 DSM (30 m)"}
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
    write(out / f"{iid}.tif", terrain, surface, tf, crs, tags)
    n += 1
    print(f"{iid:28s} {terrain.shape}  terrain {np.nanmin(terrain):7.1f}..{np.nanmax(terrain):7.1f} m"
          f"  surface {np.nanmin(surface):7.1f}..{np.nanmax(surface):7.1f} m")
total = sum(p.stat().st_size for p in out.glob("*.tif")) / 1e6
print(f"{n} packs, {total:.1f} MB -> {out}")
