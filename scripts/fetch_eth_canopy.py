#!/usr/bin/env python3
"""Phase 4 candidate (iv): ETH Global Canopy Height 2020, 10 m (Lang et al. 2023),
GEE community asset users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1, fetched onto
each accepted tile's exact UTM grid (the RGB GeoTIFF's CRS/transform/shape) via
ee.data.computePixels with bilinear resampling. Metres of canopy height above
ground. Data acquisition only. Writes data/sentinel2_benchmark/eth_canopy_2020/<tile>_eth.npy"""
import io, sys
from pathlib import Path
import ee, numpy as np, pandas as pd, rasterio
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sentinel2_benchmark/eth_canopy_2020"
ASSET = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
ee.Initialize(project=proj)
img = ee.Image(ASSET).select(0).resample("bilinear")
m = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv").set_index("tile_id")
v = pd.read_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv").set_index("tile_id")
for t in (v.index if "--all-tiles" in sys.argv else v[~v["flagged"]].index):  # --all-tiles: A2/A3
    p = OUT / f"{t}_eth.npy"
    if p.exists():
        continue
    with rasterio.open(ROOT / m.loc[t, "rgb_path"]) as s:
        T, crs, h, w = s.transform, s.crs.to_string(), s.height, s.width
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": w, "height": h},
                 "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                     "shearY": T.d, "scaleY": T.e, "translateY": T.f},
                 "crsCode": crs}})
    a = np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr, dtype=np.float32)
    a[(a < 0) | (a > 100)] = np.nan  # asset nodata is 255
    np.save(p, a)
    print(f"{t}: {a.shape} nan={np.isnan(a).mean():.3f} median={np.nanmedian(a):.2f} p98={np.nanpercentile(a,98):.2f} max={np.nanmax(a):.1f}", flush=True)
