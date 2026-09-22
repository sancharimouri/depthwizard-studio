#!/usr/bin/env python3
"""A2/A3 (2026-09-23): FABDEM (Hawker et al. 2022; GLO-30 with forests and buildings
removed; EGM2008 orthometric; data licence CC BY-NC-SA 4.0) from the GEE community
catalog `projects/sat-io/open-datasets/FABDEM` (band b1), mosaicked and resampled
bilinearly onto each of the 32 benchmark tiles' exact 10 m UTM RGB grid via
ee.data.computePixels. Data acquisition only.
Writes data/sentinel2_benchmark/fabdem/<tile>_fabdem.npy (NaN = nodata)."""
from pathlib import Path
import ee, numpy as np, pandas as pd, rasterio
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sentinel2_benchmark/fabdem"
proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
ee.Initialize(project=proj)
_col = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
# mosaic() drops the native projection (default: WGS84 at 1 degree, ~111 km), which made the
# bilinear resample average over ~100 km -- caught 2026-09-23 (manali range 3341-3740 m vs
# GLO-30 1691-4142 m). Restore the native ~30 m projection before resampling.
img = _col.mosaic().setDefaultProjection(_col.first().projection()).select("b1").resample("bilinear")
m = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv")
OUT.mkdir(parents=True, exist_ok=True)
for _, r in m.iterrows():
    p = OUT / f"{r.tile_id}_fabdem.npy"
    if p.exists():
        continue
    with rasterio.open(ROOT / r.rgb_path) as s:
        T, crs, h, w = s.transform, s.crs.to_string(), s.height, s.width
    arr = ee.data.computePixels({"expression": img.unmask(-9999), "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": w, "height": h},
                 "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                     "shearY": T.d, "scaleY": T.e, "translateY": T.f}, "crsCode": crs}})
    a = np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr, dtype=np.float32)
    a[a <= -9000] = np.nan
    np.save(p, a)
    print(f"{r.tile_id}: nan={np.isnan(a).mean():.4f} min={np.nanmin(a):.1f} median={np.nanmedian(a):.1f} max={np.nanmax(a):.1f}", flush=True)
