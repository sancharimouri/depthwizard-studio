#!/usr/bin/env python3
"""Post-hoc check (2026-09-23): after the GAMUS-DC retrain, how much does predicted AGL rise on
non-green (ExG < 0) vs green (ExG > 0.05) pixels of each VHR crop? Old = seed-43 margin192,
new = GAMUS-DC margin192. Tests whether the leaf-off GAMUS supervision taught 'brown texture = canopy'."""
import json, sys
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import Window
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vhr_dsm_pipeline as vp
out = {}
for name in vp.CROPS:
    scene, cr, cc, _ = vp.CROPS[name]
    r0, c0 = cr * vp.CELL + (vp.CELL - vp.SIZE) // 2, cc * vp.CELL + (vp.CELL - vp.SIZE) // 2
    with rasterio.open(vp.ROOT / "data/maxar_sanity" / vp.SCENES[scene]) as s:
        r, g, b = s.read([1, 2, 3], window=Window(c0, r0, vp.SIZE, vp.SIZE)).astype(np.float64)
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    old = rasterio.open(vp.OUT / f"{name}_margin192/agl.tif").read(1)
    new = rasterio.open(vp.OUT / f"{name}_margin192_gamusdc/agl.tif").read(1)
    res = {}
    for lab, m in [("nongreen_exg_lt0", exg < 0), ("green_exg_gt0.05", exg > 0.05)]:
        m = m & np.isfinite(old) & np.isfinite(new)
        if m.sum() < 10000:
            continue
        res[lab] = {"frac": float(m.mean()), "old_med": float(np.median(old[m])), "new_med": float(np.median(new[m])),
                    "old_p95": float(np.percentile(old[m], 95)), "new_p95": float(np.percentile(new[m], 95))}
    out[name] = res
    print(name, {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res.items()})
(vp.OUT / "_diagnostics" / "nongreen_agl_old_vs_gamusdc.json").write_text(json.dumps(out, indent=1))
