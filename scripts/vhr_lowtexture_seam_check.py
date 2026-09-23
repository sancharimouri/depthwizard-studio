#!/usr/bin/env python3
"""
Cheap artifact check (2026-09-23): do the five committed VHR crops (original stride-448
feather tiling) contain featureless sub-regions where the glacier's tile-edge seam could
hide? No re-inference; reads the committed agl.tif and the crop's RGB.

Texture = std of the grayscale image in 64x64 blocks. "Featureless" threshold = the 90th
percentile of b_glacier's own block std (the scene where the seam artifact was seen).
Per crop, reports the featureless-block fraction and, if any, the seam ratio (the same
original-layout line definition as C2) restricted to featureless pixels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vhr_dsm_pipeline as vp  # noqa: E402

B = 64


def crop_rgb(name):
    scene, cr, cc, _ = vp.CROPS[name]
    r0, c0 = cr * vp.CELL + (vp.CELL - vp.SIZE) // 2, cc * vp.CELL + (vp.CELL - vp.SIZE) // 2
    with rasterio.open(vp.ROOT / "data/maxar_sanity" / vp.SCENES[scene]) as src:
        return src.read([1, 2, 3], window=Window(c0, r0, vp.SIZE, vp.SIZE)).astype(np.float32)


def block_std(rgb):
    g = rgb.mean(0)
    n = vp.SIZE // B
    return g.reshape(n, B, n, B).std(axis=(1, 3))


def main():
    thr = float(np.percentile(block_std(crop_rgb("b_glacier")), 90))
    out = {"texture_threshold_graylevel_std": thr, "crops": {}}
    lines = vp.old_layout_lines(vp.SIZE)
    for name in vp.CROPS:
        bs = block_std(crop_rgb(name))
        low = np.kron(bs < thr, np.ones((B, B), bool))
        with rasterio.open(vp.OUT / name / "agl.tif") as src:
            agl = src.read(1)
        v = np.isfinite(agl)
        r = {"featureless_block_frac": float((bs < thr).mean())}
        m = v & low
        if m.sum() > 50 * B * B:  # at least ~50 featureless blocks for a meaningful ratio
            r["seam_ratio_in_featureless"] = vp.seam_ratio(np.where(m, agl, np.nan), m, None)
        r["seam_ratio_all"] = vp.seam_ratio(np.where(v, agl, np.nan), v, None)
        # Looser, crop-relative check: the crop's own least-textured 10% of blocks.
        own = np.kron(bs <= np.percentile(bs, 10), np.ones((B, B), bool)) & v
        r["own_lowest_decile_block_std_max"] = float(np.percentile(bs, 10))
        r["frac_blocks_below_2x_threshold"] = float((bs < 2 * thr).mean())
        r["seam_ratio_in_own_lowest_decile"] = vp.seam_ratio(np.where(own, agl, np.nan), own, None)
        out["crops"][name] = r
        print(name, {k: round(x, 3) for k, x in r.items()})
    (vp.OUT / "_diagnostics" / "lowtexture_seam_check.json").write_text(json.dumps(out, indent=1))
    print("threshold", round(thr, 2), "lines", len(lines))


if __name__ == "__main__":
    main()
