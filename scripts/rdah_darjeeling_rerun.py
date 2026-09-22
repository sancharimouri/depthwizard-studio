#!/usr/bin/env python3
"""
A5 (2026-09-23): RDAH-Net Darjeeling rerun with training-exact RGB preprocessing,
scored with A4's direct height test. INFERENCE ONLY, run once.
Pre-registered rule: docs/method-audit/sentinel2/sign-flip-detector.md,
2026-09-23 (final close-out), "A5 -- RDAH Darjeeling rerun".

Model path reused unmodified: scripts/rdah_sentinel2_zeroshot.Runner (backend/rdah
engine, Swiss checkpoint, reflect-pad to 1024 + crop, depth x255, then the engine's
/255 + ImageNet Normalize, which training also applies -- Phase 2a).
PRIMARY RGB: literal loaddata.py stretch: (image-minv)/(maxv-minv+1e-7), *255, uint8,
  with minv/maxv over the whole image (all channels), as the loader does.
SECONDARY (sensitivity only; cannot pass the gate): joint 2-98 percentile stretch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import direct_height_test as dh  # noqa: E402
from rdah_sentinel2_zeroshot import Runner, periodic_score, read_rgb, SCALE_FIX  # noqa: E402

RGB = ROOT / "data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif"
DEPTH = ROOT / "data/diagnostics/darjeeling/Darjeeling_RGB_depth.npy"
OUT = ROOT / "data/sentinel2_benchmark/rdah_zeroshot/darjeeling_a5"


def minmax_literal(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    minv, maxv = x.min(), x.max()
    x = (x - minv) / (maxv - minv + 1e-7)
    return (x * 255).astype(np.uint8)


def pct_stretch(img: np.ndarray, lo=2, hi=98) -> np.ndarray:
    x = img.astype(np.float64)
    a, b = np.percentile(x, [lo, hi])
    return (np.clip((x - a) / (b - a), 0, 1) * 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb, meta = read_rgb(RGB)
    depth = np.load(DEPTH).astype(np.float32)
    runner = Runner()
    res = {"inputs": {"rgb": str(RGB.relative_to(ROOT)), "depth": str(DEPTH.relative_to(ROOT)),
                      "rgb_min_max": [int(rgb.min()), int(rgb.max())]}}
    for name, fn in (("primary_minmax_literal", minmax_literal), ("secondary_p2_p98", pct_stretch)):
        rgb_s = fn(rgb)
        native, grid = runner.forward_native(rgb_s, depth, SCALE_FIX, "pad")
        np.save(OUT / f"{name}.npy", grid)
        s = dh.score_field(grid.astype(np.float64), RGB, "darjeeling")
        s.pop("_pooled")
        res[name] = {"rgb_changed_vs_input": bool(np.any(rgb_s != rgb)),
                     "max_abs_rgb_change": int(np.abs(rgb_s.astype(int) - rgb.astype(int)).max()),
                     "output_stats": {"median": float(np.median(grid)), "p2": float(np.percentile(grid, 2)),
                                      "p98": float(np.percentile(grid, 98)), "max": float(grid.max())},
                     "scores": s,
                     "fft_descriptive": {k: v["peak_to_background"] for k, v in periodic_score(native).items()}}
    p = res["primary_minmax_literal"]["scores"]
    res["REOPEN"] = bool(p["is2"]["spearman"] >= 0.30 and p["gedi"]["spearman"] >= 0.30)
    (OUT / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
