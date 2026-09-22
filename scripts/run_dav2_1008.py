#!/usr/bin/env python3
"""Phase 4 candidate (ii): frozen DAv2 at 1008x1008 (72x14 patches) on the 25
accepted Sentinel-2 tiles. backend/depth/depth_engine.py reused unmodified;
only backend.config.MODEL_INPUT_SIZE is changed 518 -> 1008 before inference
(infer_with reads it at call time). No training."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config
config.MODEL_INPUT_SIZE = 1008
from backend.depth import depth_engine
OUT = ROOT / "data/sentinel2_benchmark/dav2_depth_1008"
m = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv").set_index("tile_id")
v = pd.read_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv").set_index("tile_id")
for t in v[~v["flagged"]].index:
    p = OUT / f"{t}_depth.npy"
    if p.exists():
        continue
    img = depth_engine.load_geotiff_rgb(ROOT / m.loc[t, "rgb_path"])
    t0 = time.perf_counter(); d = depth_engine.run_inference(img)
    np.save(p, d); print(f"{t}: {d.shape} {time.perf_counter()-t0:.2f}s", flush=True)
