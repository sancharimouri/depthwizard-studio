#!/usr/bin/env python3
"""
RDAH-Net zero-shot on Sentinel-2: does the DFC2019 input-scale fix transfer?
INFERENCE ONLY -- no training, no gradient steps.

Log: docs/method-audit/sentinel2/sign-flip-detector.md (2026-09-23 entry).

Model + preprocessing are reused unmodified from backend/rdah/rdah_engine.py
(the engine that produced the original Darjeeling checkerboard run,
reproduced bit-exactly with the Swiss checkpoint). The only things varied:
  - depth input scale: x1 (original) vs. x255 (DFC2019 fix, applied to the
    DAv2 [0,1] depth BEFORE the forward pass, as in train_rdah_quadrant_cv.py)
  - size handling: bilinear resize to 1024 (original) vs. reflect-pad to
    1024 and crop (RDAH needs H,W multiples of 128 and <=1024 for its 64x64
    positional-encoding buffer; 1024 is the only native size >= 1000)

Usage:
    python scripts/rdah_sentinel2_zeroshot.py darjeeling
    python scripts/rdah_sentinel2_zeroshot.py benchmark
    python scripts/rdah_sentinel2_zeroshot.py production
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.warp import Resampling, reproject
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backend.rdah.rdah_engine import RDAHEngine  # noqa: E402

SWISS_CKPT = PROJECT_ROOT / "external" / "RDAH-Net" / "swiss_best_model.pth"
NATIVE = 1024
SCALE_FIX = 255.0
# Periods (input px at the 1024 native grid) where an architecture-induced
# periodic artifact would sit: PixelShuffle x2 stages (2,4,8,16) and hard
# 8x8 non-overlapping block-attention boundaries at feature strides 4/8/16
# (32,64,128).
PERIODS = (2, 4, 8, 16, 32, 64, 128)
OUT_DIR = PROJECT_ROOT / "data" / "sentinel2_benchmark" / "rdah_zeroshot"


# ---------------------------------------------------------------- inference
class Runner:
    def __init__(self, ckpt: Path = SWISS_CKPT):
        self.engine = RDAHEngine(ckpt)

    def forward_native(self, rgb: np.ndarray, depth: np.ndarray, scale: float, mode: str):
        """Returns (native 1024x1024 model output, output on the input grid)."""
        h, w = depth.shape
        if mode == "resize":  # the original engine's path, verbatim
            rgb_n = np.asarray(Image.fromarray(rgb, mode="RGB").resize((NATIVE, NATIVE), Image.Resampling.BILINEAR))
            dep_n = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (NATIVE, NATIVE), Image.Resampling.BILINEAR), dtype=np.float32)
        elif mode == "pad":
            if h > NATIVE or w > NATIVE:
                raise ValueError(f"tile {h}x{w} larger than {NATIVE}")
            pad = ((0, NATIVE - h), (0, NATIVE - w))
            rgb_n = np.pad(rgb, pad + ((0, 0),), mode="reflect")
            dep_n = np.pad(depth.astype(np.float32), pad, mode="reflect")
        else:
            raise ValueError(mode)

        e = self.engine
        rgb_t = e._prepare_rgb(rgb_n).unsqueeze(0).to(e.device)
        dep_t = e._prepare_depth(dep_n * scale).unsqueeze(0).to(e.device)
        with torch.no_grad():
            out = e.model(dep_t, rgb_t).squeeze().detach().cpu().numpy().astype(np.float32)

        if mode == "resize":
            grid = np.asarray(Image.fromarray(out, mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
        else:
            grid = out[:h, :w].copy()
        return out, grid


# ---------------------------------------------------------------- metrics
def plane_design(x, y, order):
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def detrend(values, x, y, order):
    """Residual after least-squares polynomial surface fit in (x, y)."""
    xs = (x - x.mean()) / (x.std() + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    A = plane_design(xs, ys, order)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    return values - A @ coef


def corr_set(pred, truth, x, y):
    out = {"n": int(len(pred))}
    for label, order in (("raw", 0), ("plane", 1), ("quadratic", 2)):
        p = pred if order == 0 else detrend(pred, x, y, order)
        t = truth if order == 0 else detrend(truth, x, y, order)
        out[label] = {"pearson": float(stats.pearsonr(p, t)[0]),
                      "spearman": float(stats.spearmanr(p, t)[0])}
    return out


def periodic_score(img: np.ndarray) -> dict:
    """Peak-to-background power at each candidate period, on a planar-
    detrended, Hann-windowed 2D FFT. Checks axis bins (0,k),(k,0) and the
    diagonal (k,k) (checkerboard) for k = N/period; background = median
    power over the annulus of the same radius (+-10%)."""
    a = img.astype(np.float64)
    n = a.shape[0]
    assert a.shape == (n, n)
    yy, xx = np.mgrid[0:n, 0:n]
    a = detrend(a.ravel(), xx.ravel().astype(float), yy.ravel().astype(float), 1).reshape(n, n)
    win = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    c = n // 2
    fy, fx = np.mgrid[-c:n - c, -c:n - c]
    r = np.hypot(fx, fy)
    res = {}
    for p in PERIODS:
        k = n // p
        cands = [(0, k), (k, 0), (k, k), (-k, k)]
        best = 0.0
        best_bin = None
        for dy, dx in cands:
            iy, ix = (c + dy) % n, (c + dx) % n
            rad = np.hypot(dx, dy)
            ann = (r > 0.9 * rad) & (r < 1.1 * rad)
            ann &= ~((np.abs(fy - dy) <= 1) & (np.abs(fx - dx) <= 1))
            ann &= ~((np.abs(fy + dy) <= 1) & (np.abs(fx + dx) <= 1))
            bg = float(np.median(P[ann])) if ann.any() else float("nan")
            ratio = float(P[iy, ix] / bg) if bg > 0 else float("nan")
            if ratio > best:
                best, best_bin = ratio, (dy, dx)
        res[str(p)] = {"peak_to_background": best, "bin": best_bin}
    return res


# ---------------------------------------------------------------- IO helpers
def read_rgb(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as s:
        rgb = np.moveaxis(s.read([1, 2, 3]), 0, -1)
        meta = {"crs": s.crs, "transform": s.transform, "shape": (s.height, s.width)}
    if rgb.dtype != np.uint8:  # same rule as backend/depth/depth_engine.load_geotiff_rgb
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, meta


def dem_on_grid(dem_path: Path, meta: dict) -> np.ndarray:
    dst = np.full(meta["shape"], np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=meta["transform"], dst_crs=meta["crs"], dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return dst


def photons_on_grid(photons: pd.DataFrame, meta: dict):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    x, y = t.transform(photons["lon"].values, photons["lat"].values)
    cols, rows = ~meta["transform"] * (x, y)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    h, w = meta["shape"]
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    return rows[ok], cols[ok], photons["height"].values[ok], np.asarray(x)[ok], np.asarray(y)[ok]


def fetch_photons(meta: dict, out_csv: Path) -> pd.DataFrame:
    """Reuses scripts/query_icesat2_photons.py's query_tile unmodified."""
    if out_csv.exists():
        return pd.read_csv(out_csv)
    from pyproj import Transformer
    from sliderule import sliderule
    import query_icesat2_photons as q
    sliderule.init("slideruleearth.io", verbose=False)
    h, w = meta["shape"]
    T = meta["transform"]
    x0, y0 = T * (0, 0)
    x1, y1 = T * (w, h)
    bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    poly = q.utm_bbox_to_wgs84_poly(bbox, meta["crs"].to_epsg())
    df = q.query_tile(poly)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def to_png(a: np.ndarray, lo=2, hi=98) -> np.ndarray:
    v = a[np.isfinite(a)]
    l, h = np.percentile(v, lo), np.percentile(v, hi)
    return (np.clip((np.nan_to_num(a, nan=l) - l) / (h - l + 1e-9), 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- Step 2
def run_darjeeling(args):
    rgb_path = Path(args.rgb)
    depth_path = PROJECT_ROOT / "data/diagnostics/darjeeling/Darjeeling_RGB_depth.npy"
    dem_path = PROJECT_ROOT / "data/elevation/darjeeling/Darjeeling_OpenTopography_DSM.tif"
    old_path = PROJECT_ROOT / "data/diagnostics/darjeeling/rdah/Darjeeling_RDAH_nDSM.npy"
    out = OUT_DIR / "darjeeling"
    out.mkdir(parents=True, exist_ok=True)

    rgb, meta = read_rgb(rgb_path)
    depth = np.load(depth_path).astype(np.float32)
    assert depth.shape == meta["shape"], (depth.shape, meta["shape"])
    old = np.load(old_path)

    runner = Runner()
    variants = {}
    for name, scale, mode in [("orig_resize_x1", 1.0, "resize"), ("pad_x1", 1.0, "pad"),
                              ("resize_x255", SCALE_FIX, "resize"), ("pad_x255", SCALE_FIX, "pad")]:
        native, grid = runner.forward_native(rgb, depth, scale, mode)
        variants[name] = {"native": native, "grid": grid}
        np.save(PROJECT_ROOT / "data/diagnostics/darjeeling/rdah" / f"{name}.npy", grid)
    assert np.array_equal(variants["orig_resize_x1"]["grid"], old), "original run not reproduced"

    # Inputs as controls for the FFT: DAv2 depth and RGB luminance at the native grid (padded)
    pad = ((0, NATIVE - depth.shape[0]), (0, NATIVE - depth.shape[1]))
    controls = {"dav2_depth_input": np.pad(depth, pad, mode="reflect"),
                "rgb_luma_input": np.pad(rgb.astype(np.float32).mean(axis=2), pad, mode="reflect")}

    dem = dem_on_grid(dem_path, meta)
    photons = fetch_photons(meta, PROJECT_ROOT / "data/icesat2_photons/darjeeling.csv")
    pr, pc, ph, px, py = photons_on_grid(photons, meta)

    h, w = meta["shape"]
    gy, gx = np.mgrid[0:h, 0:w]
    valid = np.isfinite(dem)
    rng = np.random.RandomState(0)
    sub = rng.choice(np.flatnonzero(valid.ravel()), size=min(200_000, int(valid.sum())), replace=False)

    def score(field):
        f = field.ravel()
        d = corr_set(f[sub], dem.ravel()[sub], gx.ravel()[sub].astype(float), gy.ravel()[sub].astype(float))
        i = corr_set(field[pr, pc], ph, px, py)
        return {"dem": d, "icesat2": i}

    results = {
        "inputs": {"rgb": str(rgb_path.relative_to(PROJECT_ROOT)) if rgb_path.is_relative_to(PROJECT_ROOT) else str(rgb_path),
                   "depth": str(depth_path.relative_to(PROJECT_ROOT)), "dem": str(dem_path.relative_to(PROJECT_ROOT)),
                   "checkpoint": str(SWISS_CKPT.relative_to(PROJECT_ROOT)), "grid": list(meta["shape"]),
                   "n_photons_in_tile": int(len(ph)), "dem_valid_px": int(valid.sum()),
                   "dem_px_sampled_for_corr": int(len(sub))},
        "original_reproduced_exactly": True,
        "dem_range_m": [float(np.nanmin(dem)), float(np.nanmax(dem))],
        "icesat2_height_range_m": [float(ph.min()), float(ph.max())] if len(ph) else None,
        "variants": {}, "controls_fft": {k: periodic_score(v) for k, v in controls.items()},
    }
    for name, v in variants.items():
        g = v["grid"]
        results["variants"][name] = {
            "output_stats": {"min": float(g.min()), "p2": float(np.percentile(g, 2)), "median": float(np.median(g)),
                             "p98": float(np.percentile(g, 98)), "max": float(g.max()), "std": float(g.std())},
            "fft_native": periodic_score(v["native"]),
            "correlation": score(g),
        }
    results["dav2_reference"] = {"correlation": score(depth)}
    (out / "darjeeling_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # side-by-side PNG: old run / new run / DEM (+ DAv2 input for reference)
    panels = [to_png(variants["orig_resize_x1"]["grid"]), to_png(variants["pad_x255"]["grid"]),
              to_png(dem), to_png(depth)]
    gap = np.full((h, 8), 255, np.uint8)
    strip = np.concatenate(sum([[p, gap] for p in panels], [])[:-1], axis=1)
    Image.fromarray(strip).save(out / "darjeeling_old_new_dem_dav2.png")
    # zoomed 256x256 crop, nearest-upscaled x2, to make any grid pattern visible
    zy, zx = h // 2 - 128, w // 2 - 128
    zoom = [np.kron(p[zy:zy + 256, zx:zx + 256], np.ones((2, 2), np.uint8)) for p in panels]
    gapz = np.full((512, 8), 255, np.uint8)
    Image.fromarray(np.concatenate(sum([[p, gapz] for p in zoom], [])[:-1], axis=1)).save(out / "darjeeling_zoom_center.png")
    print(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("darjeeling")
    d.add_argument("--rgb", required=True, help="Darjeeling RGB GeoTIFF (the committed 1007x1002 version)")
    args = ap.parse_args()
    if args.cmd == "darjeeling":
        run_darjeeling(args)


if __name__ == "__main__":
    main()
