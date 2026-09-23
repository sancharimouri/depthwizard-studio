#!/usr/bin/env python3
"""
Method 6 -> absolute DSM on real VHR imagery, end to end (2026-09-23). INFERENCE ONLY.
Design + plausibility criteria (committed before running):
docs/method-audit/06-full-finetune-twin-head/vhr_dsm_pipeline.md

For each crop of a georeferenced VHR GeoTIFF:
  1. AGL  = mean of the 4 seed-43 height-balanced fold checkpoints, tiled 512 px
            (training size), stride 448, feathered blending; also the full-data
            checkpoint as a cross-check, fold spread, and predicted sigma.
  2. DTM  = FABDEM (bare earth, EGM2008) resampled BICUBICALLY onto the crop grid (EE).
  3. DSM  = backend.terrain.composer.compose_dsm(DTM, AGL).
  4. Writes GeoTIFFs, PNGs, stats, and the frontend asset set (terrain.json etc.).
  5. Plausibility stats C1-C5 (see the doc).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from backend.terrain.composer import compose_dsm  # noqa: E402
from backend.terrain.mesh_export import block_mean, write_terrain_json  # noqa: E402
import evaluate_method6_gsd_film_height_balanced as hb  # noqa: E402
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, PAD_TO, TwinHeadDav2, get_device, pad_to,
)

TILE, STRIDE, RAMP = 512, 448, 64
# --margin=M mode (seam fix): discard an M-px ring per window; cores blended over MARGIN_RAMP px.
MARGIN_RAMP = 16
SIZE = 2048
CELL = 17408 // 8
SCENES = {"C": "45_120220211230_2022-03-14_1040010073381800.tif",
          "A": "45_120220122200_2022-03-07_10300100CE8D0400.tif",
          "B": "45_120220031201_2022-03-07_10300100CF621C00.tif"}
# (scene, overview cell row, cell col, land cover); crop centred in the 2176-px overview cell
CROPS = {"c_town": ("C", 7, 7, "dense hill town"),
         "c_river": ("C", 3, 2, "braided river / sandbars / fields"),
         "c_terraces": ("C", 5, 5, "terraced farmland, scattered houses"),
         "a_forest": ("A", 3, 1, "dense conifer forest"),
         "a_valley": ("A", 7, 2, "river valley with settlement"),
         "b_glacier": ("B", 7, 5, "snow/ice (negative control)")}
OUT = ROOT / "data/vhr_dsm"
PREVIEW = ROOT / "frontend/public/data/vhr"


def load_models(device):
    folds = []
    for q in range(4):
        ck = torch.load(ROOT / f"data/dfc2019/experiments/method6_height_balanced_seed43/fold{q}.pt",
                        map_location="cpu", weights_only=False)
        m = hb.TwinHeadDav2GSD(height_scale=ck["height_scale"], init_sigma_m=5.0, log_var_max=7.0,
                               log_var_min=-8.0, enable_gsd_film=False)
        m.load_state_dict(ck["state_dict"])
        folds.append(m.to(device).eval())
    ck = torch.load(ROOT / "data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt",
                    map_location="cpu", weights_only=False)
    full = TwinHeadDav2(height_scale=ck["height_scale"], **ck["config"])
    full.load_state_dict(ck["model"])
    return folds, full.to(device).eval()


def starts(n):
    s = list(range(0, n - TILE + 1, STRIDE))
    if s[-1] != n - TILE:
        s.append(n - TILE)
    return s


def feather():
    r = np.minimum(1.0, np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)) / RAMP)
    return np.outer(r, r).astype(np.float32)


@torch.no_grad()
def tiled_predict(model, rgb: np.ndarray, device):
    """rgb (3,H,W) uint8 -> (mu, sigma) (H,W), feather-blended 512 tiles."""
    _, H, W = rgb.shape
    wts, acc_mu, acc_sg = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    fw = feather()
    for r in starts(H):
        for c in starts(W):
            x = torch.from_numpy(rgb[:, r:r + TILE, c:c + TILE] / 255.0)
            x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            mu, lv = model(pad_to(x.float(), PAD_TO)[None].to(device))
            mu = mu[0, 0, :TILE, :TILE].float().cpu().numpy()
            sg = torch.exp(0.5 * lv)[0, 0, :TILE, :TILE].float().cpu().numpy()
            acc_mu[r:r + TILE, c:c + TILE] += fw * mu
            acc_sg[r:r + TILE, c:c + TILE] += fw * sg
            wts[r:r + TILE, c:c + TILE] += fw
    return acc_mu / wts, acc_sg / wts


@torch.no_grad()
def tiled_predict_margin(model, rgb: np.ndarray, device, margin: int):
    """Seam fix (2026-09-23): reflect-pad the image by `margin` px, run the same 512 px
    windows, but keep only each window's central (512 - 2*margin) px core; the outer
    `margin` ring of every window is discarded, so no kept pixel is within `margin` px of
    a window edge. Cores overlap by MARGIN_RAMP px and are linearly blended there."""
    _, H, W = rgb.shape
    core = TILE - 2 * margin
    step = core - MARGIN_RAMP
    x_all = np.pad(rgb, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
    r1 = np.zeros(TILE, np.float32)
    i = np.arange(core)
    r1[margin:margin + core] = np.minimum(1.0, np.minimum(i + 1, core - i) / MARGIN_RAMP)
    fw = np.outer(r1, r1)
    def st(n):
        s = list(range(0, n - core + 1, step))
        if s[-1] != n - core:
            s.append(n - core)
        return s
    Hp, Wp = H + 2 * margin, W + 2 * margin
    wts, acc_mu, acc_sg = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    for r in st(H):
        for c in st(W):
            x = torch.from_numpy(x_all[:, r:r + TILE, c:c + TILE] / 255.0)
            x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            mu, lv = model(pad_to(x.float(), PAD_TO)[None].to(device))
            mu = mu[0, 0, :TILE, :TILE].float().cpu().numpy()
            sg = torch.exp(0.5 * lv)[0, 0, :TILE, :TILE].float().cpu().numpy()
            acc_mu[r:r + TILE, c:c + TILE] += fw * mu
            acc_sg[r:r + TILE, c:c + TILE] += fw * sg
            wts[r:r + TILE, c:c + TILE] += fw
    sl = (slice(margin, margin + H), slice(margin, margin + W))
    core_edges = sorted({e for s in st(H) for e in (s, s + core)})
    return acc_mu[sl] / wts[sl], acc_sg[sl] / wts[sl], core_edges


def old_layout_lines(n: int) -> list[int]:
    lines = set()
    for s in starts(n)[1:]:
        lines.update({s, s + 1})
    for s in starts(n)[:-1]:
        lines.update({s + TILE - 1, s + TILE - 2})
    return sorted(lines)


def seam_ratio(a: np.ndarray, valid: np.ndarray, lines: list[int] | None = None) -> float:
    """Mean |grad| on tile-boundary lines / elsewhere. Default lines = the original
    stride-448 layout's window edges."""
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    if lines is None:
        lines = old_layout_lines(a.shape[0])  # unchanged definition used for the committed C2 numbers
    else:
        lines = set(lines) | {x - 1 for x in lines}  # 2-px band at each core edge, same width as default
    lines = sorted(x for x in lines if 2 <= x < a.shape[0] - 2)
    seam = np.zeros_like(valid)
    seam[lines, :] = True
    seam[:, lines] = True
    return float(np.nanmean(g[seam & valid]) / np.nanmean(g[~seam & valid]))


def ee_fabdem(transform, crs, shape):
    import ee
    col = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    # bicubic, not bilinear: bilinear from 30 m leaves planar facets whose slope breaks show as a
    # 30 m grid in hillshades (post-hoc change, 2026-09-23; see the design doc).
    img = col.mosaic().setDefaultProjection(col.first().projection()).select("b1").resample("bicubic")
    T = transform
    arr = ee.data.computePixels({"expression": img.unmask(-9999), "fileFormat": "NUMPY_NDARRAY",
                                 "grid": {"dimensions": {"width": shape[1], "height": shape[0]},
                                          "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                                              "shearY": T.d, "scaleY": T.e, "translateY": T.f},
                                          "crsCode": crs.to_string()}})
    a = np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr, dtype=np.float32)
    a[a <= -9000] = np.nan
    return a


def glo30_on_grid(transform, crs, shape, lonlat_bounds):
    import math
    w, s, e, n = lonlat_bounds
    out = np.full(shape, np.nan, np.float32)
    for la in range(math.floor(s), math.floor(n) + 1):
        for lo in range(math.floor(w), math.floor(e) + 1):
            name = f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM"
            with rasterio.open(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif") as src:
                tmp = np.full(shape, np.nan, np.float32)
                reproject(rasterio.band(src, 1), tmp, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
                out = np.where(np.isfinite(tmp), tmp, out)
    return out


def write_tif(path, arr, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate") as d:
        d.write(arr.astype(np.float32), 1)


def hillshade(z, res, az=315, alt=45):
    gy, gx = np.gradient(np.where(np.isfinite(z), z, np.nanmedian(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, al = np.radians(az), np.radians(alt)
    hs = np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def sparse_lidar(poly, transform, crs, shape, agl):
    """GEDI rh98 (EE) and ICESat-2 20 m h_max_canopy (sliderule) inside the crop, vs AGL."""
    import fetch_gedi_l2a as fg
    from pyproj import Transformer
    res = {}
    try:
        g = fg.fetch(poly)
        g = g[g["rh98"] <= 80]
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(g["lon"].values, g["lat"].values)
        cols, rows = ~transform * (np.asarray(x), np.asarray(y))
        pairs = []
        for r0, c0, rh in zip(rows, cols, g["rh98"].values):
            rr, cc = np.mgrid[int(r0) - 41:int(r0) + 42, int(c0) - 41:int(c0) + 42]  # 12.5 m radius at 0.305 m
            m = (np.hypot(rr - r0, cc - c0) * 0.305 <= 12.5) & (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            if m.sum() > 100:
                v = agl[rr[m], cc[m]]
                v = v[np.isfinite(v)]
                if v.size:
                    pairs.append((float(np.percentile(v, 98)), float(rh)))
        res["gedi"] = {"n": len(pairs), "pairs_agl_p98_vs_rh98": pairs[:200]}
        if len(pairs) >= 5:
            a, b = np.array(pairs).T
            res["gedi"].update({"spearman": float(stats.spearmanr(a, b)[0]), "median_agl_p98": float(np.median(a)),
                                "median_rh98": float(np.median(b))})
    except Exception as e:  # noqa: BLE001
        res["gedi"] = {"error": f"{type(e).__name__}: {e}"}
    return res


def main(only=None):
    import ee
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    from pyproj import Transformer
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    from PIL import Image

    device = get_device()
    folds, full = load_models(device)
    summary = {}
    for name, (scene, cr, cc, cover) in CROPS.items():
        if only and name not in only:
            continue
        r0, c0 = cr * CELL + (CELL - SIZE) // 2, cc * CELL + (CELL - SIZE) // 2
        with rasterio.open(ROOT / "data/maxar_sanity" / SCENES[scene]) as src:
            rgb = src.read([1, 2, 3], window=Window(c0, r0, SIZE, SIZE))
            transform = src.window_transform(Window(c0, r0, SIZE, SIZE))
            crs = src.crs
        valid = ~np.all(rgb == 0, axis=0)
        # --- AGL: fold ensemble + full-data cross-check
        if MARGIN:
            preds = [tiled_predict_margin(m, rgb, device, MARGIN) for m in folds]
            core_edges = preds[0][2]
            fold_mu, fold_sg = [p[0] for p in preds], [p[1] for p in preds]
        else:
            fold_mu, fold_sg = zip(*[tiled_predict(m, rgb, device) for m in folds])
        fold_mu = np.stack(fold_mu)
        agl = fold_mu.mean(0)
        agl_std = fold_mu.std(0)
        sigma = np.stack(fold_sg).mean(0)
        agl_full = tiled_predict_margin(full, rgb, device, MARGIN)[0] if MARGIN else tiled_predict(full, rgb, device)[0]
        agl[~valid] = np.nan
        # --- terrain + surface references
        t2g = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = [transform.c, transform.c + SIZE * transform.a]
        ys = [transform.f, transform.f + SIZE * transform.e]
        lons, lats = t2g.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        bounds = (min(lons), min(lats), max(lons), max(lats))
        dtm = ee_fabdem(transform, crs, (SIZE, SIZE))
        glo = glo30_on_grid(transform, crs, (SIZE, SIZE), bounds)
        comp = compose_dsm(dtm, agl, valid)
        dsm = comp.dsm
        # --- write rasters
        out_name = f"{name}_margin{MARGIN}" if MARGIN else name
        d = OUT / out_name
        d.mkdir(parents=True, exist_ok=True)
        for fn, arr in (("agl", agl), ("agl_std", agl_std), ("sigma", sigma), ("agl_fullckpt", agl_full),
                        ("dtm_fabdem", dtm), ("dsm", dsm), ("glo30", glo)):
            write_tif(d / f"{fn}.tif", arr, transform, crs)
        # --- plausibility stats
        v = valid & np.isfinite(agl)
        a = agl[v]
        pct = {p: float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
        blk = SIZE // 100  # ~30.5 m blocks
        agl_b = block_mean(np.where(v, agl, np.nan), (blk, blk))
        excess_b = block_mean(glo - dtm, (blk, blk))
        okb = np.isfinite(agl_b) & np.isfinite(excess_b)
        dsm_b = block_mean(dsm, (blk, blk))
        glo_b = block_mean(glo, (blk, blk))
        okd = np.isfinite(dsm_b) & np.isfinite(glo_b)
        hi = v & (agl > 2)
        s = {"land_cover": cover, "scene": SCENES[scene], "window_row_col": [int(r0), int(c0)], "size_px": SIZE,
             "gsd_m": float(transform.a), "crs": crs.to_string(), "bounds_lonlat": bounds,
             "valid_frac": float(valid.mean()),
             "agl_percentiles_m": pct, "agl_max_m": float(np.nanmax(a)), "agl_mean_m": float(a.mean()),
             "frac_agl_negative": comp.frac_negative_agl,
             "seam_ratio": seam_ratio(np.where(v, agl, np.nan), v),
             "pearson_ensemble_vs_fullckpt": float(stats.pearsonr(a, agl_full[v])[0]),
             "median_fold_std_over_agl_where_agl_gt2": float(np.median(agl_std[hi] / agl[hi])) if hi.any() else None,
             "median_fold_std_m": float(np.median(agl_std[v])), "median_sigma_m": float(np.median(sigma[v])),
             "coarse_ref": {"n_blocks": int(okb.sum()),
                            "spearman_aglblock_vs_glo30_minus_fabdem": float(stats.spearmanr(agl_b[okb], excess_b[okb])[0]) if okb.sum() > 10 else None,
                            "median_glo30_minus_fabdem_m": float(np.median(excess_b[okb])) if okb.any() else None,
                            "median_aglblock_m": float(np.median(agl_b[okb])) if okb.any() else None,
                            "median_dsm_minus_glo30_m": float(np.median((dsm_b - glo_b)[okd])) if okd.any() else None},
             "dtm_range_m": [float(np.nanmin(dtm)), float(np.nanmax(dtm))],
             "dsm_range_m": [float(np.nanmin(dsm)), float(np.nanmax(dsm))]}
        poly = [{"lon": lo, "lat": la} for lo, la in [(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3]), (bounds[0], bounds[1])]]
        s["sparse_lidar"] = sparse_lidar(poly, transform, crs, (SIZE, SIZE), agl)
        if MARGIN:
            s["margin_px"] = MARGIN
            s["seam_ratio_own_core_edges"] = seam_ratio(np.where(v, agl, np.nan), v, [e for e in core_edges if 2 <= e < SIZE - 2])
        (d / "stats.json").write_text(json.dumps(s, indent=2) + "\n")
        summary[out_name] = s
        # --- images
        rgb8 = rgb.transpose(1, 2, 0)
        agl_img = (cm.viridis(np.clip(np.nan_to_num(agl, nan=0) / 30.0, 0, 1))[..., :3] * 255).astype(np.uint8)
        hs_dsm = (hillshade(dsm, transform.a) * 255).astype(np.uint8)
        hs_dtm = (hillshade(dtm, transform.a) * 255).astype(np.uint8)
        panel = np.concatenate([rgb8, agl_img, np.repeat(hs_dsm[..., None], 3, 2), np.repeat(hs_dtm[..., None], 3, 2)], axis=1)
        Image.fromarray(panel).resize((panel.shape[1] // 4, panel.shape[0] // 4)).save(d / "panel_rgb_agl_dsmhs_dtmhs.png")
        z = (rgb8.shape[0] // 2 - 256, rgb8.shape[1] // 2 - 256)
        zoom = np.concatenate([rgb8[z[0]:z[0] + 512, z[1]:z[1] + 512], agl_img[z[0]:z[0] + 512, z[1]:z[1] + 512],
                               np.repeat(hs_dsm[z[0]:z[0] + 512, z[1]:z[1] + 512, None], 3, 2)], axis=1)
        Image.fromarray(zoom).save(d / "zoom_rgb_agl_dsmhs.png")
        # --- frontend asset set (viewer.js schema), for the preview page only
        p = PREVIEW / out_name
        p.mkdir(parents=True, exist_ok=True)
        meta = write_terrain_json(p / "terrain.json", dsm, bounds, (512, 512))
        Image.fromarray(rgb8).resize((1024, 1024)).save(p / "satellite.png")
        dn = (dsm - np.nanmin(dsm)) / (np.nanmax(dsm) - np.nanmin(dsm))
        Image.fromarray((cm.terrain(np.nan_to_num(dn, nan=0))[..., :3] * 255).astype(np.uint8)).resize((1024, 1024)).save(p / "elevation.png")
        Image.fromarray((np.clip(np.nan_to_num(agl, nan=0) / 30.0, 0, 1) * 255).astype(np.uint8)).resize((1024, 1024)).save(p / "relative_depth.png")
        s["frontend_terrain_json"] = meta
        (d / "stats.json").write_text(json.dumps(s, indent=2) + "\n")
        print(f"{name}: AGL p50 {pct[50]:.2f} p95 {pct[95]:.2f} max {s['agl_max_m']:.1f} m | seam {s['seam_ratio']:.2f} | "
              f"r(ens,full) {s['pearson_ensemble_vs_fullckpt']:.3f} | rho(AGL, GLO-FAB) {s['coarse_ref']['spearman_aglblock_vs_glo30_minus_fabdem']} | "
              f"GEDI n={s['sparse_lidar']['gedi'].get('n')}", flush=True)
    # A full default run owns summary.json; subset/margin runs write their own file.
    fn = "summary.json" if not only and not MARGIN else "summary_" + "_".join(sorted(summary)) + ".json"
    (OUT / fn).write_text(json.dumps(summary, indent=2) + "\n")


MARGIN = 0

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--margin=")]
    MARGIN = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--margin=")), 0)
    main(set(args) or None)
