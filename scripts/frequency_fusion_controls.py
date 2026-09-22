#!/usr/bin/env python3
"""
Phase 1 (2026-09-23): DEM-only controls for frequency fusion, and a direct
test of whether DAv2's high-frequency detail tracks ground photons.
Evaluation only -- no training. Pre-registered rule:
docs/method-audit/sentinel2/sign-flip-detector.md, 2026-09-23 (continued).

Reuses scripts/run_frequency_fusion_sentinel2.py's functions unmodified
(masked_gaussian, native_resolution_m, n_egm96, rmse, constants) and
run_srtm_comparison.py's n_egm2008 / COP_NODATA. Variants:
  (i)   fusion (must reproduce frequency_fusion_results.csv)
  (ii)  dem_lowpass only
  (iii) raw SRTM dem_on_grid
  (iv)  raw GLO-30 dem_on_grid (EGM2008)
  (v)   fusion with GLO-30 as the DEM

`build_tile()` is also imported by later phases (detail-source bake-off).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_frequency_fusion_sentinel2 as ff  # noqa: E402
import run_srtm_comparison as sc  # noqa: E402
from lib_backbone_correlation import sample_depth_at_photons  # noqa: E402

ROOT = ff.ROOT
OUT_DIR = ROOT / "data/sentinel2_benchmark/frequency_fusion_controls"
COP_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"


def dem_to_grid(path: Path, rgb_path: Path, nodata=None) -> np.ndarray:
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        st, sc_, snd = src.transform, src.crs, (nodata if nodata is not None else src.nodata)
    with rasterio.open(rgb_path) as r:
        dcrs, dt, shp = r.crs, r.transform, (r.height, r.width)
    out = np.full(shp, np.nan, dtype=np.float32)
    reproject(source=dem, destination=out, src_transform=st, src_crs=sc_, src_nodata=snd,
              dst_transform=dt, dst_crs=dcrs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def fuse(dem_on_grid: np.ndarray, detail_raw: np.ndarray, sigma_px: float, fit_sign: bool = True):
    """Exactly ff.process_tile's steps 1,3,4: seed-42 50/50 split, OLS of
    dem on detail_raw (train half), matched masked Gaussian low-pass."""
    valid = ~np.isnan(dem_on_grid)
    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(ff.SEED)
    rng.shuffle(idx)
    train_idx = idx[: len(idx) // 2]
    lr = stats.linregress(detail_raw.ravel()[train_idx], dem_on_grid.ravel()[train_idx])
    a, b = lr.slope, lr.intercept
    metric = (a * detail_raw + b).astype(np.float32)
    dem_low = ff.masked_gaussian(dem_on_grid, valid, sigma_px)
    det_low = ff.masked_gaussian(metric, np.ones_like(valid, dtype=bool), sigma_px)
    highpass = metric - det_low
    raw_low = ff.masked_gaussian(detail_raw.astype(np.float32), np.ones_like(valid, dtype=bool), sigma_px)
    return {"a": a, "b": b, "dem_lowpass": dem_low, "highpass": highpass,
            "raw_highpass": detail_raw - raw_low, "fused": dem_low + highpass}


def photon_pixels(rgb_path: Path, photons: pd.DataFrame, shape) -> pd.DataFrame:
    """Photon -> pixel (row, col), same projection path as sample_depth_at_photons."""
    from pyproj import Transformer
    with rasterio.open(rgb_path) as src:
        crs, T = src.crs, src.transform
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(photons["lon"].values, photons["lat"].values)
    cols, rows = ~T * (x, y)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    ok = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return pd.DataFrame({"row": rows[ok], "col": cols[ok], "height": photons["height"].values[ok]})


def r_hf(highpass: np.ndarray, dem_low: np.ndarray, pix: pd.DataFrame, n: float) -> dict:
    """Median photon height per 10 m pixel -> orthometric (h + n, n = -N)
    -> residual vs dem_lowpass; correlate with highpass at that pixel."""
    g = pix.groupby(["row", "col"])["height"].median().reset_index()
    resid = (g["height"].values + n) - dem_low[g["row"].values, g["col"].values]
    hp = highpass[g["row"].values, g["col"].values]
    ok = np.isfinite(resid) & np.isfinite(hp)
    if ok.sum() < 10:
        return {"n_px": int(ok.sum()), "pearson": np.nan, "spearman": np.nan}
    return {"n_px": int(ok.sum()), "pearson": float(stats.pearsonr(hp[ok], resid[ok])[0]),
            "spearman": float(stats.spearmanr(hp[ok], resid[ok])[0])}


def ice_rmse(field_ortho: np.ndarray, n: float, rgb_path: Path, photons: pd.DataFrame) -> tuple[float, int]:
    s = sample_depth_at_photons(rgb_path, field_ortho - n, photons)
    return ff.rmse(s["depth_value"].values, s["height"].values), len(s)


def build_tile(tile_id: str, m: pd.Series) -> dict:
    rgb_path = ROOT / m["rgb_path"]
    lat, lon = float(m["lat"]), float(m["lon"])
    srtm = dem_to_grid(ff.SRTM_DIR / f"{tile_id}_srtm.tif", rgb_path)
    glo = dem_to_grid(COP_DIR / f"{tile_id}_dem.tif", rgb_path, nodata=sc.COP_NODATA)
    _, _, res_m = ff.native_resolution_m(lat, lon)
    sigma_px = res_m / ff.GRID_RES_M
    photons = pd.read_csv(ff.PHOTON_DIR / f"{tile_id}.csv")
    return {"tile_id": tile_id, "rgb_path": rgb_path, "lat": lat, "lon": lon, "srtm": srtm, "glo": glo,
            "sigma_px": sigma_px, "n96": ff.n_egm96(lon, lat), "n08": sc.n_egm2008(lon, lat),
            "photons": photons, "pix": photon_pixels(rgb_path, photons, srtm.shape),
            "srtm_range": float(np.nanmax(srtm) - np.nanmin(srtm)),
            "glo_range": float(np.nanmax(glo) - np.nanmin(glo)), "category": m["category"]}


def main():
    manifest = pd.read_csv(ff.MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(ff.VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    ref = pd.read_csv(ff.OUT_DIR / "frequency_fusion_results.csv").set_index("tile_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, tid in enumerate(accepted, 1):
        t = build_tile(tid, manifest.loc[tid])
        dav2 = np.load(ff.DAV2_DIR / f"{tid}_depth.npy").astype(np.float32)
        fs = fuse(t["srtm"], dav2, t["sigma_px"])
        fg = fuse(t["glo"], dav2, t["sigma_px"])
        rng_ = t["srtm_range"]
        out = {"tile_id": tid, "category": t["category"], "srtm_range_m": rng_, "glo_range_m": t["glo_range"],
               "a_srtm": fs["a"], "a_glo": fg["a"]}
        for name, field, n in [("i_fusion", fs["fused"], t["n96"]), ("ii_dem_lowpass", fs["dem_lowpass"], t["n96"]),
                               ("iii_srtm_raw", t["srtm"], t["n96"]), ("iv_glo30_raw", t["glo"], t["n08"]),
                               ("v_fusion_glo30", fg["fused"], t["n08"])]:
            r, nph = ice_rmse(field, n, t["rgb_path"], t["photons"])
            out[f"{name}_rmse_m"] = r
            out[f"{name}_pct"] = 100 * r / rng_
            out[f"{name}_n"] = nph
        # reproduce check against the saved fusion CSV
        out["i_repro_absdiff_m"] = abs(out["i_fusion_rmse_m"] - ref.loc[tid, "icesat2_rmse_m"])
        h = r_hf(fs["highpass"], fs["dem_lowpass"], t["pix"], t["n96"])
        hr = r_hf(fs["raw_highpass"], fs["dem_lowpass"], t["pix"], t["n96"])
        out.update({"rhf_n_px": h["n_px"], "rhf_pearson": h["pearson"], "rhf_spearman": h["spearman"],
                    "rhf_raw_pearson": hr["pearson"], "rhf_raw_spearman": hr["spearman"]})
        rows.append(out)
        print(f"[{i}/{len(accepted)}] {tid:13s} fus={out['i_fusion_pct']:.2f}% low={out['ii_dem_lowpass_pct']:.2f}% "
              f"srtm={out['iii_srtm_raw_pct']:.2f}% glo={out['iv_glo30_raw_pct']:.2f}% fusglo={out['v_fusion_glo30_pct']:.2f}% "
              f"repro={out['i_repro_absdiff_m']:.1e} rHF={h['pearson']:+.3f} a={fs['a']:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "controls_per_tile.csv", index=False)
    if df["i_repro_absdiff_m"].max() > 1e-6:
        raise SystemExit(f"(i) does not reproduce frequency_fusion_results.csv: max diff {df['i_repro_absdiff_m'].max()}")

    d = df["i_fusion_pct"] - df["ii_dem_lowpass_pct"]
    w = stats.wilcoxon(df["i_fusion_pct"], df["ii_dem_lowpass_pct"], alternative="two-sided")
    wins = int((d < 0).sum())
    pos = int((df["rhf_pearson"] > 0).sum())
    summary = {
        "n_tiles": len(df),
        "max_repro_absdiff_m": float(df["i_repro_absdiff_m"].max()),
        "median_pct": {c: float(df[f"{c}_pct"].median()) for c in
                       ["i_fusion", "ii_dem_lowpass", "iii_srtm_raw", "iv_glo30_raw", "v_fusion_glo30"]},
        "median_rmse_m": {c: float(df[f"{c}_rmse_m"].median()) for c in
                          ["i_fusion", "ii_dem_lowpass", "iii_srtm_raw", "iv_glo30_raw", "v_fusion_glo30"]},
        "fusion_vs_lowpass": {"wilcoxon_stat": float(w.statistic), "wilcoxon_p_two_sided": float(w.pvalue),
                              "fusion_wins": wins, "median_diff_pct": float(d.median()),
                              "PASS": bool(w.pvalue < 0.05 and wins >= 15)},
        "rhf": {"median_pearson": float(df["rhf_pearson"].median()), "median_spearman": float(df["rhf_spearman"].median()),
                "n_positive_pearson": pos, "sign_test_p_one_sided": float(stats.binomtest(pos, len(df), 0.5, alternative="greater").pvalue),
                "median_raw_pearson": float(df["rhf_raw_pearson"].median()),
                "n_positive_raw_pearson": int((df["rhf_raw_pearson"] > 0).sum()),
                "PASS": bool(df["rhf_pearson"].median() > 0.10 and pos >= 18)},
        "srtm_vs_glo30_raw": {"srtm_wins": int((df["iii_srtm_raw_pct"] < df["iv_glo30_raw_pct"]).sum()),
                              "wilcoxon_p": float(stats.wilcoxon(df["iii_srtm_raw_pct"], df["iv_glo30_raw_pct"]).pvalue)},
        "fusion_vs_srtm_raw": {"fusion_wins": int((df["i_fusion_pct"] < df["iii_srtm_raw_pct"]).sum()),
                               "wilcoxon_p": float(stats.wilcoxon(df["i_fusion_pct"], df["iii_srtm_raw_pct"]).pvalue)},
        "lowpass_vs_srtm_raw": {"lowpass_wins": int((df["ii_dem_lowpass_pct"] < df["iii_srtm_raw_pct"]).sum()),
                                "wilcoxon_p": float(stats.wilcoxon(df["ii_dem_lowpass_pct"], df["iii_srtm_raw_pct"]).pvalue)},
        "by_category": {},
    }
    for cat, g in df.groupby("category"):
        summary["by_category"][cat] = {
            "n": len(g), **{f"median_{c}_pct": float(g[f"{c}_pct"].median()) for c in
                            ["i_fusion", "ii_dem_lowpass", "iii_srtm_raw", "iv_glo30_raw", "v_fusion_glo30"]},
            "fusion_beats_lowpass": int((g["i_fusion_pct"] < g["ii_dem_lowpass_pct"]).sum()),
            "median_rhf_pearson": float(g["rhf_pearson"].median()),
            "rhf_positive": int((g["rhf_pearson"] > 0).sum())}
    (OUT_DIR / "controls_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
