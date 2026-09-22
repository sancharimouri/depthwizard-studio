#!/usr/bin/env python3
"""
A2 + A3 (2026-09-23): DEM-only baselines and FABDEM (+canopy) products on all 32
Sentinel-2 benchmark tiles. Evaluation only. Pre-registered rules:
docs/method-audit/sentinel2/sign-flip-detector.md, 2026-09-23 (final close-out),
"A2 / A3 / A4 / A5 -- PRE-REGISTRATION".

Products (orthometric, bilinear onto each tile's 10 m RGB grid):
  srtm (EGM96), glo30 (EGM2008), fabdem (EGM2008),
  fabdem_eth = fabdem + ETH GCH 2020 canopy (m), fabdem_chmv2 = fabdem + DINOv3-CHMv2 (m)
References (per-10 m-pixel median of ellipsoidal heights):
  ground (ICESat-2 ground photons), surface_is2 (20 m segments h_te_median+h_max_canopy,
  Phase 4 filter), surface_gedi (elev_lowestmode + rh98, rh98 <= 80)
Geoid: PRIMARY per point -- N of each product's own geoid at every reference pixel
centre; SECONDARY tile-centre constant (the Phase 1/4 convention).
"""
from __future__ import annotations

import os

# Must precede ANY pyproj import (incl. transitively): otherwise PROJ silently uses a
# ballpark geoid with N = 0 everywhere. Caught 2026-09-23 on this script's first run
# (every product showed a ~+60 m bias = |N|); that run's outputs were discarded.
os.environ["PROJ_NETWORK"] = "ON"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frequency_fusion_controls as fc  # noqa: E402  (imports ff first: PROJ_NETWORK=ON)
import run_frequency_fusion_sentinel2 as ff  # noqa: E402

ROOT = ff.ROOT
BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "dem_baselines_32"
T96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)
T08 = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)
PRODUCTS = {"srtm": "egm96", "glo30": "egm2008", "fabdem": "egm2008",
            "fabdem_eth": "egm2008", "fabdem_chmv2": "egm2008"}
REFS = ("ground", "surface_is2", "surface_gedi")
B, SEED = 10_000, 0


def ref_pixels(t: dict, ref: str, rgb_path: Path, shape) -> pd.DataFrame:
    if ref == "ground":
        pix = t["pix"]
    else:
        if ref == "surface_is2":
            d = pd.read_csv(ROOT / "data/icesat2_segments20m" / f"{t['tile_id']}.csv")
            d = d[(d["gnd_ph_count"] > 0) & (d["landcover"] != 255) & (d["h_max_canopy"] >= 0) & (d["h_max_canopy"] <= 60)]
            d = d.assign(height=d["h_te_median"] + d["h_max_canopy"])
        else:
            d = pd.read_csv(ROOT / "data/gedi_l2a" / f"{t['tile_id']}.csv")
            d = d[d["rh98"] <= 80].assign(height=lambda x: x["elev_lowestmode"] + x["rh98"])
        pix = fc.photon_pixels(rgb_path, d[["lat", "lon", "height"]], shape)
    g = pix.groupby(["row", "col"])["height"].median().reset_index()
    with rasterio.open(rgb_path) as s:
        T, crs = s.transform, s.crs
    x, y = T * (g["col"].values + 0.5, g["row"].values + 0.5)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    # script convention: n = geoid height OF the ellipsoid = -N; ellipsoidal = ortho - n
    g["n96"] = T96.transform(lon, lat, np.zeros_like(lon))[2]
    g["n08"] = T08.transform(lon, lat, np.zeros_like(lon))[2]
    return g


def tile_metrics(field: np.ndarray, g: pd.DataFrame, n_point: np.ndarray, n_centre: float, rng_m: float) -> dict:
    v = field[g["row"].values, g["col"].values]
    out = {}
    for tag, n in (("", n_point), ("_tc", n_centre)):
        e = (v - n) - g["height"].values
        e = e[np.isfinite(e)]
        out.update({f"n_px{tag}": int(len(e)), f"rmse{tag}": float(np.sqrt(np.mean(e ** 2))),
                    f"bias{tag}": float(np.mean(e)), f"brmse{tag}": float(np.std(e)),
                    f"medae{tag}": float(np.median(np.abs(e))), f"pct{tag}": float(100 * np.sqrt(np.mean(e ** 2)) / rng_m)})
    return out


def boot_median(x: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return [float(v) for v in np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])]


def paired(a: pd.Series, b: pd.Series) -> dict:
    """a vs b per tile (lower is better). Wilcoxon two-sided + wins + bootstrap CI of median diff."""
    d = (a - b).values
    return {"n": int(len(d)), "wins": int((d < 0).sum()),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
            "median_diff": float(np.median(d)), "median_diff_ci95": boot_median(d)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def main():
    assert abs(T96.transform(75.15, 30.28, 0.0)[2]) > 1 and abs(T08.transform(75.15, 30.28, 0.0)[2]) > 1, \
        "geoid transform returned ~0 (PROJ ballpark) -- PROJ_NETWORK not active"
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ff.MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(ff.VERDICTS_CSV).set_index("tile_id")
    rows = []
    for i, tid in enumerate(manifest.index, 1):
        m = manifest.loc[tid]
        t = fc.build_tile(tid, m)
        rgb_path = ROOT / m["rgb_path"]
        fab = np.load(BENCH / "fabdem" / f"{tid}_fabdem.npy")
        eth = np.nan_to_num(np.load(BENCH / "eth_canopy_2020" / f"{tid}_eth.npy"), nan=0.0)
        chm = np.load(BENCH / "dinov3_depth" / f"{tid}_depth.npy")
        fields = {"srtm": t["srtm"], "glo30": t["glo"], "fabdem": fab, "fabdem_eth": fab + eth, "fabdem_chmv2": fab + chm}
        for ref in REFS:
            g = ref_pixels(t, ref, rgb_path, t["srtm"].shape)
            for p, geoid in PRODUCTS.items():
                npt = g["n96"].values if geoid == "egm96" else g["n08"].values
                nc = t["n96"] if geoid == "egm96" else t["n08"]
                rows.append({"tile_id": tid, "category": m["category"], "accepted25": not bool(verdicts.loc[tid, "flagged"]),
                             "reference": ref, "product": p, **tile_metrics(fields[p], g, npt, nc, t["srtm_range"])})
        print(f"[{i}/32] {tid}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_tile.csv", index=False)

    summary = {"primary_geoid": "per_point", "bootstrap": {"B": B, "seed": SEED}, "tests": {}, "descriptive": {}}
    for tiles_name, sel in (("32", df), ("25", df[df["accepted25"]])):
        for ref in REFS:
            for p in PRODUCTS:
                s = sel[(sel["reference"] == ref) & (sel["product"] == p)]
                summary["descriptive"][f"{tiles_name}|{ref}|{p}"] = {
                    "n": len(s), **{f"median_{k}": float(s[k].median()) for k in ("rmse", "pct", "bias", "brmse", "medae", "rmse_tc")},
                    "median_rmse_ci95": boot_median(s["rmse"].values),
                    "by_category": {c: {"n": len(x), "median_rmse": float(x["rmse"].median()), "median_brmse": float(x["brmse"].median()),
                                        "median_bias": float(x["bias"].median())} for c, x in s.groupby("category")}}

        def cmp(a, b, ref, metric):
            A = sel[(sel["reference"] == ref) & (sel["product"] == a)].set_index("tile_id")[metric]
            Bv = sel[(sel["reference"] == ref) & (sel["product"] == b)].set_index("tile_id")[metric]
            return paired(A, Bv.loc[A.index])

        need = 20 if tiles_name == "32" else 16  # 25-tile subset: ceil(20/32*25)=16, reported only
        # A2: GLO-30 vs SRTM, decisive on surface_is2 (32 tiles); all refs reported
        for ref in REFS:
            r = {m: cmp("glo30", "srtm", ref, m) for m in ("rmse", "brmse", "rmse_tc", "brmse_tc")}
            r["PASS_R4"] = all(r[m]["wilcoxon_p"] < 0.05 and r[m]["wins"] >= need for m in ("rmse", "brmse"))
            summary["tests"][f"A2|{tiles_name}|{ref}|glo30_vs_srtm"] = r
        # A3 terrain: FABDEM vs GLO-30 on ground
        for ref in REFS:
            r = {m: cmp("fabdem", "glo30", ref, m) for m in ("rmse", "brmse", "rmse_tc", "brmse_tc")}
            r["PASS_R4"] = all(r[m]["wilcoxon_p"] < 0.05 and r[m]["wins"] >= need for m in ("rmse", "brmse"))
            summary["tests"][f"A3terrain|{tiles_name}|{ref}|fabdem_vs_glo30"] = r
        # A3 surface: FABDEM+canopy vs raw GLO-30, Holm m=2 per metric
        for ref in REFS:
            res = {c: {m: cmp(c, "glo30", ref, m) for m in ("rmse", "brmse")} for c in ("fabdem_eth", "fabdem_chmv2")}
            for m in ("rmse", "brmse"):
                adj = holm([res["fabdem_eth"][m]["wilcoxon_p"], res["fabdem_chmv2"][m]["wilcoxon_p"]])
                res["fabdem_eth"][m]["p_holm"], res["fabdem_chmv2"][m]["p_holm"] = adj
            for c in res:
                res[c]["PASS_R4"] = all(res[c][m]["p_holm"] < 0.05 and res[c][m]["wins"] >= need for m in ("rmse", "brmse"))
                summary["tests"][f"A3surface|{tiles_name}|{ref}|{c}_vs_glo30"] = res[c]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in summary["tests"].items():
        print(k, "PASS" if v["PASS_R4"] else "fail",
              {m: (v[m]["wins"], v[m]["n"], round(v[m].get("p_holm", v[m]["wilcoxon_p"]), 4), round(v[m]["median_diff"], 3)) for m in ("rmse", "brmse")})


if __name__ == "__main__":
    main()
