#!/usr/bin/env python3
"""
A4 (2026-09-23): direct height-above-ground test, no DEM. Evaluation only.
Pre-registered rule: docs/method-audit/sentinel2/sign-flip-detector.md,
2026-09-23 (final close-out), "A2 / A3 / A4 / A5 -- PRE-REGISTRATION".

References (already relative heights):
  ICESat-2 20 m segments: h_max_canopy (PhoREAL max canopy height above h_te_median;
    use_abs_h=False), filter gnd_ph_count>0, landcover!=255, 0<=h_max_canopy<=60;
    candidate sampled at the 10 m pixel containing the segment centre.
  GEDI rh98 (<= 80 m): candidate averaged over pixels whose centres lie within 12.5 m of
    the footprint centre.
Candidates (metres, no fitting): DINOv3-CHMv2, ETH GCH 2020 (GEDI comparison non-independent).
Also importable: `score_field(field, rgb_path, tile_id)` for A5 (RDAH on Darjeeling).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "direct_height_test"
CANDS = {"dinov3_chmv2": lambda t: np.load(BENCH / "dinov3_depth" / f"{t}_depth.npy"),
         "eth_gch2020": lambda t: np.load(BENCH / "eth_canopy_2020" / f"{t}_eth.npy")}


def is2_ref(tile_id: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/icesat2_segments20m" / f"{tile_id}.csv")
    return d[(d["gnd_ph_count"] > 0) & (d["landcover"] != 255) & (d["h_max_canopy"] >= 0) & (d["h_max_canopy"] <= 60)]


def gedi_ref(tile_id: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/gedi_l2a" / f"{tile_id}.csv")
    return d[d["rh98"] <= 80]


def _xy(rgb_path, lon, lat):
    with rasterio.open(rgb_path) as s:
        T, crs, h, w = s.transform, s.crs, s.height, s.width
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return T, h, w, np.asarray(x), np.asarray(y)


def sample_is2(field, rgb_path, d):
    T, h, w, x, y = _xy(rgb_path, d["lon"].values, d["lat"].values)
    c, r = ~T * (x, y)
    c, r = np.floor(c).astype(int), np.floor(r).astype(int)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    v = np.full(len(d), np.nan)
    v[ok] = field[r[ok], c[ok]]
    return v


def sample_gedi(field, rgb_path, d, radius=12.5):
    T, h, w, x, y = _xy(rgb_path, d["lon"].values, d["lat"].values)
    c0, r0 = ~T * (x, y)
    out = np.full(len(d), np.nan)
    offs = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)]
    vals = np.zeros((len(d), len(offs))); cnt = np.zeros((len(d), len(offs)))
    for k, (dr, dc) in enumerate(offs):
        r = np.floor(r0).astype(int) + dr; c = np.floor(c0).astype(int) + dc
        cx, cy = T * (c + 0.5, r + 0.5)
        inside = (np.hypot(np.asarray(cx) - x, np.asarray(cy) - y) <= radius) & (r >= 0) & (r < h) & (c >= 0) & (c < w)
        rr, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
        fv = field[rr, cc]
        good = inside & np.isfinite(fv)
        vals[:, k] = np.where(good, fv, 0.0); cnt[:, k] = good
    n = cnt.sum(axis=1)
    out[n > 0] = vals.sum(axis=1)[n > 0] / n[n > 0]
    return out


def corr(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 10 or np.std(p) == 0:
        return {"n": int(len(p)), "spearman": float("nan"), "pearson": float("nan")}
    return {"n": int(len(p)), "spearman": float(stats.spearmanr(p, y)[0]), "pearson": float(stats.pearsonr(p, y)[0])}


def score_field(field: np.ndarray, rgb_path: Path, tile_id: str) -> dict:
    """Per-tile scores of one field against both references (used by A5 too)."""
    a, g = is2_ref(tile_id), gedi_ref(tile_id)
    pa, pg = sample_is2(field, rgb_path, a), sample_gedi(field, rgb_path, g)
    ya, yg = a["h_max_canopy"].values, g["rh98"].values
    veg_a, veg_g = ya > 2, yg > 2
    return {"is2": corr(pa, ya), "is2_veg": corr(pa[veg_a], ya[veg_a]),
            "gedi": corr(pg, yg), "gedi_veg": corr(pg[veg_g], yg[veg_g]),
            "_pooled": {"is2": (pa, ya), "gedi": (pg, yg)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(BENCH / "manifest.csv").set_index("tile_id")
    verdicts = pd.read_csv(BENCH / "sign_flip_detector_verdicts.csv").set_index("tile_id")
    rows, pooled = [], {}
    for tid in manifest.index:
        rgb = ROOT / manifest.loc[tid, "rgb_path"]
        for cname, load in CANDS.items():
            s = score_field(load(tid).astype(np.float64), rgb, tid)
            for ref in ("is2", "gedi"):
                pooled.setdefault((cname, ref), []).append((tid, *s["_pooled"][ref]))
            rows.append({"tile_id": tid, "category": manifest.loc[tid, "category"],
                         "accepted25": not bool(verdicts.loc[tid, "flagged"]), "candidate": cname,
                         **{f"{k}_{m}": s[k][m] for k in ("is2", "is2_veg", "gedi", "gedi_veg") for m in ("n", "spearman", "pearson")}})
        print(tid, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_tile.csv", index=False)

    summary = {"reference_field_is2": "h_max_canopy", "reference_field_gedi": "rh98", "results": {}}
    for tiles_name, keep in (("25", set(df[df["accepted25"]]["tile_id"])), ("32", set(df["tile_id"]))):
        for cname in CANDS:
            sub = df[(df["candidate"] == cname) & (df["tile_id"].isin(keep))]
            res = {}
            for ref in ("is2", "gedi"):
                P = np.concatenate([p for t, p, y in pooled[(cname, ref)] if t in keep])
                Y = np.concatenate([y for t, p, y in pooled[(cname, ref)] if t in keep])
                pc = corr(P, Y); veg = Y > 2; pv = corr(P[veg], Y[veg])
                pos = int((sub[f"{ref}_spearman"] > 0).sum()); n = int(sub[f"{ref}_spearman"].notna().sum())
                res[ref] = {"pooled_spearman": pc["spearman"], "pooled_pearson": pc["pearson"], "pooled_n": pc["n"],
                            "pooled_spearman_veg": pv["spearman"], "pooled_n_veg": pv["n"],
                            "tiles_positive": pos, "n_tiles": n,
                            "sign_test_p": float(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue),
                            "median_tile_spearman": float(sub[f"{ref}_spearman"].median()),
                            "median_tile_spearman_veg": float(sub[f"{ref}_veg_spearman"].median()),
                            "by_category": {c: {"median_spearman": float(x[f"{ref}_spearman"].median()),
                                                "positive": int((x[f"{ref}_spearman"] > 0).sum()), "n": len(x)}
                                            for c, x in sub.groupby("category")}}
            need = 18 if tiles_name == "25" else 23  # 32-tile version: ceil(18/25*32)=23, reported only
            res["PASS"] = bool(res["is2"]["pooled_spearman"] >= 0.30 and res["is2"]["tiles_positive"] >= need)
            summary["results"][f"{tiles_name}|{cname}"] = res
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, r in summary["results"].items():
        print(k, "PASS" if r["PASS"] else "fail",
              {ref: (round(r[ref]["pooled_spearman"], 3), f"{r[ref]['tiles_positive']}/{r[ref]['n_tiles']}",
                     round(r[ref]["pooled_spearman_veg"], 3), round(r[ref]["median_tile_spearman"], 3)) for ref in ("is2", "gedi")})


if __name__ == "__main__":
    main()
