#!/usr/bin/env python3
"""
Phase 4 (2026-09-23): no-training detail-source bake-off. Evaluation only.
Pre-registered rule: docs/method-audit/sentinel2/sign-flip-detector.md,
2026-09-23 (continued), "Phase 4 -- PRE-REGISTRATION".

Reuses scripts/frequency_fusion_controls.py (build_tile, fuse, photon_pixels)
and therefore run_frequency_fusion_sentinel2.py's split/sigma/geoid/grid.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frequency_fusion_controls as fc  # noqa: E402
import run_frequency_fusion_sentinel2 as ff  # noqa: E402

ROOT = ff.ROOT
BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "detail_source_bakeoff"
MIN_PX = 200
CANDS = {
    "dav2_518": ("relative", lambda t: np.load(BENCH / "dav2_depth" / f"{t}_depth.npy")),
    "dav2_1008": ("relative", lambda t: np.load(BENCH / "dav2_depth_1008" / f"{t}_depth.npy")),
    "dinov3_chmv2": ("metric", lambda t: np.load(BENCH / "dinov3_depth" / f"{t}_depth.npy")),
    "eth_gch2020": ("metric", lambda t: np.load(BENCH / "eth_canopy_2020" / f"{t}_eth.npy")),
}


def ref_pixels(t: dict, ref: str) -> pd.DataFrame:
    """Per-10 m-pixel median reference height (EGM96 orthometric, h + n)."""
    if ref == "ground":
        pix = t["pix"].rename(columns={"height": "h"})
    else:
        if ref == "surface_is2":
            d = pd.read_csv(ROOT / "data/icesat2_segments20m" / f"{t['tile_id']}.csv")
            d = d[(d["gnd_ph_count"] > 0) & (d["landcover"] != 255) & (d["h_max_canopy"] >= 0) & (d["h_max_canopy"] <= 60)]
            d = d.assign(height=d["h_te_median"] + d["h_max_canopy"])
        else:
            d = pd.read_csv(ROOT / "data/gedi_l2a" / f"{t['tile_id']}.csv")
            d = d[d["rh98"] <= 80]
            d = d.assign(height=d["elev_lowestmode"] + d["rh98"])
        pix = fc.photon_pixels(t["rgb_path"], d[["lat", "lon", "height"]], t["srtm"].shape).rename(columns={"height": "h"})
    g = pix.groupby(["row", "col"])["h"].median().reset_index()
    g["h_ortho"] = g["h"] + t["n96"]
    return g


def score(product: np.ndarray, control: np.ndarray, detail: np.ndarray, raw: np.ndarray | None, g: pd.DataFrame) -> dict:
    r, c = g["row"].values, g["col"].values
    y = g["h_ortho"].values
    p, q, d = product[r, c], control[r, c], detail[r, c]
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(q) & np.isfinite(d)
    y, p, q, d = y[ok], p[ok], q[ok], d[ok]
    resid = y - q
    out = {"n_px": int(ok.sum()),
           "rmse_product": float(np.sqrt(np.mean((p - y) ** 2))),
           "rmse_control": float(np.sqrt(np.mean((q - y) ** 2))),
           "rhf": float(stats.pearsonr(d, resid)[0]) if np.std(d) > 0 else float("nan"),
           "rhf_spearman": float(stats.spearmanr(d, resid)[0]) if np.std(d) > 0 else float("nan")}
    if raw is not None:
        rv = raw[r, c][ok]
        out["raw_vs_resid"] = float(stats.pearsonr(rv, resid)[0]) if np.std(rv) > 0 else float("nan")
    return out


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def main(dem: str = "srtm"):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ff.MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(ff.VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    refs = ("ground", "surface_is2", "surface_gedi")
    rows = []
    for i, tid in enumerate(accepted, 1):
        t = fc.build_tile(tid, manifest.loc[tid])
        dem_grid = t["srtm"] if dem == "srtm" else t["glo"]
        if dem == "glo30":
            t = {**t, "n96": t["n08"]}  # GLO-30 is EGM2008
        valid = ~np.isnan(dem_grid)
        dem_low = ff.masked_gaussian(dem_grid, valid, t["sigma_px"])
        allv = np.ones_like(valid, dtype=bool)
        gs = {ref: ref_pixels(t, ref) for ref in refs}
        for cname, (kind, load) in CANDS.items():
            raw = load(tid).astype(np.float32)
            if kind == "relative":
                f = fc.fuse(dem_grid, raw, t["sigma_px"])
                detail, surf_product = f["highpass"], f["fused"]
            else:
                rawf = np.nan_to_num(raw, nan=0.0)
                detail = rawf - ff.masked_gaussian(rawf, allv, t["sigma_px"])
                surf_product = dem_low + rawf
            for ref in refs:
                product = dem_low + detail if ref == "ground" else surf_product
                s = score(product, dem_low, detail, raw if kind == "metric" else None, gs[ref])
                rows.append({"tile_id": tid, "category": t["category"], "candidate": cname, "reference": ref, **s})
        print(f"[{i}/{len(accepted)}] {tid} " + " ".join(f"{r}:{len(gs[r])}px" for r in refs), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"per_tile_{dem}.csv", index=False)

    summary = {"dem": dem, "min_px": MIN_PX, "by_reference": {}}
    for ref in refs:
        sub = df[(df["reference"] == ref) & (df["n_px"] >= MIN_PX)]
        pA, pB, res = {}, {}, {}
        for cname in CANDS:
            c = sub[sub["candidate"] == cname]
            n = len(c)
            wins = int((c["rmse_product"] < c["rmse_control"]).sum())
            pA[cname] = float(stats.wilcoxon(c["rmse_product"], c["rmse_control"]).pvalue) if n > 0 and (c["rmse_product"] != c["rmse_control"]).any() else 1.0
            pos = int((c["rhf"] > 0).sum())
            pB[cname] = float(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue) if n else 1.0
            res[cname] = {"n_tiles": n, "wins": wins, "wins_needed": math.ceil(0.6 * n),
                          "median_rmse_product": float(c["rmse_product"].median()),
                          "median_rmse_control": float(c["rmse_control"].median()),
                          "wilcoxon_p": pA[cname],
                          "median_rhf": float(c["rhf"].median()), "rhf_positive": pos,
                          "rhf_positive_needed": math.ceil(0.72 * n), "sign_p": pB[cname],
                          "median_rhf_spearman": float(c["rhf_spearman"].median())}
            if "raw_vs_resid" in c and c["raw_vs_resid"].notna().any():
                res[cname]["median_raw_vs_resid"] = float(c["raw_vs_resid"].median())
            res[cname]["by_category"] = {cat: {"n": len(g), "wins": int((g["rmse_product"] < g["rmse_control"]).sum()),
                                               "median_rhf": float(g["rhf"].median())} for cat, g in c.groupby("category")}
        hA, hB = holm(pA), holm(pB)
        for cname, r in res.items():
            r["wilcoxon_p_holm"], r["sign_p_holm"] = hA[cname], hB[cname]
            r["pass_A"] = bool(r["wilcoxon_p_holm"] < 0.05 and r["wins"] >= r["wins_needed"])
            r["pass_B"] = bool(r["median_rhf"] > 0.10 and r["rhf_positive"] >= r["rhf_positive_needed"] and r["sign_p_holm"] < 0.05)
            r["PASS"] = r["pass_A"] and r["pass_B"]
        summary["by_reference"][ref] = res
    eth = summary["by_reference"]
    summary["eth_ceiling_fails_surface"] = not (eth["surface_is2"]["eth_gch2020"]["PASS"] or eth["surface_gedi"]["eth_gch2020"]["PASS"])
    (OUT / f"summary_{dem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({ref: {c: {k: v for k, v in r.items() if k != "by_category"} for c, r in res.items()}
                      for ref, res in summary["by_reference"].items()}, indent=1))
    print("ETH fails surface:", summary["eth_ceiling_fails_surface"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "srtm")
