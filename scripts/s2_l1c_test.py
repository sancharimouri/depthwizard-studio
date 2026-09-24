#!/usr/bin/env python3
"""Sentinel-2 L1C (top-of-atmosphere) vs L2A (surface reflectance) rank-loss test (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "L1C vs L2A test").

Same 32 tiles, same acquisition (manifest date_acquired), same bbox, same CDSE Process API call as
scripts/sentinel_benchmark_download.py; only the collection differs (sentinel-2-l1c vs sentinel-2-l2a).
Arms (cache format of s2_token_grid_phase_c; fab/ICESat-2 copied from token_grid_test/cache):
  l2a_gain / l1c_gain       : backend.cdse TRUE_COLOR_EVALSCRIPT (2.5 x B04/B03/B02, uint8)  -- primary pair
  l2a_stretch / l1c_stretch : FLOAT32 reflectance, per-tile per-band 2-98 % stretch to uint8   -- secondary pair
The L2A gain fetch is verified pixel-for-pixel against the existing benchmark GeoTIFFs.

  .venv/bin/python scripts/s2_l1c_test.py fetch
  .venv/bin/python scripts/s2_l1c_test.py run --arm {l2a_gain,l1c_gain,l2a_stretch,l1c_stretch}
  .venv/bin/python scripts/s2_l1c_test.py compare
Output: data/sentinel2_benchmark/l1c_test/
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from backend.cdse import client as cdse  # noqa: E402

BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "l1c_test"
S2_CACHE = BENCH / "token_grid_test/cache"
ARMS = ("l2a_gain", "l1c_gain", "l2a_stretch", "l1c_stretch")
FLOAT_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B04", "B03", "B02"] }],
        output: { bands: 3, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""


def process(collection, bbox, epsg, date, evalscript):
    body = {"input": {"bounds": {"bbox": bbox, "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                      "data": [{"type": collection,
                                "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}}}]},
            "output": {"width": 1000, "height": 1000, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript}
    for attempt in range(5):
        r = httpx.post(cdse.PROCESS_URL, json=body, headers={"Authorization": f"Bearer {cdse.get_access_token()}"},
                       timeout=180.0)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1)); continue
        if r.status_code >= 400:
            raise RuntimeError(f"{collection} {date}: {r.status_code} {r.text[:300]}")
        with rasterio.open(io.BytesIO(r.content)) as s:
            return s.read()
    raise RuntimeError("rate-limited")


def stretch(b):
    lo, hi = np.nanpercentile(b, [2, 98])
    return np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).round().astype(np.uint8)


def fetch(args):
    man = pd.read_csv(BENCH / "manifest.csv")
    for a in ARMS:
        (OUT / a / "cache").mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    rep = []
    for _, m in man.iterrows():
        tid, date, epsg, bbox = m.tile_id, m.date_acquired, int(m.bbox_utm_epsg), ast.literal_eval(m.bbox_utm)
        rawp = OUT / "raw" / f"{tid}.npz"
        if rawp.exists():
            z = np.load(rawp); g2, g1, f2, f1 = z["l2a_gain"], z["l1c_gain"], z["l2a_float"], z["l1c_float"]
        else:
            g2 = process("sentinel-2-l2a", bbox, epsg, date, cdse.TRUE_COLOR_EVALSCRIPT)
            g1 = process("sentinel-2-l1c", bbox, epsg, date, cdse.TRUE_COLOR_EVALSCRIPT)
            f2 = process("sentinel-2-l2a", bbox, epsg, date, FLOAT_EVALSCRIPT)
            f1 = process("sentinel-2-l1c", bbox, epsg, date, FLOAT_EVALSCRIPT)
            np.savez_compressed(rawp, l2a_gain=g2, l1c_gain=g1, l2a_float=f2, l1c_float=f1)
        with rasterio.open(ROOT / m.rgb_path) as s:
            existing = s.read()
        diff = np.abs(existing.astype(int) - g2.astype(int))
        rec = {"tile": tid, "date": date, "l2a_refetch_identical_frac": float((diff == 0).mean()),
               "l2a_refetch_max_abs_diff": int(diff.max()),
               "l1c_minus_l2a_float_median": [float(np.median(f1[i] - f2[i])) for i in range(3)],
               "clip_frac_l2a_gain": float((g2 == 255).mean()), "clip_frac_l1c_gain": float((g1 == 255).mean())}
        rep.append(rec)
        base = dict(np.load(S2_CACHE / f"{tid}.npz"))
        for a, rgb in (("l2a_gain", g2), ("l1c_gain", g1),
                       ("l2a_stretch", np.stack([stretch(b) for b in f2])), ("l1c_stretch", np.stack([stretch(b) for b in f1]))):
            z = dict(base); z["rgb"] = rgb.astype(np.uint8)
            np.savez_compressed(OUT / a / "cache" / f"{tid}.npz", **z)
        print(f"{tid:14s} {date} L2A refetch identical {rec['l2a_refetch_identical_frac']:.4f} (max|d| {rec['l2a_refetch_max_abs_diff']}); "
              f"L1C-L2A median refl {np.round(rec['l1c_minus_l2a_float_median'], 4).tolist()}; "
              f"clip L2A {rec['clip_frac_l2a_gain']:.3f} L1C {rec['clip_frac_l1c_gain']:.3f}", flush=True)
    pd.DataFrame(rep).to_csv(OUT / "fetch_report.csv", index=False)


def run(args):
    import torch
    from PIL import Image
    import s2_rank_loss_test as R
    import s2_rank_raw_spearman as RS
    import s2_token_grid_phase_c as C
    from evaluate_method6_finetune_twinhead import get_device
    from backend.depth.depth_engine import run_inference
    A = OUT / args.arm
    C.CACHE, R.OUT, R.DAV2 = A / "cache", A / "rank_loss", A / "dav2_depth"
    R.OUT.mkdir(parents=True, exist_ok=True); R.DAV2.mkdir(parents=True, exist_ok=True)
    tids = sorted(p.stem for p in C.CACHE.glob("*.npz"))
    assert len(tids) == 32
    for t in tids:
        o = R.DAV2 / f"{t}_depth.npy"
        if not o.exists():
            np.save(o, run_inference(Image.fromarray(np.load(C.CACHE / f"{t}.npz")["rgb"].transpose(1, 2, 0), "RGB")))
    device = get_device()
    allrows = []
    for fold in range(4):
        t0 = time.time()
        model, hs, run_ = R.fit(fold, tids, 600, 4, device)
        recs = [{"tile": t, "model": R.calibrated_eval(R.scores_all(model, t, device), t, fold),
                 "oracle": R.calibrated_eval(R.oracle_scores(t), t, fold)} for t in tids]
        (R.OUT / f"eval_rank_fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "steps": 600, "batch": 4, "height_scale": hs, "final_loss100": float(np.mean(run_[-100:])),
             "train_s": time.time() - t0, "records": recs}))
        rows = RS.cells_for_fold(model, tids, fold, device)
        for r in rows:
            r["fold"] = fold
        allrows += rows
        print(f"[{args.arm}] fold{fold} done {time.time()-t0:.0f}s", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    R.analyze(None)
    (R.OUT / "raw_spearman").mkdir(exist_ok=True)
    RS.summarize(allrows, tids, None, R.OUT / "raw_spearman")


def compare(args):
    from scipy.stats import wilcoxon

    def boot(v, B=10000):
        rng = np.random.default_rng(0)
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    L = {a: (json.loads((OUT / a / "rank_loss/raw_spearman/summary.json").read_text()),
             json.loads((OUT / a / "rank_loss/summary.json").read_text())) for a in ARMS}
    tiles = sorted(L["l2a_gain"][0]["per_tile"])

    def paired(a, b, key, raw=True, hib=True):
        get = (lambda x, t: L[x][0]["per_tile"][t][key]) if raw else (lambda x, t: L[x][1]["checks"]["icesat2"]["per_tile"][t][key])
        x = np.array([get(a, t) for t in tiles], float); y = np.array([get(b, t) for t in tiles], float); d = x - y
        return {f"{a}_mean": float(x.mean()), f"{b}_mean": float(y.mean()), "diff_mean": float(d.mean()), "ci95": boot(d),
                f"{a}_better": int(((d > 0) if hib else (d < 0)).sum()), "n": len(d), "wilcoxon_p": float(wilcoxon(x, y).pvalue)}

    S = {"primary_raw_model_gain": paired("l1c_gain", "l2a_gain", "within_crop_model"),
         "raw_oracle_gain": paired("l1c_gain", "l2a_gain", "within_crop_oracle"),
         "calibrated_rmse_model_gain": paired("l1c_gain", "l2a_gain", "model", raw=False, hib=False),
         "secondary_raw_model_stretch": paired("l1c_stretch", "l2a_stretch", "within_crop_model"),
         "raw_oracle_stretch": paired("l1c_stretch", "l2a_stretch", "within_crop_oracle"),
         "calibrated_rmse_model_stretch": paired("l1c_stretch", "l2a_stretch", "model", raw=False, hib=False),
         "per_arm": {a: {"raw_within_crop_model": v[0]["means"]["within_crop_model"],
                         "raw_within_crop_oracle": v[0]["means"]["within_crop_oracle"],
                         "raw_within_crop_fab": v[0]["means"]["within_crop_fab"],
                         "calib_implied_r": v[0]["means"]["calib_implied_r"],
                         "icesat2": {k: v[1]["checks"]["icesat2"][k] for k in
                                     ("model_mean_rmse", "oracle_mean_rmse", "model_wins", "wilcoxon_p_holm", "flat_mean_rmse")},
                         "verdict_real_signal": v[1]["verdict_real_signal"], "case_raw": v[0]["case"]} for a, v in L.items()}}

    def decide(p, c):
        higher = p["diff_mean"] >= 0.10 and list(p.values())[4] > p["n"] / 2 and p["wilcoxon_p"] < 0.05
        rd = np.sign(p["diff_mean"]) if p["wilcoxon_p"] < 0.05 else 0
        cd = -np.sign(c["diff_mean"]) if c["wilcoxon_p"] < 0.05 else 0
        return higher, bool(rd != 0 and cd != 0 and rd != cd)

    hp, dp = decide(S["primary_raw_model_gain"], S["calibrated_rmse_model_gain"])
    hs, ds = decide(S["secondary_raw_model_stretch"], S["calibrated_rmse_model_stretch"])
    S["divergence_primary"], S["divergence_secondary"] = dp, ds
    S["decision"] = ("L1C meaningfully higher -> atmospheric correction implicated as erasing signal" if hp else
                     "L1C not meaningfully higher -> atmospheric correction ruled out as the cause")
    S["secondary_consistent"] = bool(hs == hp)
    if dp:
        S["decision"] += " | FLAG: raw vs calibrated disagree (primary) -> revisit calibration first"
    (OUT / "comparison.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "compare"])
    ap.add_argument("--arm", choices=ARMS)
    a = ap.parse_args()
    {"fetch": fetch, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
