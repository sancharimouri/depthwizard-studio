#!/usr/bin/env python3
"""Sentinel-2 token-grid test, Phase C (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md).

Controlled real-Sentinel-2 comparison: the ONLY manipulated variable is the ViT token-grid geometry.
Both arms see identical 60x60 px (600 m) Sentinel-2 crops, the identical FABDEM target, identical tiles, folds,
seed, crop order, steps, loss and optimiser.
  Arm P: crop reflect-padded 60 -> 70 px  => 5x5 tokens, 140 m / token (true pixel count, like protocol P)
  Arm R: crop bilinear-upsampled 60 -> 540 (exactly 9x), reflect-padded to 546 => 39x39 tokens, 15.4 m / token
         (like protocol R: upsample to a round size, pad to x14); output cropped to 540 and 9x9-average-pooled to 60
Target (interpolation-safe, as the native-30m retry): raw FABDEM warped straight onto an exact 3x coarsening of the
tile's 10 m grid (333x333 at 30 m); the loss is computed on 3x3-average-pooled predictions (20x20 per crop) against
that 30 m grid, so no target value finer than FABDEM's own resolution ever exists. Quantity learned: within-crop
relief (FABDEM minus its crop mean), since absolute elevation is unidentifiable from a 600 m RGB crop; the crop mean
is supplied from FABDEM identically to both arms at evaluation.
Model: Method 6's TwinHeadDav2 (DAv2-Small, full fine-tune), mu head only; mean-removed Huber loss.

  PROJ_NETWORK=ON .venv/bin/python scripts/s2_token_grid_phase_c.py prep
  .venv/bin/python scripts/s2_token_grid_phase_c.py train --arm {P,R} [--folds 0 1 2 3] [--steps N]
  .venv/bin/python scripts/s2_token_grid_phase_c.py analyze
Output: data/sentinel2_benchmark/token_grid_test/
"""
from __future__ import annotations

import os

os.environ["PROJ_NETWORK"] = "ON"
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, TwinHeadDav2, get_device, pad_to, seed_everything,
)

BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "token_grid_test"
CACHE = OUT / "cache"
N = 60            # crop, 10 m px (600 m)
K = 3             # 10 m -> 30 m
P_PAD = 70        # 60 -> next multiple of 14
R_UP = 540        # 9 x 60
R_PAD = 546       # 39 x 14
T08 = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)  # ellipsoid -> EGM2008 (FABDEM's datum)


# ============================================================================ prep
def prep(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(BENCH / "manifest.csv").set_index("tile_id")
    for tid in man.index:
        out = CACHE / f"{tid}.npz"
        if out.exists():
            continue
        with rasterio.open(ROOT / man.loc[tid, "rgb_path"]) as s:
            rgb = s.read([1, 2, 3]); T10, crs = s.transform, s.crs
        h30 = rgb.shape[1] // K
        T30 = T10 * Affine.scale(K)
        fab30 = np.full((h30, h30), np.nan, np.float32)
        with rasterio.open(BENCH / "fabdem_raw" / f"{tid}_fabdem.tif") as s:
            reproject(source=s.read(1).astype(np.float32), destination=fab30, src_transform=s.transform,
                      src_crs=s.crs, src_nodata=s.nodata, dst_transform=T30, dst_crs=crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        fab10 = np.load(BENCH / "fabdem" / f"{tid}_fabdem.npy").astype(np.float32)
        ph = pd.read_csv(ROOT / "data/icesat2_photons" / f"{tid}.csv")
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(ph["lon"].values, ph["lat"].values)
        cols, rows = ~T10 * (x, y)
        ph["row"], ph["col"] = np.floor(rows).astype(int), np.floor(cols).astype(int)
        ph = ph[(ph.row >= 0) & (ph.row < rgb.shape[1]) & (ph.col >= 0) & (ph.col < rgb.shape[2])]
        g = ph.groupby(["row", "col", "rgt"])["height"].median().reset_index()
        xc, yc = T10 * (g["col"].values + 0.5, g["row"].values + 0.5)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xc, yc)
        ortho = T08.transform(lon, lat, g["height"].values)[2]
        assert len(g) == 0 or np.all(np.abs(g["height"].values - ortho) > 1), f"{tid}: geoid N~0"
        np.savez_compressed(out, rgb=rgb.astype(np.uint8), fab30=fab30, fab10=fab10,
                            ph_row=g["row"].values, ph_col=g["col"].values, ph_h=ortho.astype(np.float64))
        print(f"{tid}: fab30 {fab30.shape} finite {np.isfinite(fab30).mean():.3f}, icesat2 cells {len(g)}", flush=True)


def crop_origins(q: int, size: int = 1000):
    """64 crop origins (10 m px) inside quadrant q, aligned to the 30 m grid (multiples of 3)."""
    half = size // 2
    r_start = 0 if q in (0, 1) else 501
    c_start = 0 if q in (0, 2) else 501
    return [(r_start + N * i, c_start + N * j) for i in range(8) for j in range(8)]


def load_crops(tids, quads):
    items = []
    for tid in tids:
        z = np.load(CACHE / f"{tid}.npz")
        rgb, fab30 = z["rgb"], z["fab30"]
        for q in quads:
            for r, c in crop_origins(q):
                t = fab30[r // K:(r + N) // K, c // K:(c + N) // K]
                if t.shape != (N // K, N // K) or not np.isfinite(t).all():
                    continue
                items.append((tid, q, r, c, rgb[:, r:r + N, c:c + N], t))
    return items


# ============================================================================ model / forward
def to_input(rgb_u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
    return (x - IMAGENET_MEAN[0].float()) / IMAGENET_STD[0].float()


def forward60(model, x, arm: str):
    """x (B,3,60,60) normalised -> mu on the 60x60 10 m grid."""
    if arm == "P":
        mu, _ = model(pad_to(x, P_PAD))
        return mu[:, :, :N, :N]
    mu, _ = model(pad_to(F.interpolate(x, size=(R_UP, R_UP), mode="bilinear", align_corners=False), R_PAD))
    return F.avg_pool2d(mu[:, :, :R_UP, :R_UP], R_UP // N)


def relief_loss(mu60, tgt20):
    p = F.avg_pool2d(mu60, K)[:, 0]
    p = p - p.mean(dim=(1, 2), keepdim=True)
    t = tgt20 - tgt20.mean(dim=(1, 2), keepdim=True)
    return F.smooth_l1_loss(p, t, beta=1.0)


def train(args):
    device = get_device()
    tids = sorted(p.stem for p in CACHE.glob("*.npz"))
    for fold in args.folds:
        tag = f"{args.arm}_fold{fold}"
        res_path = OUT / f"eval_{tag}.json"
        if res_path.exists():
            print(f"{tag}: exists, skip"); continue
        seed_everything(42)
        tr = load_crops(tids, [q for q in range(4) if q != fold])
        rel = np.concatenate([(t - t.mean()).ravel() for *_, t in tr])
        hs = max(float(rel.std()), 1.0)
        model = TwinHeadDav2(height_scale=hs, init_sigma_m=hs * 0.3).to(device)
        opt = torch.optim.AdamW(model.param_groups(5e-6, 2.5e-4, 1e-4))
        warm = max(10, int(args.steps * 0.05))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else max(0.05, (args.steps - s) / max(1, args.steps - warm)))
        order = np.random.default_rng(42 + fold).permutation(np.resize(np.arange(len(tr)), args.steps * args.batch))
        print(f"[{tag}] device={device} train crops {len(tr)} ({len(tids)} tiles) relief std {hs:.2f} m, "
              f"steps {args.steps} x batch {args.batch}", flush=True)
        model.train(); t0 = time.time(); run = []
        for s in range(args.steps):
            idx = order[s * args.batch:(s + 1) * args.batch]
            x = torch.stack([to_input(tr[i][4]) for i in idx]).to(device)
            y = torch.from_numpy(np.stack([tr[i][5] for i in idx])).to(device)
            loss = relief_loss(forward60(model, x, args.arm), y)
            assert torch.isfinite(loss), f"non-finite loss at step {s}"
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run.append(float(loss.detach().cpu()))
            if (s + 1) % 100 == 0 or s == 0:
                el = time.time() - t0
                print(f"[{tag}] step {s+1}/{args.steps} loss {np.mean(run[-100:]):.4f} "
                      f"{el/(s+1):.2f}s/step ETA {el/(s+1)*(args.steps-s-1)/60:.1f} min", flush=True)
        model.eval()
        recs = evaluate(model, tids, fold, args.arm, device)
        res_path.write_text(json.dumps({"arm": args.arm, "fold": fold, "steps": args.steps, "batch": args.batch,
                                        "height_scale": hs, "train_crops": len(tr), "train_s": time.time() - t0,
                                        "final_loss100": float(np.mean(run[-100:])), "records": recs}))
        print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


@torch.no_grad()
def evaluate(model, tids, fold, arm, device):
    """Per tile on held-out quadrant `fold`: squared-error sums for the DEM (30 m) and ICESat-2 (10 m cell) checks,
    plus the flat (crop-mean) and FABDEM-10 m baselines at the same ICESat-2 cells."""
    recs = []
    for tid in tids:
        z = np.load(CACHE / f"{tid}.npz")
        items = load_crops([tid], [fold])
        if not items:
            continue
        fab10 = z["fab10"]
        prow, pcol, ph_h = z["ph_row"].astype(int), z["ph_col"].astype(int), z["ph_h"]
        pfab = fab10[prow, pcol]
        acc = dict(dem_se=0.0, dem_se_flat=0.0, dem_n=0, ice_se=0.0, ice_se_flat=0.0, ice_se_fab=0.0, ice_n=0,
                   rel_pred=[], rel_true=[])
        for b in range(0, len(items), 16):
            chunk = items[b:b + 16]
            x = torch.stack([to_input(it[4]) for it in chunk]).to(device)
            mu = forward60(model, x, arm)[:, 0].float().cpu().numpy()
            for (_, _, r, c, _, t), m in zip(chunk, mu):
                pooled = m.reshape(N // K, K, N // K, K).mean((1, 3))
                off = pooled.mean(); cm = float(t.mean())
                pred30 = pooled - off + cm
                acc["dem_se"] += float(((pred30 - t) ** 2).sum()); acc["dem_se_flat"] += float(((cm - t) ** 2).sum())
                acc["dem_n"] += t.size
                acc["rel_pred"].append((pooled - off).ravel()); acc["rel_true"].append((t - cm).ravel())
                elev10 = m - off + cm
                sel = (prow >= r) & (prow < r + N) & (pcol >= c) & (pcol < c + N) & np.isfinite(pfab)
                if sel.any():
                    h, e = ph_h[sel], elev10[prow[sel] - r, pcol[sel] - c]
                    acc["ice_se"] += float(((e - h) ** 2).sum()); acc["ice_se_flat"] += float(((cm - h) ** 2).sum())
                    acc["ice_se_fab"] += float(((pfab[sel] - h) ** 2).sum()); acc["ice_n"] += int(sel.sum())
        rp, rt = np.concatenate(acc.pop("rel_pred")), np.concatenate(acc.pop("rel_true"))
        acc["relief_pearson"] = float(np.corrcoef(rp, rt)[0, 1]) if rt.std() > 0 and rp.std() > 0 else None
        acc["relief_var_ratio"] = float(rp.var() / rt.var()) if rt.var() > 0 else None
        recs.append({"tile": tid, **{k: float(v) if not isinstance(v, (int, type(None))) else v for k, v in acc.items()}})
    return recs


# ============================================================================ analysis (pre-registered)
def analyze(args):
    from scipy.stats import wilcoxon
    rows = []
    for arm in ("P", "R"):
        for f in range(4):
            p = OUT / f"eval_{arm}_fold{f}.json"
            assert p.exists(), f"missing {p}"
            for r in json.loads(p.read_text())["records"]:
                rows.append({"arm": arm, "fold": f, **r})
    D = pd.DataFrame(rows)
    agg = D.groupby(["arm", "tile"])[["dem_se", "dem_se_flat", "dem_n", "ice_se", "ice_se_flat", "ice_se_fab",
                                      "ice_n"]].sum()
    tiles = sorted(set(D.tile))

    def tile_rmse(arm, se, n):
        a = agg.loc[arm]
        return np.array([np.sqrt(a.loc[t, se] / a.loc[t, n]) if a.loc[t, n] > 0 else np.nan for t in tiles])

    def boot(v, B=10000):
        rng = np.random.default_rng(0); v = v[np.isfinite(v)]
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    S = {"n_tiles": len(tiles), "checks": {}}
    ps = {}
    for chk, se, n, base in (("icesat2", "ice_se", "ice_n", ["ice_se_flat", "ice_se_fab"]),
                             ("dem_heldout", "dem_se", "dem_n", ["dem_se_flat"])):
        P, R = tile_rmse("P", se, n), tile_rmse("R", se, n)
        ok = np.isfinite(P) & np.isfinite(R)
        d = R[ok] - P[ok]
        p = float(wilcoxon(R[ok], P[ok]).pvalue)
        ps[chk] = p
        S["checks"][chk] = {"n_tiles_scored": int(ok.sum()), "P_mean_rmse": float(P[ok].mean()),
                            "R_mean_rmse": float(R[ok].mean()), "R_minus_P_mean": float(d.mean()),
                            "R_minus_P_ci95": boot(d), "R_wins": int((d < 0).sum()), "wilcoxon_p": p,
                            **{f"{b}_mean_rmse": float(tile_rmse("P", b, n)[ok].mean()) for b in base},
                            "per_tile": {t: {"P": float(a), "R": float(b)} for t, a, b in
                                         zip(np.array(tiles)[ok], P[ok], R[ok])}}
    keys = sorted(ps, key=ps.get); run = 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k])); S["checks"][k]["wilcoxon_p_holm"] = run
    ic = S["checks"]["icesat2"]
    S["relief_pearson_mean"] = {a: float(D[D.arm == a].relief_pearson.mean()) for a in ("P", "R")}
    S["relief_var_ratio_mean"] = {a: float(D[D.arm == a].relief_var_ratio.mean()) for a in ("P", "R")}
    S["verdict_R_helps"] = bool(ic["R_wins"] > ic["n_tiles_scored"] / 2 and ic["R_minus_P_mean"] < 0
                                and ic["wilcoxon_p_holm"] < 0.05)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "checks"}, indent=1))
    for k, v in S["checks"].items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "per_tile"}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=["P", "R"], required=True)
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch", type=int, default=4)
    sub.add_parser("analyze")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"prep": prep, "train": train, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
