#!/usr/bin/env python3
"""Sentinel-2 rank-loss test (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "Rank-loss test").

Phase C's setup with ONLY the loss changed: pure pairwise ranking (rank_pair_loss imported UNCHANGED from
scripts/evaluate_method4_v2.py -- F.margin_ranking_loss, 2000 random pairs per sample, margin 0.25, exact ties dropped;
the rank term of Method 4 v2 phase2_building_rank_v2 on DFC2019) against FABDEM-30 m order within each crop.
Everything else is imported from scripts/s2_token_grid_phase_c.py: 32 tiles, 4 quadrant folds, 60 px crops, FABDEM on
the 30 m grid (loss on 3x3-pooled output), arm-P geometry (5x5 tokens), TwinHeadDav2 (DAv2-Small), 600 steps x batch 4,
AdamW 5e-6/2.5e-4, same schedule, seed and crop order.

Metric scale is recovered post hoc, never asked of the network: per tile, a no-intercept OLS slope `a` of FABDEM
within-crop relief on score within-crop relief (30 m grid), fitted on that tile's 3 TRAINING quadrants only; held-out
prediction = crop FABDEM mean + a * (score - crop mean score). The oracle gets the identical calibration applied to
frozen DAv2-Large whole-tile depth (data/sentinel2_benchmark/dav2_depth, the project's standard oracle input).

  .venv/bin/python scripts/s2_rank_loss_test.py train [--folds ...]
  .venv/bin/python scripts/s2_rank_loss_test.py analyze
Output: data/sentinel2_benchmark/token_grid_test/rank_loss/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_token_grid_phase_c as C  # noqa: E402
from evaluate_method4_v2 import rank_pair_loss  # noqa: E402
from evaluate_method6_finetune_twinhead import TwinHeadDav2, get_device, seed_everything  # noqa: E402

OUT = C.OUT / "rank_loss"
DAV2 = ROOT / "data/sentinel2_benchmark/dav2_depth"
ARM = "P"
PAIRS, MARGIN = 2000, 0.25  # Method 4 v2 values (run_method4_sentinel2.py RANK_PAIRS_PER_PATCH / RANK_MARGIN)
N, K = C.N, C.K


def rank_loss(mu60, tgt20):
    p = F.avg_pool2d(mu60, K)                              # (B,1,20,20) score on the 30 m grid
    t = tgt20[:, None]
    return rank_pair_loss(p, t, torch.ones_like(t, dtype=torch.bool), PAIRS, MARGIN)


@torch.no_grad()
def scores_all(model, tid, device):
    """Model score on the 10 m grid for every crop of the tile (all 4 quadrants)."""
    out = {}
    items = C.load_crops([tid], range(4))
    for b in range(0, len(items), 32):
        ch = items[b:b + 32]
        x = torch.stack([C.to_input(it[4]) for it in ch]).to(device)
        mu = C.forward60(model, x, ARM)[:, 0].float().cpu().numpy()
        for it, m in zip(ch, mu):
            out[(it[1], it[2], it[3])] = (m, it[5])
    return out


def calibrated_eval(score_by_crop, tid, fold):
    """score_by_crop[(q,r,c)] = (score60 on 10 m grid, fabdem30 target 20x20). Per-tile slope fit on training quadrants."""
    num = den = 0.0
    for (q, r, c), (s, t) in score_by_crop.items():
        if q == fold:
            continue
        s30 = s.reshape(N // K, K, N // K, K).mean((1, 3)); s30 = s30 - s30.mean()
        num += float((s30 * (t - t.mean())).sum()); den += float((s30 ** 2).sum())
    a = num / den if den > 0 else 0.0
    z = np.load(C.CACHE / f"{tid}.npz")
    prow, pcol, ph = z["ph_row"].astype(int), z["ph_col"].astype(int), z["ph_h"]
    pfab = z["fab10"][prow, pcol]
    acc = dict(slope=a, dem_se=0.0, dem_n=0, ice_se=0.0, ice_se_flat=0.0, ice_se_fab=0.0, ice_n=0)
    for (q, r, c), (s, t) in score_by_crop.items():
        if q != fold:
            continue
        s30 = s.reshape(N // K, K, N // K, K).mean((1, 3)); off = s30.mean(); cm = float(t.mean())
        acc["dem_se"] += float(((cm + a * (s30 - off) - t) ** 2).sum()); acc["dem_n"] += t.size
        sel = (prow >= r) & (prow < r + N) & (pcol >= c) & (pcol < c + N) & np.isfinite(pfab)
        if sel.any():
            e = cm + a * (s[prow[sel] - r, pcol[sel] - c] - off); h = ph[sel]
            acc["ice_se"] += float(((e - h) ** 2).sum()); acc["ice_se_flat"] += float(((cm - h) ** 2).sum())
            acc["ice_se_fab"] += float(((pfab[sel] - h) ** 2).sum()); acc["ice_n"] += int(sel.sum())
    return acc


def oracle_scores(tid):
    d = np.load(DAV2 / f"{tid}_depth.npy").astype(np.float64)
    out = {}
    for q, r, c, _, t in ((it[1], it[2], it[3], None, it[5]) for it in C.load_crops([tid], range(4))):
        out[(q, r, c)] = (d[r:r + N, c:c + N], t)
    return out


def fit(fold, tids, steps, batch, device):
    """Train one fold exactly as the pre-registered run did; returns (model, height_scale, loss_trace)."""
    seed_everything(42)
    tr = C.load_crops(tids, [q for q in range(4) if q != fold])
    rel = np.concatenate([(t - t.mean()).ravel() for *_, t in tr])
    hs = max(float(rel.std()), 1.0)
    model = TwinHeadDav2(height_scale=hs, init_sigma_m=hs * 0.3).to(device)
    opt = torch.optim.AdamW(model.param_groups(5e-6, 2.5e-4, 1e-4))
    warm = max(10, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.05, (steps - s) / max(1, steps - warm)))
    order = np.random.default_rng(42 + fold).permutation(np.resize(np.arange(len(tr)), steps * batch))
    print(f"[rank_fold{fold}] device={device} train crops {len(tr)} steps {steps} x batch {batch}", flush=True)
    model.train(); t0 = time.time(); run = []
    for s in range(steps):
        idx = order[s * batch:(s + 1) * batch]
        x = torch.stack([C.to_input(tr[i][4]) for i in idx]).to(device)
        y = torch.from_numpy(np.stack([tr[i][5] for i in idx])).to(device)
        loss = rank_loss(C.forward60(model, x, ARM), y)
        assert torch.isfinite(loss), f"non-finite loss at step {s}"
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run.append(float(loss.detach().cpu()))
        if (s + 1) % 100 == 0 or s == 0:
            el = time.time() - t0
            print(f"[rank_fold{fold}] step {s+1}/{steps} loss {np.mean(run[-100:]):.4f} "
                  f"ETA {el/(s+1)*(steps-s-1)/60:.1f} min", flush=True)
    model.eval()
    return model, hs, run


def train(args):
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    tids = sorted(p.stem for p in C.CACHE.glob("*.npz"))
    for fold in args.folds:
        res_path = OUT / f"eval_rank_fold{fold}.json"
        if res_path.exists():
            print(f"fold{fold}: exists, skip"); continue
        t0 = time.time()
        model, hs, run = fit(fold, tids, args.steps, args.batch, device)
        recs = []
        for tid in tids:
            m = calibrated_eval(scores_all(model, tid, device), tid, fold)
            o = calibrated_eval(oracle_scores(tid), tid, fold)
            recs.append({"tile": tid, "model": m, "oracle": o})
        res_path.write_text(json.dumps({"fold": fold, "steps": args.steps, "batch": args.batch, "height_scale": hs,
                                        "final_loss100": float(np.mean(run[-100:])), "train_s": time.time() - t0,
                                        "records": recs}))
        print(f"[rank_fold{fold}] done {time.time()-t0:.0f}s", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


def analyze(args):
    from scipy.stats import wilcoxon
    rows = []
    for f in range(4):
        for r in json.loads((OUT / f"eval_rank_fold{f}.json").read_text())["records"]:
            for who in ("model", "oracle"):
                rows.append({"tile": r["tile"], "fold": f, "who": who, **r[who]})
    D = pd.DataFrame(rows)
    agg = D.groupby(["who", "tile"])[["dem_se", "dem_n", "ice_se", "ice_se_flat", "ice_se_fab", "ice_n"]].sum()
    tiles = sorted(set(D.tile))

    def rm(who, se, n):
        a = agg.loc[who]
        return np.array([np.sqrt(a.loc[t, se] / a.loc[t, n]) for t in tiles])

    def boot(v, B=10000):
        rng = np.random.default_rng(0)
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    # context: Phase C arm P (magnitude loss, same geometry), from the primary Kaggle run
    pc = {}
    for f in range(4):
        for r in json.loads((C.OUT / "kaggle" / f"eval_P_fold{f}.json").read_text())["records"]:
            x = pc.setdefault(r["tile"], [0.0, 0])
            x[0] += r["ice_se"]; x[1] += r["ice_n"]
    # only meaningful on the Indian benchmark tiles (e.g. absent for the Brazil benchmark)
    phaseC_P = np.array([np.sqrt(pc[t][0] / pc[t][1]) for t in tiles]) if all(t in pc for t in tiles) else None

    S = {"n_tiles": len(tiles), "checks": {}, "slopes": {
        "model_median": float(D[D.who == "model"].slope.median()), "oracle_median": float(D[D.who == "oracle"].slope.median()),
        "model_frac_negative": float((D[D.who == "model"].slope < 0).mean()),
        "oracle_frac_negative": float((D[D.who == "oracle"].slope < 0).mean())}}
    ps = {}
    for chk, se, n in (("icesat2", "ice_se", "ice_n"), ("dem_heldout", "dem_se", "dem_n")):
        M, O = rm("model", se, n), rm("oracle", se, n)
        d = M - O
        ps[chk] = float(wilcoxon(M, O).pvalue)
        S["checks"][chk] = {"model_mean_rmse": float(M.mean()), "oracle_mean_rmse": float(O.mean()),
                            "model_minus_oracle": float(d.mean()), "ci95": boot(d), "model_wins": int((d < 0).sum()),
                            "wilcoxon_p": ps[chk],
                            "per_tile": {t: {"model": float(a), "oracle": float(b)} for t, a, b in zip(tiles, M, O)}}
    keys = sorted(ps, key=ps.get); run = 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k])); S["checks"][k]["wilcoxon_p_holm"] = run
    ic = S["checks"]["icesat2"]
    ic["flat_mean_rmse"] = float(rm("model", "ice_se_flat", "ice_n").mean())
    ic["fabdem_mean_rmse"] = float(rm("model", "ice_se_fab", "ice_n").mean())
    M = rm("model", "ice_se", "ice_n")
    if phaseC_P is not None:
        ic["context_phaseC_armP_mean_rmse"] = float(phaseC_P.mean())
        ic["context_rank_minus_phaseC_P"] = {"mean": float((M - phaseC_P).mean()), "ci95": boot(M - phaseC_P),
                                            "rank_wins": int((M < phaseC_P).sum()),
                                            "wilcoxon_p": float(wilcoxon(M, phaseC_P).pvalue)}
    S["verdict_real_signal"] = bool(ic["model_wins"] > len(tiles) / 2 and ic["model_minus_oracle"] < 0
                                    and ic["wilcoxon_p_holm"] < 0.05)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "checks"}, indent=1))
    for k, v in S["checks"].items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "per_tile"}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch", type=int, default=4)
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"train": train, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
