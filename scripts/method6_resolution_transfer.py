#!/usr/bin/env python3
"""Method 6 coarse-to-fine resolution transfer (2026-09-24). Pre-registration:
docs/method-audit/06-full-finetune-twin-head/resolution-transfer.md (commit a8d2568).

  train --gsd G --proto {P,R} [--folds ...] [--tag T]   Method 6's adopted height-balanced recipe, seed 42, the same
        4-fold quadrant split; only the input is degraded to GSD G (native = 0.3 m).
        P (pixel): area-average the 512^2 quadrant to n = round(512*0.3/G) px, reflect-pad to a multiple of 14,
                   predict, crop n x n, bilinearly upsample mu and log-var to 512^2, loss vs native AGL on the 512 grid.
        R (resample-back): area-average to n px, bilinear back to 512^2, then exactly the native path (pad 518).
        G = 0.3 is the identity transform and follows the native code path exactly (recipe-identity gate).
  eval  --models ...   Phase 3 matrix; writes data/dfc2019/experiments/resolution_transfer/matrix_tiles.json
The training loop below is a verbatim copy of scripts/evaluate_method6_gsd_film_height_balanced.py main(), with
the input transform as the only change.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_method4 as m4  # noqa: E402
import evaluate_method6_gsd_film_height_balanced as hb  # noqa: E402
from evaluate_method6_finetune_twinhead import (IMAGENET_MEAN, IMAGENET_STD, PAD_TO, compute_metrics,  # noqa: E402
                                                 gaussian_nll, get_device, load_tile_rgb_agl, masked_huber,
                                                 pad_to, seed_everything, tile_ids)

NATIVE_GSD = 0.3
OUT = ROOT / "data/dfc2019/experiments/resolution_transfer"


def n_px(gsd: float, size: int = 512) -> int:
    return max(1, int(round(size * NATIVE_GSD / gsd)))


def degrade(rgb: torch.Tensor, gsd: float, proto: str) -> torch.Tensor:
    """rgb (3,S,S) float 0-255 -> degraded input (still 0-255). Identity at native GSD."""
    S = rgb.shape[-1]
    n = n_px(gsd, S)
    if n >= S:
        return rgb
    small = F.interpolate(rgb[None], size=(n, n), mode="area")[0]
    if proto == "P":
        return small
    return F.interpolate(small[None], size=(S, S), mode="bilinear", align_corners=False)[0]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]


def pad14(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    P = ((n + 13) // 14) * 14
    return pad_to(x, P) if P > n else x


class DegradedQuadrantDataset(Dataset):
    """Native path (identity or R protocol): identical tensors to QuadrantDataset. P protocol: small padded input +
    native 512 AGL/valid (no padding)."""

    def __init__(self, samples, gsd, proto):
        self.samples, self.gsd, self.proto = samples, gsd, proto
        self.native_path = n_px(gsd) >= 512 or proto == "R"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, agl, valid = self.samples[idx]
        if self.native_path:
            if n_px(self.gsd) >= 512:  # identity: exactly QuadrantDataset
                rgb_t = torch.from_numpy(rgb / 255.0)
                rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            else:
                rgb_t = normalize(degrade(torch.from_numpy(rgb).float(), self.gsd, "R"))
            rgb_t = pad_to(rgb_t, PAD_TO)
            agl_t = torch.nan_to_num(pad_to(torch.from_numpy(agl)[None], PAD_TO)[0], nan=0.0)
            valid_t = pad_to(torch.from_numpy(valid.astype(np.float32))[None], PAD_TO)[0] > 0.5
            return rgb_t, agl_t, valid_t
        x = pad14(normalize(degrade(torch.from_numpy(rgb).float(), self.gsd, "P")))
        agl_t = torch.nan_to_num(torch.from_numpy(agl), nan=0.0)
        valid_t = torch.from_numpy(valid)
        return x, agl_t, valid_t


def forward_native_grid(model, x, gsd, proto, out_size=512):
    """Returns mu, log_var on the grid the loss/scoring uses. Native/R path: the model's padded output (caller crops).
    P path: crop n x n, bilinear to out_size."""
    mu, lv = model(x)
    if proto == "P" and n_px(gsd, out_size) < out_size:
        n = n_px(gsd, out_size)
        mu = F.interpolate(mu[:, :, :n, :n], size=(out_size, out_size), mode="bilinear", align_corners=False)
        lv = F.interpolate(lv[:, :, :n, :n], size=(out_size, out_size), mode="bilinear", align_corners=False)
    return mu, lv


def train(args):
    seed_everything(args.seed)
    device = get_device()
    tag = args.tag or f"rt_{args.proto}_{args.gsd:g}m_seed{args.seed}"
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    tids = tile_ids()
    cache = {tid: load_tile_rgb_agl(tid) for tid in tids}
    print(f"[{tag}] device={device} gsd={args.gsd} proto={args.proto} n_px={n_px(args.gsd)}", flush=True)
    fold_results = []
    for held_out_q in args.folds:
        t_fold = time.time()
        train_samples, train_agl_for_scale, train_weights, eval_samples = [], [], [], []
        for tid in tids:
            rgb, agl, valid = cache[tid]
            h, w = agl.shape
            for q in range(4):
                r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
                rgb_c, agl_c, valid_c = rgb[:, r0:r1, c0:c1], agl[r0:r1, c0:c1], valid[r0:r1, c0:c1]
                if q == held_out_q:
                    eval_samples.append((tid, rgb_c, agl_c, valid_c))
                else:
                    train_samples.append((rgb_c, agl_c, valid_c))
                    if valid_c.any():
                        train_agl_for_scale.append(agl_c[valid_c])
                    train_weights.append(hb.quadrant_sample_weight(agl_c, valid_c))
        height_scale = max(float(np.percentile(np.concatenate(train_agl_for_scale), 95)), 1.0)
        model = hb.TwinHeadDav2GSD(height_scale=height_scale, init_sigma_m=5.0, log_var_max=7.0, log_var_min=-8.0,
                                   enable_gsd_film=False).to(device)
        opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_backbone * 50, 0.01))
        train_ds = DegradedQuadrantDataset(train_samples, args.gsd, args.proto)
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds), replacement=True)
        train_ld = DataLoader(train_ds, batch_size=2, sampler=sampler, num_workers=0)
        steps_per_epoch = max(1, len(train_ds) // 2)
        total_steps = steps_per_epoch * args.epochs
        warm = max(10, int(total_steps * 0.05))

        def lr_lambda(step):
            if step < warm:
                return step / warm
            return max(0.05, (total_steps - step) / max(1, total_steps - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        model.train()
        gstep = 0
        for epoch in range(1, args.epochs + 1):
            running = []
            for rgb_t, agl_t, valid_t in train_ld:
                rgb_t, agl_t, valid_t = rgb_t.to(device), agl_t.to(device), valid_t.to(device)
                mu, log_var = forward_native_grid(model, rgb_t, args.gsd, args.proto)
                if gstep < 30:
                    loss = masked_huber(mu[:, 0], agl_t, valid_t)
                else:
                    loss = gaussian_nll(mu[:, 0], log_var[:, 0], agl_t, valid_t)
                loss = loss + hb.LAMBDA_HEIGHT_WEIGHT * hb.capped_height_weighted_huber(mu[:, 0], agl_t, valid_t)
                if not torch.isfinite(loss):
                    print(f"[{tag}] fold{held_out_q}: non-finite loss at step {gstep}, aborting fold", flush=True)
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                running.append(float(loss.detach().cpu()))
                gstep += 1
            if not running or not math.isfinite(running[-1]):
                break
            if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs:
                print(f"[{tag}] fold{held_out_q} epoch {epoch:02d}/{args.epochs}: loss={np.mean(running):.4f} "
                      f"({time.time()-t_fold:.0f}s)", flush=True)
        model.eval()
        per_tile = []
        with torch.no_grad():
            for tid, rgb_c, agl_c, valid_c in eval_samples:
                mu = predict(model, rgb_c, args.gsd, args.proto, device)
                m = compute_metrics(agl_c[valid_c], mu[valid_c]); m["tile"] = tid
                per_tile.append(m)
        torch.save({"state_dict": model.state_dict(), "height_scale": height_scale, "seed": args.seed, "fold": held_out_q,
                    "gsd": args.gsd, "proto": args.proto}, outdir / f"fold{held_out_q}.pt")
        from evaluate_method6_finetune_twinhead import agg
        fold_agg = agg(per_tile)
        print(f"[{tag}] fold{held_out_q} DONE in {time.time()-t_fold:.1f}s. own-GSD metrics={fold_agg}", flush=True)
        fold_results.append({"fold": held_out_q, "height_scale": height_scale, "own_gsd_metrics": fold_agg,
                             "fold_time_sec": time.time() - t_fold, "tiles": per_tile})
        (outdir / "train_results.json").write_text(json.dumps({"tag": tag, "gsd": args.gsd, "proto": args.proto,
                                                              "seed": args.seed, "folds": fold_results}, indent=1))
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()


def predict(model, rgb_c: np.ndarray, gsd: float, proto: str, device) -> np.ndarray:
    """Held-out quadrant rgb (3,512,512) 0-255 -> mu on the native 512 grid (numpy)."""
    if n_px(gsd) >= 512 or proto == "R":
        if n_px(gsd) >= 512:  # identical to the original evaluation path
            x = torch.from_numpy(rgb_c / 255.0)
            x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        else:
            x = normalize(degrade(torch.from_numpy(rgb_c).float(), gsd, "R"))
        x = pad_to(x, PAD_TO)[None].to(device)
        mu, _ = model(x)
        return mu[0, 0, :512, :512].float().cpu().numpy()
    x = pad14(normalize(degrade(torch.from_numpy(rgb_c).float(), gsd, "P")))[None].to(device)
    mu, _ = forward_native_grid(model, x, gsd, "P")
    return mu[0, 0].float().cpu().numpy()


# ============================================================================ Phase 3: evaluation matrix
EVAL_GSDS = [0.3, 1.2, 2.0, 2.4, 3.0, 5.0]


def metrics5(y, p):
    m = compute_metrics(y, p)
    ok = np.isfinite(y) & np.isfinite(p)
    vy = float(np.var(y[ok]))
    m["var_ratio"] = float(np.var(p[ok]) / vy) if vy > 0 else None
    return m


def load_fold_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = hb.TwinHeadDav2GSD(height_scale=ck["height_scale"], init_sigma_m=5.0, log_var_max=7.0, log_var_min=-8.0,
                           enable_gsd_film=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def oracle_depth(rgb_tile: np.ndarray, gsd: float, proto: str) -> np.ndarray:
    """Frozen DAv2-Large (demo engine, same as the DFC2019 oracle) on the whole degraded tile -> depth on native grid."""
    from PIL import Image
    from backend.depth.depth_engine import _DEVICE, _MODEL, _PROCESSOR, infer_with
    S = rgb_tile.shape[-1]
    x = degrade(torch.from_numpy(rgb_tile).float(), gsd, proto)
    img = Image.fromarray(np.clip(np.rint(x.numpy()), 0, 255).astype(np.uint8).transpose(1, 2, 0), "RGB")
    d = infer_with(img, _PROCESSOR, _MODEL, _DEVICE)
    if d.shape[0] != S:
        d = F.interpolate(torch.from_numpy(d)[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
    return d


def evaluate(args):
    device = get_device()
    tids = tile_ids()
    cache = {tid: load_tile_rgb_agl(tid) for tid in tids}
    models = {"native": (ROOT / "data/dfc2019/experiments/method6_height_balanced_seed42_ckpt", ["P", "R"])}
    for g in args.train_gsds:
        for pr in ("P", "R"):
            d = OUT / f"rt_{pr}_{g:g}m_seed42"
            if all((d / f"fold{q}.pt").exists() for q in range(4)):
                models[f"{pr}{g:g}"] = (d, [pr])
    eval_gsds = sorted(set(EVAL_GSDS + list(args.train_gsds)))
    out_path = OUT / "matrix_tiles.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            r = json.loads(l); done.add((r["model"], r["proto"], r["eval_gsd"], r["tile"], r["q"]))
    fout = out_path.open("a")
    for name, (d, protos) in models.items():
        for q in range(4):
            model = load_fold_model(d / f"fold{q}.pt", device)
            for pr in protos:
                for e in eval_gsds:
                    for tid in tids:
                        if (name, pr, e, tid, q) in done:
                            continue
                        rgb, agl, valid = cache[tid]
                        r0, r1, c0, c1 = m4.quadrant_bounds(*agl.shape, q)
                        with torch.no_grad():
                            mu = predict(model, rgb[:, r0:r1, c0:c1], e, pr, device)
                        yv = agl[r0:r1, c0:c1]; vv = valid[r0:r1, c0:c1]
                        rec = {"model": name, "proto": pr, "eval_gsd": e, "tile": tid, "q": q, **metrics5(yv[vv], mu[vv])}
                        fout.write(json.dumps(rec) + "\n")
            fout.flush()
            del model
            print(f"eval {name} fold{q} done", flush=True)
    # oracle per protocol x eval GSD
    for pr in ("P", "R"):
        for e in eval_gsds:
            for tid in tids:
                if ("oracle", pr, e, tid, 3) in done:
                    continue
                rgb, agl, valid = cache[tid]
                dep = oracle_depth(rgb, e, pr)
                for q in range(4):
                    qm = np.zeros(agl.shape, bool)
                    r0, r1, c0, c1 = m4.quadrant_bounds(*agl.shape, q); qm[r0:r1, c0:c1] = True
                    ok = valid & np.isfinite(dep)
                    tr, te = ok & ~qm, ok & qm
                    A = np.vstack([dep[tr], np.ones(tr.sum())]).T
                    coef = np.linalg.lstsq(A, agl[tr], rcond=None)[0]
                    pred = coef[0] * dep[te] + coef[1]
                    rec = {"model": "oracle", "proto": pr, "eval_gsd": e, "tile": tid, "q": q, **metrics5(agl[te], pred)}
                    fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"oracle {pr} {e} m done", flush=True)
    fout.close()


# ============================================================================ Phase 4: pre-registered analysis
def analyze(args):
    from scipy.stats import wilcoxon
    recs = [json.loads(l) for l in (OUT / "matrix_tiles.jsonl").read_text().splitlines()]
    import collections
    T = collections.defaultdict(lambda: collections.defaultdict(list))  # (model,proto,e) -> tile -> [quadrant metrics]
    for r in recs:
        T[(r["model"], r["proto"], r["eval_gsd"])][r["tile"]].append(r)
    tiles = sorted({r["tile"] for r in recs})
    MET = ("mae", "rmse", "pearson", "spearman", "var_ratio")
    LOWER = {"mae": True, "rmse": True, "pearson": False, "spearman": False}

    def tile_vec(key, m):
        return np.array([np.nanmean([x[m] if x[m] is not None else np.nan for x in T[key][t]]) for t in tiles])

    def boot(v, B=10000):
        rng = np.random.default_rng(0); v = v[np.isfinite(v)]
        mm = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

    def holm(ps):
        keys = sorted(ps, key=ps.get); out, run = {}, 0.0
        for i, k in enumerate(keys):
            run = max(run, min(1.0, (len(keys) - i) * ps[k])); out[k] = run
        return out

    S = {"n_tiles": len(tiles), "aggregation": "tile = mean of its 4 held-out quadrants; headline = mean of tiles",
         "matrix": {}, "rules": {}}
    for key in sorted(T):
        name, pr, e = key
        S["matrix"][f"{name}|{pr}|{e:g}"] = {m: {"mean": float(np.nanmean(tile_vec(key, m))), "ci95": boot(tile_vec(key, m))}
                                            for m in MET}
    gs = sorted({float(k[0][1:]) for k in T if k[0][0] in "PR" and k[0] != "native" and k[0] != "oracle"})
    for pr in ("P", "R"):
        R1, ps_sym = {}, {}
        for m in ("mae", "rmse", "pearson", "spearman"):
            pass
        for g in gs:
            name = f"{pr}{g:g}"
            if (name, pr, 0.3) not in T:
                continue
            res = {}
            wins_ci = 0
            for m in ("mae", "rmse", "pearson", "spearman"):
                a, b = tile_vec((name, pr, 0.3), m), tile_vec(("oracle", pr, 0.3), m)
                ca, cb = boot(a), boot(b)
                better = np.nanmean(a) < np.nanmean(b) if LOWER[m] else np.nanmean(a) > np.nanmean(b)
                nonover = (ca[1] < cb[0]) if LOWER[m] else (ca[0] > cb[1])
                ok = np.isfinite(a) & np.isfinite(b)
                res[m] = {"model": float(np.nanmean(a)), "oracle": float(np.nanmean(b)), "better": bool(better),
                          "ci_nonoverlap": bool(nonover), "wins": int(((a < b) if LOWER[m] else (a > b))[ok].sum()),
                          "wilcoxon_p": float(wilcoxon(a[ok], b[ok]).pvalue)}
                wins_ci += int(better and nonover)
            res["n_metrics_better_ci_nonoverlap"] = wins_ci
            res["transfer_works"] = wins_ci >= 3
            R1[name] = res
            # symmetry: g-trained at 0.3 vs native at g, per metric
            sym = {}
            for m in ("mae", "rmse", "pearson", "spearman"):
                a, b = tile_vec((name, pr, 0.3), m), tile_vec(("native", pr, g), m)
                ok = np.isfinite(a) & np.isfinite(b)
                sym[m] = {"coarse_to_fine": float(np.nanmean(a)), "fine_to_coarse": float(np.nanmean(b)),
                          "diff_c2f_minus_f2c": float(np.nanmean(a[ok] - b[ok])), "diff_ci95": boot(a[ok] - b[ok]),
                          "wilcoxon_p": float(wilcoxon(a[ok], b[ok]).pvalue)}
                if m == "pearson":
                    ps_sym[name] = sym[m]["wilcoxon_p"]
            R1[name]["symmetry"] = sym
        for m in ("mae", "rmse", "pearson", "spearman"):
            h = holm({n: R1[n][m]["wilcoxon_p"] for n in R1})
            for n in R1:
                R1[n][m]["wilcoxon_p_holm"] = h[n]
        hs = holm(ps_sym)
        for n in R1:
            p = hs.get(n)
            d = R1[n]["symmetry"]["pearson"]["diff_c2f_minus_f2c"]
            R1[n]["symmetry"]["pearson"]["p_holm"] = p
            R1[n]["symmetry"]["verdict"] = ("asymmetric: coarse->fine better" if d > 0 else "asymmetric: fine->coarse better") \
                if p is not None and p < 0.05 else "no evidence of asymmetry"
        # dose-response on native-eval Pearson / MAE
        curve = [("native", 0.3)] + [(f"{pr}{g:g}", g) for g in gs if (f"{pr}{g:g}", pr, 0.3) in T]
        pear = [float(np.nanmean(tile_vec((n, pr, 0.3), "pearson"))) for n, _ in curve]
        mae = [float(np.nanmean(tile_vec((n, pr, 0.3), "mae"))) for n, _ in curve]
        drops = [pear[i] - pear[i + 1] for i in range(len(pear) - 1)]
        total = pear[0] - pear[-1] if len(pear) > 1 else 0.0
        cliff = [f"{curve[i][1]:g}->{curve[i+1][1]:g} m" for i, d in enumerate(drops) if total > 0 and d >= 0.6 * total]
        S["rules"][pr] = {"transfer": R1, "dose_response": {"train_gsd": [g for _, g in curve], "native_eval_pearson": pear,
                                                            "native_eval_mae": mae, "monotonic": all(d > 0 for d in drops),
                                                            "cliff": cliff or None}}
    (OUT / "matrix_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S["rules"], indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--gsd", type=float, required=True)
    t.add_argument("--proto", choices=["P", "R"], required=True)
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--lr-backbone", type=float, default=5e-6)
    t.add_argument("--tag", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--train-gsds", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
