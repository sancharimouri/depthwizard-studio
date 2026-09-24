#!/usr/bin/env python3
"""Kaggle build of scripts/terrain_relief_positive_control.py `run` (terrain-relief positive control; 2026-09-24).
Pre-registration: docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md,
"Terrain-relief positive control".

Self-contained: no repo imports.
  - VERBATIM: seed_everything, TwinHeadDav2 and pad_to (from scripts/evaluate_method6_finetune_twinhead.py via
    scripts/s2_token_grid_phase_c_kaggle.py); rank_pair_loss (from scripts/evaluate_method4_v2.py, SID branch removed,
    unused here); crop layout / load_crops / fit / calibrated_eval / cells_for_fold logic (from
    s2_token_grid_phase_c.py, s2_rank_loss_test.py, s2_rank_raw_spearman.py with the geometry patch of
    terrain_relief_positive_control.py: N 600, K 30, P_PAD 602, 6 x 6 crops per quadrant, tile 7200 px).
  - Changed for the GPU, not in substance:
      input normalisation runs on the GPU (same float32 ops as to_input);
      calibration sums, the calibrated errors and the per-crop Spearman (ties averaged, = scipy rankdata 'average')
      run in float64 on cuda:0; only the final Wilcoxon / bootstrap summary (a few dozen numbers) runs on the CPU;
      the reference-slot DTM is shipped only at the truth cells (ph_fab) instead of the full 1 m grid;
      the oracle depth is shipped at 3600^2 and repeated 2x on the GPU (the local run stores it already repeated).
  - Added: resume. Every 100 steps a checkpoint (model, optimiser, schedule, loss trace, CPU + CUDA RNG) goes to
    --out; a restarted run continues from it, and finished folds (eval_rank_fold*.json) are skipped.
    A heartbeat line is printed every 25 steps and per scored tile, for the notebook's stall watchdog.
Pinned: transformers==5.17.0 (TwinHeadDav2 uses backbone.forward_with_filtered_kwargs).

  python terrain_relief_kaggle.py run --data BUNDLE --out /kaggle/working/terrain_ctrl
  python terrain_relief_kaggle.py summarize --data BUNDLE --out /kaggle/working/terrain_ctrl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
PATCH_SIZE = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODEL_ID = None   # set in main(): BUNDLE/dav2_small (local HF snapshot)
CACHE = None      # set in main(): BUNDLE/cache
ORACLE = None     # set in main(): BUNDLE/oracle
OUT = None        # set in main(): --out


def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TwinHeadDav2(nn.Module):
    """DA-V2-Small backbone/neck + pretrained mean head + fresh log-variance head.

    Independent reimplementation of the twin-head idea in
    zaidnansari2011/sih2026-depthwizard/depthwizard/model.py (read, not
    imported or executed).
    """

    def __init__(self, model_id: str = None, height_scale: float = 30.0,
                 init_sigma_m: float = 5.0, log_var_max: float = 7.0, log_var_min: float = -8.0,
                 init_mu: str = "pretrained", init_mu_m: float = 0.0):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        base = AutoModelForDepthEstimation.from_pretrained(model_id or MODEL_ID)
        cfg = base.config

        self.backbone = base.backbone
        self.neck = base.neck
        self.patch_size = int(getattr(cfg, "patch_size", PATCH_SIZE))
        self.head_in_index = int(getattr(cfg, "head_in_index", -1))
        self.height_scale = float(height_scale)
        self.log_var_max = float(log_var_max)
        self.log_var_min = float(log_var_min)

        self.conv1 = base.head.conv1
        self.conv2 = base.head.conv2
        self.activation1 = base.head.activation1
        self.conv_mu = base.head.conv3  # pretrained 32 -> 1, reused as the mean readout

        # "pretrained" reproduces DA-V2's own disparity-scale readout, which is fine when
        # height_scale is small (DFC2019 AGL, ~O(30m)). "constant" re-inits conv_mu to emit
        # a spatially-uniform init_mu_m everywhere -- needed when height_scale is large and
        # variable across tiles (e.g. absolute SRTM elevation across India, -112m to +4125m):
        # otherwise the pretrained readout's O(1)-normalized output times a large height_scale
        # produces a huge initial residual on every pixel, which is exactly the mechanism the
        # source repo (zaidnansari2011/model.py) documents killing its own run05 (a 666m
        # residual railing log_var at init). Same fix, applied for the same reason.
        if init_mu == "constant":
            nn.init.zeros_(self.conv_mu.weight)
            nn.init.constant_(self.conv_mu.bias, float(init_mu_m) / float(height_scale))
        elif init_mu != "pretrained":
            raise ValueError(f"init_mu must be 'pretrained' or 'constant', got {init_mu!r}")

        hidden = self.conv_mu.in_channels
        self.conv_log_var = nn.Conv2d(hidden, 1, kernel_size=1)
        # Start as a spatially-uniform sigma = init_sigma_m; only the data should
        # make it spatially varying (same reasoning as the source repo).
        nn.init.zeros_(self.conv_log_var.weight)
        nn.init.constant_(self.conv_log_var.bias,
                          2.0 * math.log(max(init_sigma_m, 1e-6) / self.height_scale))
        del base

    def forward(self, pixel_values: torch.Tensor):
        _, _, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"{H}x{W} not a multiple of patch size {self.patch_size}"
        ph, pw = H // self.patch_size, W // self.patch_size

        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False
        )
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]

        x = self.conv1(feat)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))

        mu = self.conv_mu(x) * self.height_scale
        log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        return mu, log_var

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        head = nn.ModuleList([self.neck, self.conv1, self.conv2, self.conv_mu, self.conv_log_var])
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": [p for p in head.parameters() if p.requires_grad],
             "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
        ]


def pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Reflect-pad the last two dims up to `size` (pads bottom/right only)."""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = size - h, size - w
    assert ph >= 0 and pw >= 0
    return F.pad(x, (0, pw, 0, ph), mode="reflect")



# ============================================================================ geometry (terrain_relief_positive_control)
N, K, P_PAD = 600, 30, 602           # 600 m crop at 1.0 m; 30 m target cells (20 x 20); 43 x 43 tokens
PER_Q, Q0 = 6, 3600                  # 6 x 6 crops per quadrant; tile 7200 px
TILE = 2 * Q0
PAIRS, MARGIN = 2000, 0.25           # Method 4 v2 rank_pair_loss settings
MIN_CELLS = 10


def crop_origins(q: int, size: int = TILE):
    r0 = 0 if q in (0, 1) else Q0
    c0 = 0 if q in (0, 2) else Q0
    return [(r0 + N * i, c0 + N * j) for i in range(PER_Q) for j in range(PER_Q)]


_Z = {}


def npz(tid):
    """Tile arrays, loaded once (the CPU only decodes; ~160 MB per tile)."""
    if tid not in _Z:
        z = np.load(CACHE / f"{tid}.npz")
        _Z[tid] = {k: z[k] for k in z.files}
    return _Z[tid]


def load_crops(tids, quads):
    """Verbatim logic of s2_token_grid_phase_c.load_crops (same order, same finite-target filter)."""
    items = []
    for tid in tids:
        z = npz(tid)
        rgb, fab30 = z["rgb"], z["fab30"]
        for q in quads:
            for r, c in crop_origins(q):
                t = fab30[r // K:(r + N) // K, c // K:(c + N) // K]
                if t.shape != (N // K, N // K) or not np.isfinite(t).all():
                    continue
                items.append((tid, q, r, c, rgb[:, r:r + N, c:c + N], t))
    return items


def to_input_gpu(u8: torch.Tensor) -> torch.Tensor:
    """= s2_token_grid_phase_c.to_input, applied on the GPU to a (B,3,H,W) uint8 batch (same float32 ops)."""
    x = u8.float() / 255.0
    return (x - IMAGENET_MEAN.to(x.device).float()) / IMAGENET_STD.to(x.device).float()


def forwardN(model, x):
    """Arm P (= s2_token_grid_phase_c.forward60 with arm 'P'): reflect-pad to P_PAD, crop back to N."""
    mu, _ = model(pad_to(x, P_PAD))
    return mu[:, :, :N, :N]


# ---------------------------------------------------------------------------- loss (verbatim from evaluate_method4_v2)
def rank_pair_loss(pred, target, mask, pairs_per_sample, margin):
    B = pred.shape[0]
    losses = []
    for b in range(B):
        m = mask[b, 0]
        idx = torch.nonzero(m, as_tuple=False)
        n = idx.shape[0]
        if n < 2:
            continue
        k = min(pairs_per_sample, n * (n - 1) // 2)
        if k <= 0:
            continue
        i1 = torch.randint(0, n, (k,), device=pred.device)
        i2 = torch.randint(0, n, (k,), device=pred.device)
        keep = i1 != i2
        i1, i2 = i1[keep], i2[keep]
        if i1.numel() == 0:
            continue
        r1, c1 = idx[i1, 0], idx[i1, 1]
        r2, c2 = idx[i2, 0], idx[i2, 1]
        y1 = target[b, 0, r1, c1]
        y2 = target[b, 0, r2, c2]
        p1 = pred[b, 0, r1, c1]
        p2 = pred[b, 0, r2, c2]
        target_sign = torch.sign(y1 - y2)
        tie_mask = target_sign != 0
        if tie_mask.sum() == 0:
            continue
        p1, p2, target_sign = p1[tie_mask], p2[tie_mask], target_sign[tie_mask]
        losses.append(F.margin_ranking_loss(p1, p2, target_sign, margin=margin))
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def rank_loss(mu, tgt):
    p = F.avg_pool2d(mu, K)
    t = tgt[:, None]
    return rank_pair_loss(p, t, torch.ones_like(t, dtype=torch.bool), PAIRS, MARGIN)


# ---------------------------------------------------------------------------- training (= s2_rank_loss_test.fit + resume)
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit(fold, tids, steps, batch, device):
    seed_everything(42)
    tr = load_crops(tids, [q for q in range(4) if q != fold])
    rel = np.concatenate([(t - t.mean()).ravel() for *_, t in tr])
    hs = max(float(rel.std()), 1.0)
    model = TwinHeadDav2(height_scale=hs, init_sigma_m=hs * 0.3).to(device)
    opt = torch.optim.AdamW(model.param_groups(5e-6, 2.5e-4, 1e-4))
    warm = max(10, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.05, (steps - s) / max(1, steps - warm)))
    order = np.random.default_rng(42 + fold).permutation(np.resize(np.arange(len(tr)), steps * batch))
    ck = OUT / f"ckpt_fold{fold}.pt"
    start, run = 0, []
    if ck.exists():  # resume: model, optimiser, schedule, loss trace and both RNG streams
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, run = st["step"], st["run"]
        torch.set_rng_state(st["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(st["rng_dev"])
        log(f"[rank_fold{fold}] resumed at step {start}")
    log(f"[rank_fold{fold}] device={device} train crops {len(tr)} steps {steps} x batch {batch}")
    tr_u8 = [torch.from_numpy(np.ascontiguousarray(it[4])) for it in tr]
    tr_t = [torch.from_numpy(np.ascontiguousarray(it[5])) for it in tr]
    model.train(); t0 = time.time()
    for s in range(start, steps):
        idx = order[s * batch:(s + 1) * batch]
        x = to_input_gpu(torch.stack([tr_u8[i] for i in idx]).to(device, non_blocking=True))
        y = torch.stack([tr_t[i] for i in idx]).to(device)
        loss = rank_loss(forwardN(model, x), y)
        assert torch.isfinite(loss), f"non-finite loss at step {s}"
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run.append(float(loss.detach().cpu()))
        if (s + 1) % 25 == 0 or s == start:
            el = time.time() - t0; done = s + 1 - start
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            log(f"[rank_fold{fold}] step {s+1}/{steps} loss {np.mean(run[-100:]):.4f} {el/done:.2f}s/step "
                f"ETA {el/done*(steps-s-1)/60:.1f} min GPU peak {mem:.1f} GiB")
        if (s + 1) % 100 == 0 or s + 1 == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": s + 1, "run": run, "rng_cpu": torch.get_rng_state(),
                        "rng_dev": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ck.with_suffix(".tmp"))
            ck.with_suffix(".tmp").replace(ck)
    model.eval()
    return model, hs, run


# ---------------------------------------------------------------------------- scoring (GPU, float64)
@torch.no_grad()
def scores_all(model, tid, device, bs):
    """Model score on the 1 m grid for every crop of the tile (all 4 quadrants), kept on the GPU."""
    out = {}
    items = load_crops([tid], range(4))
    for b in range(0, len(items), bs):
        ch = items[b:b + bs]
        x = to_input_gpu(torch.stack([torch.from_numpy(np.ascontiguousarray(it[4])) for it in ch]).to(device))
        mu = forwardN(model, x)[:, 0].float()
        for it, m in zip(ch, mu):
            out[(it[1], it[2], it[3])] = (m.double(), torch.from_numpy(it[5]).to(device).double())
    return out


def oracle_scores(tid, device):
    d2 = torch.from_numpy(np.load(ORACLE / f"{tid}_depth_half.npy")).to(device)
    d = d2.repeat_interleave(2, 0).repeat_interleave(2, 1).double()   # = the local 2x-repeated float16 depth
    out = {}
    for it in load_crops([tid], range(4)):
        q, r, c, t = it[1], it[2], it[3], it[5]
        out[(q, r, c)] = (d[r:r + N, c:c + N], torch.from_numpy(t).to(device).double())
    return out


def pool30(s):
    return s.reshape(N // K, K, N // K, K).mean((1, 3))


def cells(tid, r, c, device):
    """Indices of this crop's truth cells (the 'ICESat-2' slot) and their values, on the GPU."""
    z = npz(tid)
    prow, pcol = z["ph_row"], z["ph_col"]
    sel = (prow >= r) & (prow < r + N) & (pcol >= c) & (pcol < c + N) & np.isfinite(z["ph_fab"])
    return (torch.from_numpy(prow[sel] - r).to(device), torch.from_numpy(pcol[sel] - c).to(device),
            torch.from_numpy(z["ph_h"][sel]).to(device).double(), torch.from_numpy(z["ph_fab"][sel].astype(np.float64)).to(device))


def calibrated_eval(score_by_crop, tid, fold, device):
    """= s2_rank_loss_test.calibrated_eval, in float64 on the GPU."""
    num = den = torch.zeros((), dtype=torch.float64, device=device)
    for (q, r, c), (s, t) in score_by_crop.items():
        if q == fold:
            continue
        s30 = pool30(s); s30 = s30 - s30.mean()
        num = num + (s30 * (t - t.mean())).sum(); den = den + (s30 ** 2).sum()
    a = float(num / den) if float(den) > 0 else 0.0
    acc = dict(slope=a, dem_se=0.0, dem_n=0, ice_se=0.0, ice_se_flat=0.0, ice_se_fab=0.0, ice_n=0)
    for (q, r, c), (s, t) in score_by_crop.items():
        if q != fold:
            continue
        s30 = pool30(s); off = s30.mean(); cm = t.mean()
        acc["dem_se"] += float(((cm + a * (s30 - off) - t) ** 2).sum()); acc["dem_n"] += t.numel()
        rr, cc, h, pf = cells(tid, r, c, device)
        if rr.numel():
            e = cm + a * (s[rr, cc] - off)
            acc["ice_se"] += float(((e - h) ** 2).sum()); acc["ice_se_flat"] += float(((cm - h) ** 2).sum())
            acc["ice_se_fab"] += float(((pf - h) ** 2).sum()); acc["ice_n"] += int(rr.numel())
    return acc


def cells_for_fold(sm, so, a_m, tid, fold, device):
    """= s2_rank_raw_spearman.cells_for_fold (held-out crops of one tile), arrays returned to the CPU for saving."""
    rows = []
    for ci, ((q, r, c), (s, t)) in enumerate(sm.items()):
        if q != fold:
            continue
        rr, cc, h, pf = cells(tid, r, c, device)
        if not rr.numel():
            continue
        d = so[(q, r, c)][0]
        rows.append(dict(tile=tid, crop=ci, h=h, model=s[rr, cc], oracle=d[rr, cc], fab=pf,
                         calib=t.mean() + a_m * (s[rr, cc] - pool30(s).mean())))
    return rows


def rankdata_avg(x):
    """scipy.stats.rankdata(method='average') on the GPU (ties averaged)."""
    srt, perm = torch.sort(x)
    uniq, inv, cnt = torch.unique_consecutive(srt, return_inverse=True, return_counts=True)
    end = torch.cumsum(cnt, 0).double()
    avg = end - (cnt.double() - 1) / 2
    r = torch.empty_like(x, dtype=torch.float64)
    r[perm] = avg[inv]
    return r


def spearman(x, h):
    a, b = rankdata_avg(x), rankdata_avg(h)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def crop_rho(x, h):
    if len(h) < MIN_CELLS or float(h.max() - h.min()) == 0 or float(x.max() - x.min()) == 0:
        return None
    return spearman(x, h)


def run_fold(fold, tids, args, device):
    res = OUT / f"eval_rank_fold{fold}.json"
    if res.exists():
        log(f"fold{fold}: exists, skip"); return
    t0 = time.time()
    model, hs, run_ = fit(fold, tids, args.steps, args.batch, device)
    recs, rows = [], []
    for tid in tids:
        sm, so = scores_all(model, tid, device, args.eval_bs), oracle_scores(tid, device)
        m = calibrated_eval(sm, tid, fold, device); o = calibrated_eval(so, tid, fold, device)
        recs.append({"tile": tid, "model": m, "oracle": o})
        rows += cells_for_fold(sm, so, m["slope"], tid, fold, device)
        log(f"fold{fold}: scored {tid}")
    # per-crop raw Spearman on the GPU; store per-crop results + raw cells for the summary
    crop_recs = []
    for r in rows:
        crop_recs.append({"tile": r["tile"], "crop": r["crop"], "n": int(len(r["h"])),
                          **{f"rho_{k}": crop_rho(r[k], r["h"]) for k in ("model", "oracle", "fab")},
                          "sse": float(((r["calib"] - r["h"]) - (r["calib"] - r["h"]).mean()).pow(2).sum()),
                          "sst": float((r["h"] - r["h"].mean()).pow(2).sum())})
    np.savez_compressed(OUT / f"cells_fold{fold}.npz", **{f"{i}_{k}": r[k].cpu().numpy() for i, r in enumerate(rows)
                                                           for k in ("h", "model", "oracle", "fab", "calib")},
                        tiles=np.array([r["tile"] for r in rows]))
    (OUT / f"crops_fold{fold}.json").write_text(json.dumps(crop_recs))
    res.write_text(json.dumps({"fold": fold, "steps": args.steps, "batch": args.batch, "height_scale": hs,
                               "final_loss100": float(np.mean(run_[-100:])), "train_s": time.time() - t0,
                               "records": recs}))
    log(f"fold{fold}: done {time.time()-t0:.0f}s, {len(rows)} held-out crops with cells")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================================ summary (= s2_rank_loss_test.analyze + s2_rank_raw_spearman.summarize)
def boot(v, B=10000):
    rng = np.random.default_rng(0)
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(args):
    from scipy.stats import wilcoxon
    recs = {f: json.loads((OUT / f"eval_rank_fold{f}.json").read_text())["records"] for f in range(4)}
    tiles = sorted({r["tile"] for f in recs for r in recs[f]})
    # --- calibrated (analyze)
    agg = {w: {t: {k: 0.0 for k in ("dem_se", "dem_n", "ice_se", "ice_se_flat", "ice_se_fab", "ice_n")} for t in tiles}
           for w in ("model", "oracle")}
    slopes = {"model": [], "oracle": []}
    for f in recs:
        for r in recs[f]:
            for w in ("model", "oracle"):
                for k in agg[w][r["tile"]]:
                    agg[w][r["tile"]][k] += r[w][k]
                slopes[w].append(r[w]["slope"])

    def rm(w, se, n):
        return np.array([np.sqrt(agg[w][t][se] / agg[w][t][n]) for t in tiles])

    S = {"n_tiles": len(tiles), "checks": {}, "slopes": {
        "model_median": float(np.median(slopes["model"])), "oracle_median": float(np.median(slopes["oracle"])),
        "model_frac_negative": float(np.mean(np.array(slopes["model"]) < 0)),
        "oracle_frac_negative": float(np.mean(np.array(slopes["oracle"]) < 0))}}
    ps = {}
    for chk, se, n in (("icesat2", "ice_se", "ice_n"), ("dem_heldout", "dem_se", "dem_n")):
        M, O = rm("model", se, n), rm("oracle", se, n); d = M - O
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
    S["verdict_real_signal"] = bool(ic["model_wins"] > len(tiles) / 2 and ic["model_minus_oracle"] < 0
                                    and ic["wilcoxon_p_holm"] < 0.05)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    # --- raw Spearman (summarize)
    crops = [c for f in range(4) for c in json.loads((OUT / f"crops_fold{f}.json").read_text())]
    per_tile = {}
    for t in tiles:
        cs = [c for c in crops if c["tile"] == t]
        rec = {}
        for k in ("model", "oracle", "fab"):
            ok = [c for c in cs if c[f"rho_{k}"] is not None]
            w = sum(c["n"] for c in ok)
            rec[f"within_crop_{k}"] = (sum(c[f"rho_{k}"] * c["n"] for c in ok) / w) if w else None
        rec["n_cells_scored"] = sum(c["n"] for c in cs if c["rho_model"] is not None)
        valid = [c for c in cs if c["n"] >= MIN_CELLS and c["sst"] > 0]
        sse, sst = sum(c["sse"] for c in valid), sum(c["sst"] for c in valid)
        r2 = 1 - sse / sst if sst > 0 else None
        rec["calib_relief_r2"] = r2
        rec["calib_implied_r"] = float(np.sqrt(max(0.0, r2))) if r2 is not None else None
        per_tile[t] = rec

    def vec(k):
        return np.array([per_tile[t][k] for t in tiles if per_tile[t][k] is not None], float)

    R = {"reproduction": "not a re-fit: scores from the run itself", "min_cells_per_crop": MIN_CELLS,
         "n_tiles": len(tiles), "means": {}, "tests": {}, "per_tile": per_tile}
    for k in ("within_crop_model", "within_crop_oracle", "within_crop_fab", "calib_implied_r"):
        v = vec(k); R["means"][k] = {"mean": float(v.mean()), "ci95": boot(v), "n": len(v)}
    for k in ("model", "oracle", "fab"):
        ok = [c for c in crops if c[f"rho_{k}"] is not None]
        R["means"][f"pooled_within_crop_{k}"] = sum(c[f"rho_{k}"] * c["n"] for c in ok) / sum(c["n"] for c in ok)
    ok = [t for t in tiles if per_tile[t]["within_crop_model"] is not None and per_tile[t]["within_crop_oracle"] is not None]
    a = np.array([per_tile[t]["within_crop_model"] for t in ok]); b = np.array([per_tile[t]["within_crop_oracle"] for t in ok])
    g = np.array([per_tile[t]["calib_implied_r"] for t in ok])
    R["tests"]["model_vs_oracle"] = {"diff_mean": float((a - b).mean()), "ci95": boot(a - b), "model_higher": int((a > b).sum()),
                                     "n": len(ok), "wilcoxon_p": float(wilcoxon(a, b).pvalue)}
    R["tests"]["model_spearman_minus_calib_implied"] = {"diff_mean": float((a - g).mean()), "ci95": boot(a - g),
                                                        "spearman_higher": int((a > g).sum()), "n": len(ok),
                                                        "wilcoxon_p": float(wilcoxon(a, g).pvalue)}
    m = R["means"]["within_crop_model"]
    R["pre_registered_verdict"] = ("PASS" if m["mean"] >= 0.30 and m["ci95"][0] > 0.134 else "FAIL") + \
        (" (strong)" if m["mean"] >= 0.50 else "")
    (OUT / "raw_spearman").mkdir(exist_ok=True)
    (OUT / "raw_spearman" / "summary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "checks"}, indent=1))
    for k, v in S["checks"].items():
        print(k, json.dumps({a_: b_ for a_, b_ in v.items() if a_ != "per_tile"}))
    print(json.dumps({k: v for k, v in R.items() if k != "per_tile"}, indent=1))


def main():
    global MODEL_ID, CACHE, ORACLE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/terrain_ctrl"))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-bs", type=int, default=8, help="inference batch (does not change results)")
    ap.add_argument("--tiles", nargs="*", help="subset of tiles (local parity check only)")
    ap.add_argument("--device", default="cuda", help="cuda (Kaggle) or cpu (local parity check only)")
    a = ap.parse_args()
    MODEL_ID, CACHE, ORACLE, OUT = str(a.data / "dav2_small"), a.data / "cache", a.data / "oracle", a.out
    OUT.mkdir(parents=True, exist_ok=True)
    tids = sorted(json.loads((a.data / "tiles.json").read_text()))
    assert sorted(p.stem for p in CACHE.glob("*.npz")) == tids, "cache does not match tiles.json"
    if a.tiles:
        tids = sorted(a.tiles)
    if a.cmd == "summarize":
        summarize(a); return
    if a.device == "cuda":
        assert torch.cuda.is_available(), "No GPU: set Accelerator = GPU P100 (or T4) in the notebook settings"
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)
        log(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    else:
        device = torch.device("cpu")
    log(f"tiles {len(tids)}: {tids}")
    for fold in a.folds:
        run_fold(fold, tids, a, device)
    if device.type == "cuda":
        log(f"GPU peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if not a.tiles and all((OUT / f"eval_rank_fold{f}.json").exists() for f in range(4)):
        summarize(a)
    log("DONE")


if __name__ == "__main__":
    main()
