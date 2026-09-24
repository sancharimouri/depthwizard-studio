#!/usr/bin/env python3
"""Kaggle build of scripts/s2_token_grid_phase_c.py (Sentinel-2 token-grid test, Phase C; 2026-09-24).

Self-contained: no repo imports. TwinHeadDav2, pad_to and seed_everything are copied VERBATIM from
scripts/evaluate_method6_finetune_twinhead.py; crop_origins / load_crops / to_input / forward60 / relief_loss /
train / evaluate / analyze are copied VERBATIM from scripts/s2_token_grid_phase_c.py. The only changes are:
  - paths: one root argument --data BUNDLE (holds cache/*.npz and the DAv2-Small HF snapshot in dav2_small/);
    outputs go to --out (default /kaggle/working); nothing else is an absolute path;
  - device: CUDA asserted and pinned to cuda:0, cudnn.benchmark on (--device cpu is for the local parity check only);
  - prep is not included (the bundle ships the prepared cache).
Pre-registration (unchanged): docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, Phase C.
Pinned: transformers==5.17.0 (TwinHeadDav2 uses backbone.forward_with_filtered_kwargs).

  python s2_token_grid_phase_c_kaggle.py train --data BUNDLE --arm P
  python s2_token_grid_phase_c_kaggle.py train --data BUNDLE --arm R
  python s2_token_grid_phase_c_kaggle.py analyze --data BUNDLE
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
PATCH_SIZE = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N = 60            # crop, 10 m px (600 m)
K = 3             # 10 m -> 30 m
P_PAD = 70        # 60 -> next multiple of 14
R_UP = 540        # 9 x 60
R_PAD = 546       # 39 x 14
MODEL_ID = None   # set in main(): BUNDLE/dav2_small (local HF snapshot, no internet needed)
CACHE = None      # set in main(): BUNDLE/cache
OUT = None        # set in main(): --out
DEVICE = None     # set in main()


def get_device() -> torch.device:
    return DEVICE


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
        if device.type == "cuda":
            torch.cuda.empty_cache()


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
    global MODEL_ID, CACHE, OUT, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "analyze"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--arm", choices=["P", "R"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda", help="cuda (Kaggle) or cpu (local parity check only)")
    a = ap.parse_args()
    MODEL_ID, CACHE, OUT = str(a.data / "dav2_small"), a.data / "cache", a.out
    OUT.mkdir(parents=True, exist_ok=True)
    assert len(list(CACHE.glob("*.npz"))) == 32, f"expected 32 tiles in {CACHE}"
    if a.cmd == "train":
        if a.device == "cuda":
            assert torch.cuda.is_available(), "No GPU: set Accelerator = GPU T4 x2 (or P100) in the notebook settings"
            DEVICE = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
        else:
            DEVICE = torch.device("cpu")
        assert a.arm, "--arm required for train"
        train(a)
        if DEVICE.type == "cuda":
            print(f"GPU peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
