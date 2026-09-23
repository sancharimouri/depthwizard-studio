#!/usr/bin/env python3
"""Kaggle port of scripts/gamus_zeroshot_eval.py (Part B, 2026-09-23). Self-contained: same model code
(TwinHeadDav2GSD non-FiLM path, copied verbatim), same frozen DAv2-Large preprocessing as
backend/depth/depth_engine.py (infer_with, copied), same per-tile scoring, same jsonl record format.
Differences, by design: (1) DFC2019 block hashes are loaded from dfc2019_block_hashes.npy
(precomputed locally with the identical block_hashes()); (2) tiles are processed in REVERSE order of the
HF listing and tiles in done_tiles.txt are skipped, so this run and the local run meet in the middle;
(3) device = CUDA (a single GPU, e.g. P100): model inference AND all per-tile scoring (Pearson,
Spearman with averaged tie ranks, OLS oracle, class/bin sums) run on the GPU in float64; the CPU only
downloads, decodes h5 and hashes blocks. Output: /kaggle/working/zeroshot_tiles_kaggle.jsonl (resumable).

Kaggle notebook (GPU, Internet ON):
  !pip -q install "transformers==5.17.0" h5py
  !python /kaggle/input/<dataset>/gamus_zeroshot_kaggle.py --data /kaggle/input/<dataset>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr, spearmanr

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DAV2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
PATCH_SIZE, PAD_TO, DAV2_INPUT = 14, 518, 518
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
REPO_API = "https://huggingface.co/api/datasets/earthflow/GAMUS"
URL = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"
CLASSES = {0: "unlabelled", 1: "ground", 2: "low_veg", 3: "building", 4: "water", 5: "road", 6: "tree"}
BINS = [0, 2, 5, 10, 20, 30, 50, np.inf]
QUAD = [(0, 0), (0, 512), (512, 0), (512, 512)]
SEEDS = ["m6_s42", "m6_s43", "m6_s44"]


# ---- model (verbatim from scripts/evaluate_method6_gsd_film_height_balanced.py, enable_gsd_film=False) ----
class TwinHeadDav2GSD(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, height_scale: float = 30.0,
                 init_sigma_m: float = 5.0, log_var_max: float = 7.0, log_var_min: float = -8.0):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        base = AutoModelForDepthEstimation.from_pretrained(model_id)
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
        self.conv_mu = base.head.conv3
        hidden = self.conv_mu.in_channels
        self.conv_log_var = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.conv_log_var.weight)
        nn.init.constant_(self.conv_log_var.bias, 2.0 * math.log(max(init_sigma_m, 1e-6) / self.height_scale))
        self.film_bottleneck = None
        self.film_dec = None
        del base

    def forward(self, pixel_values):
        _, _, H, W = pixel_values.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        out = self.backbone.forward_with_filtered_kwargs(pixel_values, output_hidden_states=False, output_attentions=False)
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]
        x = self.conv1(feat)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size), mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))
        mu = self.conv_mu(x) * self.height_scale
        log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)
        return mu, torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)


def pad_to(x, size):
    h, w = x.shape[-2], x.shape[-1]
    return F.pad(x, (0, size - w, 0, size - h), mode="reflect")


# ---- frozen DAv2-Large (verbatim logic of backend/depth/depth_engine.py infer_with + normalize_depth) ----
def load_dav2(device):
    from transformers import AutoModelForDepthEstimation
    try:
        from transformers import DPTImageProcessorPil as Proc
    except ImportError:  # older transformers
        from transformers import DPTImageProcessor as Proc
    proc = Proc.from_pretrained(DAV2_LARGE)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_LARGE, use_safetensors=True).to(device).eval()
    return proc, model


def infer_dav2(image, proc, model, device):
    src_w, src_h = image.size
    inputs = proc(images=image.convert("RGB"), return_tensors="pt", do_resize=True, keep_aspect_ratio=False,
                  size={"height": DAV2_INPUT, "width": DAV2_INPUT})
    assert tuple(inputs["pixel_values"].shape[-2:]) == (DAV2_INPUT, DAV2_INPUT)
    with torch.no_grad():
        pd_ = model(pixel_values=inputs["pixel_values"].to(device)).predicted_depth
    d = F.interpolate(pd_.reshape(1, 1, *pd_.shape[-2:]), size=(src_h, src_w), mode="bicubic",
                      align_corners=False).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    if not np.isfinite(d).all():
        return np.zeros_like(d)
    lo, hi = float(d.min()), float(d.max())
    return np.zeros_like(d) if hi == lo else np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)


# ---- scoring (verbatim from scripts/gamus_zeroshot_eval.py) ----
def block_hashes(rgb_hwc):
    out = set()
    for r in range(0, rgb_hwc.shape[0] - 31, 32):
        for c in range(0, rgb_hwc.shape[1] - 31, 32):
            b = rgb_hwc[r:r + 32, c:c + 32]
            if b.std() > 2.0:
                out.add(hashlib.blake2b(np.ascontiguousarray(b).tobytes(), digest_size=12).digest())
    return out


def metrics(y, p):
    e = p - y
    vy = float(np.var(y))
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "pearson": float(pearsonr(y, p).statistic) if vy > 0 and np.var(p) > 0 else None,
            "spearman": float(spearmanr(y, p).statistic) if vy > 0 and np.var(p) > 0 else None,
            "var_ratio": float(np.var(p) / vy) if vy > 0 else None, "bias": float(np.mean(e)), "n": int(len(y))}


def sums(y, p, keys, labels):
    e = p - y
    out = {}
    for s, lab in zip(keys, labels):
        if s.any():
            out[lab] = [int(s.sum()), float(np.abs(e[s]).sum()), float((e[s] ** 2).sum()), float(e[s].sum()),
                        float(p[s].sum()), float(y[s].sum())]
    return out


# ---- GAMUS streaming (scripts/gamus_io.py) ----
TOK = os.environ.get("HF_TOKEN")


def get(rel, tries=6):
    for k in range(tries):
        fd, p = tempfile.mkstemp(suffix=".h5"); os.close(fd)
        try:
            req = urllib.request.Request(URL.format(rel), headers={"Authorization": f"Bearer {TOK}"} if TOK else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
            with h5py.File(p) as f:
                return f["image"][...]
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
        finally:
            os.remove(p)


def tile_paths(img_rel):
    stem = img_rel.split("/", 1)[1].rsplit("_", 1)[0]
    return f"heights/{stem}_AGL.h5", f"classes/{stem}_CLS.h5"


# ---- GPU scoring (float64 on the device). Numerically equivalent to the numpy/scipy path in
# scripts/gamus_zeroshot_eval.py: MAE/RMSE/bias/var-ratio are plain reductions; Pearson = cov/(sd*sd);
# Spearman = Pearson of average ranks (ties averaged, as scipy.stats.spearmanr); the 2-parameter
# lstsq fit a*x+b is solved in closed form. Parity is checked in README (CPU run vs local run).
def _pearson(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    return (yc * pc).sum() / torch.sqrt((yc * yc).sum() * (pc * pc).sum())


def _avg_rank(x):
    xs, idx = torch.sort(x)
    _, counts = torch.unique_consecutive(xs, return_counts=True)
    ends = torch.cumsum(counts, 0).to(torch.float64)
    avg = ends - (counts.to(torch.float64) - 1) / 2.0  # 1-based average rank of each tie group
    r = torch.empty_like(x)
    r[idx] = torch.repeat_interleave(avg, counts)
    return r


def metrics_t(y, p):
    e = p - y
    vy, vp = y.var(unbiased=False), p.var(unbiased=False)
    ok = bool(vy > 0) and bool(vp > 0)
    vals = torch.stack([e.abs().mean(), torch.sqrt((e * e).mean()),
                        _pearson(y, p) if ok else torch.zeros((), dtype=y.dtype, device=y.device),
                        _pearson(_avg_rank(y), _avg_rank(p)) if ok else torch.zeros((), dtype=y.dtype, device=y.device),
                        vp / vy if bool(vy > 0) else torch.zeros((), dtype=y.dtype, device=y.device),
                        e.mean()]).cpu().tolist()
    return {"mae": vals[0], "rmse": vals[1], "pearson": vals[2] if ok else None, "spearman": vals[3] if ok else None,
            "var_ratio": vals[4] if bool(vy > 0) else None, "bias": vals[5], "n": int(y.numel())}


def sums_t(y, p, keys, labels):
    e = p - y
    K = torch.stack(keys).to(torch.float64)  # (k, n)
    S = torch.stack([K.sum(1), (K * e.abs()).sum(1), (K * e * e).sum(1), (K * e).sum(1), (K * p).sum(1), (K * y).sum(1)], 1).cpu().tolist()
    return {lab: [int(s[0])] + s[1:] for lab, s in zip(labels, S) if s[0] > 0}


def ols_predict(x_tr, y_tr, x_te):
    xm, ym = x_tr.mean(), y_tr.mean()
    a = ((x_tr - xm) * (y_tr - ym)).sum() / ((x_tr - xm) ** 2).sum()
    return a * x_te + (ym - a * xm)


def score_tile(rel, rgb, agl, cls, models, proc, dav2, dev, dfc_hash):
    agl = agl.astype(np.float32)
    valid_np = np.isfinite(agl) & (agl >= 0)
    rec = {"tile": rel, "city": rel.split("/")[-1].split("_")[0], "valid_frac": float(valid_np.mean()),
           "neg_frac": float((np.isfinite(agl) & (agl < 0)).mean()),
           "shared_dfc_blocks": len(block_hashes(rgb) & dfc_hash), "runner": "kaggle"}
    if valid_np.sum() < 400:
        rec["skipped"] = "too few valid px"
        return rec
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    batch = torch.stack([pad_to(x[:, r:r + 512, c:c + 512], PAD_TO) for r, c in QUAD]).float().to(dev, non_blocking=True)
    full = {}
    with torch.no_grad():
        for tag, ms in models.items():
            mu = torch.stack([m(batch)[0][:, 0, :512, :512] for m in ms]).mean(0)  # stays on GPU
            P = torch.zeros((1024, 1024), dtype=torch.float32, device=dev)
            for i, (r, c) in enumerate(QUAD):
                P[r:r + 512, c:c + 512] = mu[i]
            full[tag] = P
    d = torch.from_numpy(infer_dav2(Image.fromarray(rgb, "RGB"), proc, dav2, dev)).to(dev)
    aglt = torch.from_numpy(agl).to(dev)
    valid = torch.from_numpy(valid_np).to(dev)
    clst = torch.from_numpy(cls.astype(np.int64)).to(dev)
    rec["quads"] = []
    pooled = {}
    for i, (r0, c0) in enumerate(QUAD):
        qm = torch.zeros((1024, 1024), dtype=torch.bool, device=dev)
        qm[r0:r0 + 512, c0:c0 + 512] = True
        te, tr = valid & qm, valid & ~qm
        if int(te.sum()) < 100 or int(tr.sum()) < 100:
            continue
        y = aglt[te].double()
        preds = {tag: P[te].double() for tag, P in full.items()}
        preds["oracle"] = ols_predict(d[tr].double(), aglt[tr].double(), d[te].double())
        for tag, P in full.items():
            preds[tag + "_ols"] = ols_predict(P[tr].double(), aglt[tr].double(), P[te].double())
        qrec = {"q": i, "m": {k: metrics_t(y, p) for k, p in preds.items()}}
        dm = metrics_t(y, d[te].double())
        qrec["m"]["dav2_raw"] = {"pearson": dm["pearson"], "spearman": dm["spearman"], "n": dm["n"]}
        c = clst[te]
        ckeys = [c == k for k in CLASSES]
        bkeys = [(y >= lo) & (y < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]
        blabs = [f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])]
        tk = [(c == 6) & b for b in bkeys]
        base = [k for k in preds if not k.endswith("_ols")]
        qrec["cls"] = {k: sums_t(y, preds[k], ckeys, list(CLASSES.values())) for k in base}
        qrec["bin"] = {k: sums_t(y, preds[k], bkeys, blabs) for k in base}
        qrec["tree_bin"] = {k: sums_t(y, preds[k], tk, blabs) for k in base}
        rec["quads"].append(qrec)
        for k, p in preds.items():
            v = torch.stack([torch.tensor(float(y.numel()), dtype=torch.float64, device=dev), p.sum(), (p * p).sum(),
                             y.sum(), (y * y).sum()]).cpu().numpy()
            pooled[k] = pooled.get(k, np.zeros(5)) + v
    rec["pooled_moments"] = {k: v.tolist() for k, v in pooled.items()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with ckpt/, dfc2019_block_hashes.npy, done_tiles.txt")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/zeroshot_tiles_kaggle.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="cuda (Kaggle) | cpu (local parity check)")
    ap.add_argument("--only", default=None, help="score just this tile (parity check)")
    args = ap.parse_args()
    if args.device == "cuda":
        assert torch.cuda.is_available(), "No CUDA GPU visible: set the notebook accelerator to GPU P100"
        dev = torch.device("cuda:0")  # a single GPU (P100); everything below runs on it
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        print("device", dev, torch.cuda.get_device_name(0), flush=True)
    else:
        dev = torch.device(args.device)
        print("device", dev, "(parity/debug mode)", flush=True)
    torch.set_num_threads(2)  # CPU only does downloads, h5 decode, block hashing
    models = {}
    for tag in SEEDS:
        ms = []
        for f in range(4):
            ck = torch.load(args.data / "ckpt" / tag / f"fold{f}.pt", map_location="cpu", weights_only=False)
            m = TwinHeadDav2GSD(height_scale=ck["height_scale"])
            m.load_state_dict(ck["state_dict"]); ms.append(m.to(dev).eval())
        models[tag] = ms
    proc, dav2 = load_dav2(dev)
    dfc_hash = {bytes(x) for x in np.load(args.data / "dfc2019_block_hashes.npy")}
    print(f"{len(dfc_hash)} DFC2019 blocks", flush=True)

    files = sorted(s["rfilename"] for s in json.loads(urllib.request.urlopen(REPO_API).read())["siblings"])
    imgs = [f for f in files if f.startswith("images/test/")]
    skip = set((args.data / "done_tiles.txt").read_text().split())
    if args.out.exists():
        skip |= {json.loads(l)["tile"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [args.only] if args.only else [f for f in reversed(imgs) if f not in skip][: args.limit]
    print(f"test tiles {len(imgs)}, skipped {len(skip)}, todo {len(todo)} (reverse order)", flush=True)

    q = queue.Queue(maxsize=32)

    def fetch(rel):
        h, c = tile_paths(rel)
        return rel, get(rel), get(h), get(c)

    def producer():
        with ThreadPoolExecutor(16) as ex:
            for item in ex.map(fetch, todo):
                q.put(item)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, n = time.time(), 0
    fout = args.out.open("a")
    while (item := q.get()) is not None:
        rel, rgb, agl, cls = item
        rec = score_tile(rel, rgb, agl, cls, models, proc, dav2, dev, dfc_hash)
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
            print(f"{n}/{len(todo)} tiles, {el/n:.2f} s/tile, ETA {(len(todo)-n)*el/n/3600:.2f} h, "
                  f"GPU peak mem {mem:.1f} GiB, queue {q.qsize()}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
