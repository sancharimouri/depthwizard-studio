"""Method 6 checkpoint audit (docs/method6-checkpoint-audit.md): for each of the 17 local checkpoints
  - SHA-256, and the metadata saved IN the checkpoint (height_scale, seed, fold; the full model's config);
  - a strict load into the adopted architecture (TwinHeadDav2GSD(enable_gsd_film=False), which is TwinHeadDav2);
  - a forward pass on 3 DFC2019 tiles, on the quadrant the checkpoint did NOT train on (fold models); the full
    model saw every quadrant, so its 3-tile numbers are in-sample (sanity only);
  - for the full model: agreement with each fold model on that fold's held-out quadrants of all 50 tiles
    (Pearson and MAE between the two predictions). Never the full model's DFC2019 accuracy.
And the 5 checkpoints in the private HF repo: their LFS SHA-256 from the repo metadata (read-only, no download).

Read-only on data/. Output: build/method6_audit/audit.json (gitignored). The DAv2-Small base config/weights needed
to build the architecture are fetched into build/hf (HF_HOME), never into the repo.

  .venv/bin/python scripts/method6_checkpoint_audit.py
  .venv/bin/python scripts/method6_checkpoint_audit.py --checkpoint <path.pt> --agree-with hb_seed42
      one checkpoint (a retrain): strict load, 3-tile sanity, recipe match (its results JSON + train log), and the
      pre-registered agreement rule vs the given seed's 4 fold models on their held-out quadrants
      (docs/method6-checkpoint-audit.md §7.2: median Pearson >= 0.95 AND mean abs difference <= 0.9 m)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/method6_audit"
os.environ.setdefault("HF_HOME", str(ROOT / "build/hf"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_method4 as m4  # noqa: E402
from evaluate_method6_finetune_twinhead import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, MODEL_ID, PAD_TO, load_tile_rgb_agl, pad_to, tile_ids,
)
from evaluate_method6_gsd_film_height_balanced import TwinHeadDav2GSD  # noqa: E402

EXP = ROOT / "data/dfc2019/experiments"
SETS = {
    "hb_seed42": EXP / "method6_height_balanced_seed42_ckpt",
    "hb_seed43": EXP / "method6_height_balanced_seed43",
    "hb_seed44": EXP / "method6_height_balanced_seed44",
    "hb_gamusdc_seed43": EXP / "method6_hb_gamusdc_seed43",
}
FULL = EXP / "method6_full_checkpoint/method6_full_dfc2019.pt"
HF_REPO = "sancharimouri/depthwizard2-method6"
HF_MAP = {"full_dfc2019/method6_full_dfc2019.pt": FULL,
          **{f"height_balanced_seed43/fold{q}.pt": SETS["hb_seed43"] / f"fold{q}.pt" for q in range(4)}}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(height_scale: float) -> torch.nn.Module:
    return TwinHeadDav2GSD(height_scale=height_scale, init_sigma_m=5.0, log_var_max=7.0, log_var_min=-8.0,
                           enable_gsd_film=False)


def load(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict") or ck.get("model")
    meta = {k: v for k, v in ck.items() if k not in ("state_dict", "model")}
    model = build(float(ck["height_scale"]))
    res = model.load_state_dict(sd, strict=True)  # raises on any missing / unexpected key
    n_params = sum(v.numel() for v in sd.values())
    return model.to(device).eval(), meta, {"strict_load": "ok", "missing": list(res.missing_keys),
                                           "unexpected": list(res.unexpected_keys), "n_tensors": len(sd),
                                           "n_params": int(n_params),
                                           "has_gsd_film": any("film" in k.lower() for k in sd)}


@torch.no_grad()
def predict(model, rgb_c: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(rgb_c / 255.0).float()
    x = ((x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]).float()
    mu, _ = model(pad_to(x, PAD_TO)[None].to(device))
    return mu[0, 0, : rgb_c.shape[1], : rgb_c.shape[2]].float().cpu().numpy()


def quad(arrs, q):
    rgb, agl, valid = arrs
    h, w = agl.shape
    r0, r1, c0, c1 = m4.quadrant_bounds(h, w, q)
    return rgb[:, r0:r1, c0:c1], agl[r0:r1, c0:c1], valid[r0:r1, c0:c1]


def sanity(model, tiles_data, quads, device) -> dict:
    ys, ps = [], []
    for arrs, q in zip(tiles_data, quads):
        rgb_c, agl_c, valid_c = quad(arrs, q)
        p = predict(model, rgb_c, device)
        ys.append(agl_c[valid_c].astype(np.float64))
        ps.append(p[valid_c].astype(np.float64))
    Y, P = np.concatenate(ys), np.concatenate(ps)
    ok = np.isfinite(P).all()
    return {"finite": bool(ok), "pred_min": float(P.min()), "pred_p50": float(np.median(P)),
            "pred_p99": float(np.percentile(P, 99)), "pred_max": float(P.max()),
            "gt_p99": float(np.percentile(Y, 99)), "variance_ratio": float(np.var(P) / np.var(Y)),
            "pearson": float(np.corrcoef(Y, P)[0, 1]), "mae_m": float(np.mean(np.abs(P - Y))),
            "n_pixels": int(len(Y))}


AGREE_MIN_MEDIAN_PEARSON, AGREE_MAX_MEAN_ABS_M = 0.95, 0.9  # pre-registered 2026-10-01 (§7.2)
SANE_RANGE_M, SANE_VR = (-5.0, 60.0), (0.4, 1.2)


def audit_one(ckpt: Path, agree_with: str) -> dict:
    """The §7.2 pre-registered audit of one retrained checkpoint."""
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tids = tile_ids()
    cache = {t: load_tile_rgb_agl(t) for t in tids}
    sample_tiles = [tids[0], tids[len(tids) // 2], tids[-1]]
    model, meta, arch = load(ckpt, device)
    s = sanity(model, [cache[t] for t in sample_tiles], [0] * 3, device)  # in-sample for a full model: sanity only
    sane = s["finite"] and SANE_RANGE_M[0] <= s["pred_min"] and s["pred_max"] <= SANE_RANGE_M[1] \
        and SANE_VR[0] <= s["variance_ratio"] <= SANE_VR[1]
    # recipe: the run's own results JSON (config) + train log
    res_json = next(ckpt.parent.glob("*_results.json"), None)
    cfgs = [json.loads(p.read_text()).get("config", {}) for p in ckpt.parent.glob("*_results.json")]
    cfg = next((c for c in cfgs if c.get("checkpoint") == ckpt.name), {})
    log = next((p for p in ckpt.parent.glob("*train.log")), None)
    log_txt = log.read_text() if log else ""
    recipe = {"seed": (cfg.get("seed"), 42), "epochs": (cfg.get("epochs"), 12), "batch": (cfg.get("batch"), 2),
              "lr_backbone": (cfg.get("lr_backbone"), 5e-6), "lr_head": (cfg.get("lr_head"), 2.5e-4),
              "weight_decay": (cfg.get("weight_decay"), 0.01), "warmup_steps": (cfg.get("warmup_steps"), 30),
              "enable_height_balanced": (cfg.get("enable_height_balanced"), True),
              "enable_gsd_film": (cfg.get("enable_gsd_film"), False), "extra_train_npz": (cfg.get("extra_train_npz"), None),
              "train_on_all": (cfg.get("train_on_all"), True),
              "log: base revision 5426e4f": ("train-on-all: base revision 5426e4f" in log_txt, True),
              "log: 200 train quadrants": ("200 train quadrants, 0 eval quadrants" in log_txt, True),
              "log: height-balanced sampler": ("height-balanced sampler" in log_txt, True),
              "log: reached epoch 12/12": ("epoch 12/12" in log_txt, True),
              "log: no non-finite abort": ("non-finite" not in log_txt, True),
              "saved seed": (meta.get("seed"), 42), "saved fold": (meta.get("fold"), None)}
    mismatches = {k: v for k, v in recipe.items() if not (v[0] == v[1] or (isinstance(v[1], float) and v[0] is not None
                                                                          and abs(v[0] - v[1]) < 1e-12))}
    agree = {}
    for q in range(4):
        fm, _, _ = load(SETS[agree_with] / f"fold{q}.pt", device)
        a, b = [], []
        for t in tids:
            rgb_c, _agl, v = quad(cache[t], q)
            a.append(predict(fm, rgb_c, device)[v]); b.append(predict(model, rgb_c, device)[v])
        A, Bv = np.concatenate(a).astype(np.float64), np.concatenate(b).astype(np.float64)
        agree[f"fold{q}"] = {"pearson": float(np.corrcoef(A, Bv)[0, 1]), "mean_abs_diff_m": float(np.mean(np.abs(A - Bv))),
                             "mean_full_minus_fold_m": float(np.mean(Bv - A)), "n_pixels": int(len(A))}
        print("agreement", q, {k: round(v, 4) for k, v in agree[f"fold{q}"].items() if k != "n_pixels"}, flush=True)
        del fm
    med_r = float(np.median([v["pearson"] for v in agree.values()]))
    mean_mad = float(np.mean([v["mean_abs_diff_m"] for v in agree.values()]))
    rule_pass = med_r >= AGREE_MIN_MEDIAN_PEARSON and mean_mad <= AGREE_MAX_MEAN_ABS_M
    out = {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": sha256(ckpt), "size_bytes": ckpt.stat().st_size,
           "saved_metadata": meta, "architecture": arch, "sanity_3_tiles_in_sample": s, "sane": bool(sane),
           "recipe_check": {k: {"got": v[0], "want": v[1]} for k, v in recipe.items()},
           "recipe_mismatches": list(mismatches), "results_json": str(res_json.relative_to(ROOT)) if res_json else None,
           "agreement_vs": agree_with, "agreement": agree,
           "rule": {"median_pearson": med_r, "mean_abs_diff_m": mean_mad,
                    "thresholds": [AGREE_MIN_MEDIAN_PEARSON, AGREE_MAX_MEAN_ABS_M], "pass": bool(rule_pass)},
           "verdict": "PASS" if (rule_pass and sane and not mismatches and arch["strict_load"] == "ok") else "FAIL"}
    (OUT / f"retrain_{ckpt.stem}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--agree-with", default="hb_seed42", choices=list(SETS))
    a = ap.parse_args()
    if a.checkpoint:
        out = audit_one(a.checkpoint.resolve(), a.agree_with)
        print(json.dumps({k: out[k] for k in ("sha256", "sane", "recipe_mismatches", "rule", "verdict")}, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tids = tile_ids()
    assert len(tids) == 50, len(tids)
    sample_tiles = [tids[0], tids[len(tids) // 2], tids[-1]]  # one early JAX, one mid, one late (OMA) tile
    cache = {t: load_tile_rgb_agl(t) for t in tids}
    res = {"device": str(device), "sample_tiles": sample_tiles, "checkpoints": {}, "started": time.ctime()}

    # base weights: the recipe loads MODEL_ID unpinned; record the repo's revision + last modification
    from huggingface_hub import HfApi
    mi = HfApi().model_info(MODEL_ID)
    res["base_model"] = {"id": MODEL_ID, "revision_sha": mi.sha, "last_modified": str(mi.last_modified)}

    entries = [(f"{k}/fold{q}", SETS[k] / f"fold{q}.pt", q) for k in SETS for q in range(4)]
    entries.append(("full_dfc2019", FULL, None))
    full_model = None
    for name, path, q in entries:
        t0 = time.time()
        model, meta, arch = load(path, device)
        # fold models: the quadrant they never trained on; full model: in-sample, so use quadrant 0 (labelled)
        held = q if q is not None else 0
        s = sanity(model, [cache[t] for t in sample_tiles], [held] * 3, device)
        s["held_out"] = q is not None
        res["checkpoints"][name] = {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
                                    "sha256": sha256(path), "saved_metadata": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                                               for k, v in meta.items()},
                                    "architecture": arch, "sanity_3_tiles": s, "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps({"hs": meta.get("height_scale"), "seed": meta.get("seed"), "fold": meta.get("fold"),
                                "vr": round(s["variance_ratio"], 3), "r": round(s["pearson"], 3),
                                "range": [round(s["pred_min"], 1), round(s["pred_max"], 1)]}), flush=True)
        if q is None:
            full_model = model
        else:
            del model

    # full model vs every adopted-recipe fold model, on that fold's held-out quadrants of all 50 tiles
    agree = {}
    for k in ("hb_seed42", "hb_seed43", "hb_seed44"):
        for q in range(4):
            fm, _, _ = load(SETS[k] / f"fold{q}.pt", device)
            pf, pF = [], []
            for t in tids:
                rgb_c, agl_c, valid_c = quad(cache[t], q)
                a, b = predict(fm, rgb_c, device), predict(full_model, rgb_c, device)
                pf.append(a[valid_c]); pF.append(b[valid_c])
            A, Bv = np.concatenate(pf).astype(np.float64), np.concatenate(pF).astype(np.float64)
            agree[f"{k}/fold{q}"] = {"pearson": float(np.corrcoef(A, Bv)[0, 1]), "mae_between_m": float(np.mean(np.abs(A - Bv))),
                                     "mean_full_minus_fold_m": float(np.mean(Bv - A)), "n_pixels": int(len(A))}
            print("agreement", k, q, {kk: round(v, 3) for kk, v in agree[f'{k}/fold{q}'].items() if kk != "n_pixels"}, flush=True)
            del fm
    res["full_vs_fold_agreement"] = agree

    # the private HF copies: LFS SHA-256 from the repo metadata (read-only, nothing downloaded)
    from backend.storage.hf_checkpoints import token
    info = HfApi(token=token()).model_info(HF_REPO, files_metadata=True)
    hf = {}
    for s in info.siblings:
        if s.rfilename in HF_MAP:
            local = res["checkpoints"][next(n for n, e in res["checkpoints"].items() if e["path"] == str(HF_MAP[s.rfilename].relative_to(ROOT)))]
            hf[s.rfilename] = {"hf_sha256": s.lfs.sha256 if s.lfs else None, "hf_size": s.size,
                               "local": local["path"], "local_sha256": local["sha256"],
                               "match": bool(s.lfs and s.lfs.sha256 == local["sha256"])}
    res["hf_repo"] = {"repo": HF_REPO, "revision": info.sha, "files": hf}
    res["finished"] = time.ctime()
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, default=str))
    print("HF:", {k: v["match"] for k, v in hf.items()})
    print(f"-> {OUT / 'audit.json'}")


if __name__ == "__main__":
    main()
