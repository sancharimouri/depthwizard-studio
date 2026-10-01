#!/usr/bin/env python3
"""Upload the Method 6 checkpoints (seed-43 folds, the production full model, the archived old full model) to a PRIVATE HF model repo.

Private on purpose: Method 6 was trained on DFC2019, whose contest terms forbid
distributing the data (docs/STORAGE.md), so the weights are not published.
Re-runnable; verifies each file's sha256 against the Hub's LFS record.

  uv run python scripts/hf_upload_checkpoints.py            # HF_TOKEN from env or .env
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.storage import hf_checkpoints as hc  # noqa: E402

CARD = """---
license: other
tags: [depth-estimation, remote-sensing, research]
---

# Depth Wizard 2 — Method 6 checkpoints (private, research only)

Method 6 = full fine-tune of Depth-Anything-V2-Small with a twin (mean, log-variance) head and the
height-balanced recipe (SIH26175 Depth Wizard 2, research track).

**Validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS cities by RMSE.**
The product itself is DEM-only; this model is research.

## Production model: `full_dfc2019/method6_full_dfc2019_hb_seed42.pt`

- **Recipe (the adopted one):**
  - DAv2-Small (`depth-anything/Depth-Anything-V2-Small-hf`, revision `5426e4f0f36572d16453bbda7a8389317b1bef99`);
  - a twin head: `mu` = AGL in metres (× `height_scale` 16.427 = the training p95), `log_var`;
  - 30-step masked-Huber warm-up, then Gaussian NLL + 0.35 × capped height-weighted Huber;
  - height-balanced `WeightedRandomSampler` (1 + 3·frac ≥ 10 m + 2·frac 4–25 m);
  - AdamW 5e-6 / 2.5e-4, wd 0.01, clip 1.0, linear warm-up + decay, batch 2;
  - **12 fixed epochs, final weights** (no validation selection);
  - input: 512 px quadrants, ImageNet normalisation, reflect-padded to 518; no augmentation.
- **Training data:** all 4 quadrants of all 50 DFC2019 Track-1 tiles (no holdout); **seed 42**.
- **Training commit:** `821792e` (`scripts/evaluate_method6_gsd_film_height_balanced.py --enable-height-balanced
  --epochs 12 --seed 42 --train-on-all`), 1,070 s on Apple MPS.
- **SHA-256:** `69a29e10b965b282891d12468f7a994c2374031cbbe42130a226250a8b7d7d63`.
- **Expected accuracy (DFC2019, 3-seed cross-validation of the same recipe, 4-fold spatial-quadrant holdout):**
  **MAE 1.990 m, RMSE 3.504 m, Pearson 0.743, Spearman 0.656.** This model has seen every DFC2019 tile, so its own
  DFC2019 numbers are not an estimate.
- **GAMUS comparison** (pre-registered, unseen data, GAMUS LiDAR nDSM, 2,848 test tiles, DC / NYC / PHL):
  - this model scores **MAE 3.156, RMSE 4.692, Pearson 0.624**;
  - the mean of the 4 seed-42 fold models scores 3.175 / 4.660 / 0.626;
  - **PASS**: within 5% on MAE and RMSE, and Pearson −0.02, pooled and in 3 of 3 cities.
  - Both lose RMSE to a per-tile-OLS DAv2-Large oracle (4.426): out of domain, aerial imagery.

## Other files

- `height_balanced_seed43/fold{0..3}.pt`: the 4 spatial-quadrant fold models of seed 43 (the ensemble used by
  `scripts/vhr_dsm_pipeline.py` for the Maxar AGL).
- `archive/method6_full_dfc2019_pre_height_balanced.pt`: the **superseded** full model, from the pre-adoption recipe
  (no height-balanced loss or sampler, warm-up 40). Archived 2026-10-01; kept for reproducibility, not for use.

**Not for redistribution.** Trained on the 2019 IEEE GRSS Data Fusion Contest data (DigitalGlobe / IARPA / JHU APL),
whose terms forbid dissemination of the data. Base model: Depth-Anything-V2-Small (Apache-2.0).
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=hc.token())
    api.create_repo(hc.REPO_ID, repo_type=hc.REPO_TYPE, private=True, exist_ok=True)
    info = api.repo_info(hc.REPO_ID, repo_type=hc.REPO_TYPE)
    assert info.private, f"{hc.REPO_ID} is not private; refusing to upload"
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=hc.REPO_ID, repo_type=hc.REPO_TYPE)
    total = 0
    for repo_file, local in hc.FILES.items():
        total += local.stat().st_size
        api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_file, repo_id=hc.REPO_ID,
                        repo_type=hc.REPO_TYPE, commit_message=f"add {repo_file}")
        print(f"up  {repo_file}  {local.stat().st_size:,d}", flush=True)
    remote = {f.path: f for f in api.list_repo_tree(hc.REPO_ID, repo_type=hc.REPO_TYPE, recursive=True)
              if getattr(f, "lfs", None)}
    for repo_file, local in hc.FILES.items():
        assert remote[repo_file].lfs.sha256 == sha256(local), f"sha256 mismatch: {repo_file}"
    print(f"verified {len(hc.FILES)} files ({total:,d} bytes) in private repo {hc.REPO_ID}")


if __name__ == "__main__":
    main()
