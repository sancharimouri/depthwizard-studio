#!/usr/bin/env python3
"""Upload the 5 Method 6 checkpoints to a PRIVATE Hugging Face model repo.

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

Full DAv2-Small fine-tune with a twin (mean, log-variance) head and the height-balanced
recipe (SIH26175 Depth Wizard 2, research track). DFC2019 research best only: it does
**not** generalize (GAMUS, 3DEP, Sentinel-2), and the product itself is DEM-only.

Files (what `scripts/vhr_dsm_pipeline.py` loads):
- `height_balanced_seed43/fold{0..3}.pt` — the 4 spatial-quadrant fold models (ensembled)
- `full_dfc2019/method6_full_dfc2019.pt` — trained on all 50 tiles (cross-check)

**Not for redistribution.** Trained on the 2019 IEEE GRSS Data Fusion Contest data
(DigitalGlobe / IARPA / JHU APL), whose terms forbid dissemination of the data. Base model:
Depth-Anything-V2-Small (Apache-2.0).
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
