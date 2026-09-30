"""Method 6 checkpoints on the Hugging Face Hub (private model repo).

The 5 checkpoint files code actually loads live in a PRIVATE model repo under
the project's HF account; scripts/hf_upload_checkpoints.py puts them there.
They are private because Method 6 was trained on DFC2019, whose contest terms
forbid distributing the data (docs/STORAGE.md).

  checkpoint("height_balanced_seed43/fold0.pt") -> local Path

uses the local file when this machine has it, else hf_hub_download (cached in
the HF cache) with HF_TOKEN from the environment or .env. Same pattern as the
other hub downloads in scripts/ (e.g. prepare_method3_semantic_inputs.py).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = os.environ.get("DW2_CKPT_REPO", "sancharimouri/depthwizard2-method6")
REPO_TYPE = "model"

# repo file -> local path (the files scripts/vhr_dsm_pipeline.py loads)
FILES = {
    **{f"height_balanced_seed43/fold{q}.pt":
       ROOT / f"data/dfc2019/experiments/method6_height_balanced_seed43/fold{q}.pt" for q in range(4)},
    "full_dfc2019/method6_full_dfc2019.pt":
        ROOT / "data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt",
}


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    env = ROOT / ".env"
    if os.environ.get("DW2_NO_DOTENV") != "1" and env.exists():  # never a .env in the container or the desktop app
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    return None


def checkpoint(repo_file: str, prefer_local: bool = True) -> Path:
    local = FILES.get(repo_file)
    if prefer_local and local is not None and local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type=REPO_TYPE, token=token()))
