"""Central configuration for DepthWizard2."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model/cache
# ---------------------------------------------------------------------------

HF_HOME_DIR = REPO_ROOT / "models"

MODEL_IDS = {
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}

ACTIVE_MODEL_SIZE = "large"
ACTIVE_MODEL_ID = MODEL_IDS[ACTIVE_MODEL_SIZE]

MODEL_INPUT_SIZE = 518

# Prevent excessively large image inputs.
MAX_INPUT_PIXELS = 50_000_000

# ---------------------------------------------------------------------------
# Data/output paths
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device() -> str:
    """Select CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"
