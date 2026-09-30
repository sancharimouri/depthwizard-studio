"""The --train-on-all run resolves to EXACTLY the adopted Method 6 recipe (docs/method6-checkpoint-audit.md §1);
the only difference from a fold run is the data split."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hb", ROOT / "scripts/evaluate_method6_gsd_film_height_balanced.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)

ADOPTED = {"epochs": 12, "batch": 2, "lr_backbone": 5e-6, "lr_head": 2.5e-4, "weight_decay": 0.01,
           "init_sigma": 5.0, "log_var_max": 7.0, "log_var_min": -8.0, "warmup_steps": 30, "seed": 42,
           "enable_height_balanced": True, "enable_gsd_film": False, "extra_train_npz": None,
           "gamus_test_npz": None, "tiles_limit": None, "max_minutes": 0.0}


def resolved(extra):
    return vars(hb.resolve_args(hb.build_parser().parse_args(
        ["--outdir", "x", "--tag", "t", "--enable-height-balanced", "--epochs", "12", "--seed", "42", *extra])))


def test_train_on_all_resolves_to_the_adopted_recipe():
    cfg = resolved(["--train-on-all"])
    for k, v in ADOPTED.items():
        assert cfg[k] == pytest.approx(v) if isinstance(v, float) else cfg[k] == v, (k, cfg[k], v)
    assert cfg["train_on_all"] is True


def test_only_the_data_split_differs_from_a_fold_run():
    fold = resolved(["--save-checkpoints"])  # the command that produced the seed-42 fold checkpoints
    allq = resolved(["--train-on-all"])
    differing = {k for k in fold if fold[k] != allq[k]}
    # train_on_all selects the split; save_checkpoints is set by train-on-all at run time (it always saves)
    assert differing == {"train_on_all", "save_checkpoints"}, differing


def test_fixed_recipe_constants():
    assert hb.MODEL_ID == "depth-anything/Depth-Anything-V2-Small-hf"
    assert hb.BASE_REVISION.startswith("5426e4f")
    assert hb.PAD_TO == 518 and hb.PATCH_SIZE == 14
    assert hb.IMAGENET_MEAN.flatten().tolist() == pytest.approx([0.485, 0.456, 0.406])
    assert hb.IMAGENET_STD.flatten().tolist() == pytest.approx([0.229, 0.224, 0.225])
    assert (hb.TALL_THRESH_M, hb.CANOPY_LO_M, hb.CANOPY_HI_M) == (10.0, 4.0, 25.0)
    assert (hb.HEIGHT_LOSS_ALPHA, hb.HEIGHT_LOSS_MAX_WEIGHT, hb.LAMBDA_HEIGHT_WEIGHT) == (0.08, 4.0, 0.35)
    # the sampler weight: 1 + 3*frac(>=10 m) + 2*frac(4..25 m)
    agl = np.array([[0.0, 5.0], [12.0, 30.0]], np.float32)
    assert hb.quadrant_sample_weight(agl, np.ones_like(agl, bool)) == pytest.approx(1 + 3 * 0.5 + 2 * 0.5)
