> **Repo layout note (2026-09-24):** this manual moved here from the deleted `kaggle/` / `kaggle_phase2.5_package/` folders. The script now lives in `scripts/`. To re-stage the upload bundle, see `data/REGENERATION.md`, "Repo folder consolidation". Paths below that start with `/kaggle/` are Kaggle runtime paths and are unchanged.

# DepthWizard2 — Method 4 v2, SID ordinal-constraint follow-up (Kaggle package)

**One experiment this round**, on top of the current best config
(`phase2_building_rank_v2` — see
`docs/method-audit/04-learned-scale-modulation/v2-results.md` §"Post-verdict
follow-up" in the main repo for the full method writeup and citation). This
does not reopen general tuning — it tests the one specific "different loss
design" direction that was identified as the only lever left after the
bounded Phase 2.5 Round 2 closed out weight-sweep tuning on this
architecture.

**`phase2_building_rank_v2_sid`** — adds an ordinal height-discretization
constraint (Spacing-Increasing Discretization, adapted from weakIM2H —
Chen, Shi, Zhu 2025, arXiv:2506.02534 eq. 7) into the *existing*
pairwise rank-loss mechanism, on top of `phase2_building_rank_v2`
unchanged otherwise (building channel, rank-weight 0.5, no ground-plane).
Not a full reimplementation of that paper — see the main repo's
`v2-results.md` for exactly what was and wasn't ported, and how the
h_min-near-zero log(0) issue (which the paper itself doesn't address) was
handled.

Uses the same fixed training script as prior rounds — the data-loading fix
(`--num-workers`, `pin_memory`, non-blocking transfers, per-epoch timing)
and the new `--sid-bins`/`--sid-eps` flags are both already in
`scripts/evaluate_method4_v2_kaggle.py` in this package.

Trains from scratch (zero-init heads, same as every prior phase) — no
checkpoint included or needed.

## Contents

```
data/
  RGB/<tile>_RGB.tif        (50 files — the DAv2 50-tile DFC2019 benchmark)
  Truth/<tile>_AGL.tif      (50 files)
  depth/<tile>_depth.npy    (50 files — cached DAv2 relative-depth output)
  building/<tile>.npy       (50 files — Method 3's HOTOSM building-probability maps)
  manifest.csv              (the 50-tile benchmark manifest)
scripts/
  evaluate_method4.py           (unmodified — reused for Tile/quadrant_bounds/etc.)
  evaluate_method4_v2_kaggle.py (training/eval script, now with --sid-bins/--sid-eps)
tile_list.txt                (the 50 tile IDs, for reference)
README.md                    (this file)
```

## 1. What to upload as a Kaggle dataset

Upload the zip file (`kaggle_phase2.5_package.zip`) as a **new Kaggle
Dataset** (Create → New Dataset → upload the zip; Kaggle extracts it
automatically). Name it whatever you like — e.g. `depthwizard2-sid`. Note
the exact slug Kaggle assigns; it determines the mount path below.

Then create a **new Notebook**, attach that dataset, and turn on a **GPU
accelerator** (a single T4 is enough — this round is one config, one GPU,
no need for the 2-GPU split the last round used).

## 2. Notebook cells, in order

**Cell 1 — setup.** Adjust `DATASET_SLUG` (and `KAGGLE_USERNAME` if
needed). Checks both mount-path conventions Kaggle has used
(`/kaggle/input/<slug>/...` and `/kaggle/input/datasets/<username>/<slug>/...`)
rather than hardcoding one.

```python
import subprocess, sys, os

try:
    import rasterio  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)

DATASET_SLUG = "depthwizard2-sid"          # <-- change if Kaggle assigned a different slug
KAGGLE_USERNAME = "sancharimouri"           # <-- only used by the fallback path below

candidate_roots = [
    f"/kaggle/input/{DATASET_SLUG}",
    f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}",
]
ROOT = next((r for r in candidate_roots if os.path.isdir(r)), None)
if ROOT is None:
    print("None of the expected mount paths exist. Contents of /kaggle/input:")
    for p in os.listdir("/kaggle/input"):
        print(" ", p)
    raise FileNotFoundError(f"Checked: {candidate_roots}")

DATA_ROOT = f"{ROOT}/data"
SCRIPT_DIR = f"{ROOT}/scripts"
SCRIPT = f"{SCRIPT_DIR}/evaluate_method4_v2_kaggle.py"

assert os.path.isdir(DATA_ROOT), f"Data root not found: {DATA_ROOT}"
assert os.path.isfile(SCRIPT), f"Script not found: {SCRIPT} -- available: {os.listdir(SCRIPT_DIR)}"
print("Mount root:", ROOT)
print("Data root OK:", DATA_ROOT)
print("Script OK:", SCRIPT)

import torch
n_gpus = torch.cuda.device_count()
print(f"CUDA available: {torch.cuda.is_available()} - {n_gpus} GPU(s)")
for i in range(n_gpus):
    print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")
if n_gpus == 0:
    print("WARNING: no GPU detected. This will be extremely slow on CPU — "
          "check Settings -> Accelerator before continuing.")
```

**Cell 2 — run `phase2_building_rank_v2_sid`.**

```python
import subprocess, sys, time

device = "cuda:0" if n_gpus >= 1 else "cpu"

cmd = [
    sys.executable, "-u", SCRIPT,
    "--data-root", DATA_ROOT,
    "--outdir", "/kaggle/working/phase2_building_rank_v2_sid",
    "--tag", "phase2_building_rank_v2_sid",
    "--patch-mode", "dense",
    "--epochs", "60",
    "--smoothness-weight", "0.01",
    "--extra-channel", "building",
    "--rank-weight", "0.5", "--rank-pairs-per-patch", "2000", "--rank-margin", "0.25",
    "--sid-bins", "10", "--sid-eps", "0.1",
    "--folds", "0", "1", "2", "3",
    "--device", device,
    "--num-workers", "4",
]
print("Running:", " ".join(cmd))
t0 = time.time()
result = subprocess.run(cmd)
print(f"Exited with code {result.returncode} after {time.time()-t0:.0f}s")
```

**Cell 3 — summarize the result.**

```python
import json

path = "/kaggle/working/phase2_building_rank_v2_sid/phase2_building_rank_v2_sid_results.json"
with open(path) as f:
    data = json.load(f)
print("Overall:", data["overall"])
print("Calibration:", data["overall_calibration"])
print(f"Total time: {data['total_time_sec']:.1f}s")

print("\nReference points (from Phase 2.5 Round 2, "
      "docs/method-audit/04-learned-scale-modulation/v2-results.md):")
print("  Corrected baseline:                  MAE 3.3924  RMSE 4.5787  Pearson 0.5824  Spearman 0.5093")
print("  phase2_building_rank_v2 (current best): MAE 2.8803  RMSE 4.7751  Pearson 0.5835  Spearman 0.5438")
```

## 3. Where the output lands

```
/kaggle/working/phase2_building_rank_v2_sid/
  phase2_building_rank_v2_sid_fold{0,1,2,3}.pt
  phase2_building_rank_v2_sid_results.json     <- RESULT FILE
```

Same top-level shape as every prior phase (`overall`, `overall_calibration`,
`folds` with per-tile Pearson/Spearman) — drops straight into
`v2-results.md`'s existing table format.

**To download afterward**: Kaggle's notebook "Output" tab (right sidebar,
after the run finishes) lists every file under `/kaggle/working/`. Only
`phase2_building_rank_v2_sid_results.json` is actually needed back — the
`.pt` checkpoints can be skipped.

Bring the downloaded JSON back; the doc will be updated with whether this
closes any more of the remaining 4.3% RMSE gap. Either outcome is useful —
if RMSE/Pearson don't move, that's a real negative result for the
"different loss design" direction, not a failed attempt.

## Rough cost estimate

Same cost profile as `phase2_building_rank_v2` was last round (one
rank-loss config, ~60-70 min on a single T4/P100) — the SID bin lookup adds
negligible per-step overhead (a few scalar log/floor ops per sampled pair,
on top of the existing pair-sampling loop that already dominates rank-loss
cost).
