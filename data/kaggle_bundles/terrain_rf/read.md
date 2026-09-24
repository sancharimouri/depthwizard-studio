> **Repo layout note (2026-09-24):** this manual moved here from the deleted `kaggle/` / `kaggle_phase2.5_package/` folders. The script now lives in `scripts/`. To re-stage the upload bundle, see `data/REGENERATION.md`, "Repo folder consolidation". Paths below that start with `/kaggle/` are Kaggle runtime paths and are unchanged.

# Part F on Kaggle: full manual (Sentinel-2 terrain RF residual)

**What this runs.** This is only the random-forest stage of the pre-registered Part F (`scripts/terrain_rf_residual.py`, pre-registration commit `3a4cf84`).
- **Method:** Song, Chen & Yokoya (2026), *ISPRS J.* 232:155–171, arXiv 2505.06905: a handcrafted-feature RF on the residual FABDEM − ICESat-2 ground height.
- **Features:** already extracted locally into `samples.parquet` (1,241,260 samples = per 10 m pixel × ICESat-2 track, 32 tiles, 63 tracks).
- **Split:** held-out ICESat-2 tracks, `GroupKFold(5)` on RGT.
- **Model:** `RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)`, otherwise defaults. **One run, no tuning.**
- **Variants:** A has 18 features; B has 32 (A + WorldCover fractions + entropy + ETH + CHMv2).
- **Parity, checked before shipping:** the sample filter and fold assignment are byte-identical to the local code
  (fold sizes 248,332 / 248,049 / 248,314 / 248,268 / 248,297).
- **What stays local:** the linear baseline, per-tile metrics and the pre-registered Wilcoxon/Holm decision. They are cheap and run after you bring the outputs back.
- **Hardware:** sklearn's RF runs on the **CPU only**, so no GPU is needed.
  - Use accelerator **None**: 4 vCPU and about 30 GB RAM, and no GPU quota used.
  - The RAM is the point: 100 full-depth trees on ~1 M rows need several GB, which made the local Mac swap.

Bundle contents (the folder `bundle_f/` inside the zip): `terrain_rf_kaggle.py`, `samples.parquet` (212 MB), `read.md`.

## 0. Choose the engine: sklearn (CPU, pre-registered) or xgb (GPU, faster, a documented deviation)

| | `--engine sklearn` (**default**) | `--engine xgb` (opt-in) |
|---|---|---|
| Model | The exact pre-registered sklearn `RandomForestRegressor` | XGBoost random-forest mode: 100 parallel trees, 1 round, `device="cuda"` |
| Hardware | **CPU**. sklearn has no GPU implementation, so a GPU would sit idle | **GPU**, P100 or T4 |
| Time (estimate) | About 1.5–2.5 h with two notebooks in parallel | Minutes per variant |
| Fidelity | Exactly as pre-registered | **Approximation.** Histogram splits (256 bins), row sampling without replacement (0.632), depth cap 20. Results are logged as a deviation from the pre-registration |
| Output files | `rf_oof_*`, `rf_run_meta.json` | `xgbrf_oof_*`, `xgbrf_run_meta.json` (so the two can never be mixed at merge) |

If you have time, prefer **sklearn**, because it is the pre-registered model. Use **xgb** if you need it fast; the write-up
will then say the RF was the XGBoost approximation. Either way, `folds.npy` is identical, and the merge checks that.

**GPU (xgb) quick recipe.** One notebook, `gamus-f-rf-gpu`: **Accelerator GPU P100** (or T4), **Internet Off**.
XGBoost with CUDA is preinstalled on Kaggle images, so skip the pip cell.
```python
import glob, os
hits = glob.glob("/kaggle/input/**/terrain_rf_kaggle.py", recursive=True); assert len(hits) == 1, hits
BUNDLE = os.path.dirname(hits[0]); OUT = "/kaggle/working"; LOG = "/kaggle/working/xgbrf_run_log.txt"
!nvidia-smi --query-gpu=name,memory.total --format=csv
# 1) smoke test: 20,000 rows, fold 0, a few seconds; must end with DONE
!python -u {BUNDLE}/terrain_rf_kaggle.py --data {BUNDLE} --variant AB --engine xgb --smoke --out {OUT}
# 2) the real run: both variants, all 5 folds
!python -u {BUNDLE}/terrain_rf_kaggle.py --data {BUNDLE} --variant AB --engine xgb --out {OUT} 2>&1 | tee -a {LOG}
```
If step 1 fails on the P100 with a CUDA or architecture error, switch the accelerator to **T4** and rerun.
Download these into `DepthWizard2/data/sentinel2_benchmark/terrain_rf_residual/kaggle/`:
`xgbrf_oof_A_fold0..4.npy`, `xgbrf_oof_B_fold0..4.npy`, `folds.npy`, `xgbrf_run_meta.json`, `xgbrf_run_log.txt`.
Ignore the `smoke_*` files. Then tell Claude: "Part F GPU (xgb) output is in `terrain_rf_residual/kaggle/`." Claude runs
`scripts/terrain_rf_residual.py --from-kaggle=data/sentinel2_benchmark/terrain_rf_residual/kaggle --kaggle-engine=xgb`.

The sections below describe the **sklearn (CPU)** path.

---

## 1. Upload the dataset

1. Go to kaggle.com → **Datasets** → **+ New Dataset** → drag in **`gamus_f_kaggle_bundle.zip`**. Kaggle auto-extracts it.
2. **Title:** `gamus-f-rf` (slug `gamus-f-rf`). Visibility: **Private**. Click **Create** and wait for processing.
3. The files appear read-only somewhere under `/kaggle/input/…/bundle_f/`. The notebook finds them by glob, so the exact nesting doesn't matter.

## 2. Notebooks: two, run at the same time, to halve the wall time

Create **two** notebooks, both with the `gamus-f-rf` dataset attached (**+ Add Input**):

| notebook title | variant | expected time (estimate) |
|---|---|---|
| `gamus-f-rf-A` | `A` (18 features) | about 1–1.5 h |
| `gamus-f-rf-B` | `B` (32 features) | about 1.5–2.5 h |

Settings for both: **Accelerator: None**, **Internet: Off** (not needed), **Persistence: Files only**.
To use only one notebook, pass `--variant AB`; it runs both in sequence (about 3–4 h).

## 3. Cells (identical in both notebooks except `VARIANT`)

**Cell 1: variant and paths**
```python
VARIANT = "A"          # "B" in the gamus-f-rf-B notebook
import glob, os
hits = glob.glob("/kaggle/input/**/terrain_rf_kaggle.py", recursive=True)
assert len(hits) == 1, hits
BUNDLE = os.path.dirname(hits[0])
OUT = "/kaggle/working"
LOG = f"/kaggle/working/rf_run_log_{VARIANT}.txt"
print(BUNDLE, os.listdir(BUNDLE), "| CPUs:", os.cpu_count())
```

**Cell 2: dependencies.** scikit-learn is pinned to the local version.
```python
!pip -q install "scikit-learn==1.9.0" pyarrow
```
If pip can't install 1.9.0 on Kaggle's Python, skip this cell. The script records the version it used in
`rf_run_meta.json`; a different sklearn version is a documented deviation, not a blocker.

**Cell 3: resume.** Only needed after an interruption.
If an earlier version produced some `rf_oof_*_fold*.npy`, add that version's output as an input (**+ Add Input → Your Work**)
and copy those files back. Finished folds are then skipped.
```python
import shutil
for f in glob.glob("/kaggle/input/**/rf_oof_*_fold*.npy", recursive=True):
    shutil.copy(f, OUT); print("resume:", os.path.basename(f))
```

**Cell 4: run**
```python
!python -u {BUNDLE}/terrain_rf_kaggle.py --data {BUNDLE} --variant {VARIANT} --out {OUT} 2>&1 | tee -a {LOG}
```

**Expected output.** First, a meta line: `{'n_samples': 1241260, 'n_tiles': 32, 'n_rgt': 63, ...}`. Then one line per fold:
```
variant A fold 0: n_train 992928, n_test 248332, NNN s
...
DONE
```
- Run it as **Save Version → Save & Run All (Commit)**, so it survives the browser idling.
- Each fold's predictions are saved as soon as that fold finishes, so an interruption loses at most one fold.

## 4. Bring the outputs back into the repo

From each notebook version's **Output** tab, download:

| from | files |
|---|---|
| `gamus-f-rf-A` | `rf_oof_A_fold0.npy` … `rf_oof_A_fold4.npy`, `folds.npy`, `rf_run_meta.json`, `rf_run_log_A.txt` |
| `gamus-f-rf-B` | `rf_oof_B_fold0.npy` … `rf_oof_B_fold4.npy`, `rf_run_log_B.txt` (its `folds.npy` and meta are identical; either copy is fine) |

Put **all** of them, names unchanged, in this one folder (create it):
```
DepthWizard2/data/sentinel2_benchmark/terrain_rf_residual/kaggle/
```
It should end up with 10 `rf_oof_*.npy`, 1 `folds.npy`, 1 `rf_run_meta.json` and 2 logs. Then tell Claude:
"Part F Kaggle output is in `terrain_rf_residual/kaggle/`." Claude will then:
1. Run `scripts/terrain_rf_residual.py --from-kaggle=data/sentinel2_benchmark/terrain_rf_residual/kaggle`. It asserts
   that `folds.npy` matches the local fold assignment exactly, loads the RF predictions, fits the linear baseline and
   applies the pre-registered rule.
2. Write up the result and finish the session's final documentation step.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `assert len(hits) == 1` fails | The dataset isn't attached: **+ Add Input** → `gamus-f-rf`. |
| Kernel dies or runs out of memory | Accelerator must be **None**, the 30 GB CPU machine. Don't run A and B in the *same* notebook with other jobs. Re-run with Cell 3 to resume. |
| Much slower than estimated | Check `cpu_count` in the meta line; the CPU machine should show 4. |
| `Kaggle fold assignment differs` (at merge, locally) | A different `samples.parquet` was used. Re-upload this bundle. |
