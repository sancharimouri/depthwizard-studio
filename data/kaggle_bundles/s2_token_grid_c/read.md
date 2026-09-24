# Phase C on Kaggle — Sentinel-2 token-grid test (P vs R), GPU run

**What this runs.** Phase C of `docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md`: both arms,
all 4 folds, 8 trainings. It runs on one Kaggle GPU. The script is `s2_token_grid_phase_c_kaggle.py`, the
self-contained build of `scripts/s2_token_grid_phase_c.py`. Its model and Phase C code are copied verbatim.

- **Arm P:** 60 × 60 px Sentinel-2 crops padded to 70 px, giving a 5 × 5 token grid.
- **Arm R:** the same crops ×9-upsampled to 540 and padded to 546, giving a 39 × 39 token grid.
- **Same everything else:** tiles, folds, target (FABDEM at 30 m), loss, steps (600 × batch 4), seed and crop order.

Run it on a Kaggle GPU because it is about 5× faster than the Mac's MPS, and the Mac is busy with Phase B.

**Parity (checked locally before shipping, CPU, 2026-09-24).**
- **Arm P**, 3 training steps plus the full fold-0 evaluation on 32 tiles: the two scripts' JSON is structurally
  identical and all 288 numeric fields match. Max relative difference **0.0**; final loss 40.80353991190592 in both.
- **Arm R**, forward pass plus loss on one batch with identical weights: max absolute difference **0.0**.

**Why Kaggle runs *all* of Phase C.** Both arms must come from the same device, otherwise the device becomes a second
manipulated variable. **Never mix** Kaggle `eval_*.json` files with local ones: they go to a separate folder, see
Downloading below.

**Bundle contents** (the folder `bundle_c/` inside the zip, 265 MB, stored uncompressed):

| File | What |
|---|---|
| `s2_token_grid_phase_c_kaggle.py` | the script (`train` / `analyze`) |
| `cache/*.npz` | 32 tiles: Sentinel-2 RGB (uint8), FABDEM on the 30 m and 10 m grids, ICESat-2 ground cells (EGM2008) |
| `dav2_small/` | DAv2-Small HF snapshot (`config.json`, `model.safetensors`, sha1 `b991d2040b22…`). Loaded offline. |
| `read.md` | this file |

---

## Setup (once)

1. kaggle.com → **Datasets** → **+ New Dataset** → drag in **`s2_token_grid_c_bundle.zip`**.
   - Title: `s2-token-grid-c`.
   - Kaggle extracts the zip automatically; don't unzip it yourself.
2. **+ New Notebook** titled `s2-token-grid-phase-c`. In the right-hand panel:
   - **Add Input** → your dataset `s2-token-grid-c`.
   - **Settings → Accelerator → GPU T4 x2.** One GPU is used, pinned to `cuda:0`. **GPU P100** also works if your
     image's PyTorch still supports it; see Troubleshooting.
   - **Settings → Internet → On.** Only Cell 1's `pip install` needs it. The model loads from the bundle.
   - **Persistence:** "Files only" (optional; this helps resume).

## Cells (paste in order)

**Cell 1: pin transformers, check the GPU.**
```python
!pip install -q "transformers==5.17.0"
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| CUDA", torch.cuda.is_available())
assert transformers.__version__ == "5.17.0", "restart the kernel (Run > Restart) after the pip install, then re-run Cell 1"
assert torch.cuda.is_available(), "No GPU: Settings > Accelerator > GPU T4 x2"
print(torch.cuda.get_device_name(0))
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2: find the bundle. Don't hard-code the path; Kaggle's mount nesting varies.**
```python
import glob, os
hits = glob.glob("/kaggle/input/**/s2_token_grid_phase_c_kaggle.py", recursive=True)
assert len(hits) == 1, hits          # 0 = dataset not attached, 2 = attached twice
BUNDLE = os.path.dirname(hits[0])
SCRIPT = hits[0]
OUT = "/kaggle/working"
LOG = "/kaggle/working/phase_c_kaggle_log.txt"
assert len(glob.glob(f"{BUNDLE}/cache/*.npz")) == 32 and os.path.exists(f"{BUNDLE}/dav2_small/model.safetensors")
print(BUNDLE)
```

**Cell 3: resume. Copy finished folds from a previous version's output, if you attached one.** This is safe to run
every time.
```python
import shutil
prev = glob.glob("/kaggle/input/**/eval_[PR]_fold[0-3].json", recursive=True)
for p in prev:
    dst = os.path.join(OUT, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst)
print("resumed folds:", sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/eval_*_fold*.json")))
```
The script skips any fold whose `eval_<arm>_fold<k>.json` already exists in `/kaggle/working`. Each fold's file is
written only when that fold is finished, so an interrupted fold just re-runs from scratch. That costs about 5 min at most.

**Cell 4: arm P (≈ 1–2 min per fold on a T4).**
```python
!python -u {SCRIPT} train --data {BUNDLE} --out {OUT} --arm P 2>&1 | grep -v -i "warn" | tee -a {LOG}
```

**Cell 5: arm R (≈ 5–6 min per fold on a T4).**
```python
!python -u {SCRIPT} train --data {BUNDLE} --out {OUT} --arm R 2>&1 | grep -v -i "warn" | tee -a {LOG}
```

**Cell 6: the pre-registered analysis.** It writes `summary.json` and fails loudly if any of the 8 folds is missing.
```python
!python -u {SCRIPT} analyze --data {BUNDLE} --out {OUT} 2>&1 | tee -a {LOG}
```

**Run it:** **Save Version → Save & Run All (Commit)**. This runs in the background, so you can close the tab.
Expected total: about 30–40 min on a T4.

## Expected log lines

```
GPU: Tesla T4, torch 2.x
[P_fold0] device=cuda:0 train crops 6144 (32 tiles) relief std 34.47 m, steps 600 x batch 4
[P_fold0] step 100/600 loss ... 0.1s/step ETA ... min
[P_fold0] done ...s
...
GPU peak memory ... GiB
[R_fold0] device=cuda:0 train crops 6144 ...
...
{ "n_tiles": 32, ..., "verdict_R_helps": true|false }
icesat2 {...}
dem_heldout {...}
```

Checks:
- `device=cuda:0` must appear. If it says `cpu`, the GPU isn't attached.
- `relief std 34.47 m` is fold 0's value, the same as locally. Other folds differ.

## Downloading the results into the repo

Open the notebook, go to the committed version and open the **Output** tab (`/kaggle/working`). Download:
- `eval_P_fold0.json` … `eval_P_fold3.json`
- `eval_R_fold0.json` … `eval_R_fold3.json`
- `summary.json`
- `phase_c_kaggle_log.txt`

Put them into **`DepthWizard2/data/sentinel2_benchmark/token_grid_test/kaggle/`**, a new folder:
- Do **not** put them in `token_grid_test/` itself.
- Do not rename or overwrite any local file there.

Then tell Claude: "Phase C Kaggle output is in `token_grid_test/kaggle/`." Claude will then:
1. check that all 8 fold files are present;
2. re-run `analyze` on them locally;
3. confirm it reproduces the Kaggle `summary.json`;
4. write up the result.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `assert len(hits) == 1` fails with `[]` | The dataset isn't attached. Use **Add Input** → `s2-token-grid-c`. |
| `assert len(hits) == 1` fails with 2 paths | The dataset is attached twice; remove one. |
| `No GPU` assertion | Settings → Accelerator → **GPU T4 x2**, then restart the session. |
| `CUDA error: no kernel image is available` / "sm_60 not supported" | Your image's PyTorch dropped P100. Switch the accelerator to **GPU T4 x2**. |
| Cell 1 says transformers isn't 5.17.0 | **Run → Restart & clear outputs**, then run Cell 1 again (pip needs a kernel restart). |
| `pip install` fails | Settings → **Internet → On**. |
| `AttributeError: ... forward_with_filtered_kwargs` | The wrong transformers version is loaded; see the Cell 1 row above. |
| `CUDA out of memory` (not expected: DAv2-Small at 546² × batch 4 needs about 3–4 GB) | Another notebook is holding the GPU. Restart the session. Don't change `--batch`: it is pre-registered. |
| Session died mid-run | Make a new version, attach the previous version's **Output** as an input, and run all. Cell 3 resumes finished folds. |
| `analyze` says `missing .../eval_R_fold3.json` | A fold didn't finish. Re-run Cells 2–6; finished folds are skipped. |
