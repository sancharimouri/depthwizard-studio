# Terrain-relief positive control on Kaggle: full manual

**What this runs.** It is the main run of `scripts/terrain_relief_positive_control.py run`: the rank-loss /
raw-Spearman pipeline on VHR NAIP (1.0 m) against USGS 3DEP LiDAR terrain, in 600 m crops.
- Pre-registration: `docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md`, "Terrain-relief positive
  control".
- The work is 4 quadrant folds. Each fold is 600 training steps of DAv2-Small (full fine-tune, batch 4, 602 × 602 px
  crops), then scoring of every tile.

**Where it runs:** everything numeric runs on **one GPU**, `cuda:0`:
- training;
- inference;
- the calibration sums;
- the calibrated errors;
- per-crop Spearman (ties averaged, float64).

The CPU only reads the `.npz` files and runs the final Wilcoxon and bootstrap summary, which covers a few dozen numbers.

**Parity, checked before shipping:** the Kaggle script was run on the CPU and compared with the original imported pipeline, on the
Tahoe tile, fold 0, 5 steps.
- The loss trace is identical.
- All 14 calibrated-eval fields match structurally, with a maximum relative difference of 2.1e-7.
- All 36/36 held-out crops have the same Spearman None-pattern, with a maximum |Δρ| of 3.8e-5 (float32 vs float64 near-ties).
- Resuming from a checkpoint reproduced the fold result exactly.
- Expect CUDA-vs-MPS training noise on the GPU. That's the same class of difference as between local reruns, not a bug.

**What's in the bundle** (the folder `bundle/` inside the zip):
```
bundle/
  terrain_relief_kaggle.py     the script (self-contained, no repo imports)
  tiles.json                   the 8 tiles, in order
  cache/<tile>.npz             NAIP RGB 7200x7200 uint8, 30 m DTM target, truth cells (row, col, height, ref)
  oracle/<tile>_depth_half.npy frozen DAv2-Large depth, 3600x3600 float16 (repeated 2x on the GPU)
  dav2_small/                  DAv2-Small HF snapshot (config.json, model.safetensors), so no HF download is needed
  read.md                      this file
```

---

## 1. Upload the bundle as a Kaggle Dataset

1. kaggle.com → **Datasets** → **+ New Dataset**.
2. Drag in **`terrain_relief_kaggle_bundle.zip`**. Kaggle extracts it by itself; don't unzip it first.
3. **Dataset title:** `terrain-relief-ctrl-bundle`, which gives the slug `terrain-relief-ctrl-bundle`. Keep it exactly.
4. Visibility: **Private** → **Create**. Wait until processing finishes; the bundle is about 1.5 GB.

The files land **read-only** somewhere under `/kaggle/input/`. The notebook finds them by searching for the script, so
it doesn't matter how deep Kaggle nests the folder.

## 2. Create the notebook

1. On the dataset page click **New Notebook**, or go to **Code → + New Notebook → + Add Input** and search for
   `terrain-relief-ctrl-bundle`.
2. **Notebook title:** `terrain-relief-control` (File → rename, top-left).
3. **Settings** (right-hand panel):
   - **Accelerator:** `GPU P100`. `GPU T4 x2` also works, but the script uses only one GPU, `cuda:0`, so P100 is the
     better choice.
   - **Internet:** **On**. It's needed only for Cell 2's `pip install`; the model weights are in the bundle.
   - **Persistence:** `Files only`.

## 3. Notebook cells (paste each into its own cell, in order)

**Cell 1: confirm the GPU**
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
It must list a GPU (`Tesla P100-PCIE-16GB` or `Tesla T4`). If it errors, the accelerator isn't set to a GPU.

**Cell 2: dependencies.** transformers is pinned to the version used locally.
```python
!pip -q install "transformers==5.17.0"
```

**Cell 3: locate the bundle and set paths**
```python
import glob, os
hits = glob.glob("/kaggle/input/**/terrain_relief_kaggle.py", recursive=True)
assert len(hits) == 1, hits          # 0 = dataset not attached, 2 = attached twice
BUNDLE = os.path.dirname(hits[0])
OUT = "/kaggle/working/terrain_ctrl"
LOG = "/kaggle/working/terrain_ctrl_run_log.txt"
os.makedirs(OUT, exist_ok=True)
print("BUNDLE =", BUNDLE); print(os.listdir(BUNDLE))
```

**Cell 4: resume from a previous version.** Skip this cell on the first run.
- If an earlier version of this notebook was interrupted, first attach its output: **+ Add Input → Your Work →
  terrain-relief-control**.
- Then run this cell. It copies the previous checkpoints and results back, so the run continues where it stopped: at
  the last 100-step checkpoint, with finished folds skipped.
```python
import shutil
prev = [p for p in glob.glob("/kaggle/input/**/terrain_ctrl", recursive=True) if os.path.isdir(p)]
assert len(prev) <= 1, prev
if prev:
    for f in os.listdir(prev[0]):
        src, dst = os.path.join(prev[0], f), os.path.join(OUT, f)
        if f.endswith(".tmp"):
            continue                  # a half-written checkpoint: drop it
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)
    print("resumed from", prev[0], sorted(os.listdir(OUT)))
else:
    print("no previous output found: fresh start")
```

**Cell 5: run, with automatic restart if it stalls or crashes.** This is the long cell.
- The script prints a line at least every 25 training steps and once per scored tile.
- If the log gets no new line for `STALL_MIN` minutes, or the process dies with an error, this cell kills it and
  starts it again.
- The restart resumes from the last checkpoint, so at most 100 steps are repeated.
- It gives up after `MAX_RESTARTS` restarts.
```python
import subprocess, time, sys
STALL_MIN, MAX_RESTARTS = 15, 5
cmd = [sys.executable, "-u", f"{BUNDLE}/terrain_relief_kaggle.py", "run", "--data", BUNDLE, "--out", OUT]
for attempt in range(MAX_RESTARTS + 1):
    print(f"=== launch {attempt} ===", flush=True)
    with open(LOG, "a") as lf:
        lf.write(f"=== launch {attempt} {time.ctime()} ===\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        os.set_blocking(p.stdout.fileno(), False)
        last = time.time(); done = False
        while True:
            line = p.stdout.readline()
            if line:
                print(line, end="", flush=True); lf.write(line); lf.flush(); last = time.time()
                done = done or line.strip().endswith("DONE")
                continue
            if p.poll() is not None:
                break
            if time.time() - last > STALL_MIN * 60:
                print(f"!!! no output for {STALL_MIN} min: killing and restarting", flush=True)
                lf.write("!!! stall: killed\n"); p.kill(); p.wait(); break
            time.sleep(2)
    if done and p.returncode == 0:
        print("=== finished ==="); break
    print(f"=== exited with code {p.returncode}; restarting in 30 s ===", flush=True); time.sleep(30)
else:
    raise RuntimeError("gave up after MAX_RESTARTS restarts; see LOG")
```

**What you should see:**
```
[..] GPU: Tesla P100-PCIE-16GB, torch 2.x
[..] tiles 8: [...]
[..] [rank_fold0] device=cuda:0 train crops ... steps 600 x batch 4
[..] [rank_fold0] step 25/600 loss 0.2xxx 0.xxs/step ETA x.x min GPU peak y.y GiB
...
[..] fold0: scored <tile>          (one per tile)
[..] fold0: done ...s, ... held-out crops with cells
...                                (folds 1, 2, 3)
{ ... summary ... }
[..] DONE
```
- **The GPU check:** `s/step` should be well under 1 s on a P100, and `GPU peak` should be several GiB.
- If `s/step` is several seconds and `GPU peak` is about 0, it isn't running on the GPU. Stop and check Cell 1.

**Cell 6: check the outputs**
```python
print(sorted(os.listdir(OUT)))
print(open(f"{OUT}/raw_spearman/summary.json").read()[:1500])
```
It must list `eval_rank_fold0..3.json`, `crops_fold0..3.json`, `cells_fold0..3.npz`, `summary.json` and
`raw_spearman/summary.json`.

**Run it as a committed version, not interactively.** An interactive session stops when the browser tab idles.
- **Save Version** (top right) → **Save & Run All (Commit)** → Save.
- It runs in the background. Kaggle allows up to 12 h per session; this run should need well under 2 h.
- Follow it under **View Active Events** or the version's **Logs**.
- If the version is killed anyway, start a new version with Cell 4 (it resumes).

## 4. Put the output back into the repo

1. Open the notebook → the finished version → **Output** tab.
2. Download the whole `terrain_ctrl` folder and `terrain_ctrl_run_log.txt`. The `ckpt_fold*.pt` files (about 100 MB
   each) are optional, needed only to resume, so you can skip them.
3. Put them **exactly** here:
   ```
   DepthWizard2/data/terrain_relief_control/kaggle_out/terrain_ctrl/...          (the folder's contents)
   DepthWizard2/data/terrain_relief_control/kaggle_out/terrain_ctrl_run_log.txt
   ```
   - **Don't** put them in `data/terrain_relief_control/rank_loss/`; that folder belongs to a local run.
   - Don't rename any file.
4. Tell Claude: "Kaggle output is in `data/terrain_relief_control/kaggle_out/`." Claude will then:
   - check that all 4 folds and all tiles are present;
   - apply the pre-registered verdict;
   - write the result section, HANDOFF.md and CLAUDE.md.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `No GPU: set Accelerator ...` | Settings → Accelerator → GPU P100, then restart the session. |
| `assert len(hits) == 1` fails | 0 hits: attach the dataset (**+ Add Input** → `terrain-relief-ctrl-bundle`). 2 hits: it's attached twice, so remove one. |
| `ImportError` / `forward_with_filtered_kwargs` | Cell 2 didn't run. Re-run it, then **Restart session** and run all cells again. |
| `pip` fails with a network error | Internet is off. Settings → Internet → On (the toggle needs a phone-verified account). |
| `CUDA out of memory` during scoring | Add `"--eval-bs", "4"` to `cmd` in Cell 5. It doesn't change results. |
| `CUDA out of memory` during training | Restart the session so nothing else holds GPU memory. Batch 4 at 602 px needs well under 16 GB. |
| Cell 5 prints `stall: killed` repeatedly | Check the last lines of `LOG`. A repeated error means a real bug: stop and send the log to Claude. |
| The version hit the 12 h limit or was killed | New version: attach the old output, run Cell 4, then Cell 5. It resumes. |
| `cache does not match tiles.json` | The upload is incomplete. Re-upload the zip as a new dataset version. |
