# Part B on Kaggle: full manual

**What this runs:** Method 6 zero-shot on the GAMUS test split (2,861 tiles). It is the same computation as
`scripts/gamus_zeroshot_eval.py`, which is running locally.
- **Models:** the 12 Method 6 fold models (seeds 42, 43, 44 × 4 folds) plus frozen DAv2-Large for the oracle.
- **GAMUS input:** tiles are streamed from Hugging Face (`earthflow/GAMUS`). No GAMUS data is inside this bundle.
- **What runs where:** everything numeric runs on **one GPU (P100)**:
  - model inference;
  - Pearson and Spearman (tied ranks averaged, as in scipy);
  - the OLS oracle;
  - the class and height-bin sums (float64).
  The CPU only downloads, decodes files and hashes blocks.
- **Parity, checked before shipping.** The scoring code was run on the CPU for tile `DC_05_28` and compared with the local run:
  - all 1,513 fields have the same structure;
  - the largest relative difference is 1.2e-4, on one per-class error sum; headline metrics differ by less than 1e-5.
  - The difference is float32 inference noise between MPS and CPU. Expect the same order between the P100 and local runs.
- **Division of work:**
  - Kaggle processes tiles in **reverse** order and skips the tiles listed in `done_tiles.txt` (102 were done locally when the zip was built).
  - The local run keeps going **forward**, so the two meet in the middle.
  - Duplicates are dropped at merge; the local copy wins.

Bundle contents (the folder `bundle/` inside the zip):

```
bundle/
  gamus_zeroshot_kaggle.py        the script
  ckpt/m6_s42/fold0..3.pt         Method 6 fold checkpoints (12 files, 1.1 GB)
  ckpt/m6_s43/fold0..3.pt
  ckpt/m6_s44/fold0..3.pt
  dfc2019_block_hashes.npy        leakage check vs. DFC2019 (51,125 block hashes)
  done_tiles.txt                  tiles already scored locally (skipped on Kaggle)
  read.md                         this file
```

---

## 1. Upload the bundle as a Kaggle Dataset

1. Go to kaggle.com → **Datasets** → **+ New Dataset**.
2. Drag in **`gamus_b_kaggle_bundle.zip`**. Kaggle extracts zips automatically; don't unzip it yourself.
3. **Dataset title:** `gamus-b-bundle`. That makes the URL slug `gamus-b-bundle`. Keep it exactly, because the paths below assume it.
4. Visibility: **Private**. Click **Create** and wait until processing finishes. 1.2 GB takes a few minutes.

After processing, the files live **read-only** under:
```
/kaggle/input/gamus-b-bundle/bundle/
```
- Kaggle sometimes drops or doubles the top folder. The notebook below finds the script automatically, so the exact nesting doesn't matter.
- `/kaggle/input` is read-only. Everything the run writes goes to `/kaggle/working`, which becomes the notebook's **Output**.

## 2. Create the notebook

1. Open the dataset page → **New Notebook**, or go to **Code → + New Notebook** and then **+ Add Input** and search `gamus-b-bundle`.
2. **Title the notebook:** `gamus-b-zeroshot` (File → rename, top-left).
3. Right-hand **Settings** panel (Session options):
   - **Accelerator:** `GPU P100`. Not T4 ×2; the script uses exactly one GPU, `cuda:0`.
   - **Internet:** **On**. It is required, because tiles stream from Hugging Face. If the toggle is greyed out, verify your phone number in Kaggle account settings.
   - **Persistence:** `Files only` is fine.
4. **Optional, recommended:** a Hugging Face token raises download rate limits.
   - Add-ons → **Secrets** → **Add secret**, with label `HF_TOKEN` and your HF read token as the value. Tick it for this notebook.

## 3. Notebook cells (paste each into its own cell, in order)

**Cell 1: confirm the GPU**
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
It must show `Tesla P100-PCIE-16GB`. If it errors, the accelerator is not set to GPU.

**Cell 2: dependencies.** The transformers version is pinned to match the local run.
```python
!pip -q install "transformers==5.17.0" h5py
```

**Cell 3: locate the bundle and set paths**
```python
import glob, os
hits = glob.glob("/kaggle/input/**/gamus_zeroshot_kaggle.py", recursive=True)
assert len(hits) == 1, hits
BUNDLE = os.path.dirname(hits[0])
OUT = "/kaggle/working/zeroshot_tiles_kaggle.jsonl"
LOG = "/kaggle/working/zeroshot_kaggle_run_log.txt"
print("BUNDLE =", BUNDLE); print(os.listdir(BUNDLE))
```

**Cell 4: HF token.** Skip this if you added no secret.
```python
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

**Cell 5: resume support.** Run it only when restarting after an interrupted run; otherwise skip it.
If a previous version of this notebook produced a partial `zeroshot_tiles_kaggle.jsonl`, add that version's output as an
input (**+ Add Input → Your Work → gamus-b-zeroshot**), then copy the file back so the script resumes where it stopped:
```python
prev = glob.glob("/kaggle/input/**/zeroshot_tiles_kaggle.jsonl", recursive=True)
if prev:
    import shutil; shutil.copy(prev[0], OUT); print("resuming from", prev[0])
```

**Cell 5b: drop a partial last line.** Always run this after Cell 5. If the previous session died mid-write, its
last line may be cut off, and resuming would crash on it.
```python
import json
if os.path.exists(OUT):
    good = []
    for l in open(OUT).read().splitlines():
        try:
            json.loads(l); good.append(l)
        except Exception:
            print("dropped a partial line")
    open(OUT, "w").write("\n".join(good) + ("\n" if good else ""))
    print("tiles already done on Kaggle:", len(good))
```
Check: the next run's `skipped` count must equal len(`done_tiles.txt`) + this number. It was verified on
2026-09-23: 102 + 1,611 = 1,713 skipped, 1,148 to do, 2,861 total.

**Cell 6: run** (the long cell)
```python
!python -u {BUNDLE}/gamus_zeroshot_kaggle.py --data {BUNDLE} --out {OUT} 2>&1 | tee -a {LOG}
```

**What you should see.** In the first minutes:
```
device cuda:0 Tesla P100-PCIE-16GB
51125 DFC2019 blocks
test tiles 2861, skipped 102, todo 2759 (reverse order)
```
After that, every 25 tiles:
```
25/2759 tiles, x.xx s/tile, ETA y.yy h, GPU peak mem z.z GiB, queue q
```
- `queue` near 0 means downloads are the bottleneck.
- A full queue (32) means the GPU is the bottleneck, which is the intended state.
- The run ends with `DONE`.

**Run it as a committed version, not interactively.** Interactive sessions stop when the browser idles.
- Click **Save Version** (top right) → **Save & Run All (Commit)** → Save.
- It runs in the background for up to 12 h. Kaggle's GPU limit is 12 h per session and about 30 h per week.
- Follow progress under **View Active Events** or the version's **Logs**.

You don't have to wait for all 2759 tiles. The local run is also advancing, so the two meet somewhere in the
middle. You can stop Kaggle whenever the tiles it has done plus the local ones cover all 2,861. The merge step checks this.

## 4. Get the output back into the repo

When the committed version shows **Complete**, or once you stop it:

1. Open the notebook → the version → **Output** tab (the right-hand file list under `/kaggle/working`).
2. Download these two files:
   - `zeroshot_tiles_kaggle.jsonl`: the results, about 36 KB per tile (roughly 100 MB if it does most of the split).
   - `zeroshot_kaggle_run_log.txt`: the run log.
3. Put them **exactly** here, keeping the names:
   ```
   DepthWizard2/data/gamus_eval/zeroshot_tiles_kaggle.jsonl
   DepthWizard2/data/gamus_eval/zeroshot_kaggle_run_log.txt
   ```
   It's the same folder as the local `zeroshot_tiles.jsonl`. **Do not overwrite or rename `zeroshot_tiles.jsonl`**: that is the local run's file.
4. Tell Claude: "Kaggle output is in `data/gamus_eval/`." Claude will then:
   - union the two files by tile, with the local record winning on duplicates;
   - check that all 2,861 test tiles are present exactly once;
   - stop the local job once coverage is complete;
   - run `scripts/gamus_zeroshot_aggregate.py` on the merged file.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `AssertionError: No CUDA GPU visible` | Settings → Accelerator → GPU P100, then restart the session. |
| `urlopen error` / HTTP 429 | Internet is off, or you're rate-limited. Turn Internet on and add the `HF_TOKEN` secret (Cell 4). The script retries each file 6 times. |
| `ImportError ... DPTImageProcessorPil` | Cell 2 didn't run. Re-run it and restart the kernel. |
| `assert len(hits) == 1` fails | The dataset isn't attached. Use **+ Add Input** → `gamus-b-bundle`. |
| Run interrupted (12 h limit, crash) | Start a new version with Cell 5; it resumes from the partial jsonl. |
| `CUDA out of memory` | Shouldn't happen on 16 GB (13 small models plus DAv2-Large at batch 4). Restart the session so nothing else holds GPU memory. |
