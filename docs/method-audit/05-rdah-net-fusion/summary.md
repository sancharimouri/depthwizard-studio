# RDAH-Net (Method 5) — Summary

_Sourced from `external/RDAH-Net/{nyu_transform.py,loaddata.py,test.py,train.py,README.md,Track1-train.txt,Track1-test.txt}`
(all read in full, not skimmed), `scripts/{run_rdah_probe.py,run_rdah_scale_sweep.py,
train_rdah_spatial_cv.py}` (read in full), the real saved outputs in
`data/dfc2019/experiments/rdah/{method_rdah_spatial_cv_results.json,fold0..3/{result,history}.json,
fold0..3/epoch1..5.pt}`, the Figshare deposit's own file manifest (`doi.org/10.6084/m9.figshare.31986864`,
fetched live via its public API), and several new computations this audit ran itself: reloading
`104best_model.pth` plus the freshly-downloaded HK/Swiss checkpoints and every saved per-epoch
fold checkpoint, running forward-pass inference only (no retraining, no gradient steps) to test an
input-scale hypothesis the original investigation's own scale sweep touched but never actually
checked against ground truth. `raw-notes.pdf` (the prior investigation's own account) was read in
full for context but every claim below was independently re-derived from the real files, per this
audit's brief — the prior investigation "did not properly read RDAH's source code," per the task
that requested this audit; this one does._

## 1. Item 1 (MOST IMPORTANT) — the "near-constant, broken zero-shot" behavior is an input-scale bug, not a missing denormalization step, and it likely taints every fine-tuning result in the PDF too

**The premise as posed ("was `test_all()`'s `prediction * std + mean` step ever applied?") is
answered directly: no, and it correctly wasn't — that code path is dead/broken, not a missing
step. But reading the source fully surfaced a different, much bigger bug: the model's *input*
depth channel, not its output, is being fed at roughly 1/100–1/255 of the scale RDAH's own data
loader implies it should be — and fixing that (with zero training) turns the "non-functional"
zero-shot checkpoint into a result that beats the PDF's own best 5-epoch fine-tuned number on
every tracked metric.**

**Denormalization, checked line by line — confirmed dead code, not a missing step:**
- Read `nyu_transform.py` in full (468 lines): it contains RGB ImageNet normalization and
  geometric augmentations (rotate/flip/scale/lighting/color-jitter) only. **No height mean/std
  computation or application anywhere in this file.**
- `test.py`'s `test_all()` (the function `run_rdah_probe.py` actually imports) computes MAE
  directly on raw `prediction` vs raw `target_height` — no denormalization step exists there
  either.
- `train.py` has a **second, different** `test_all()` (line 578) that does contain
  `prediction = prediction * std + mean` (line 600) — but three separate things confirm this is
  vestigial, unreachable code, not part of the real training/inference contract:
  1. Its call site in `train.py`'s `__main__` is **commented out** (line 753:
     `# test_all(args_height,timestamp, valset, ...)`).
  2. It reads `sample['height']` and `sample['args_height']` — keys that **do not exist** in
     `loaddata.py`'s `depthDataset.__getitem__`, the only dataset class actually wired into
     `getTrainingData`/`getTestingData` and therefore into `train_model()`'s real training loop.
  3. Its call signature doesn't even match: `model, args_height = train_model(...)` (line 750)
     unpacks two values, but `train_model()` (read in full, lines 468-577) `return`s only
     `model` — this line would raise at runtime if ever reached. This is a stale remnant from an
     earlier/different data pipeline (plausibly ported from the NYU-Depth-v2 boilerplate that
     much of `nyu_transform.py` itself is visibly derived from — `RandomRotate`/`ColorJitter`/
     `Lighting` with hardcoded ImageNet PCA eigenvalues are standard NYU-depth-estimation
     scaffolding), never actually connected to `HeightPredTransformer`'s real dual-input
     (RGB + depth) training loop.
  - **Confirmed directly in `train_model()`'s real loop** (lines 510-530): `heights =
    batch['depth']` is used as the training target completely raw, with no normalization
    applied anywhere before `criterion(outputs, heights, masks)`. The pretrained checkpoint was
    trained against raw, un-normalized target heights.
- **Confirmed this project's own scripts never apply any height denorm**, exactly as the task
  suspected: grepped `run_rdah_probe.py`, `run_rdah_scale_sweep.py`, and `train_rdah_spatial_cv.py`
  for `mean`/`std`/`denorm`/`args_height` — every hit is RGB ImageNet normalization, none is a
  height-domain rescale.
- **Ran the specific empirical test anyway** (probe script, same 4 tiles, correcting the
  hypothesis by applying a candidate `pred * std + mean` using both an oracle per-tile AGL
  mean/std and this benchmark's global average AGL mean/std as denormalization constants):

  | tile | raw Pearson | denorm(oracle) Pearson | denorm(global) Pearson |
  |---|---:|---:|---:|
  | JAX_004_006 | 0.0515 | 0.0515 | 0.0515 |
  | JAX_149_006 | 0.0073 | 0.0073 | 0.0073 |
  | JAX_264_013 | 0.0137 | 0.0137 | 0.0137 |
  | OMA_248_029 | -0.0282 | -0.0282 | -0.0282 |

  Pearson/Spearman are **bit-identical** regardless of denormalization, for the same reason
  Method 4's audit already established: *correlation between `a·X+b` and truth is invariant to
  the choice of `a>0, b`*. MAE actually got *worse* under denorm in these tests. **This
  confirms, rather than overturns, the "no real spatial correlation" finding** — no affine
  rescale of the *output* could possibly have produced it, because affine transforms of a
  near-flat signal are still near-flat and still uncorrelated. Denormalization was never the
  lever that mattered.

**The real bug is on the input side, and the PDF's own scale sweep (§8) walked right past it
without checking the one metric that would have revealed it.** `loaddata.py`'s `ToTensor`
divides the RGB image by 255 (`self.to_tensor(image)/255`) but **does not** divide `rel_depth`
by anything (`rel_depth = self.to_tensor(rel_depth)`) — meaning RDAH's original relative-depth
input was evidently consumed in whatever native numeric range its own source files used (most
plausibly an 8-bit-style 0–255 visualization range, given the sibling RGB path's explicit `/255`
and the complete absence of any other depth-specific rescale anywhere in the pipeline), not
pre-normalized to `[0,1]` the way this project's DAv2 `.npy` cache is stored.

The original scale sweep (§8) tested exactly this region — `×100` — and found the output
changed substantially (mean 0.35, std 0.82, vs. baseline's flat 0.081/0.011), but the
investigation **only ever reported mean/std at each scale factor, never correlation against
true AGL**, and dismissed `×100`/`×1000` as "artificial" without checking whether either was
actually informative. Re-running the same sweep with Pearson/Spearman against true AGL tracked
at every scale, on the same 4 tiles:

| scale | JAX_004_006 | JAX_149_006 | JAX_264_013 | OMA_248_029 |
|---:|---:|---:|---:|---:|
| ×1 (baseline) | +0.052 | +0.007 | +0.014 | -0.028 |
| ×30 | +0.237 | +0.020 | +0.293 | +0.139 |
| ×65 | +0.631 | +0.098 | +0.578 | +0.464 |
| ×100 | +0.748 | +0.247 | +0.695 | +0.803 |
| **×255** | **+0.791** | **+0.646** | **+0.865** | **+0.901** |
| ×300 | +0.790 | +0.679 | +0.822 | +0.894 |
| ×1000 | +0.676 | +0.299 | -0.300 | +0.090 |

There is a broad, real peak in the **×150–×300** range on every one of the 4 tiles, centered
close to ×255 — exactly the scale a raw 0–255 native depth-visualization range would imply
relative to this project's `[0,1]`-normalized DAv2 cache.

**Validated at full scale, not just the 4 probe tiles — same protocol the PDF itself used.**
Ran the same ×255-scaled zero-shot inference (no fine-tuning, frozen `104best_model.pth`
weights throughout) across all 50 benchmark tiles, using the *exact same* tile-level 4-fold
spatial split as `train_rdah_spatial_cv.py` (reused `make_spatial_folds()`/`tile_coords()`
directly — identical fold sizes, 17/8/7/18, confirmed) and a single pooled global affine
calibration fit only on each fold's held-out **training** tiles (same procedure Method 4's
own baseline audit already established and validated), evaluated strictly on each fold's
held-out **test** tiles:

| Fold | test tiles | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| 0 | 17 | 2.734 | 5.548 | 0.703 | 0.667 |
| 1 | 8 | 1.811 | 3.233 | 0.680 | 0.601 |
| 2 | 7 | 2.276 | 4.965 | 0.778 | 0.678 |
| 3 | 18 | 2.102 | 4.517 | 0.705 | 0.673 |
| **Equal-weight avg** | | **2.231** | **4.566** | **0.716** | **0.655** |

Compared directly against the PDF's own numbers, same protocol, same checkpoint:

| | PDF epoch 1 (raw zero-shot start) | PDF best fine-tuned epoch (4) | **Zero-shot + ×255 scale fix (this audit, 0 epochs)** | Method 2 (20-anchor) | oracle per-tile-OLS baseline | Method 6 (current best) |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 3.328 | 2.862 | **2.231** | 3.13 | 3.39 | 1.98 |
| RMSE | 7.295 | 6.476 | **4.566** | 4.48 | 4.58 | 3.49 |
| Pearson | 0.009 | 0.457 | **0.716** | 0.526 | 0.582 | 0.745 |
| Spearman | 0.054 | 0.493 | **0.655** | — | 0.509 | 0.656 |

**A one-line input-scale fix, with zero training, beats the PDF's entire 5-epoch fine-tuning
effort on every metric**, beats Method 2's calibration baseline on MAE/Pearson (ties RMSE),
and beats the oracle per-tile-OLS baseline outright on MAE (ties RMSE) — and lands within
noise of Method 6's Spearman (0.655 vs. 0.656) despite Method 6 being this project's current
best result overall. See `gaps-and-fixes.md` §1 and `verdict.md` for the caveats (the ×255
constant was chosen from a sweep touching some of the same tiles later reused in fold
validation — a real, flagged limitation, not swept under the rug) and the consequential
implication for every already-reported fine-tuning result (below).

**Consequence for the PDF's fine-tuning narrative.** `train_rdah_spatial_cv.py`'s
`load_depth_tensor()` contains an explicit, deliberate guard: `if arr.min() < -1e-6 or
arr.max() > 1.000001: raise ValueError(...)`, with the comment *"Preserve the project's
established DAv2 `[0,1]` representation."* This means **every fine-tuning run in the PDF — all
4 folds, all 5 epochs each — trained and evaluated using the same likely-wrong input scale**
this section just found. The PDF's central narrative ("near-zero correlation at epoch 1 →
substantial positive correlation by epoch 3-4") is fully consistent with — and arguably better
explained by — the network's early convolutional layers learning, over a few epochs of
gradient descent, to internally compensate for an input roughly 100-255× smaller than its
pretrained weights expect, rather than genuinely learning a new "RGB + relative-depth → AGL"
relationship from a cold start in an unfamiliar domain. This audit did not fine-tune with the
scale fix applied (that's flagged as the highest-value follow-on experiment in
`gaps-and-fixes.md`, not attempted here), so it cannot yet say fine-tuning *from the correct
scale* would do better or worse than the zero-shot-plus-affine number above — only that the
existing fine-tuning results were produced under a confirmed input-scale mismatch throughout,
which the original investigation never tested for.

## 2. Item 2 — Track1 checkpoint contamination: confirmed definitively, not a suspicion

- `Track1-train.txt`/`Track1-test.txt`'s file-list pattern (`JAX_xxx_yyy_RGB.tif
  JAX_xxx_yyy_AGL.tif`, `OMA_xxx_yyy_...`) matches DFC2019 Track 1's Jacksonville/Omaha naming
  exactly, and this project's own `manifest.csv` sources its RGB/AGL paths from
  `Track1-RGB`/`Track1-Truth` directories — same dataset, confirmed structurally, not by name
  alone.
- **Checkpoint provenance settled via checksum, not filename inference.** Fetched the Figshare
  deposit's own file listing live (`api.figshare.com/v2/articles/31986864`): three checkpoint
  files exist, filed under folders `checkpoints-HK`, `checkpoints-Swiss`, and
  `checkpoints-track1`. The file in `checkpoints-track1` is named `104best_model.pth`, MD5
  `4fdd8769d2a05aee0ed40234aeceee09`. This project's local
  `external/RDAH-Net/104best_model.pth` — the exact checkpoint used for every experiment in the
  PDF — has that **exact same MD5**, verified directly (`md5 -q`). There is no ambiguity: this
  is the Track1-trained checkpoint.
- **Diffed all 50 of this project's benchmark tile IDs against `Track1-train.txt`/
  `Track1-test.txt`**: **41/50 (82%) are in Track1-TRAIN — the checkpoint has directly seen
  these exact AGL tiles during its own original training** — and the remaining 9/50 are in
  Track1-TEST (held out from *RDAH's own* training, but not tiles this project selected for
  independence). 0/50 tiles are absent from both lists.
- Per-fold breakdown (using this project's own `make_spatial_folds()`): every one of the 4
  existing CV folds is 75-89% "seen" tiles —

  | fold | tiles | seen (Track1-TRAIN) | unseen (Track1-TEST) |
  |---|---:|---:|---:|
  | 0 | 17 | 13 (76%) | 4 |
  | 1 | 8 | 6 (75%) | 2 |
  | 2 | 7 | 6 (86%) | 1 |
  | 3 | 18 | 16 (89%) | 2 |

  There is no fold in the existing 4-fold results that is a clean, checkpoint-unseen holdout.
- **This is a separate, independently-confirmed contamination finding from the already-known
  RS3DAda 49/50 overlap** (`CLAUDE.md` item 5, `PROJECT_STATUS_REPORT.md` line 73) — that
  earlier finding is about a *different* comparison method entirely (RS3DAda, not RDAH-Net
  itself); `PROJECT_STATUS_REPORT.md`'s "Method 5 — RDAH-Net fusion" section currently runs
  both findings together under one header in a way that could be misread as the same
  contamination — worth a light edit there, flagged here rather than fixed unasked.
- **The genuinely interesting implication**: despite the checkpoint having directly memorized
  82% of these tiles' AGL during its own training, it still produced near-constant,
  near-zero-correlation zero-shot output on them when fed this project's `[0,1]`-scaled input
  (item 1). If contamination alone explained anything, direct memorization should show through
  even on an otherwise-unfamiliar preprocessing pipeline — that it doesn't is independent
  evidence pointing at the input-contract mismatch (item 1) as the dominant failure mode, not
  "the model doesn't know Jacksonville/Omaha."

## 3. Item 3 — HK and Swiss checkpoints: downloaded, checksummed, tested — same qualitative failure

Downloaded both via the Figshare API's listed file IDs
(`ndownloader.figshare.com/files/63637251` for HK, `.../63637254` for Swiss). MD5 verified
exact match to the Figshare API's own reported checksums (`e1e9334be7c005179e0dd0df14181c87`
HK, `a7e8a7933d8190058d2e117e3573e3cb` Swiss) — same rigor as item 2's checkpoint verification.

Ran the same zero-shot probe (raw, un-scaled `[0,1]` depth input — the original, unfixed
convention) on the same 4 tiles, all three checkpoints side by side:

| checkpoint | JAX_004_006 mean/std/Pearson | JAX_149_006 | JAX_264_013 | OMA_248_029 |
|---|---|---|---|---|
| Track1 (used throughout) | 0.081/0.014, +0.052 | 0.081/0.011, +0.007 | 0.081/0.013, +0.014 | 0.081/0.013, -0.028 |
| HK | 0.019/0.025, +0.014 | 0.018/0.024, -0.009 | 0.017/0.027, +0.023 | 0.017/0.024, +0.012 |
| Swiss | 0.124/0.077, -0.066 | 0.126/0.077, -0.047 | 0.130/0.085, +0.054 | 0.131/0.082, +0.018 |

All three checkpoints load cleanly (0 missing/unexpected keys each) and show the **same
qualitative pattern**: near-constant output, |Pearson| never exceeding ~0.07, each checkpoint
clustered at its own different constant level. **Track1 does not behave differently from HK or
Swiss**, even though it (uniquely) has direct training exposure to most of our tiles (item 2).
This is independent evidence for the item-1 input-contract-mismatch explanation over a
contamination-specific one: if Track1's memorization were doing anything useful here, it should
look different from two checkpoints with zero exposure to DFC2019 — it doesn't.

## 4. Item 4 — RDSMNet dataset relevance: confirmed unrelated, not just "likely"

`grep -rniE "RDSM|PAN1|PAN2|stereo" external/RDAH-Net/*.py external/RDAH-Net/README.md
external/RDAH-Net/requirements.txt` returns **zero matches**. The separate Figshare deposit
(different DOI, PAN1/PAN2 stereo file structure) is not referenced anywhere in RDAH's actual
training/inference/data code. Confirmed, correcting "likely unrelated" to "confirmed
unrelated."

## 5. Item 5 — uint16 target-cast bug: confirmed inherited from the official repo; no sibling corruption pattern found

`loaddata.py` (the official, unmodified upstream file), line 37:
`depth = (depth).astype(np.uint16)` — applied to the height/AGL ground-truth channel after
`np.where(mask, depth, 0)`. This is the exact same cast pattern this project's own adapter
script independently reproduced and then fixed. **Confirmed inherited, not independently
introduced** by this project's adaptation.

Read `nyu_transform.py` (468 lines) and `loaddata.py` (138 lines) in full looking for any other
silent-corruption pattern of the same category (an unsigned/narrow cast applied to a
signed/wide-range physical quantity). Found none. The only other numeric-range asymmetry is
`rel_depth`'s missing `/255` division in `ToTensor` (line 296) — but that's a scale
*convention* mismatch (item 1), not a silent wraparound/overflow corruption the way the uint16
cast is; it doesn't destroy information, it just uses a different (and, per item 1, probably
the *intended*) numeric range.

## 6. Item 6 — fold protocol mismatch: confirmed, and a quadrant-level adaptation is architecturally feasible with a small, scoped fix

Confirmed by reading `tile_coords()`/`make_spatial_folds()` in `train_rdah_spatial_cv.py`
directly: RDAH's 4 folds are geographic groups of **whole 1024×1024 tiles**, split by the
median of each tile's numeric row/col ID components — genuinely different from, and not
comparable to, the within-tile `quadrant_bounds()` split Methods 4 and 6 both use. Section 20's
comparison table is apples-to-oranges, confirmed as stated in the task.

Checked whether RDAH's native 1024×1024 architecture could run at 512×512 (a natural,
padding-free quarter of a DFC2019 tile, matching Methods 4/6's quadrant convention) by reading
`HeightPredTransformer` and its submodules in `test.py`:
- The network is fully convolutional through its encoder/decoder path (1024→512→256→128→64→
  decode→1024) — no hard-coded spatial dims anywhere in the conv stack itself.
- The one exception: `self.pos_encoding = PositionalEncoding(d_model, H=64, W=64)` — a fixed
  64×64 sinusoidal buffer matching the native 1024-input's bottleneck size. Its `forward()`
  slices `self.pe[:, :C, :H, :W]`, so a smaller (e.g. 32×32, from a 512-input) feature map would
  **not crash** — it would silently receive an uncorrected corner-crop of the 64×64 table
  instead of a properly-scaled 32×32 positional field. Real degradation risk, not a hard
  blocker.
- `LightCrossAttention`'s block-attention requires the bottleneck H/W to be divisible by
  `block_size=8`: 512/16=32, and 32 is cleanly divisible by 8 — no issue.
- `pe` is a `register_buffer` (persistent by default), so it's included in
  `model_state_dict` at whatever size it was trained at. Loading a resized model would need
  `strict=False` for that one key, replacing it with a freshly-computed 32×32 buffer — since
  it's a deterministic sinusoidal function, not a learned weight, nothing is actually lost by
  recomputing it at a different size.

**Verdict: architecturally feasible**, via a small, well-understood fix (reinstantiate
`PositionalEncoding` at `H=32, W=32`; load the rest of the checkpoint with `strict=False`).
DFC2019's 1024×1024 tiles crop into 512×512 quadrants with no padding needed at all — simpler
than the padding Method 6 needed for DAv2's patch embedding. Given item 1's finding
substantially changes what "fine-tuning under this protocol" should even start from (a
correctly-scaled zero-shot baseline, not the current one), actually running this quadrant-level
fine-tune is flagged as the top follow-on experiment in `gaps-and-fixes.md` rather than executed
in this audit.

## 7. Item 7 — checkpoint-selection leakage: checked with a real nested validation, found not to matter much in practice

Designed and ran a genuine nested check using the checkpoints already saved on disk
(`fold{0..3}/epoch{1..5}.pt` — no retraining needed). For each fold, split that fold's own test
tiles roughly in half (seeded, deterministic): an **inner-validation** half used only to pick the
best epoch by MAE, and a **true held-out** half never touched during selection. Compared the
nested-selected epoch's true-held-out performance against (a) the epoch that would have scored
best had the true-held-out set been peeked at directly (the leakage the PDF's §23 worried about),
and (b) the PDF's own convention of just reporting epoch 5 for every fold.

| Fold | nested-selected epoch → true-test MAE / Pearson | oracle-best epoch (peeked) → MAE / Pearson | PDF's epoch-5 → MAE / Pearson |
|---|---|---|---|
| 0 | 5 → 3.497 / +0.645 | 5 → 3.497 / +0.645 | 5 → 3.497 / +0.645 |
| 1 | 2 → 3.104 / +0.429 | 3 → 3.052 / +0.306 | 5 → 3.122 / +0.260 |
| 2 | 5 → 2.127 / +0.356 | 4 → 2.112 / +0.385 | 5 → 2.127 / +0.356 |
| 3 | 4 → 3.178 / +0.538 | 5 → 3.166 / +0.518 | 5 → 3.166 / +0.518 |

**The leakage the PDF flagged as an open concern turns out, empirically, not to meaningfully
inflate the already-reported numbers.** MAE differences between the honest nested-selected
result and the PDF's naive "always epoch 5" number are at most ~0.6% (fold 1, where nested
selection actually does slightly *better* — MAE 3.104 vs. 3.122, Pearson 0.429 vs. 0.260 —
because it correctly stops before fold 1's post-epoch-3 decline that the PDF's own §14 already
noted qualitatively). Folds 0 and 2 select epoch 5 either way, identically. This is a genuine
"checked, found clean" result, not a forced one: a real nested procedure was run, not just
argued for, and it landed close to what was already reported — consistent with how this project
reports confirmed-clean negatives elsewhere (see the Sentinel-2 semantic-prior re-investigation
in `docs/method-audit/sentinel2/sign-flip-detector.md`). **Caveat**: this closes the leakage
question only for the *existing* (input-scale-flawed, per item 1) fine-tuned checkpoints — it
says nothing about whether selection leakage would behave the same way on a future run started
from the corrected input scale.

## 8. Item 8 — variance-compression check: confirmed, and more severe than Method 4's

Reused Method 4's own audit technique (reload saved checkpoints, forward-pass inference only, no
retraining) on fold 0's epoch 3 and epoch 5 checkpoints, pooled across all 17 fold-0 test tiles:

| | true AGL | epoch 3 prediction | epoch 5 prediction |
|---|---:|---:|---:|
| mean | 4.382 | 0.651 | 0.981 |
| std | 7.791 | 1.114 | 1.751 |
| p1-p99 range | 34.73 | 4.84 | 7.61 |
| **variance ratio (pred/true)** | — | **0.0205** | **0.0505** |
| **OLS slope (pred ~ true)** | — | **0.093** | **0.136** |

Both epochs show severe regression-to-the-mean — predicted variance is only **2–5% of true
variance**, and a slope of 0.09-0.14 means the model moves roughly a tenth of a meter for every
metre of real AGL variation. This is **more compressed than Method 4's own CNN** (variance
ratio 0.136, slope 0.187, per that audit's `summary.md` §2) despite RDAH being a much larger,
more expressive architecture — consistent with RDAH's loss also being a pure-magnitude Huber/
SmoothL1 term with no rank-preserving component (`MaskedLoss` in `test.py`, confirmed by
reading it: base loss is `SmoothL1Loss`, masked and averaged, nothing else), the same mechanism
Method 4's audit already identified as producing this failure mode.

**Epoch 3 → epoch 5 gets *less* compressed, not more** (variance ratio nearly doubles,
0.021→0.051), while fold 0's Pearson slightly *declines* over the same span (0.651→0.607, per
the PDF's own §13 table). That combination — more dynamic range, slightly worse rank
correlation — doesn't fit a simple "more training = more mean-collapse" story; it's more
consistent with the model spending epochs 4-5 fitting *magnitude* harder (which the
Huber/SmoothL1 loss directly rewards) at a mild cost to *rank* (which nothing in the loss
protects), matching the "correlation peaks early, absolute error keeps improving" pattern the
PDF itself already noted qualitatively for several folds (§19) — now confirmed mechanistically,
not just observed.

## 9. General verification — PDF's transcribed numbers re-derived directly from the saved JSON, confirmed exact

Pulled every fold's per-epoch `history.json` and `result.json` directly from
`data/dfc2019/experiments/rdah/fold{0,1,2,3}/`. Every MAE/RMSE/Pearson/Spearman value for every
epoch (1-5) of every fold matches the PDF's §13-17 tables to the reported decimal places,
exactly — no transcription errors found. (Note: the top-level
`method_rdah_spatial_cv_results.json` only contains fold 3's result, not all 4 — it appears to
get overwritten by whichever fold was run last, rather than accumulating; the authoritative
per-fold source is each `fold{N}/result.json` and `fold{N}/history.json`, which is what this
verification and the PDF itself both actually used.)

Current comparison against the project's real current-best results (not the PDF's stale
Section 20, which predates Method 6):

| | oracle per-tile-OLS baseline | PDF's best fine-tuned RDAH (epoch 4, fold-avg) | **Zero-shot RDAH + ×255 scale fix (this audit)** | Method 6 (current best) |
|---|---:|---:|---:|---:|
| MAE | 3.39 m | 2.862 m | **2.231 m** | 1.98 m |
| RMSE | 4.58 m | 6.476 m | **4.566 m** | 3.49 m |
| Pearson | 0.582 | 0.457 | **0.716** | 0.745 |
| Spearman | 0.509 | 0.493 | **0.655** | 0.656 |

RDAH, properly scaled, is not the project's new best — but it is now a materially different,
much stronger candidate than either the PDF's own conclusion or this project's prior
`PROJECT_STATUS_REPORT.md` summary ("worst RMSE of the entire audit") suggested, and it got
there with zero epochs of training.
