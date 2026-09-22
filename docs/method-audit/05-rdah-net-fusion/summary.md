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

## 10. 2026-09-23 — RDAH-FT-2: the three-fixes-stacked fine-tune, all 4 folds

_Sourced from `scripts/train_rdah_quadrant_cv.py` (read in full), its four per-fold outputs
`data/dfc2019/experiments/rdah_quadrant_cv/fold{0..3}/result.json` (fold index confirmed from each
file's own `"fold"` field, not its directory name), and
`data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`, written by
`scripts/aggregate_rdah_ft2.py` (arithmetic only: no inference, no retraining). Every number
below comes from that aggregate JSON. The training run finished last session, but the session hit
its usage limit before any doc was written, so no earlier FT-2 section exists to extend._

### What changed from FT-1 to FT-2

FT-2 is the "single combined experiment" that `verdict.md` §4 called for. It runs on the same
50-tile DFC2019 benchmark for 5 epochs at Adam lr 1e-5, and changes four things:

| | RDAH-FT-1 (`train_rdah_spatial_cv.py`) | RDAH-FT-2 (`train_rdah_quadrant_cv.py`) |
|---|---|---|
| Init checkpoint | Track1 `104best_model.pth` (41/50 tiles in its own training list, §2) | Swiss `swiss_best_model.pth` (0/50 overlap, see below) |
| Depth input scale | unscaled `[0,1]` (§1's bug) | per-fold constant from a Pearson sweep on **training quadrants only**, with the 4 original probe tiles excluded: ×255 for folds 0–1, ×300 for folds 2–3 |
| Holdout | whole-tile geographic folds (17/8/7/18 tiles) | quadrant-level, `quadrant_bounds()` reused from `evaluate_method4.py`, matching Methods 4/6. 150 train / 50 test samples per fold. PositionalEncoding rebuilt at 32×32 for 512² inputs |
| Loss | masked SmoothL1 only | masked SmoothL1 + 0.5 × `rank_pair_loss` (from Method 4 v2's `phase2_building_rank_v2`, 2000 pairs/patch, margin 0.25) |
| Checkpoint selection | epoch 5, fixed | nested: select epoch by MAE on one half of the held-out quadrant, report on the other half |

**Nested selection doesn't use inner-train data, and this needs stating plainly.**
`run_fold()` shuffles the fold's 50 held-out-quadrant samples (seed `42 + fold`), selects the
epoch by MAE on 25 of them (`inner_val_n = 25`), and reports on the other 25
(`true_test_n = 25`). The two halves are disjoint tiles, so the reported samples never influence
selection. But the selection set comes from the held-out quadrant, not from the training
quadrants. This is the same discipline §7 used for FT-1, not a stricter one. In practice it
barely matters. The oracle epoch (peeking at the reported half) and the fixed epoch 5 land within
0.5% MAE of the nested choice:

| | MAE (pixel-weighted) | RMSE (pixel-weighted) | Pearson (mean of folds) | Spearman (mean of folds) |
|---|---:|---:|---:|---:|
| nested-selected (reported) | 2.499 | 5.598 | 0.571 | 0.572 |
| oracle epoch (peeked) | 2.497 | 5.602 | 0.572 | 0.572 |
| fixed epoch 5 | 2.510 | 5.614 | 0.568 | 0.566 |

**Side effect: every FT-2 number is measured on half of each held-out quadrant** (25 samples,
about 6.54M valid pixels per fold). Methods 4 and 6 report on all 50 samples. The per-epoch
`history` entries in each `result.json` use all 50, but those include the selection half.

### The import bug that was fixed

`train_rdah_quadrant_cv.py` loads `evaluate_method4.py`, `evaluate_method4_v2.py` and
`train_rdah_spatial_cv.py` through `importlib.util.spec_from_file_location`. `evaluate_method4.py`
uses `@dataclass`. During class construction, `dataclasses` calls
`sys.modules.get(cls.__module__)`, which returns `None` for a module that hasn't been registered
yet, and that crashes with `AttributeError`. The fix, `import_module()`, registers
`sys.modules[name] = mod` **before** `spec.loader.exec_module(mod)`. The script's own comment at
that function records this. No separate crash log was saved, so this account comes from the fixed
code and its comment, not from a traceback.

### Per-fold results (nested-selected epoch, true-test half)

| Fold | scale | selected epoch (oracle) | valid px | MAE | RMSE | Pearson | Spearman | variance ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | ×255 | 4 (4) | 6,553,024 | 2.012 | 5.375 | 0.580 | 0.560 | 0.192 | 0.192 |
| 1 | ×255 | 3 (2) | 6,536,997 | 2.359 | 5.490 | 0.607 | 0.612 | 0.186 | 0.198 |
| 2 | ×300 | 3 (3) | 6,545,567 | 3.042 | 6.063 | 0.503 | 0.499 | 0.196 | 0.228 |
| 3 | ×300 | 4 (5) | 6,540,036 | 2.585 | 5.435 | 0.593 | 0.618 | 0.231 | 0.221 |
| **min–max** | | | | 2.012–3.042 | 5.375–6.063 | **0.503–0.607** | 0.499–0.618 | 0.186–0.231 | 0.192–0.228 |

### Aggregate, labelled by method

| Aggregation | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Pixel-weighted (MAE = Σn·MAE/Σn, RMSE = √(Σn·RMSE²/Σn); n = 26,175,624) | **2.499** | **5.598** | — | — |
| Mean of folds (each fold pixel-pooled within fold) | 2.499 | 5.591 | **0.571** | **0.572** |
| Pixel-count-weighted mean of fold correlations | — | — | 0.571 | 0.572 |
| **Mean of per-sample metrics, then mean of folds** (Method 6's aggregation) | **2.500** | **4.294** | **0.640** | **0.506** |

A **true pooled Pearson/Spearman can't be computed.** The JSONs store no per-fold means,
variances, covariance, or predictions. The correlation rows above are means of fold values and
are labelled that way. The recomputation agrees with last session's in-chat figures on MAE ≈2.50,
RMSE ≈5.60, Pearson ≈.571 and Spearman ≈.572. **It disagrees on the variance-ratio range.** At the
selected epochs the range is **0.186–0.231**, not ≈.13–.22. The .13 figure matches fold 1's
full-quadrant per-epoch `history` (0.117–0.138), which covers different samples. The JSON values
are the ones used here.

### Fold stability (HANDOFF §3.1's question)

FT-1's fold-level Pearson ranged over 0.254–0.607, a width of 0.353. That is epoch 5, pixel-pooled
within each fold. FT-2's Pearson range under the same within-fold pooling is **0.503–0.607, a width
of 0.104**. Spearman narrows from 0.307–0.573 to 0.499–0.618. RMSE narrows from 4.623–7.723 to
5.375–6.063. **The instability FT-1 showed doesn't reproduce under FT-2's recipe.** Several things
changed together (quadrant folds of 50 samples instead of tile folds of 7–18 tiles, correct input
scale, rank loss, a different checkpoint), so this run can't say which one fixed it. The most
likely single contributor is the fold construction: FT-1's 7- and 8-tile folds are small enough
for a few tiles to swing a fold's correlation. That attribution is a hypothesis.

### Variance ratio: better, but still severe underdispersion

Going from FT-1's 0.02–0.05 (§8, fold 0) to FT-2's 0.186–0.231 (mean 0.201) is a real
improvement, roughly 4–11× (0.186/0.050 to 0.231/0.021). The rank term (the fix §8 prescribed) is the most plausible cause.
**It is still severe underdispersion.** Predicted variance is about a fifth of true variance, and
the OLS slope of prediction on truth is 0.19–0.23, so the model moves about 0.2 m for every metre
of real AGL. For comparison, Method 4's CNN had a variance ratio of 0.136 and a slope of 0.187 (§8),
so FT-2 is only modestly less compressed than the simplest learned model in this audit.

**RMSE is high relative to MAE, and the per-sample data shows where.** The pixel-pooled RMSE/MAE
ratio is 2.24 (5.598/2.499), while the mean of per-sample RMSEs is only 4.294. That gap means a
few samples with very large errors dominate the pooled RMSE. In every fold, the three worst-RMSE
samples come from the same handful of Jacksonville tiles. Those samples have **high correlation
but large absolute error**:

| Fold | sample | MAE | RMSE | Pearson |
|---|---|---:|---:|---:|
| 0 | JAX_166_006_q0 | 9.93 | 17.20 | 0.835 |
| 0 | JAX_214_023_q0 | 6.77 | 14.71 | 0.745 |
| 1 | JAX_164_008_q1 | 11.66 | 19.51 | 0.761 |
| 1 | JAX_214_015_q1 | 6.22 | 10.37 | 0.825 |
| 2 | JAX_166_006_q2 | 7.22 | 17.15 | 0.887 |
| 2 | JAX_214_023_q2 | 10.26 | 13.49 | 0.750 |
| 3 | JAX_214_015_q3 | 6.99 | 14.65 | 0.688 |
| 3 | JAX_166_006_q3 | 6.31 | 10.18 | 0.744 |

Good ranking with large magnitude error is the signature you'd expect if FT-2 **under-predicts
tall structures**: a slope of about 0.2 stretched over a large AGL range. **This is a hypothesis,
not a measured result.** The JSONs contain no per-height-bin error and no per-sample AGL
distribution, so they can't show that these samples are the tall ones or that the error sits at
the top of each sample's height range. Testing it needs saved predictions binned by true AGL. That
is an inference re-run, so it wasn't done this session.

### Comparison against the rest of the project

Aggregation differs across these rows, and mixing them quietly would mislead. Here is what each
row's own script does:
- **Method 6**, `evaluate_method6_finetune_twinhead.py` `agg()`: mean of per-tile(-quadrant)
  metrics within each fold, then mean over folds.
- **Oracle per-tile-OLS** (Method 3's baseline as re-derived in `04-learned-scale-modulation/
  summary.md` §1) and **Method 2 Grid+Huber+20** (`evaluate_sparse_anchor_regression.py`): mean of
  per-tile metrics.
- **RDAH-FT-1's "pooled" 2.906/6.659/.513/.527** (`evaluate_rdah_pooled_cv.py` →
  `pooled_cv_epoch5_report.json`): true pixel-level pooling over all 52.4M pixels. Pearson is
  exact. Spearman is computed on a 2M-pixel deterministic reservoir sample. Recomputing FT-1's
  pixel-weighted MAE/RMSE from its per-fold files reproduces 2.906/6.659 exactly.
- **RDAH zero-shot on DFC2019**: equal-weight mean of 4 fold values (§1). The within-fold
  aggregation isn't recorded, and **the script and result JSON for the 50-tile ×255 run weren't
  located in the repo** (`run_rdah_scale_sweep.py` is the 4-tile probe sweep, not this run). The
  numbers come from §1 of this document and are **unverified against an artifact**.

The like-for-like FT-2 row is therefore the **per-sample mean** (2.500/4.294/0.640/0.506). The
pixel-pooled row is also shown.

| | MAE | RMSE | Pearson | Spearman | aggregation | protocol / checkpoint |
|---|---:|---:|---:|---:|---|---|
| **RDAH-FT-2** (per-sample mean) | **2.500** | **4.294** | **0.640** | **0.506** | per-sample mean → fold mean | quadrant, Swiss, 25/50 samples per fold |
| RDAH-FT-2 (pixel-pooled) | 2.499 | 5.598 | 0.571* | 0.572* | pixel-weighted; *mean of folds | same |
| RDAH-FT-1 (per-tile mean) | 2.864 | 5.502 | 0.498 | — | per-tile mean → fold mean | tile-level, Track1, unscaled, epoch 5 |
| RDAH-FT-1 (pooled, as reported) | 2.906 | 6.659 | 0.513 | 0.527 | true pixel pooling | same |
| RDAH zero-shot on DFC2019, ×255 + per-fold affine | 2.231 | 4.566 | 0.716 | 0.655 | mean of 4 folds; artifact not located | tile-level, **Track1** |
| Method 6 + height-balanced (`method6_height_balanced/m6_heightbal_results.json`) | 1.980 | 3.492 | 0.745 | 0.656 | per-tile mean → fold mean | quadrant, all 50 samples |
| Oracle per-tile-OLS (`04-…/summary.md` §1) | 3.39 | 4.58 | 0.582 | 0.509 | per-tile mean | quadrant |
| Method 2 Grid+Huber+20 (HANDOFF's "oracle" row) | 2.929 | 4.718 | 0.532 | 0.471 | per-tile mean | per-tile anchors, no CV |

**Does FT-2 (per-sample mean) beat each reference?**

| | MAE | RMSE | Pearson | Spearman |
|---|---|---|---|---|
| vs. Method 6 (height-balanced) | ✗ | ✗ | ✗ | ✗ |
| vs. Method 2 Grid+Huber+20 (HANDOFF §7 bar) | ✓ | ✓ | ✓ | ✓ |
| vs. oracle per-tile-OLS (3.39/4.58/.582/.509) | ✓ | ✓ | ✓ | ✗ (0.506 vs 0.509) |

Three corrections to the numbers this session was briefed with:
1. **Method 6's height-balanced Pearson/Spearman are 0.7445/0.6563** in its result JSON. The
   briefed 0.737/0.654 are the original Method 6 recipe's values (2.053/3.531/0.737/0.654). FT-2
   loses to both.
2. **"Oracle per-tile-OLS" names two different rows in this project.** CLAUDE.md and HANDOFF use
   2.929/4.718/.532/.471, which is Method 2's Grid+Huber+20 sparse-anchor result
   (`02-gcp-regression/summary.md`). The per-tile-OLS oracle that Methods 4 and 6 are compared
   against in their own audits is 3.39/4.58/.582/.509. Both rows are shown above.
3. **"FT-2 fails the oracle bar on RMSE" holds only under pixel pooling** (5.598 vs 4.718/4.58).
   Every baseline row uses per-tile means, and under that aggregation FT-2's RMSE of 4.294 clears
   both bars. The JSONs support the like-for-like reading, so this document uses it. It doesn't
   change the verdict, because FT-2 still loses to Method 6 on all four metrics under either
   aggregation.

### Separating the four RDAH results

1. **Zero-shot on Sentinel-2 (Darjeeling)**: rejected for checkerboard artifacts
   (`PROJECT_STATUS_REPORT.md`). Probably fed unscaled `[0,1]` depth (`verdict.md` §5). The
   ×255/×300 rescale has not been re-checked on Sentinel-2.
2. **Zero-shot on DFC2019, corrected input scale**: 2.231/4.566/0.716/0.655, Track1 checkpoint,
   tile-level folds, per-fold affine calibration fitted on training tiles. Strong, but it comes
   from a contaminated checkpoint, and the artifact wasn't located.
3. **RDAH-FT-1**: 2.906/6.659/0.513/0.527 pooled. Track1, unscaled input, tile folds, epoch 5,
   unstable across folds.
4. **RDAH-FT-2**: 2.500/4.294/0.640/0.506 (per-sample mean). Swiss, corrected scale, quadrant
   folds, rank loss, nested selection. Stable across folds, still underdispersed.

### Open question: why does zero-shot beat both fine-tuned runs?

Result 2 beats FT-2 on MAE, Pearson and Spearman. Two explanations are live:
- **(a) Fine-tuning damages the pretrained model** on a benchmark this small (150 training
  samples per fold, 5 epochs).
- **(b) The zero-shot score is inflated by pretraining overlap with DFC2019.** The repo documents
  each checkpoint's training data, so this can be read directly without running anything:
  `external/RDAH-Net/Track1-train.txt` (2,226 lines, DFC2019 JAX/OMA tile pairs) contains **41/50**
  of this benchmark's tiles, and `Track1-test.txt` contains the other 9. `Swiss-{train,test}.txt`
  (8,823 / 2,204 lines) and `HK-{train,test}.txt` (1,179 / 293 lines) list GF-7 ortho tiles
  (`GF07_DLC_…_Ortho_*.tif`), and **0/50** benchmark tiles appear in either. The zero-shot row used
  Track1 and FT-2 used Swiss. The same pattern appeared for another method: RS3DAda had 49/50
  benchmark tiles in its training split (`stage0-gates/rs3dada-audit.md`).

This comparison is confounded on three axes at once: checkpoint (Track1 vs. Swiss), protocol
(tile-level vs. quadrant) and output mapping (per-fold affine calibration vs. raw fine-tuned
output). The check that separates (a) from (b) is **Swiss zero-shot at the fold-derived scale,
under the quadrant protocol, with the same per-fold affine calibration**. If it lands near
2.23/0.72, contamination doesn't explain the zero-shot score and fine-tuning is the problem. If it
lands at or below FT-2, the zero-shot number was inflated by contamination. **Not run this
session** (documentation-only).

## 11. 2026-09-23 (later) — rescued artifacts: where ×255 came from, and Swiss zero-shot on DFC2019

_Sourced from session `fa0035ab`'s scratchpad (under `/private/tmp`, cleared on reboot), now
copied to `scripts/diag/` and `data/dfc2019/experiments/rdah_zeroshot/` (see its `README.md`).
Files marked TRANSCRIBED come from the session transcript, because those scripts printed their
results instead of saving them._

**§10's "artifact not located" is resolved.** The 50-tile Track1 zero-shot run is
`scripts/diag/diag_rdah_x255_fullcv.py`. Its saved per-fold CSV,
`rdah_x255_zeroshot_fold_results.csv`, reproduces **2.231 / 4.566 / 0.716 / 0.655** exactly
(mean of folds). Pearson and Spearman there are computed on *uncalibrated* predictions pooled
within each fold; only MAE/RMSE use the per-fold affine calibration.

**Provenance of ×255: picked on Track1, on tiles Track1 trained on.**
- `diag_rdah_scale_corr.py` swept ×1–×1000 on 4 probe tiles with `CHECKPOINT` imported from
  `run_rdah_probe.py`, which is `104best_model.pth` (Track1). Output in `scale_corr_stdout.txt`.
- Three of those probe tiles (JAX_004_006, JAX_264_013, OMA_248_029) are in `Track1-train.txt`;
  JAX_149_006 is in `Track1-test.txt`.
- FT-2's clean re-derivation (Swiss, training quadrants only, probe tiles excluded) sits on a
  **broad plateau, not a sharp peak**. Fold 0's sweep reads ×200 +0.437, ×255 +0.469,
  ×300 +0.456, ×500 +0.469, ×1000 +0.436.
- Per fold it chose ×255 for folds 0–1 and ×300 for folds 2–3. Only fold 0's full sweep line
  survives. Fold 3's selected Pearson (+0.4212) is assigned by elimination.

**Swiss zero-shot on DFC2019** (`diag_swiss_zeroshot_per_fold.py`, run 2026-09-22 19:37 UTC,
never previously written up). Setup: quadrant protocol, per-fold scale, **FT-2's own 25-sample
true-test halves**, raw output with no calibration. Source: `swiss_zeroshot_fold_results.csv`,
TRANSCRIBED.

| Fold | scale | MAE | RMSE | Pearson | Spearman | var. ratio | FT-2 Pearson (same samples) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | ×255 | 2.486 | 5.915 | 0.505 | 0.529 | 0.72 | 0.580 |
| 1 | ×255 | 3.127 | 6.660 | 0.482 | 0.571 | 1.02 | 0.607 |
| 2 | ×300 | 3.369 | 6.746 | 0.484 | 0.487 | 0.84 | 0.503 |
| 3 | ×300 | 3.149 | 6.364 | 0.499 | 0.579 | 0.93 | 0.593 |
| **mean of folds** | | **3.033** | **6.421** | **0.492** | **0.542** | | 0.571 |

**FT-2 improves on Swiss zero-shot on all four metrics in 4/4 folds, on identical samples.**
The "fine-tuning damages the pretrained model" branch of §10's open question is **resolved: no.**

The zero-shot variance ratio of 0.72–1.02 against FT-2's 0.19–0.23 is informative in its own
right. Fine-tuning *compressed* the output's variance while improving every accuracy metric,
consistent with the SmoothL1 + rank loss trading dispersion for error.

**Contamination.** Track1 zero-shot Pearson 0.716 vs. Swiss 0.492 is consistent with a
training-data advantage for Track1, but the comparison isn't clean:
- The protocols differ: tile-level folds vs. quadrant halves.
- Track1's advantage mixes two things: memorised tiles, and in-domain training (the same
  WorldView sensor and the same two cities).

Separating those two needs Track1 zero-shot on its 9 `Track1-test` tiles vs. its 41
`Track1-train` tiles. The per-tile predictions were never saved, so that is a fresh inference
run. **Open, low priority.** Not a training run.

**Correction to the 2026-09-23 Sentinel-2 entry** (`sentinel2/sign-flip-detector.md`, "RDAH-Net
zero-shot on Sentinel-2", Step 2). That step's correlation criterion correlated RDAH output with
**terrain** references (the DEM and ground-classified ICESat-2 photons). RDAH predicts **height
above ground**, so that half of the stop condition is **uninformative**: a working nDSM model
would also score ~0 there. The entry flagged this caveat a priori but still applied the
criterion. **The checkerboard half stands** on its own. Revisited under Phase 2 of the same
day's follow-up (sign-flip-detector.md, 2026-09-23 "(continued)" entries).

## 12. 2026-09-23 (later) — Phase 2: RDAH on Sentinel-2 closed; the checkerboard is intrinsic

_Full entry, with the pre-registered rule: `sentinel2/sign-flip-detector.md`, 2026-09-23
"(continued)", Phase 2. Script `scripts/rdah_resolution_sweep.py`; outputs
`data/dfc2019/experiments/rdah_zeroshot/resolution_sweep/`._

- **Preprocessing.** Training applies ImageNet `Normalize` to RGB only, which
  `backend/rdah/rdah_engine.py` matches. The Sentinel-2 renders are **2–4× darker** than DFC2019
  (median channel 21–39 vs. 82–157), while GF-7 training input was per-image min-max stretched.
  That's a real mismatch. The Figshare Swiss data is one 14.7 GB archive, so the Swiss depth
  convention and a positive control stayed untested.
- **Resolution sweep** (Swiss zero-shot, 8 DFC2019 Jacksonville tiles, block-averaged, DAv2
  re-run at each resolution): pooled Pearson **0.483 / 0.589 / 0.589 / 0.242** at
  0.3 / 0.6 / 1.2 / 2.4 m.
  - The pre-registered close-if-8×-below-half-of-1× test **misses by 0.0006** (ratio 0.5006).
  - RDAH is closed on Sentinel-2 through the rule's **default** clause. The curve holds to 1.2 m
    and then drops 59% in one step, which is not graceful; that reading is recorded as a
    post-hoc judgment.
- **The checkerboard is intrinsic, not a failure signature.** On in-domain DFC2019 at native
  resolution, where Pearson is 0.48, the median period-2/4 peak-to-background is 1,576 / 3,758.
  Darjeeling's scale-fixed output is the same order (6,972 / 1,592). Combined with §11's
  correction, **neither half of the earlier Sentinel-2 stop condition was valid evidence**. The
  Sentinel-2 closure now rests on the resolution cliff (DFC2019 correlation collapses by 2.4 m;
  Sentinel-2 is 10 m) and the rule's default clause.
