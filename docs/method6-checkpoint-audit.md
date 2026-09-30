# Method 6 checkpoint audit (2026-10-01, branch `release-candidate-2026-09-30`)

- **Scope:** read-only on `data/`. Measurements come from `scripts/method6_checkpoint_audit.py` → `build/method6_audit/audit.json` (gitignored).
- **Evidence:** saved configs, logs, the metadata inside each checkpoint, and git history. File dates are never used as evidence.
- **Hardware:** Apple Silicon MPS.

## 1. The ADOPTED recipe (source of truth)

**Sources:**
- `docs/method-audit/06-full-finetune-twin-head/verdict.md`:
  - §5 (2026-09-22 ablation): "Adopt height-balanced loss/sampling … Do not adopt GSD-FiLM";
  - §6 (C1–C3, 3 seeds: 1.990 ± 0.010 / 3.504 ± 0.026 / 0.743 ± 0.002 / 0.656 ± 0.0003).
- **Training script used for the 3 seeds:** `scripts/evaluate_method6_gsd_film_height_balanced.py`. It imports the
  architecture, losses, data and schedule from `scripts/evaluate_method6_finetune_twinhead.py`.
- **The exact command** (`scripts/run_method6_seeds.sh`):
  `--enable-height-balanced --epochs 12 --seed <42|43|44> --save-checkpoints`, with all other options at their
  defaults.

| Parameter | Adopted value |
|---|---|
| Base weights | `depth-anything/Depth-Anything-V2-Small-hf` via `AutoModelForDepthEstimation.from_pretrained`, **unpinned**. The HF repo's current revision is `5426e4f0f365…`, last modified **2024-07-05**, before any training here (2026-09-22/23), so every checkpoint started from this revision. Apache-2.0, 24.8 M parameters |
| Architecture | DAv2-Small backbone + neck + head conv1/conv2 + **twin head**: `conv_mu` (the pretrained 32 → 1 readout) × `height_scale`, and a fresh `conv_log_var` (1 × 1). No GSD-FiLM (`enable_gsd_film=False`) |
| Twin-head output | `mu` = AGL in metres = `conv_mu(x) × height_scale`; `log_var` = `conv_log_var(x) + 2·ln(height_scale)`, clamped to [−8, 7]. `log_var` is initialised to a uniform σ = 5 m |
| `height_scale` | p95 of the training AGL pixels, computed per fold (≈ 16.2–16.6 m) |
| Input and normalisation | one sample = one whole **512 × 512 quadrant** of a DFC2019 tile, RGB / 255, **ImageNet mean/std**, **reflect-padded to 518** (37 × 14). Prediction is cropped back to 512 |
| Loss | 30 steps of masked Huber (`--warmup-steps 30`, head only), then **Gaussian NLL** (mask-then-compute) **+ 0.35 × capped height-weighted Huber** (α = 0.08, max weight 4.0) |
| Sampler | `WeightedRandomSampler` over training quadrants, weight = 1 + 3·frac(AGL ≥ 10 m) + 2·frac(4 ≤ AGL ≤ 25 m), with replacement (logged: weights 1.00–3.67, mean 2.00) |
| Optimiser | AdamW, two groups: backbone LR **5e-6**, head (neck + convs) LR **2.5e-4** (50×), weight decay 0.01, gradient clip 1.0 |
| Schedule | linear warmup over max(10, 5% of steps), then linear decay to a 0.05 floor |
| Epochs and selection | **fixed 12 epochs; the FINAL weights are saved. No selection by validation** |
| Batch | 2 |
| Augmentation | **none** |
| Folds | 4-fold spatial-quadrant holdout: fold q trains on the other 3 quadrants of all 50 tiles (150) and evaluates on quadrant q (50) |
| Seeds | 42 (the 1.980 headline), 43, 44 |

**Expected accuracy** is the **3-seed cross-validation** result: **MAE 1.990 ± 0.010, RMSE 3.504 ± 0.026, Pearson
0.743 ± 0.002, Spearman 0.656 ± 0.0003** (mean of 4 fold means per seed).

## 2. Per checkpoint (17 local files)

**Provenance:**
- **Adopted-recipe fold checkpoints** (seeds 42 / 43 / 44): script `scripts/evaluate_method6_gsd_film_height_balanced.py`
  (added in `4e1689d`, 2026-09-22), with `--save-checkpoints` added in `9977b14` (2026-09-23).
  - Seeds 43 and 44 ran via `scripts/run_method6_seeds.sh`.
  - Seed 42 ran with "the identical command plus `--save-checkpoints`", and reproduces the headline **bit-for-bit**:
    MAE 1.979971505587631 in both results JSONs (07 `log.md`, 2026-09-23).
- **GAMUS-DC folds:** the same script with `--extra-train-npz data/gamus_dc/train_quadrants.npz`, an option added in
  `2d896f0`.
- **Full model:** `scripts/train_method6_full_dfc2019.py`. It was **untracked when it ran** (2026-09-22) and first
  committed in `99f0d29`. Its `meta.json` (50 tiles, 200 samples, 12 epochs, `height_scale` 16.427) matches the
  checkpoint's saved `height_scale`.

**Checks:**
- **"Strict load":** `load_state_dict(strict=True)` into the adopted architecture, with 0 missing and 0 unexpected
  keys.
- **The 3-tile check:** uses tiles `JAX_004_006`, the middle tile and the last tile of the 50-tile list, and each
  fold model's **held-out** quadrant. The metrics are pooled over the 3 tiles.

| # | checkpoint | what | `height_scale` / seed / fold (saved in the file) | strict load | 3-tile check: range (m) · var ratio · Pearson · MAE | SHA-256 (first 16) | in HF repo |
|---|---|---|---|---|---|---|---|
| 1 | `method6_height_balanced_seed42_ckpt/fold0.pt` | adopted recipe, seed 42 (= the 1.980 headline, re-run bit-for-bit), fold 0 (held-out quadrant 0) | 16.62 / 42 / 0 | ok, 24.79 M params, no GSD-FiLM | -2.8 … 21.1 · 0.63 · 0.806 · 1.78 (held-out) | `0e48066a83a057fe` | no |
| 2 | `method6_height_balanced_seed42_ckpt/fold1.pt` | adopted recipe, seed 42 (= the 1.980 headline, re-run bit-for-bit), fold 1 (held-out quadrant 1) | 16.40 / 42 / 1 | ok, 24.79 M params, no GSD-FiLM | -0.3 … 28.4 · 0.80 · 0.801 · 1.83 (held-out) | `608876eea2eba913` | no |
| 3 | `method6_height_balanced_seed42_ckpt/fold2.pt` | adopted recipe, seed 42 (= the 1.980 headline, re-run bit-for-bit), fold 2 (held-out quadrant 2) | 16.25 / 42 / 2 | ok, 24.79 M params, no GSD-FiLM | -1.6 … 21.8 · 0.56 · 0.836 · 2.10 (held-out) | `e9aa240ba3c3dd7f` | no |
| 4 | `method6_height_balanced_seed42_ckpt/fold3.pt` | adopted recipe, seed 42 (= the 1.980 headline, re-run bit-for-bit), fold 3 (held-out quadrant 3) | 16.42 / 42 / 3 | ok, 24.79 M params, no GSD-FiLM | -2.2 … 24.9 · 0.78 · 0.775 · 2.02 (held-out) | `b4d5017f01f99a12` | no |
| 5 | `method6_height_balanced_seed43/fold0.pt` | adopted recipe, seed 43, fold 0 (held-out quadrant 0) | 16.62 / 43 / 0 | ok, 24.79 M params, no GSD-FiLM | -0.9 … 20.4 · 0.58 · 0.811 · 1.74 (held-out) | `1cecee4823db8a58` | yes, SHA-256 matches |
| 6 | `method6_height_balanced_seed43/fold1.pt` | adopted recipe, seed 43, fold 1 (held-out quadrant 1) | 16.40 / 43 / 1 | ok, 24.79 M params, no GSD-FiLM | -0.7 … 26.4 · 0.89 · 0.791 · 1.92 (held-out) | `1b81d3d7bf514607` | yes, SHA-256 matches |
| 7 | `method6_height_balanced_seed43/fold2.pt` | adopted recipe, seed 43, fold 2 (held-out quadrant 2) | 16.25 / 43 / 2 | ok, 24.79 M params, no GSD-FiLM | -1.5 … 22.9 · 0.57 · 0.854 · 1.93 (held-out) | `bce363f2994629e1` | yes, SHA-256 matches |
| 8 | `method6_height_balanced_seed43/fold3.pt` | adopted recipe, seed 43, fold 3 (held-out quadrant 3) | 16.42 / 43 / 3 | ok, 24.79 M params, no GSD-FiLM | -1.8 … 27.1 · 0.76 · 0.762 · 2.08 (held-out) | `50556adb1b3ed60c` | yes, SHA-256 matches |
| 9 | `method6_height_balanced_seed44/fold0.pt` | adopted recipe, seed 44, fold 0 (held-out quadrant 0) | 16.62 / 44 / 0 | ok, 24.79 M params, no GSD-FiLM | -0.7 … 20.8 · 0.61 · 0.809 · 1.77 (held-out) | `5eb917112bea7966` | no |
| 10 | `method6_height_balanced_seed44/fold1.pt` | adopted recipe, seed 44, fold 1 (held-out quadrant 1) | 16.40 / 44 / 1 | ok, 24.79 M params, no GSD-FiLM | -0.8 … 27.9 · 0.82 · 0.783 · 1.94 (held-out) | `e976a86d123c6eb0` | no |
| 11 | `method6_height_balanced_seed44/fold2.pt` | adopted recipe, seed 44, fold 2 (held-out quadrant 2) | 16.25 / 44 / 2 | ok, 24.79 M params, no GSD-FiLM | -1.2 … 23.8 · 0.59 · 0.844 · 2.01 (held-out) | `53c2edfc961d3078` | no |
| 12 | `method6_height_balanced_seed44/fold3.pt` | adopted recipe, seed 44, fold 3 (held-out quadrant 3) | 16.42 / 44 / 3 | ok, 24.79 M params, no GSD-FiLM | -0.6 … 24.2 · 0.73 · 0.763 · 2.06 (held-out) | `d1af9d82967e674f` | no |
| 13 | `method6_hb_gamusdc_seed43/fold0.pt` | adopted recipe + 300 GAMUS-DC quadrants, seed 43, fold 0 (held-out quadrant 0) | 24.66 / 43 / 0 | ok, 24.79 M params, no GSD-FiLM | -0.8 … 20.2 · 0.63 · 0.789 · 1.74 (held-out) | `8557e12af819936c` | no |
| 14 | `method6_hb_gamusdc_seed43/fold1.pt` | adopted recipe + 300 GAMUS-DC quadrants, seed 43, fold 1 (held-out quadrant 1) | 24.45 / 43 / 1 | ok, 24.79 M params, no GSD-FiLM | -0.2 … 27.0 · 0.83 · 0.795 · 1.85 (held-out) | `c7d82c0a3d576a95` | no |
| 15 | `method6_hb_gamusdc_seed43/fold2.pt` | adopted recipe + 300 GAMUS-DC quadrants, seed 43, fold 2 (held-out quadrant 2) | 24.56 / 43 / 2 | ok, 24.79 M params, no GSD-FiLM | -1.3 … 22.1 · 0.52 · 0.844 · 1.94 (held-out) | `3c40dbeeaeea4d26` | no |
| 16 | `method6_hb_gamusdc_seed43/fold3.pt` | adopted recipe + 300 GAMUS-DC quadrants, seed 43, fold 3 (held-out quadrant 3) | 24.65 / 43 / 3 | ok, 24.79 M params, no GSD-FiLM | -0.5 … 23.7 · 0.77 · 0.755 · 2.08 (held-out) | `c272d78d01becbde` | no |
| 17 | `method6_full_checkpoint/method6_full_dfc2019.pt` | FULL model: all 50 tiles, no holdout | 16.43 / — / — | ok, 24.79 M params, no GSD-FiLM | -0.7 … 20.5 · 0.64 · 0.869 · 1.46 (**in-sample sanity only, NOT an accuracy estimate**) | `24d69822e048151b` | yes, SHA-256 matches |

Full model vs each adopted-recipe fold model, on that fold's held-out quadrants of all 50 tiles:

| fold model | Pearson (full vs fold) | MAE between predictions (m) | mean(full − fold) (m) |
|---|---|---|---|
| hb_seed42/fold0 | 0.895 | 1.114 | -0.071 |
| hb_seed42/fold1 | 0.922 | 1.233 | +0.086 |
| hb_seed42/fold2 | 0.915 | 1.200 | +0.235 |
| hb_seed42/fold3 | 0.895 | 1.230 | -0.050 |
| hb_seed43/fold0 | 0.898 | 1.126 | -0.023 |
| hb_seed43/fold1 | 0.930 | 1.178 | +0.046 |
| hb_seed43/fold2 | 0.919 | 1.203 | +0.383 |
| hb_seed43/fold3 | 0.890 | 1.343 | -0.114 |
| hb_seed44/fold0 | 0.894 | 1.120 | -0.020 |
| hb_seed44/fold1 | 0.917 | 1.280 | +0.010 |
| hb_seed44/fold2 | 0.916 | 1.201 | +0.226 |
| hb_seed44/fold3 | 0.880 | 1.328 | +0.009 |

| reference: seed 42 vs seed 43, same fold | 0.984 / 0.979 / 0.978 / 0.982 | 0.502 / 0.655 / 0.604 / 0.585 | |

HF repo sancharimouri/depthwizard2-method6 revision 676f908c746072723ee943b1a9ecbd011380d653
- `full_dfc2019/method6_full_dfc2019.pt` ↔ `data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt`: HF `24d69822e048151b…` vs local `24d69822e048151b…` → match
- `height_balanced_seed43/fold0.pt` ↔ `data/dfc2019/experiments/method6_height_balanced_seed43/fold0.pt`: HF `1cecee4823db8a58…` vs local `1cecee4823db8a58…` → match
- `height_balanced_seed43/fold1.pt` ↔ `data/dfc2019/experiments/method6_height_balanced_seed43/fold1.pt`: HF `1b81d3d7bf514607…` vs local `1b81d3d7bf514607…` → match
- `height_balanced_seed43/fold2.pt` ↔ `data/dfc2019/experiments/method6_height_balanced_seed43/fold2.pt`: HF `bce363f2994629e1…` vs local `bce363f2994629e1…` → match
- `height_balanced_seed43/fold3.pt` ↔ `data/dfc2019/experiments/method6_height_balanced_seed43/fold3.pt`: HF `50556adb1b3ed60c…` vs local `50556adb1b3ed60c…` → match

full sha256:
- `hb_seed42/fold0`: `0e48066a83a057fe932808fd676a6d3135996735fe6b8e3329912e092dcee9c6`
- `hb_seed42/fold1`: `608876eea2eba913066be2884234a397ca91c0f4f9a61e274ddf66b17dc96318`
- `hb_seed42/fold2`: `e9aa240ba3c3dd7fe6f426730d36e73eb7bc298f8a66da9e6f19d75f5fd666ce`
- `hb_seed42/fold3`: `b4d5017f01f99a12ad47cc7609f499bf40b1b62d5a32fa9193d4e40fe102f032`
- `hb_seed43/fold0`: `1cecee4823db8a584961969e03266f77f115552a73071386e259bdb6a327ee65`
- `hb_seed43/fold1`: `1b81d3d7bf514607510607b279643355a5db93e7d63ce139532610df7480a50b`
- `hb_seed43/fold2`: `bce363f2994629e1b6f3b5cf261c59f88696d800967484fef66ee210094efe41`
- `hb_seed43/fold3`: `50556adb1b3ed60c3257a8aa04cc2dc4568f363f4b20a0fdc31acb17e3e18362`
- `hb_seed44/fold0`: `5eb917112bea7966c0bd438578305bd1ce723854ae306a0162f93fb2cc8f3627`
- `hb_seed44/fold1`: `e976a86d123c6eb099b63c9171722c668e7168a0715936df5de60822e5123b49`
- `hb_seed44/fold2`: `53c2edfc961d307887cb0c2aa384620491cab1b6b2c0c5286980503965fce7cb`
- `hb_seed44/fold3`: `d1af9d82967e674f15a00372a14e53ebf4e24994c27aee9995b7c7fa68347fdb`
- `hb_gamusdc_seed43/fold0`: `8557e12af819936c3ca04596afe17c55c223629b1604e07e57a357e347ae4c55`
- `hb_gamusdc_seed43/fold1`: `c7d82c0a3d576a956f5e82558acdbc7b0e89808d551bcfb89267f88dcfdb708d`
- `hb_gamusdc_seed43/fold2`: `3c40dbeeaeea4d266831f5b3a2d914ccbc5a7a04703f7189a902c58dc920c8d2`
- `hb_gamusdc_seed43/fold3`: `c272d78d01becbde92233c71429005a735b3c32932a4e61e53ede9552446a5a5`
- `full_dfc2019`: `24d69822e048151b31ea832f0c862c89e5f296841a9dfa235f8779038894ab7f`

**Parameter-by-parameter comparison against the adopted recipe.** Sources: the results-JSON `config`, `train.log` and
the saved checkpoint metadata. ✓ = matches.

| Parameter | seed 42 / 43 / 44 folds (12) | GAMUS-DC folds (4) | full model (1) |
|---|---|---|---|
| Base weights / architecture / twin head | ✓ (strict load) | ✓ | ✓ |
| Height-balanced **loss** (+0.35 × capped HW-Huber) | ✓ `enable_height_balanced: true` | ✓ | ✗ **MISMATCH: absent** (NLL only) |
| Height-balanced **sampler** | ✓ (logged "height-balanced sampler, weight range [1.00, 3.67]") | ✓ | ✗ **MISMATCH: plain shuffle** |
| Huber warmup steps | ✓ 30 (default, not overridden) | ✓ 30 | ✗ **MISMATCH: 40** (`WARMUP_STEPS = 40`) |
| Epochs 12, final weights | ✓ (all 4 folds logged "epoch 12/12"; 0 non-finite aborts, 0 budget stops) | ✓ | ✓ fixed 12 |
| LR 5e-6 / 2.5e-4, wd 0.01, batch 2, clip 1.0, schedule | ✓ | ✓ | ✓ |
| Input 512 quadrant → reflect-pad 518, ImageNet norm, no augmentation | ✓ | ✓ | ✓ |
| `height_scale` = training p95 | ✓ 16.25–16.62 | ✓ by rule, but **24.45–24.66** (GAMUS pixels included) | ✓ 16.43 (all 200 quadrants) |
| Training data | ✓ 150 DFC2019 quadrants per fold | ✗ **+300 GAMUS-DC quadrants** (by design) | 200 DFC2019 quadrants: all tiles, **no holdout** |
| Seed | ✓ 42 / 43 / 44 saved in the file | 43 | not saved (script default 42) |

## 3. Verdicts

| Checkpoint(s) | Verdict | Why |
|---|---|---|
| `method6_height_balanced_seed42_ckpt/fold0–3` | **production-valid** (as the 4-fold ensemble) | the exact adopted recipe; the headline model, reproduced bit-for-bit |
| `method6_height_balanced_seed43/fold0–3` (also in the private HF repo, SHA-256 match) | **production-valid** (as the 4-fold ensemble) | the exact adopted recipe |
| `method6_height_balanced_seed44/fold0–3` | **production-valid** (as the 4-fold ensemble) | the exact adopted recipe |
| `method6_hb_gamusdc_seed43/fold0–3` | **valid for evaluation only** | adopted recipe + 300 GAMUS-DC quadrants; **not adopted**. It failed its own pre-registered criterion (ii): 20–30 m trees predicted at 14.1 m vs. an 18 m target. Single seed |
| `method6_full_checkpoint/method6_full_dfc2019.pt` (also in the HF repo, SHA-256 match) | **mismatch** | the **pre-adoption** recipe (2026-09-22): no height-balanced loss, no height-balanced sampler, warmup 40 vs. 30 |

**"Production-valid" means:**
- A per-seed average of its 4 fold models. Each fold model saw 3 of the 4 quadrants of every tile, so the ensemble has
  seen all 50 tiles.
- This is how the project's own GAMUS (07 Part B) and VHR pipelines already use Method 6 ("per seed, the 4 fold
  models are averaged").
- Its expected accuracy is the 3-seed CV result above.
- Method 6 is the **DFC2019 research best**, not a product model. It does not generalize to GAMUS (07 Part B) or to
  10 m Sentinel-2.

### The full model specifically

- **Trained on all 50 tiles?** Yes: 50 tiles, 200 quadrants, no holdout (`meta.json`, `n_tiles: 50, n_samples: 200`).
- **A fixed epoch count, and where it came from?** Yes, a fixed 12 (`EPOCHS = 12`), taken from the validated fold
  recipe's default (`--epochs 12`), whose fold models also keep their final weights. No validation-based selection
  was possible or used.
  - **Caveat:** 200 quadrants vs. 150 per fold means about 1.33× more optimiser steps over the same 12 epochs. The
    LR schedule scales with total steps.
- **The same recipe as the 3-seed runs?** **No.** It is the pre-adoption recipe: no height-balanced loss or sampler,
  and warmup 40. The 07 GAMUS pre-registration had already excluded it for this reason ("its recipe differs").
- **Agreement with the fold models on their held-out quadrants** (all 50 tiles, each of the 12 adopted-recipe fold
  models):
  - **Pearson 0.880–0.930, MAE between predictions 1.11–1.34 m**, mean offset −0.11 to +0.38 m.
  - For reference, two adopted-recipe seeds on the same held-out quadrants agree at **Pearson 0.978–0.984, MAE-between
    0.50–0.66 m**.
  - So the full model disagrees with the adopted models about **twice as much** as seeds disagree with each other.
  - This is consistent with its recipe difference, but **confounded**: it has also *trained* on those quadrants.
  - Its own DFC2019 "accuracy" is in-sample and is **not** reported as an estimate. The 3-tile row above is a sanity
    check only.

**Retrain plan (NOT run).** A full model with the adopted recipe:
1. **Code:** add a `--train-all` option to `scripts/evaluate_method6_gsd_film_height_balanced.py`: one "fold" with all
   200 quadrants, no evaluation loop, save the final weights. That reuses the adopted loss, sampler, warmup 30 and
   schedule unchanged, instead of patching the older `train_method6_full_dfc2019.py`.
2. **Command:**
   `python scripts/evaluate_method6_gsd_film_height_balanced.py --outdir data/dfc2019/experiments/method6_hb_full_seed42 --tag m6_hb_full_seed42 --enable-height-balanced --epochs 12 --seed 42 --save-checkpoints --train-all`
3. **Epoch count:** fixed **12**, the recipe's number. Owner's choice: 9 epochs would match the folds' total step
   count (≈ 75 × 12 = 900 steps). Either way it is fixed in advance, with no validation selection.
4. **Wall time:**
   - The adopted fold runs took 3,733 s for 4 folds (≈ 933 s per fold at 150 quadrants, including evaluation).
   - The old full run took 1,029 s at 200 quadrants.
   - **Estimate ≈ 20–25 min on this Mac's MPS** for one seed. Three seeds ≈ 1–1.3 h.
5. **Acceptance, fixed in advance:** strict-load, sane output, and agreement with the adopted fold models at least as
   close as the old full model's. No DFC2019 accuracy claim, since it is in-sample.

   **Alternative that needs no retraining:** ship the seed-42 (or seed-43) **4-fold ensemble**, which is already
   production-valid.

## 4. The private HF repo (`sancharimouri/depthwizard2-method6`)

It holds 5 of the 17 checkpoints. Each HF file's LFS SHA-256 comes from the repo metadata (read-only, no download)
and was compared with the local file:
- `full_dfc2019/method6_full_dfc2019.pt` = the local full model (**mismatch** recipe; the SHA-256 matches);
- `height_balanced_seed43/fold0–3.pt` = the local seed-43 folds (**production-valid**; SHA-256 matches, 4/4).

**All 5 HF copies are byte-identical to the local files.** The repo has neither seed 42 (the headline) nor seed 44
nor the GAMUS-DC folds.

## 5. GAMUS-DC (the 4 `method6_hb_gamusdc_seed43` checkpoints)

- **What GAMUS-DC is:**
  - the **Washington DC** city of GAMUS (`earthflow/GAMUS` on HF: aerial RGB + nDSM, ≈ 0.33 m, CC-BY-4.0), the
    SIH brief's recommended dataset;
  - the HF release has 3 cities (DC, NYC, PHL); the paper's JAX and OMA tiles *were* DFC2019 tiles and were removed
    upstream.
- **Recipe:** the adopted seed-43 height-balanced recipe, unchanged, plus the 300 GAMUS-DC quadrants appended to each
  fold's training set (`--extra-train-npz`). `height_scale` rose to 24.5–24.7 m. Confounds: about 3× more steps per
  epoch; GSD not resampled.
- **Training data:** the DFC2019 fold's 150 quadrants plus 300 GAMUS-DC quadrants.
  - The GAMUS quadrants come from a **leakage-safe split** (`scripts/prepare_gamus_dc.py`): test = DC grid rows ≤ 14
    (163 tiles), rows 15–16 dropped as a buffer, train drawn from rows ≥ 17.
  - Checks: minimum grid distance 3; 0 of 166,892 test 32 px blocks shared.
- **Evaluation:** on each DFC2019 held-out quadrant, plus GAMUS-DC's held-out test (descriptive).
- **Where it's documented:** `docs/method-audit/06-full-finetune-twin-head/vhr_dsm_pipeline.md` ("Retrain:
  Method 6 + GAMUS-DC", pre-registration and results); the results JSON is
  `data/dfc2019/experiments/method6_hb_gamusdc_seed43/m6_hb_gamusdc_seed43_results.json`.
- **Results:**
  - DFC2019: **1.920 / 3.425 / 0.744 / 0.652**, all 4 folds better than seed 43's 1.989. **Criterion (i) PASS.**
  - 20–30 m DFC2019 trees at **14.1 m** vs. the 18 m target. **Criterion (ii) FAIL.**
  - GAMUS-DC test MAE 3.04–3.07 m. Tall GAMUS trees *were* learned (20–30 m predicted at 21–22 m), but this did not
    transfer to Jacksonville's trees.
  - The VHR forest ceiling didn't rise.
  - Explanation: **GAMUS imagery is leaf-off** (tree-greenness check). **Not adopted.**
- **The earlier GAMUS plan (07, Parts A–C): all ran and are documented** in
  `docs/method-audit/07-gamus-generalization/{summary,log,verdict}.md`:
  - **A (acquisition and leakage):** the HF GAMUS is the 3-city version, 8,724 tiles, 80 GB. **Leakage against
    Method 6's training set: 0** (by provenance, and confirmed by byte hash on all 2,861 test tiles).
  - **B (zero-shot, 2,861 test tiles, 3 seeds × 4 folds): pre-registered verdict "DOES NOT GENERALIZE".**
    - Method 6 3.130 / 4.583 / 0.638 / 0.583 vs. the oracle's 3.474 / **4.426** / 0.491 / 0.425.
    - It wins MAE and ranking but **loses RMSE in all 3 cities for every seed**: it compresses tall objects (pooled
      variance ratio 0.305).
  - **C (fine-tune on GAMUS): pre-registered STOP at C.0.** No GAMUS city is leaf-on (green share DC 0.06, NYC 0.03,
    PHL 0.32 vs. a 0.40 threshold), so the fine-tune was not run.

    The later GAMUS-DC retrain above was separately authorised by the user on 2026-09-23, and tested whether DC's
    tall canopy helps anyway. It didn't transfer.

## 6. ONNX export feasibility (Part 3): **skipped**

The full model is **not production-valid** (§3: recipe mismatch), so it was not exported, per the instruction.

**Implications:**
- The production-valid Method 6 is a 4-fold ensemble, i.e. **4 ONNX sessions** (or 4 sequential passes of one
  session with swapped weights): 4 × 99 MB, and about 4× the inference time of a single model.
- A retrained adopted-recipe full model (§3 plan) would be **one** ~99 MB session.
- The model-service estimate in `docs/release-candidate-2026-09-30.md` Part 7 is unchanged: DAv2-Small measured 214
  MiB idle / 354 MiB peak (arena off); + Method 6 ≈ 500 MiB estimated, plan 1 GiB. With a 4-fold ensemble loaded at
  once, add ≈ 3 × 146 MiB idle, **≈ 800 MiB**, so plan **2 GiB** or load folds sequentially.
- **Owner decision:** retrain the full model (≈ 25 min), or accept an ensemble.

---

## 7. Retrain: full model with the adopted recipe (2026-10-01)

**Owner decisions:**
- retrain ONE full model with the adopted recipe, seed 42, instead of shipping an ensemble;
- in the private HF repo, upload the new model and archive the old mismatched one (don't delete it); don't upload
  the seed-42 folds;
- wording: "validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS by RMSE."

### 7.1 `--train-on-all` (Part 1)

- **Where:** `scripts/evaluate_method6_gsd_film_height_balanced.py`, the adopted 3-seed script. **The only
  behavioural change is the data split.**
  - With `--train-on-all`, the existing fold loop runs **once** with no held-out quadrant (`held_out_q = None`), so
    all 4 quadrants of all 50 tiles (200) are training samples.
  - The evaluation block is skipped, since there is nothing held out.
  - The final weights are saved as `method6_full_dfc2019_hb_seed<seed>.pt`, refusing to overwrite an existing file.
- **Guards:**
  - It asserts that the DAv2-Small snapshot actually loaded is revision `5426e4f0f365…` (`BASE_REVISION`).
  - It refuses `--extra-train-npz` / `--gamus-test-npz`.
- **Refactors that don't change behaviour:**
  - `build_parser()`, so the config is testable;
  - `resolve_args()`, the existing 50× head-LR rule, moved into a function;
  - `per_tile / fold_agg / fold_diag` default to empty when nothing is held out;
  - the results JSON also records `warmup_steps`, `train_on_all` and `checkpoint`.
- **Unchanged, the same code path as every fold:**
  - the height-balanced loss (+0.35 × capped HW-Huber) and the weighted quadrant sampler;
  - 12 fixed epochs and the final weights;
  - the 30-step Huber warm-up then Gaussian NLL;
  - AdamW 5e-6 / 2.5e-4, wd 0.01, clip 1.0, the warmup + linear-decay schedule, batch 2;
  - `height_scale` = training p95;
  - 512 px quadrants reflect-padded to 518, ImageNet normalisation, no augmentation;
  - seed 42.
- **Test:** `backend/tests/test_method6_recipe.py`, 3 tests:
  - the `--train-on-all` config resolves to every adopted value;
  - compared with the command that produced the seed-42 folds, **only `train_on_all` (and `save_checkpoints`, which
    train-on-all forces) differ**;
  - the fixed constants: model id, revision, pad, patch, ImageNet stats, height thresholds, loss weights, and the
    sampler formula.

### 7.2 PRE-REGISTRATION (committed before training; nothing has been run)

- **Command** (background, `nohup`, logged):
  `python -u scripts/evaluate_method6_gsd_film_height_balanced.py --outdir data/dfc2019/experiments/method6_full_checkpoint --tag m6_hb_full_seed42 --enable-height-balanced --epochs 12 --seed 42 --train-on-all`
  → `data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019_hb_seed42.pt`. That is a new name next to
  the old `method6_full_dfc2019.pt`, which is never overwritten.
- **Audit criteria** (all required):
  1. every recipe parameter matches: the config in the results JSON + `train.log` + the saved metadata;
  2. `load_state_dict(strict=True)` into the adopted architecture;
  3. sane output on the 3 audit tiles:
     - all finite;
     - prediction range within [−5, 60] m;
     - pooled variance ratio in [0.4, 1.2];
     - these are in-sample and are a sanity check only.
- **Agreement rule, against the four seed-42 fold models on each fold's held-out quadrant** (all 50 tiles):
  - **PASS if the median of the 4 per-fold Pearson correlations (full vs. fold prediction) is ≥ 0.95, AND the mean
    absolute difference between the two predictions, averaged over the 4 folds, is ≤ 0.9 m.**
  - References: seed-to-seed 0.978–0.984 / 0.50–0.66 m; the old mismatched full model 0.880–0.930 / 1.11–1.34 m.
  - The full model has trained on those quadrants, so some extra deviation is expected.
- **Not reported:** its DFC2019 accuracy. The estimate stays the 3-seed CV result, 1.990 / 3.504 / 0.743 / 0.656.
- **If any criterion fails: STOP, report, and upload nothing.**
