# Last session: Method 6 coarse-to-fine resolution transfer (2026-09-24, unattended overnight run)

Full record, with the pre-registration, every phase log and the full matrix:
`docs/method-audit/06-full-finetune-twin-head/resolution-transfer.md`.

## What I decided on my own (you were asleep), and why

- **DFC2019's native GSD is 0.3 m, not 0.5 m.** The project convention is `GSD_1X_M = 0.3`, so a 512 px quadrant covers
  153.6 m. The true-GSD pixel counts are therefore 2 m → 77 px, 3 m → 51 px, 5 m → 31 px, 8 m → 19 px.
  After padding that means **6×6 / 4×4 / 3×3 / 2×2 token grids**, much coarser than the prompt's "about 7×7 at 5 m".
- **Two protocols were run, both pre-registered before any training:**
  - **P** (primary as you specified): the image is physically resized to the smaller pixel count.
  - **R** (secondary): the same area-averaged content is resampled back to 512 px, so 37×37 tokens are kept. This matches
    the existing 0.795 → 0.708 sweep and separates "not enough information" from "not enough tokens".
- Evaluation GSDs are 0.3 / 1.2 / 2 / 2.4 / 3 / 5 m (+8 m). Every model was evaluated at every GSD, and the oracle was
  recomputed at each one. All scoring is against native AGL on the native grid.

## Results

- **Phase 0 (pre-flight): PASS, natively.** DAv2's DINOv2 backbone interpolates its 37×37 position table to any token grid
  on every forward pass. It is a real interpolation, not a crop, so no fix was needed. It runs from 37×37 down to 1×1 tokens.
- **Recipe-identity gate: PASS, bit-identical.** The new script reproduces the adopted fold-0 MAE 1.8834942542525481.
- **Sanity anchors:** native Method 6 at 0.3 m = 1.98 / 3.49 / 0.745 / 0.656, and the recomputed oracle = 3.39 / 4.58 /
  0.582 / 0.509. Both are exactly the known numbers.
- **Rule 1, "coarse-to-fine transfer works"** (at 0.3 m, beats the oracle on ≥ 3/4 metrics with non-overlapping CIs):
  - **P: FAILS at 2, 3, 5 m** (and 8 m). Not one metric beats the oracle. Pearson at 0.3 m is 0.420 / 0.203 / 0.173 (/ 0.172).
  - **R: WORKS at 2, 3, 5 m** (3/4 metrics: MAE, Pearson, Spearman). Pearson at 0.3 m is 0.721 / 0.708 / 0.687, against the
    oracle's 0.582. **It fails at 8 m** (0.631, CIs overlap).
- **Rule 2, symmetry:** **asymmetric at every GSD in both protocols, and coarse→fine is better.** For example, under R at 5 m,
  a 5 m-trained model on 0.3 m input gets 0.687, while the native model on 5 m input gets 0.321.
- **Rule 3, dose-response:** monotonic in both protocols, with no cliff at or below 5 m by the pre-registered 60% rule.
  - Under P it is front-loaded: most of the loss happens by 2 m.
  - Under R it is gentle: 0.745 → 0.721 → 0.708 → 0.687. It **bends between 5 and 8 m** (→ 0.631), which is the largest
    single step (Phase 5, descriptive).
- **Main finding:** at this scene size, what kills coarse inputs is the **token count**, not the loss of ground detail.
  With the token grid kept, 5 m-content imagery still carries a learnable height signal that transfers to sharp imagery.
- **Limits:** synthetic degradation of one sensor (WorldView-3) in one city (JAX), one seed per cell, 50 tiles.
  **This is not evidence that real 10 m Sentinel-2 supports Method 6-style height estimation.**

## Incidents

- The Phase 4 analysis crashed once on a metric-key name (`mae_m` vs `mae`). I fixed it and re-ran the analysis only;
  the raw matrix records were untouched and no number changed.
- The mean-of-tiles variance ratio isn't interpretable (near-flat quadrants make it explode). A median-of-tiles ratio is
  reported instead, labelled post-hoc.
- The regenerated summary JSON applies Holm across 4 GSDs once 8 m is included. The Phase 4 write-up keeps the
  pre-registered 3-GSD values, and no verdict changed.

## Files created / modified this session (all committed except where noted)

- **Created:**
  - `docs/method-audit/06-full-finetune-twin-head/resolution-transfer.md` (the full write-up and matrix)
  - `scripts/method6_resolution_transfer.py` (train / eval / analyze)
  - `data/dfc2019/experiments/resolution_transfer/{run_chain.sh, run_chain8.sh, chain.log, chain8.log,
    matrix_tiles.jsonl, matrix_summary.json}`
  - `data/dfc2019/experiments/resolution_transfer/rt_{P,R}_{2,3,5,8}m_seed42/train_results.json`
  - `data/dfc2019/experiments/resolution_transfer/gate_native_fold0/train_results.json`
  - `data/dfc2019/experiments/resolution_transfer_gate.log`
  - `last_session.md` (this file)
- **Modified:** `docs/HANDOFF.md` (§2a, §3 item 0d, footer), `CLAUDE.md` (new resolution-transfer section),
  `data/REGENERATION.md`, `.gitignore`.
- **Created, not committed (gitignored):** `.claude/settings.local.json`, which holds the allow rule for the final shutdown command.
- **Not committed (large, regenerable):** 33 fold checkpoints under `data/dfc2019/experiments/resolution_transfer/*/fold*.pt`
  (about 3.3 GB; recipe in `data/REGENERATION.md`).
- **Git status after the final commit:** the only tracked modifications are the 6 files that were already modified before
  these sessions (4 `*_RGB.tif`, `pyproject.toml`, `uv.lock`).

## Background jobs

None. Both training chains finished (00:45–04:11 and 04:14–05:23 IST). `caffeinate` was stopped just before the shutdown.
The machine was shut down after this file was committed, as you asked.
