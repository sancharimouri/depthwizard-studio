# RDAH-Net — Gaps & Fixes

_One issue per subsection. New computation run: reloading the pretrained checkpoint, both
freshly-downloaded HK/Swiss checkpoints, and every saved per-fold per-epoch checkpoint, running
forward-pass inference only (no retraining) to test hypotheses the prior investigation's own
scale sweep and leakage concern raised but never actually resolved._

## 1. The zero-shot input depth is fed at roughly 1/100-1/255 of its likely intended scale — the single highest-value fix found in this audit

**Problem:** `104best_model.pth` (and, confirmed independently, the HK and Swiss checkpoints)
produce near-constant, near-zero-correlation output on this project's DFC2019 probe tiles when
fed DAv2 depth normalized to `[0,1]` — the convention this project uses everywhere else and
deliberately preserves (`train_rdah_spatial_cv.py`'s `load_depth_tensor()` explicitly asserts
`[0,1]` and raises if violated). The prior investigation's own scale sweep (PDF §8) tested `×100`
and `×1000`, saw the output change substantially, and rejected both as "artificial" — but only
ever checked output mean/std, never correlation against true AGL.

**Checked:** Read `loaddata.py`'s `ToTensor` in full: RGB is explicitly divided by 255
(`self.to_tensor(image)/255`); `rel_depth` is not divided by anything. This asymmetry implies
RDAH's original relative-depth input was consumed in its native file range (most plausibly a
0-255 8-bit-style range), not `[0,1]`. Re-ran the scale sweep with Pearson/Spearman against true
AGL tracked at every scale factor (not just mean/std) on the same 4 tiles: correlation rises
from ~0.01-0.05 at `×1` to a broad peak of **+0.65 to +0.90 around ×150-300**, centered near
`×255`. Validated at full scale — all 50 tiles, the exact tile-level 4-fold protocol already
established in this project (`make_spatial_folds()`, identical 17/8/7/18 fold sizes), a single
pooled global affine fit only on each fold's held-out training tiles, frozen pretrained weights,
**zero epochs of fine-tuning**: MAE 2.231m / RMSE 4.566m / Pearson 0.716 / Spearman 0.655
(equal-weight fold average). Full detail and comparison tables: `summary.md` §1.

**Outcome:** This is the load-bearing finding of this audit. A one-line input-scale correction,
with no training at all, beats the PDF's own best 5-epoch fine-tuned epoch on every metric (MAE
2.231 vs 2.862, RMSE 4.566 vs 6.476, Pearson 0.716 vs 0.457, Spearman 0.655 vs 0.493), and lands
within noise of Method 6's Spearman despite doing zero training. It also means every fine-tuning
result already reported in the PDF was produced starting from — and continuously trained
through — this confirmed input-scale mismatch, since `load_depth_tensor()`'s `[0,1]` guard
applied throughout all 4 folds × 5 epochs. **Fix, and the clear highest-value follow-on
experiment**: re-run `train_rdah_spatial_cv.py`'s fine-tuning with the depth input scaled by
`×255` (or, better, with the scale itself included as a free/fit parameter rather than a
hardcoded guess) before feeding it to the model, on the same 4-fold protocol, and see whether
fine-tuning *from* a correctly-scaled starting point does better, worse, or converges to a
similar place as the zero-shot-plus-affine number above. Not run in this audit — this is a new
training experiment, out of scope for a read-and-verify pass, and is flagged rather than
attempted.

**Caveat, stated plainly:** the `×255` constant was chosen from an exploratory sweep touching
the same 4 probe tiles that also appear (spread across different folds) in the later 50-tile
fold validation — a mild, real risk that the constant itself benefits from some information the
fold-level affine fit did not have access to. The fold-level affine *fit* itself remains a clean
held-out procedure (fit only on each fold's training tiles, evaluated only on that fold's test
tiles) — only the outer scale constant carries this caveat. Given how broad and consistent the
peak is (every one of the 4 probe tiles independently peaks in the same 150-300 range, and all 4
held-out folds in the 50-tile validation land within a tight 0.68-0.78 Pearson band, no fold
dramatically failing), this is very unlikely to be pure overfitting to 4 tiles — but a fully
nested procedure (fit the scale constant itself only on training folds, never touching any tile
that later appears in a test fold) would close this gap completely, and is the natural
companion to actually re-running fine-tuning per the fix above.

## 2. Track1 checkpoint contamination — confirmed via checksum-verified provenance, not just file-list pattern-matching

**Problem:** `104best_model.pth` was used, unquestioned, as "the pretrained RDAH checkpoint" for
every experiment in the PDF, without confirming which of RDAH's three released, dataset-specific
checkpoints (HK / Swiss / Track1) it actually is, or whether "Track1" in this repo's file names
(`Track1-train.txt`/`Track1-test.txt`) really means DFC2019 Track 1.

**Checked:** Fetched the Figshare deposit's file listing live via its public API
(`api.figshare.com/v2/articles/31986864`): three checkpoints exist, filed under
`checkpoints-HK`/`checkpoints-Swiss`/`checkpoints-track1`, each with a distinct MD5. This
project's local `104best_model.pth` MD5 matches the `checkpoints-track1` file's MD5 exactly
(`4fdd8769d2a05aee0ed40234aeceee09`) — settled by checksum, not inferred from the filename.
Confirmed `Track1-train.txt`/`Track1-test.txt`'s naming pattern matches DFC2019 Track 1
(Jacksonville/Omaha) structurally, and diffed all 50 of this project's own benchmark tile IDs
against both lists: 41/50 (82%) are in Track1-TRAIN, the remaining 9/50 in Track1-TEST, 0/50 in
neither.

**Outcome:** Every one of this project's own 4 CV folds is 75-89% dominated by tiles the
pretrained checkpoint had already seen during its own original training — there is no fold in
the existing results that is a genuinely unseen holdout with respect to the base checkpoint's
own training history. **This does not, by itself, invalidate the fine-tuning results** — they're
still a legitimate test of whether further fine-tuning improves on a (possibly already-exposed)
starting point — but it means the PDF's zero-shot "near-constant, no signal" framing cannot be
read as "the model has never seen this kind of geography," and the notably more useful reading is
the one this audit found instead: even with 82% direct prior exposure, the checkpoint still fails
zero-shot when fed the wrong input scale (item 1), and the HK/Swiss checkpoints — with zero
DFC2019 exposure at all — fail in the exact same qualitative way (item 3 below). **Fix**: none
needed beyond documenting this clearly, which this audit now does; the practical implication is
already captured by item 1's finding, since a correctly-scaled zero-shot probe is the fairer way
to characterize what the checkpoint "already knows" than the flawed-scale zero-shot probe the PDF
used.

## 3. HK/Swiss checkpoints were never obtained or tested — now downloaded, checksummed, and tested; same failure pattern as Track1

**Problem:** The PDF exclusively used `104best_model.pth` and never obtained the other two
released checkpoints, so there was no independent check on whether Track1's zero-shot failure
was specific to that checkpoint (e.g., a training-data-specific quirk) or a shared architectural/
input-contract issue.

**Checked:** Downloaded both via the Figshare API's file-ID download URLs
(`ndownloader.figshare.com/files/63637251` HK, `.../63637254` Swiss), verified MD5 against the
Figshare API's own reported checksums — exact match for both. Ran the same zero-shot probe (raw,
un-scaled `[0,1]` input — the original convention, not the item-1 fix) on all 4 probe tiles, all
3 checkpoints. All three load cleanly (0 missing/unexpected keys) and show the same qualitative
near-constant, near-zero-correlation pattern, each centered at a different constant level
(Track1≈0.081, HK≈0.017-0.019, Swiss≈0.12-0.13). Full table: `summary.md` §3.

**Outcome:** Independent, checkpoint-agnostic confirmation that the zero-shot failure is a shared
input-contract issue (item 1), not something specific to Track1's own training data or the
concern raised in item 2. **Fix**: none needed — this closes the "is it just Track1" question
cleanly. Worth re-testing HK/Swiss with the `×255` scale fix from item 1 if RDAH is pursued
further, to see whether the fix generalizes across all three checkpoints or is Track1-specific —
not done in this audit (out of scope; the DFC2019 benchmark this project uses doesn't correspond
to HK or Swiss's own training domains, so there's no ground truth to validate against for those
two beyond the same 4 DFC2019 probe tiles already used).

## 4. RDSMNet dataset relevance — confirmed unrelated, closing an open question from the prior investigation

**Problem:** The prior investigation's assessment that the separate RDSMNet Figshare deposit
(different DOI, PAN1/PAN2 stereo structure) was "likely unrelated" was never actually verified
against RDAH's source code.

**Checked:** `grep -rniE "RDSM|PAN1|PAN2|stereo" external/RDAH-Net/*.py
external/RDAH-Net/README.md external/RDAH-Net/requirements.txt` — zero matches across every
`.py`, the README, and requirements file in the RDAH repo.

**Outcome:** Confirmed, not just assumed. RDSMNet's dataset is not referenced anywhere in RDAH's
actual training, testing, or data-loading code. **Fix**: none needed; no further action on this
thread.

## 5. uint16 target-cast bug — confirmed inherited from the official repo, and confirmed to be the only silent-corruption pattern of its kind in these files

**Problem:** It was unclear whether the uint16 target-casting bug this project's own adapter
script hit and fixed was an artifact of adapting RDAH to a new dataset, or a bug already present
in RDAH's own official code — and whether other similar silent-corruption patterns existed
elsewhere in the same files that nobody had read carefully.

**Checked:** `loaddata.py` line 37 (official, unmodified upstream file):
`depth = (depth).astype(np.uint16)`, applied to the target height channel — the identical pattern
this project's own script independently reproduced and then fixed. Read `nyu_transform.py` (468
lines) and `loaddata.py` (138 lines) in full for any other narrow/unsigned cast applied to a
signed or wide-range physical quantity. Found none — the augmentation/transform classes operate
on already-loaded tensors, and the only other numeric-range asymmetry (`rel_depth` not divided
by 255 in `ToTensor`) is a scale *convention* difference (item 1), not a wraparound/overflow
corruption.

**Outcome:** Confirmed inherited, not independently introduced — correcting the item's own
framing from "presumed inherited" to "confirmed inherited." No other corruption pattern found in
either file. **Fix**: none needed beyond what this project's own adapter script already did
(preserve signed float32 AGL, don't cast to uint16) — already correct and already in place.

## 6. Fold protocol mismatch with Methods 4/6 — confirmed, and a concrete architectural path to a fair comparison identified but not executed

**Problem:** Section 20's cross-method comparison table places RDAH's tile-level 4-fold results
next to Methods 4 and 6's within-tile quadrant-level 4-fold results as if they were the same
protocol. They are not.

**Checked:** Read `tile_coords()`/`make_spatial_folds()` in `train_rdah_spatial_cv.py`: RDAH's
folds are geographic groups of whole 1024×1024 tiles; Methods 4/6 split each tile into 4
in-tile quadrants and hold out one quadrant per fold across all 50 tiles. Genuinely different
spatial-holdout unit. Read `HeightPredTransformer`'s architecture for hard-coded spatial
dependencies that would block running the model at quadrant size (512×512): the network is fully
convolutional except a fixed 64×64 sinusoidal `PositionalEncoding` buffer, whose `forward()`
crops via slicing (won't crash on a smaller input, but would silently use an unscaled corner-crop
of the table rather than a properly-sized one) and a block-attention module requiring the
bottleneck to be divisible by 8 (512's bottleneck of 32 satisfies this cleanly). Full detail:
`summary.md` §6.

**Outcome:** Confirmed architecturally feasible via a small, well-understood, two-part fix
(reinstantiate `PositionalEncoding` at `H=32, W=32`; load the rest of the pretrained checkpoint
with `strict=False` for that one buffer key, since it's a deterministic function, not a learned
weight). DFC2019's 1024×1024 tiles crop into 512×512 quadrants with no padding needed. **Fix, not
executed in this audit**: implement the resized `PositionalEncoding` and re-run fine-tuning under
the same `quadrant_bounds()` protocol Methods 4/6 use, ideally *starting from the item-1 scale
fix* rather than the current flawed-scale convention — this is now a two-part follow-on
experiment (scale fix + protocol fix), flagged as the top priority if RDAH work continues, not
attempted here since both are genuinely new training runs beyond this audit's read-and-verify
scope.

## 7. Checkpoint-selection leakage — ran a real nested validation instead of leaving it as an acknowledged-but-unresolved gap

**Problem:** The PDF's own §23 correctly identifies that selecting a "best" epoch after looking
at all 5 epochs' held-out test performance is a form of leakage, but stops at flagging it rather
than resolving it.

**Checked:** Built a genuine nested procedure using the checkpoints already saved on disk (no
retraining): split each fold's test tiles into an inner-validation half (epoch selection only)
and a true held-out half (final reporting only, never used for selection). Full per-fold table:
`summary.md` §7.

**Outcome:** The leakage concern, while valid in principle, turns out not to meaningfully change
the already-reported numbers in practice — nested-selected performance is within ~0.6% MAE of the
PDF's naive "always epoch 5," and in one fold (fold 1) the honest nested procedure actually
outperforms the naive convention by correctly stopping before that fold's post-epoch-3 decline.
This is a genuinely clean result, checked rather than assumed — consistent with how this project
elsewhere prefers a confirmed-clean negative over a forced finding. **Fix**: none needed for the
*existing* checkpoints; the honest recommendation, if a locked benchmark number is needed, is to
report the nested-selected numbers (fold 0: epoch 5; fold 1: epoch 2; fold 2: epoch 5; fold 3:
epoch 4) rather than a blanket "epoch 5 for every fold," since they're both unbiased and, in
aggregate, marginally better. This closes the item, but only for the current, input-scale-flawed
checkpoints — a future scale-corrected run would need its own nested check, not reuse this one.

## 8. Variance-compression check — confirmed, and RDAH's fine-tuned predictions are more compressed than Method 4's CNN, not less

**Problem:** The PDF observes Pearson/Spearman peaking around epoch 3-4 then softly declining in
most folds while MAE/RMSE keep improving through epoch 5, without diagnosing the mechanism —
worth checking against the same regression-to-the-mean signature already found in Method 4.

**Checked:** Reused Method 4's own diagnostic technique on fold 0's epoch 3 and epoch 5
checkpoints (forward-pass inference only, no retraining): variance ratio (pred/true) and OLS
slope of prediction-vs-truth, pooled over all 17 fold-0 test tiles. Full numbers: `summary.md`
§8.

**Outcome:** Confirmed, and more severe than Method 4: variance ratio 0.02-0.05 (Method 4: 0.136)
and slope 0.09-0.14 (Method 4: 0.187) — despite RDAH being a substantially larger, more
expressive architecture, its fine-tuned predictions are *more* compressed toward the mean than
Method 4's simpler CNN, consistent with the same root cause Method 4's audit already identified:
a pure-magnitude loss (`MaskedLoss` wraps `SmoothL1Loss` with no rank/order term — confirmed by
reading it) with nothing penalizing rank/order errors. The epoch 3→5 transition gets *less*
compressed (variance ratio nearly doubles) while Pearson *declines* slightly over the same span —
a pattern consistent with the model fitting magnitude harder at a mild cost to rank late in
training, which the PDF's §19 already observed qualitatively per-fold and this diagnostic now
explains mechanistically. **Fix, if RDAH fine-tuning is pursued further** (ideally combined with
the item-1 scale fix and item-6 protocol fix): add a rank/structure-preserving loss term
alongside the existing SmoothL1 term, the same fix direction Method 4's audit already recommended
for the same underlying reason — not executed here, flagged as a natural third component of the
combined follow-on experiment.
