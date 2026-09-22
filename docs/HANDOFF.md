# DepthWizard2 (SIH26175) — Handoff

Read this once, act on it. This is the organized reference; `docs/method-audit/sentinel2/sign-flip-detector.md`
and the per-method `docs/method-audit/*/` folders are the chronological logs — go there for
blow-by-blow detail, come here for "what's the state and what's next."

## 1. Project overview

Depth Wizard turns satellite RGB into 3D terrain visualization for SIH26175. There are
**two separate tracks** — do not conflate them:

- **DFC2019 ML research track** — offline model-improvement research using the DFC2019
  dataset, which has **dense real LiDAR ground truth**. This is a proxy-validation
  sandbox: if a correction method can't even win here, it's not worth testing on the
  real target domain.
- **Sentinel-2/India track** — the actual deployment domain (Darjeeling, Kolkata,
  Bardhaman, Sundarbans-style 10 m Sentinel-2 imagery). No dense ground truth exists
  here — only sparse ICESat-2 photon footprints and coarse DEMs (SRTM/Copernicus
  GLO-30). A method winning on DFC2019 does **not** automatically transfer (see Method 6
  below — it wins on DFC2019, loses on Sentinel-2).

Neither track is deployed. The live demo (frontend/backend) is a separate, frozen
system — see §5.

## 2. Current state per track

### 2a. DFC2019 track

Current best: **Method 6** — full DAv2-Small backbone fine-tune, twin (mean,
log-variance) head, `CappedHeightWeightedLoss` + `WeightedRandomSampler` recipe.
First method in the whole audit to beat the oracle per-tile-OLS baseline on all 4
tracked metrics simultaneously (4-fold spatial-quadrant holdout, 50-tile DFC2019
benchmark):

| | MAE | RMSE | Pearson | Spearman |
|---|---|---|---|---|
| Method 6 | 2.053m → **1.980m** (height-balanced) | 3.531m → **3.492m** | 0.737 → **0.745** | 0.654 → **0.656** |
| Oracle per-tile-OLS baseline | 2.929m | 4.718m | 0.532 | 0.471 |

_Naming note (2026-09-23): the "oracle" row above is Method 2's Grid+Huber+20 sparse-anchor
result. The per-tile-OLS oracle that Methods 4/6's own audits compare against is
3.39m/4.58m/0.582/0.509 (`04-learned-scale-modulation/summary.md` §1). Both rows are per-tile
means. See `05-rdah-net-fusion/summary.md` §10._

- Docs: `docs/method-audit/06-full-finetune-twin-head/{summary,verdict}.md`
- Eval/train scripts: `scripts/evaluate_method6_finetune_twinhead.py`,
  `scripts/evaluate_method6_gsd_film_height_balanced.py`,
  `scripts/evaluate_method6_sentinel2.py`, `scripts/train_method6_full_dfc2019.py`,
  `scripts/method6_vhr_sanity_check.py`
- Experiment outputs: `data/dfc2019/experiments/method6*/`
- **Caveat: DFC2019-only.** Staged on Sentinel-2/SRTM (item 2b) and lost.

Methods 1–5 (1–4 closed/superseded; 5 open, see §3.1; doc location in each case):
1. Global DEM-stat calibration — CLOSED, no real improvement. `docs/method-audit/01-dem-stat-anchoring/`
2. Sparse-anchor/GCP regression — CLOSED as standalone; best variant Grid+Huber+20
   anchors is the oracle baseline Method 6 is compared against. `docs/method-audit/02-gcp-regression/`
3. Semantic prior (linear building-probability term) — CLOSED, overfit, didn't
   generalize spatially. `docs/method-audit/03-semantic-prior/`
4. Learned CNN scale-modulation — superseded by Method 6, not current best; still the
   best frozen-feature approach (MAE 2.8803m/RMSE 4.7751m/Pearson 0.5835/Spearman
   0.5438). `docs/method-audit/04-learned-scale-modulation/` (full table in `v2-results.md`)
5. RDAH-Net fusion — **open, not rejected** (§3.1). `docs/method-audit/05-rdah-net-fusion/`.
   Four separate results:
   - zero-shot on Sentinel-2: rejected for checkerboard artifacts, probably fed unscaled input
   - zero-shot on DFC2019 with corrected ×255 input: 2.231/4.566/0.716/0.655. Track1 checkpoint
     (41/50 tiles contaminated), tile-level folds; the result artifact wasn't located
   - RDAH-FT-1: 2.906/6.659/0.513/0.527 pooled, unstable across folds
   - **RDAH-FT-2** (2026-09-23): Swiss init, per-fold input scale, quadrant folds, rank loss,
     nested selection. **2.500/4.294/0.640/0.506** (per-sample mean, Method 6's aggregation);
     2.499/5.598 pixel-pooled. **Not adopted**: loses to Method 6 on all four metrics. Fold
     Pearson range 0.503–0.607 (FT-1: 0.254–0.607). Variance ratio 0.19–0.23, still severe
     underdispersion. Aggregate: `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`
     (`scripts/aggregate_rdah_ft2.py`)

### 2b. Sentinel-2/India track

Current deployable baseline: **frequency fusion** (real DEM low-frequency trend +
DAv2 high-frequency detail, matched-filter subtraction — no training, so it can't
memorize). Beats plain per-tile linear calibration on **21/25 tiles**, median
ICESat-2 error cut from 10.88% → 3.57% of elevation range. Validated two ways: the
original single-split protocol and leave-one-**block**-out (LOBO, 25 spatial
blocks/tile) — same 21/25 result under both, zero flips.

- **Single running log for all Sentinel-2 work**: `docs/method-audit/sentinel2/sign-flip-detector.md`
  (calibration, sign-flip detection, frequency fusion, evidence-gating/LOBO,
  semantic-prior phase 2.3 incl. its closing re-investigation — all entries, dated, most
  recent 2026-09-22)
- Fusion script: `scripts/run_frequency_fusion_sentinel2.py`
- Benchmark data: `data/sentinel2_benchmark/` — `manifest.csv` (25 accepted tiles of
  32 originally selected), `icesat2_coverage.csv`, `srtm_raw/`, `copernicus_dem_raw/`
- **Content-QC history, so it isn't rediscovered as a surprise**: the benchmark started
  at 32 tiles (8 each: agricultural/coastal/hilly/urban) selected via real ICESat-2
  ATL08 coverage. A content-quality audit (`content_audit.csv` →
  `content_audit_corrected_32.csv`) found real problems in 14/32 tiles, which were
  replaced/corrected. Separately, the sign-flip detector then flagged 7 of the
  resulting set as unreliable, leaving **25 tiles as the actual working set**. Two
  different filters, two different reasons — both already resolved.
- Backbone comparison (DAv2 vs DINOv3 SAT493M+CHMv2, closed, DINOv3 wins pooled but
  DAv2 wins on agricultural/hilly specifically): `docs/method-audit/sentinel2/backbone-comparison.md`
- Three independent CNN-correction attempts on top of frequency fusion all failed
  differently (SRTM-interpolation memorization, native-30m data starvation,
  Open-Buildings-target memorization) — **CNN correction is not pursued further
  without new evidence**, this is a deliberate stop, not an open thread. Detail in
  sign-flip-detector.md.
- **RDAH-Net zero-shot on Sentinel-2 (2026-09-23): clean negative, line closed.** Tested whether
  the DFC2019 input-scale fix (×255 on DAv2 depth) rescues the Darjeeling checkerboard. The
  original run (Swiss checkpoint, reproduced bit-exactly) got the fix plus reflect-pad-to-1024
  sizing. The artifact's FFT peaks drop 2–3 orders of magnitude but persist (diagonal peaks at
  periods 8/16/32). The output has no terrain correlation, raw or detrended (ICESat-2 plane-
  detrended −0.139/−0.132). Resize vs. pad was ruled out as the cause. Stopped at Step 2, so no
  benchmark scoring, fusion swap or fine-tuning. Entry: sign-flip-detector.md, "2026-09-23 —
  RDAH-Net zero-shot on Sentinel-2"; script `scripts/rdah_sentinel2_zeroshot.py`.

## 3. Long-term plan — open items, priority order

1. **RDAH-Net: contamination vs. fine-tuning damage** — **NOT rejected.** FT-1's
   fold-instability question is answered: under FT-2's quadrant protocol the fold Pearson range
   narrows from 0.254–0.607 to 0.503–0.607. **RDAH-FT-2 is not adopted** (2.500/4.294/0.640/
   0.506 per-sample mean; loses to Method 6 on all four metrics; `05-rdah-net-fusion/verdict.md`
   §6). What's still open: zero-shot on DFC2019 (2.231/4.566/0.716/0.655) beats both fine-tuned
   runs. Either fine-tuning damages the pretrained model on this small benchmark, or the
   zero-shot score is inflated because the Track1 checkpoint it used has 41/50 benchmark tiles in
   `Track1-train.txt`. The Swiss checkpoint FT-2 used has 0/50. **Next check**: Swiss zero-shot at
   the fold-derived scale, under the quadrant protocol, with per-fold affine calibration. Also
   re-create the zero-shot run's script/JSON, which weren't found in the repo. Neither has
   been run. The 2026-09-23 Sentinel-2 test (§2b) also used the clean Swiss checkpoint but
   **doesn't answer this**: it fails for reasons confounded with the question (GSD, sensor and
   nDSM-vs-terrain gaps). Scope: this item is **DFC2019-only**, and RDAH on Sentinel-2 is closed
   (§4).
2. **TSE-Net (self-training)** — untouched, no code or docs exist for it yet. Next
   candidate after the RDAH contamination check.
3. **Sparse-LiDAR 27-feature RF version** — untried. Low priority: frequency fusion
   already beats it on deployability grounds and the DFC2019 feasibility check
   (`docs/method-audit/stage0-gates/sparse-lidar-feasibility.md`) found the approach is
   **blocked on DFC2019** specifically (no recoverable georeferencing for the sparse
   LiDAR source) — the Sentinel-2 domain wasn't blocked the same way, so if this is
   revisited, prototype directly on Sentinel-2, not DFC2019.

**Closed since the last handoff (2026-09-23):** semantic-prior phase 2.3. The original test and
both follow-up mechanism checks came back clean. Approach A (building-aware confidence
weighting) is a genuine no-op, not an integration bug: `building_prob` is correctly wired with
real variation, but `dav2_highpass` is sub-centimetre (abs-mean 0.0083 m at chennai), so the
0.0005pp delta is physical. Approach B (Open Buildings height blending) has no units/datum
mismatch: Google's catalog documents `building_height` as AGL relative to terrain, the blend
already adds it to `dem_lowpass`, and the hypothesized full-weight "fix" roughly doubles the
ICESat-2 overshoot at chennai (+8.28 m vs. +4.52 m mean error). The negative stands. Record:
`docs/method-audit/sentinel2/sign-flip-detector.md`, "Re-investigation" subsection of the
2026-09-22 phase 2.3 entry (commit `5cc5e80`).

## 4. Rejected list (don't redo these)

- Global/pooled calibration (Method 1)
- RANSAC for sparse anchors (both rounds — `sparse_anchor_ransac_v2` and predecessor)
- Spatial-local calibration variants: quadrant-wise, smooth RBF residual
  (`sparse_anchor_spatial*`, `sparse_anchor_smooth_residual`)
- Semantic-prior as a **linear interaction term** (Method 3's original DFC2019 form —
  note phase 2.3's closed-form building-aware tests on Sentinel-2 are a *different*,
  also-negative, already-closed result, not a redo of this)
- All CNN-based Sentinel-2 correction — 3 independently diagnosed failures:
  SRTM-interpolation-memorization, native-30m-resolution data-starvation,
  Open-Buildings-target memorization
- GSD-FiLM conditioning on Method 6 (DFC2019 has no real per-tile GSD variation —
  clean wash, not adopted)
- arpitparashar06's non-regression scale derivation
- ArnabTechiee's shadow-geometry photogrammetry (fails at 10m GSD, both physics and
  empirically — `data/sentinel2_benchmark/shadow_photogrammetry_plausibility.csv`)
- RS3DAda (contaminated on DFC2019 — 49/50 benchmark tiles were in its own training
  split; degenerate on Sentinel-2)
- Sparse-LiDAR-Guided-Correction's **DFC2019** feasibility specifically — blocked, no
  recoverable georeferencing (Sentinel-2 domain not equally blocked, see §3.3)
- RDAH-Net **on Sentinel-2**, zero-shot or fine-tuned (2026-09-23). The input-scale fix
  doesn't remove the checkerboard, and there's no terrain correlation after detrending. See
  sign-flip-detector.md.

**Explicitly excluded from this list: RDAH-Net on DFC2019.** It is open, see §3.1. Only its Sentinel-2 use is closed (above). The specific
**RDAH-FT-2 recipe** (Swiss init + per-fold scale + quadrant folds + rank loss, 5 epochs) was
not adopted (loses to Method 6 on all four metrics). That closes one recipe, not the method.

## 5. Frontend/backend status (live demo — frozen, separate from all of the above)

The demo stays intentionally separate from every ML research finding above, including
Method 6's DFC2019 win. **A real DSM is the terrain source; DAv2 output is shown only
as a labeled relative-depth *visualization* layer, never as the thing that produced the
terrain.** None of the research-track work is deployed into it.

- **Stack**: single-page Three.js + Vite app (`frontend/src/main.js`, `viewer.js`), dev
  server at `localhost:5173`. FastAPI backend (`backend/main.py`) with a CDSE router
  mounted at `/api/cdse` (`backend/api/routes.py`, `backend/cdse/client.py`).
- **Not a multi-page app** — there is no separate "Explore" or "Workbench" page. It's
  one viewer with:
  - A region rail — all 4 regions (Darjeeling, Kolkata, Bardhaman, Sundarbans) are
    interactive, each with satellite/relative-depth/elevation layer switching
    (`activateLayer`/`selectLayer` in `main.js`) and a real per-region stats strip.
  - A mock "RUN RECONSTRUCTION" pipeline (`runReconstruction()`) with staged progress
    and pipeline-step highlighting, labeled inline as a prototype flow.
  - A scene search/upload modal (`openSceneSearchModal`/`openSceneUploadModal`) with
    real Nominatim geocoding (`queryNominatim`) and CDSE scene search/selection
    (`runSceneSearch`, `selectScene`) hitting the backend `/api/cdse` routes — this is
    the "CDSE Scene Input integration."
  - OrbitControls flythrough (damped drag-orbit, idle auto-rotate) and a flood-overlay
    toggle (`setFloodActive`).
- **Assets**: `frontend/public/data/{darjeeling,kolkata,bardhaman,sundarbans}/` all
  populated (satellite/relative-depth/elevation textures + terrain.json per region).
- **Working state**: last verified interactive for all 4 regions per CLAUDE.md Session 2
  outcomes — confirm with a fresh `npm run dev` + visual check before relying on this,
  don't assume it's still green without checking.

## 6. Credentials/access inventory (`.env`)

| Key | Status |
|---|---|
| `EARTHENGINE_PROJECT` | Working. |
| `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` | **Sentinel-Hub-scoped only, NOT OData/download-scoped.** This caused a real credentials-audience error earlier — if you hit an audience/scope error against the CDSE OData download endpoint, this is why; the fix is a different credential type, not a retry. |
| `HF_TOKEN` | DINOv3 gated access, granted. |
| `NICFI_API_KEY` | Present, but **account lacks program entitlement** — confirmed via a real HTTP 200 + empty list response, not an auth error. Don't mistake this for a broken key. |
| `OPENTOPOGRAPHY_API_KEY` | 401, unresolved. Copernicus GLO-30 via AWS is the working substitute (see CLAUDE.md's "what actually exists" section for where it's already wired in). |
| `ICESAT_API_KEY` | Present, used for ICESat-2 ATL08 pulls (Sentinel-2 benchmark ground truth). |
| Bhoonidhi | No working API — manual portal fetch only, not automatable. |

## 7. Standing methodology conventions (follow, don't rediscover)

- **4-fold spatial-quadrant holdout** is the standard evaluation protocol for any new
  DFC2019-track method.
- **Always compare against the oracle per-tile-OLS baseline** (Method 2's Grid+Huber+20
  result) — a method that doesn't beat this isn't worth adopting.
- **Independent-of-training-reference validation is required.** ICESat-2 for Sentinel-2,
  real held-out LiDAR for DFC2019. A DEM-only or same-source check alone is **not
  trusted** — this project has hit the same memorization-detection pattern three
  separate times on Sentinel-2 CNN attempts (§2b) by trusting a same-source check first.
- **Staged/gated testing with an explicit stop condition** before committing to a full
  run (see Method 6's Sentinel-2 staging: stopped at fold 0 per a pre-agreed protocol
  once it lost decisively, rather than burning budget on folds 1–3).
- **A clean negative result gets the same write-up rigor as a positive one** — never
  silently dropped. Semantic-prior phase 2.3, RDAH-Net zero-shot, and all three CNN
  Sentinel-2 failures are documented this way; keep doing that.
- **Every summary response ends with an explicit list of files created/modified.**

## 8. Skills

`superpowers` (verification-before-completion, systematic-debugging, etc.) and
`claude-scientific-skills` should both be active — confirm they still appear in this
session's skill listing; if either is missing, that's a regression to flag, not
something to work around silently.

## 9. External reference repos already cloned (`external/`)

Clone-on-demand-only policy: don't re-clone or re-investigate any of these from
scratch — they're already here and already read.

| Folder | Source | Contribution |
|---|---|---|
| `sih2026-depthwizard` | `zaidnansari2011/sih2026-depthwizard` | Independently validated the full-DAv2-fine-tune idea (on its own non-comparable split) — this is where **Method 6's approach was sourced from**. |
| `depthwizard` | `blakc-coffee/depthwizard` | Real-DEM-low-frequency + model-high-frequency fusion mechanism — source of the **current Sentinel-2 deployable baseline** (frequency fusion, §2b). |
| `arpitparashar06-depthwizard` | `arpitparashar06/depthwizard` | Also does DEM+model frequency fusion (independently flagged alongside blakc-coffee); its non-regression scale-derivation approach was tried and rejected (§4). |
| `DepthWizard-SIH26175` | `devendrakushwah80/DepthWizard-SIH26175` | Source of Method 6's two follow-on ablation ideas: GSD-FiLM conditioning (rejected) and height-balanced loss+sampling (**adopted**, current Method 6 recipe). |
| `amogh-hub-depthwizard` | `amogh-hub/depthwizard` | Source of evidence-gating + leave-one-out validation, adapted onto frequency fusion as LOBO (§2b). |
| `ArnabTechiee-depthwizard` | `ArnabTechiee/depthwizard` | Shadow-geometry photogrammetry approach — tested and rejected, fails at 10m GSD (§4). |
| `RDAH-Net` | (upstream RDAH-Net repo) | Source of the RDAH-Net fusion method itself (§2a Method 5, §3.1 open item). |
| `SynRS3D` | `JTRNEO/SynRS3D` | Synthetic RS 3D dataset/method referenced during the RDAH-Net investigation; not independently adopted. |
| `yats0x7-depthwizard` | `yats0x7/DepthWizard` | Ground-trend scale approach — code-read and compared against blakc-coffee's fusion (`docs/method-audit/` — code-read comparison entry); converges with blakc-coffee, no separate adoption. |
| `madhu-mitha-e-depthwizard` | `madhu-mitha-e/DepthWizard` | Part of the 20-repo competitive audit (`COMPETITIVE_REPO_AUDIT.md`) — reviewed, no method adopted from it directly. |
| `gowthamkrishna27-elevate3d` | `gowthamkrishna27/Elevate3d` | Part of the competitive audit — reviewed, no method adopted from it directly. |
| `dinov3` | (Meta DINOv3) | Backbone used for the DAv2-vs-DINOv3 comparison (§2b, `backbone-comparison.md`); required a HF-checkpoint state-dict conversion to run SAT493M — see that doc for the conversion steps if reused. |

Full 20-repo audit with claimed-numbers verification: `COMPETITIVE_REPO_AUDIT.md`
(repo root). Overall research-track state snapshot: `PROJECT_STATUS_REPORT.md`
(repo root).

---

**Consistency check against CLAUDE.md** (updated 2026-09-23): CLAUDE.md's "one rule that
overrides everything else" (quota discipline, act don't ask) still applies and this doc doesn't
change it. CLAUDE.md's ML-research-track summary (as of 2026-09-23) and this doc agree on all
facts above, including RDAH-FT-2 (not adopted, method still open on DFC2019), RDAH-on-Sentinel-2
(closed 2026-09-23, clean negative), and phase 2.3 (closed). This
doc reorganizes the same information by "what's next" instead of chronology, and adds the
frontend feature inventory (§5) and credentials inventory (§6), which CLAUDE.md covers in less
detail. One naming inconsistency is flagged rather than silently fixed: both docs call Method 2's
Grid+Huber+20 result (2.929/4.718/0.532/0.471) the "oracle per-tile-OLS baseline", while the
per-method audits use that name for 3.39/4.58/0.582/0.509 (see the §2a note).
