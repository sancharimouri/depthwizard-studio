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

Methods 1–5 (1–4 closed/superseded; 5 open on DFC2019 only, see §3.2; doc location in each case):
1. Global DEM-stat calibration — CLOSED, no real improvement. `docs/method-audit/01-dem-stat-anchoring/`
2. Sparse-anchor/GCP regression — CLOSED as standalone; best variant Grid+Huber+20
   anchors is the oracle baseline Method 6 is compared against. `docs/method-audit/02-gcp-regression/`
3. Semantic prior (linear building-probability term) — CLOSED, overfit, didn't
   generalize spatially. `docs/method-audit/03-semantic-prior/`
4. Learned CNN scale-modulation — superseded by Method 6, not current best; still the
   best frozen-feature approach (MAE 2.8803m/RMSE 4.7751m/Pearson 0.5835/Spearman
   0.5438). `docs/method-audit/04-learned-scale-modulation/` (full table in `v2-results.md`)
5. RDAH-Net fusion — **open on DFC2019 (low priority), closed on Sentinel-2** (§3.2, §4). `docs/method-audit/05-rdah-net-fusion/`.
   Five separate results:
   - zero-shot on Sentinel-2: closed. The resolution cliff on DFC2019 is the basis; the
     checkerboard is intrinsic and isn't (§2b)
   - zero-shot on DFC2019 with corrected ×255 input: 2.231/4.566/0.716/0.655. Track1 checkpoint
     (41/50 tiles contaminated), tile-level folds; artifact rescued to
     `data/dfc2019/experiments/rdah_zeroshot/`
   - Swiss zero-shot on DFC2019, FT-2's samples: 3.033/6.421/0.492/0.542 (TRANSCRIBED from a
     session log)
   - RDAH-FT-1: 2.906/6.659/0.513/0.527 pooled, unstable across folds
   - **RDAH-FT-2** (2026-09-23): Swiss init, per-fold input scale, quadrant folds, rank loss,
     nested selection. **2.500/4.294/0.640/0.506** (per-sample mean, Method 6's aggregation);
     2.499/5.598 pixel-pooled. **Not adopted**: loses to Method 6 on all four metrics. Fold
     Pearson range 0.503–0.607 (FT-1: 0.254–0.607). Variance ratio 0.19–0.23, still severe
     underdispersion. Aggregate: `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`
     (`scripts/aggregate_rdah_ft2.py`)

### 2b. Sentinel-2/India track

Current deployable baseline (restated 2026-09-23 after DEM-only controls): **raw Copernicus
GLO-30 reprojected to the 10 m grid (EGM2008 geoid), DEM only.** Median ICESat-2 ground-photon
RMSE is **2.32% of elevation range**. It beats the previous "frequency fusion" baseline (3.57%)
on **23/25 tiles** (Wilcoxon p = 6.6e-6).

**The DEM carries the frequency-fusion result; DAv2 detail adds no measurable value against
ground photons.** This is the pre-registered wording.
- Fusion vs. its own DEM-only control (SRTM low-pass, DAv2 detail zeroed): 10/25 wins,
  p = 0.853.
- DAv2 high-pass vs. the ground residual: median r = −0.037, positive on 8/25 tiles.
- Fusion's old 21/25 win (10.88% → 3.57%) was against *linear-calibrated DAv2*, a baseline
  weaker than the raw DEM. Most of that gain is the DEM itself (raw SRTM alone: 3.70%); the
  low-pass smoothing of SRTM's resampling noise supplies the rest (3.70% → 3.57%).

**No detail source adds signal against a surface reference either.** Tested: DAv2 @518/@1008,
DINOv3-CHMv2, ETH canopy height, against ICESat-2 20 m canopy-top segments and GEDI rh98. All
fail the pre-registered bar (max median r_HF 0.089 vs. 0.10). So no "DEM + detail" product is
recommended. Details: sign-flip-detector.md, "2026-09-23 (continued)", Phases 1 and 4; scripts
`scripts/frequency_fusion_controls.py`, `scripts/detail_source_bakeoff.py`.

- **Single running log for all Sentinel-2 work**: `docs/method-audit/sentinel2/sign-flip-detector.md`
  (calibration, sign-flip detection, frequency fusion, evidence-gating/LOBO,
  semantic-prior phase 2.3 incl. its closing re-investigation, RDAH zero-shot, DEM-only
  controls, surface references, detail-source bake-off — all entries dated, most recent
  2026-09-23)
- Fusion script (superseded as the recommendation, kept as the reference pipeline):
  `scripts/run_frequency_fusion_sentinel2.py`. Result CSVs, now committed:
  `frequency_fusion_results/`, `srtm_3way_comparison.csv`, `frequency_fusion_controls/`.
- Surface references (2026-09-23, 25 tiles + Darjeeling):
  - ICESat-2 20 m PhoREAL segments `data/icesat2_segments20m/`: not committed, regenerable with
    `scripts/fetch_icesat2_segments20m.py`. Sliderule's `ats` had to be lowered to 5.
  - GEDI L2A `data/gedi_l2a/` (committed).
  - Counts: `data/sentinel2_benchmark/surface_reference_counts.csv`.
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
- **RDAH-Net on Sentinel-2 (2026-09-23): CLOSED.**
  - The first test's two stop criteria were both later shown non-diagnostic:
    - The terrain correlation was the wrong reference for an above-ground-height model.
    - The checkerboard also appears on DFC2019 at native resolution, where RDAH works.
  - The closure rests on a pre-registered resolution sweep on DFC2019. Pearson goes 0.483 /
    0.589 / 0.589 / 0.242 at 0.3 / 0.6 / 1.2 / 2.4 m; Sentinel-2 is 10 m. The rule's
    mechanism clause missed by 0.0006, and its default clause closes the line.
  - Also found: the Sentinel-2 RGB renders are 2–4× darker than RDAH's training input.
  - Entries: sign-flip-detector.md, "2026-09-23 — RDAH-Net zero-shot on Sentinel-2" (with its
    correction) and "(continued)" Phase 2. Scripts: `scripts/rdah_sentinel2_zeroshot.py`,
    `scripts/rdah_resolution_sweep.py`.

## 3. Long-term plan — open items, priority order

1. **TSE-Net (self-training)** — untouched, no code or docs exist for it yet. Now the top
   open item.
2. **RDAH-Net on DFC2019: memorised tiles vs. in-domain training (low priority, inference
   only).**
   - "Fine-tuning damages the model" is **resolved: no**. Swiss zero-shot on FT-2's exact
     samples scores 3.033/6.421/0.492/0.542, and FT-2 beats it on every metric in 4/4 folds.
   - The Track1 zero-shot artifact was rescued (`data/dfc2019/experiments/rdah_zeroshot/`) and
     reproduces 2.231/4.566/0.716/0.655. Its ×255 was picked on 3 tiles from Track1's own
     training list; Swiss re-derivation finds a broad ×200–×1000 plateau.
   - Remaining: Track1 0.716 vs. Swiss 0.492 Pearson fits a training-data advantage. To split
     memorised tiles from in-domain sensor/city, run Track1 zero-shot on its 9 `Track1-test`
     vs. 41 `Track1-train` tiles.
   - RDAH-FT-2 is not adopted (`05-rdah-net-fusion/verdict.md` §6–8). RDAH on Sentinel-2 is
     closed (§4).
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
- RDAH-Net **on Sentinel-2**, zero-shot or fine-tuned (2026-09-23). Resolution cliff on
  DFC2019 (Pearson 0.589 at 1.2 m → 0.242 at 2.4 m, vs. Sentinel-2's 10 m), closed by the
  pre-registered rule's default clause. The checkerboard is *not* the reason; it's intrinsic.
- **DAv2 (or any tested no-training source) as a high-frequency detail add-on for Sentinel-2**
  (2026-09-23): DAv2 @518/@1008, DINOv3-CHMv2, ETH canopy height. None passes against ground
  photons, ICESat-2 20 m surface or GEDI (max median r_HF 0.089 < 0.10). Frequency fusion's gain
  was the DEM low-pass.
- **Learned Sentinel-2 + GEDI canopy/surface-height route** (2026-09-23): not started and not
  recommended. The pre-registered ETH ceiling check fired: ETH GCH 2020, itself a global
  Sentinel-2 + GEDI model, doesn't pass either surface reference, and worsens the ICESat-2
  surface product by double-counting with SRTM (8.32 vs. 5.00 m).
  - That route *would* address both earlier failure modes: ~10⁵–10⁶ direct GEDI labels across
    many tiles instead of one tile's DEM, and a sparse direct-lidar target instead of an
    interpolated one.
  - **Gate to reopen:** a canopy/object-height source passing Test B against ICESat-2 surface,
    e.g. in a `bare-earth DTM + canopy` product form (FABDEM + ETH), which is untested.

**Explicitly excluded from this list: RDAH-Net on DFC2019.** It is open (low priority), see §3.2. Only its Sentinel-2 use is closed (above). The specific
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
- **Fusion variants must be compared against a DEM-only control** (the same pipeline with the
  added component zeroed, plus the raw DEM(s)). Beating a weaker learned baseline isn't enough:
  the frequency-fusion headline (21/25 vs. linear-calibrated DAv2) turned out to be the DEM's
  own result (2026-09-23).
- **Detail sources are scored against a surface reference, not only ground photons**
  (ICESat-2 20 m canopy-top segments, GEDI rh98). Scoring an above-ground-height product against
  terrain truth is uninformative, as the first RDAH Sentinel-2 test showed.
- **Pre-register each phase's decision rule in the log and commit it before running**, and label
  any post-hoc judgment of an un-operationalized term as such.

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
| `RDAH-Net` | (upstream RDAH-Net repo) | Source of the RDAH-Net fusion method itself (§2a Method 5, §3.2 open item). |
| `SynRS3D` | `JTRNEO/SynRS3D` | Synthetic RS 3D dataset/method referenced during the RDAH-Net investigation; not independently adopted. |
| `yats0x7-depthwizard` | `yats0x7/DepthWizard` | Ground-trend scale approach — code-read and compared against blakc-coffee's fusion (`docs/method-audit/` — code-read comparison entry); converges with blakc-coffee, no separate adoption. |
| `madhu-mitha-e-depthwizard` | `madhu-mitha-e/DepthWizard` | Part of the 20-repo competitive audit (`COMPETITIVE_REPO_AUDIT.md`) — reviewed, no method adopted from it directly. |
| `gowthamkrishna27-elevate3d` | `gowthamkrishna27/Elevate3d` | Part of the competitive audit — reviewed, no method adopted from it directly. |
| `dinov3` | (Meta DINOv3) | Backbone used for the DAv2-vs-DINOv3 comparison (§2b, `backbone-comparison.md`); required a HF-checkpoint state-dict conversion to run SAT493M — see that doc for the conversion steps if reused. |

Full 20-repo audit with claimed-numbers verification: `COMPETITIVE_REPO_AUDIT.md`
(repo root). Overall research-track state snapshot: `PROJECT_STATUS_REPORT.md`
(repo root).

---

**Consistency check against CLAUDE.md** (updated 2026-09-23, later): CLAUDE.md's "one rule
that overrides everything else" (quota discipline, act don't ask) still applies and this doc
doesn't change it. CLAUDE.md's ML-research-track summary and this doc agree on all facts above:
- RDAH-FT-2 not adopted; RDAH open on DFC2019 only (low priority); RDAH closed on Sentinel-2
- the Sentinel-2 deployable baseline restated as raw GLO-30 DEM-only
- no detail source passes, and the learned GEDI route is not recommended
- phase 2.3 closed

This doc reorganizes the same information by "what's next" instead of chronology, and adds the
frontend (§5) and credentials (§6) inventories. One naming inconsistency is flagged rather than
silently fixed: both docs call Method 2's Grid+Huber+20 result (2.929/4.718/0.532/0.471) the
"oracle per-tile-OLS baseline", while the per-method audits use that name for
3.39/4.58/0.582/0.509 (see the §2a note).
