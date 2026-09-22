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

**Current deployable baselines** (final, 2026-09-23; all 32 benchmark tiles, independent ICESat-2,
per-point geoid, R4 offset-guarded; `data/sentinel2_benchmark/dem_baselines_32/summary.json`):

| use | product | median RMSE vs. ICESat-2 [95% tile-bootstrap CI] | vs. runner-up |
|---|---|---|---|
| **terrain (bare earth)** | **FABDEM** (EGM2008) | ground photons: **1.782 m** [1.068, 3.417] (1.87% of range) | beats GLO-30 on 32/32 tiles, RMSE and bias-removed RMSE, p = 4.7e-10 |
| **surface (DSM)** | **Copernicus GLO-30** (EGM2008) | 20 m canopy-top segments: **4.484 m** [3.363, 8.846] | beats SRTM: RMSE 24/32 (p = 6.6e-4), bias-removed 31/32 |

- **No depth-model or canopy-model add-on beats these.**
  - Frequency fusion equals its DEM-only control (10/25, p = 0.853), and DAv2 detail has median
    r_HF −0.037. **The DEM carries the frequency-fusion result.**
  - No detail source passes any reference (Phase 4).
  - No DEM + canopy product passes (A3).
- **Height-above-ground signal at 10 m exists but is weak within a scene.** CHMv2 and ETH pass a
  pre-registered direct test (pooled Spearman 0.641 / 0.378 vs. ICESat-2). Within-tile it's only
  0.28 / 0.23 (post-hoc).
- The earlier "frequency fusion 21/25, 10.88% → 3.57%" headline compared against
  linear-calibrated DAv2, which is weaker than the raw DEM. It's kept only as history.
- Everything, with independence flags: `docs/method-audit/final-comparison.md` §3.

- **Single running log for all Sentinel-2 work**: `docs/method-audit/sentinel2/sign-flip-detector.md`
  (calibration, sign-flip detection, frequency fusion, evidence-gating/LOBO,
  semantic-prior phase 2.3 incl. its closing re-investigation, RDAH zero-shot, DEM-only
  controls, surface references, detail-source bake-off, datum audit, 32-tile DEM baselines,
  FABDEM, direct height test, RDAH rerun — all entries dated, most recent 2026-09-23 "(final
  close-out)")
- Fusion script (superseded as the recommendation, kept as the reference pipeline):
  `scripts/run_frequency_fusion_sentinel2.py`. Result CSVs, now committed:
  `frequency_fusion_results/`, `srtm_3way_comparison.csv`, `frequency_fusion_controls/`.
- Surface references (2026-09-23, all 32 tiles + Darjeeling):
  - ICESat-2 20 m PhoREAL segments `data/icesat2_segments20m/` (committed). Sliderule's `ats` was
    lowered to 5; the A1.3 sensitivity check changes no conclusion. 25 ATL03 granules failed
    server-side (`data/icesat2_segments20m_failed_granules.csv`).
  - GEDI L2A `data/gedi_l2a/` (committed).
  - FABDEM and ETH on each tile's grid: not committed, regeneration in `data/REGENERATION.md`.
- **Datum:** SRTM is EGM96; GLO-30 and FABDEM are EGM2008 (verified A0). Use the per-point geoid
  (`scripts/dem_baselines_32.py`).
  - **PROJ_NETWORK must be set before any pyproj import**, or N silently becomes 0.
    `frequency_fusion_controls.build_tile` now raises if it does.
- **Darjeeling DEM note:** CLAUDE.md records GLO-30 for Darjeeling as "all-NaN" earlier. GLO-30
  fetched fine for all 32 tiles, and FABDEM for all 32, this session, so the earlier failure was
  most likely a fetch bug. Switching the demo's Darjeeling DEM to GLO-30 or FABDEM is an
  **optional future demo change, not made**.
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
- **RDAH-Net on Sentinel-2: CLOSED by the pre-registered rule (A5).**
  - The Darjeeling rerun with training-exact preprocessing scores Spearman +0.026 vs. ICESat-2
    canopy height and −0.010 vs. GEDI rh98, against a reopen bar of ≥ 0.30 on both.
  - This replaces the earlier post-hoc "default clause" closure.
  - The checkerboard is intrinsic (it appears in-domain) and was never a valid criterion.
  - Scripts: `scripts/rdah_darjeeling_rerun.py`, `scripts/rdah_resolution_sweep.py`.

## 3. Long-term plan — open items, priority order

1. **TSE-Net (self-training)** — untouched, no code or docs. The only substantive open method.
2. **RDAH on DFC2019: Track1 seen-vs-unseen split** — low priority, inference only.
   - Run Track1 zero-shot on its 9 `Track1-test` vs. 41 `Track1-train` tiles, to separate
     memorised tiles from in-domain training.
   - "Fine-tuning damages the model" is resolved (no): Swiss zero-shot 0.492 Pearson, FT-2
     better in 4/4 folds.
   - FT-2 is not adopted.
3. **Learned Sentinel-2 + GEDI route — GATED, not recommended.**
   - Why: the pre-registered ETH ceiling fired. ETH GCH 2020, a global Sentinel-2 + GEDI model,
     fails both surface references as a DEM add-on, and FABDEM + ETH fails too (A3).
   - **Reopen gate:** a canopy/object-height source passing Test B against ICESat-2 surface.
   - What it would fix: many tiles of direct GEDI labels (not one tile's DEM) and a sparse lidar
     target (not an interpolated one), i.e. both diagnosed CNN failure modes.
4. **Sparse-LiDAR 27-feature RF** — untried, low priority. Blocked on DFC2019 (no
   georeferencing, `stage0-gates/sparse-lidar-feasibility.md`). If revisited, prototype on
   Sentinel-2 against the new DEM-only baselines, not against fusion.

**Closed, 2026-09-23:**
- semantic-prior phase 2.3 (clean negative, both mechanism checks clean; log, commit `5cc5e80`)
- RDAH on Sentinel-2 (A5)
- detail add-ons (Phase 4)
- DEM + canopy products (A3)

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
  recoverable georeferencing (Sentinel-2 domain not equally blocked, see §3.4)
- RDAH-Net **on Sentinel-2**, zero-shot or fine-tuned (2026-09-23). Closed by the pre-registered
  A5 rule: Darjeeling Spearman +0.026 / −0.010 vs. ICESat-2 / GEDI height above ground. The
  checkerboard is *not* the reason; it's intrinsic.
- **DEM + canopy-height products** (FABDEM + ETH, FABDEM + CHMv2; 2026-09-23, A3). Both lose to
  raw GLO-30 on ICESat-2 surface heights under the R4 offset guard (4/32 and 0/32).
- **Frequency fusion as a recommendation** (2026-09-23). It equals its DEM-only control.
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
  - **Gate to reopen:** a canopy/object-height source passing Test B against ICESat-2 surface.
    The `bare-earth DTM + canopy` form (FABDEM + ETH) has now been tested and fails (A3). Also
    listed as gated in §3.3.

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
- **DEM-only comparisons use all benchmark tiles, not DAv2-filtered subsets.** The 7 sign-flip-
  excluded tiles were removed by a DAv2 criterion that is irrelevant to DEM products (A2 uses 32).
- **R4 offset guard:** a DEM or product "pass" must hold on BOTH RMSE and bias-removed RMSE
  (report bias and median absolute error too). This blocks constant-offset "wins" like CHMv2's
  (Phase 4) and FABDEM + ETH's on GEDI (A3).
- **Set `PROJ_NETWORK=ON` before any pyproj import, and assert |N| > 1 m.** Otherwise the geoid
  silently becomes 0.
- **Commit result files when the result is written up**, not later (see the gaps in
  `00-audit-log.md`).

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
| `depthwizard` | `blakc-coffee/depthwizard` | Real-DEM-low-frequency + model-high-frequency fusion mechanism — source of frequency fusion, **retired 2026-09-23** (fusion = its DEM-only control; §2b). |
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
