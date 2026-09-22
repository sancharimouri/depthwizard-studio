# Audit Log

Chronological index of the ML research track. It was **built on 2026-09-23 from `git log`**
(91 commits at build time) and from the dated headers of the per-method docs, not from memory.
Each row points to the doc that holds the detail and the commit that first recorded it. The
frontend/demo track's commits (2026-09-10 to 09-13, 32 commits: `660ecb6` … `a6b03fc`) are
grouped into one row; that track is documented in CLAUDE.md "Session 1/2" and HANDOFF §5.

Earlier pre-git-era work (Methods 1–4, the RS3DAda and sparse-LiDAR gates) is dated from its own
docs and `PROJECT_STATUS_REPORT.md`. See **Gaps** at the bottom for every place where a result and
its commit don't line up.

| date | track / method | action | result | doc | commit |
|---|---|---|---|---|---|
| 2026-09-10 → 13 | frontend/demo | multi-region demo, layers, flythrough, CDSE search | shipped demo (real DSM terrain; DAv2 as a labelled relative-depth layer) | CLAUDE.md, HANDOFF §5 | `660ecb6` … `a6b03fc` (32 commits) |
| before 2026-09-20 | DFC2019 M1 | global DEM-stat calibration | CLOSED, no real improvement | `01-dem-stat-anchoring/verdict.md` | first in git `0638569` (09-23) ⚠ |
| before 2026-09-20 | DFC2019 M2 | sparse-anchor/GCP regression (Grid/Random/Spatial × OLS/Huber/RANSAC) | CLOSED standalone; Grid+Huber+20 = 2.929/4.718/0.532/0.471 | `02-gcp-regression/*` | `0638569` ⚠ |
| before 2026-09-20 | DFC2019 M3 | semantic prior (building-probability term); holds the per-tile-OLS oracle baseline 3.39/4.58/0.582/0.509 | CLOSED, overfit | `03-semantic-prior/*` | `0638569` ⚠; results `b168dca` ⚠ |
| before 2026-09-20 | DFC2019 M4 | learned CNN scale modulation, v2 phases | superseded by M6; best v2 2.8803/4.7751/0.5835/0.5438 | `04-learned-scale-modulation/*` | `0638569` ⚠; results `b168dca` ⚠ |
| before 2026-09-20 | gates | RS3DAda contamination audit; sparse-LiDAR feasibility | RS3DAda 49/50 contaminated; sparse-LiDAR blocked on DFC2019 | `stage0-gates/*` | `0638569` ⚠ |
| 2026-09-20 | CLAUDE.md | fixed the stale "ML pipeline frozen" framing | — | CLAUDE.md | `89c017d` |
| 2026-09-20 | S2 backbone | ICESat-2 photon query; DAv2 vs. DINOv3 frozen comparison | DINOv3 "wins" pooled (+0.30/+0.35 vs. +0.04/−0.05); **later corrected** (see 09-23) | `sentinel2/backbone-comparison.md` | `8f7dddc`, `92e16b6`, `ee30f7f`, `93c4f81`, `e41a6fb`, `cb7948e`, `da06e96` |
| 2026-09-20 → 21 | S2 backbone | terrain-failure diagnostics; held-out calibration; aspect/built-up follow-ups | negatives | `sentinel2/backbone-terrain-analysis.md` | `2c3553b`, `2e15821` |
| 2026-09-21 | S2 calibration | detector; per-tile DEM calibration; hardening (offset proxy, tiers, FABDEM as a *calibration target*) | tiers usable; FABDEM-as-target not adopted | `sign-flip-detector.md` (top, "2026-09-21") | first in git `216ddb4` (09-22) ⚠ |
| 2026-09-21 | S2 datum | ICESat-2 "gap" was an EGM96/ellipsoid datum bug | fixed | `sign-flip-detector.md` | `216ddb4` ⚠ |
| 2026-09-21 | S2 | SRTM added, 3-way DEM comparison; residual correction; Method 4 ported (10 m, native 30 m) | CNN correction fails twice (memorisation, data starvation) | `sign-flip-detector.md` | `216ddb4` ⚠ |
| 2026-09-22 | DFC2019 M6 | full DAv2-Small fine-tune; GSD-FiLM (wash) and height-balanced (adopted) ablations | 1.980/3.492/0.7445/0.6563 | `06-full-finetune-twin-head/verdict.md` | `39e20f5` |
| 2026-09-22 | S2 M6 / M4 | Method 6 staged on S2; Method 4 with Open Buildings target | both fail (fold 0 stop; memorisation) | `sign-flip-detector.md` | `216ddb4` ⚠ |
| 2026-09-22 | S2 | frequency fusion 21/25 vs. linear-calibrated DAv2; viewing-angle test; arpitparashar06 scale; evidence-gating/LOBO; shadow photogrammetry; yats0x7 read | fusion adopted at the time; **later overturned** (09-23) | `sign-flip-detector.md` | `216ddb4`, `5abf8c2`, `298c0ac`, `e06df63`, `b5b1339` |
| 2026-09-22 | S2 semantic 2.3 | building footprints / Open Buildings; building-aware fusion + re-investigation | clean negative, CLOSED | `sign-flip-detector.md` | `405db5f`, `39e53b7`, `aa27dd5`, `5cc5e80` |
| 2026-09-22 | DFC2019 M5 | full RDAH audit: ×255 input-scale bug; Track1 contamination | zero-shot 2.231/4.566/0.716/0.655 (Track1) | `05-rdah-net-fusion/*` | `0b64c6b`; scripts `37b9122` ⚠; results `b168dca` ⚠ |
| 2026-09-22 (night) | DFC2019 M5 | RDAH-FT-2 training (Swiss, quadrant CV, rank loss) | not adopted (session hit usage limit) | `05` summary §10 | results `5bb45f7`, `b168dca` ⚠ |
| 2026-09-23 | DFC2019 M5 | FT-2 aggregate + write-up | 2.500/4.294/0.640/0.506; NOT ADOPTED | `05` summary §10, verdict §6 | `5bb45f7`, `be8ccac`, `9fb0ae2` |
| 2026-09-23 | S2 RDAH | Darjeeling zero-shot with ×255 (Steps 0–2) | checkerboard persists; correlation criterion **later invalidated** | `sign-flip-detector.md` | `348b865`, `82689ac`, `b51b46f`, `d90859f` |
| 2026-09-23 | DFC2019 M5 | rescue of the `/private/tmp` scratchpad; ×255 provenance; Swiss zero-shot | ×255 from Track1; "fine-tuning damages" resolved: no | `05` summary §11, verdict §7 | `37b9122`, `0638569` |
| 2026-09-23 | S2 | Phase 1: DEM-only controls (pre-registered) | fusion = DEM low-pass (10/25, p = 0.85); GLO-30 beats fusion 23/25 | `sign-flip-detector.md` "(continued)" | `2cc40d4` (prereg), `a15b361` |
| 2026-09-23 | S2 RDAH | Phase 2: DFC2019 resolution sweep (8 tiles) | 8×/1× = 0.5006 (rule missed by 0.0006); checkerboard intrinsic | same; `05` §12 | `3aa534c`, `2973692`, `7501f1f` |
| 2026-09-23 | S2 | Phases 3–5: ICESat-2 20 m + GEDI; no-training detail bake-off | no detail source passes; ETH ceiling fires | same | `cf83c6d`, `b921b7f`, `da6bde4`, `c652c7a` |
| 2026-09-23 | S2 | A0 datum audit | GLO-30 already on EGM2008; tile-centre geoid up to 4.56 m within tiles | "(final close-out)" | `76f9812` |
| 2026-09-23 | DFC2019 M6 | C3 prereg; 2 seeds launched (43, 44) | — | `06` verdict §6 | `6bbcd65` |
| 2026-09-23 | S2 | A1 reproducibility; A1.3 `ats` sensitivity | no conclusion changes | "(final close-out)" | `ce861af`, `d513ee6`, `6ecb1ba` |
| 2026-09-23 | S2 | A2/A3 DEM-only baselines (32 tiles), FABDEM | GLO-30 > SRTM (surface); FABDEM > GLO-30 (terrain, 32/32); no DEM + canopy passes | same | `59f0285` (prereg), `2710c1d` |
| 2026-09-23 | S2 | A4 direct height test | CHMv2 and ETH pass (pooled); within-tile weak (0.23–0.28) | same | `169504e` |
| 2026-09-23 | S2 RDAH | A5 Darjeeling rerun | Spearman ≈ 0; closed by the pre-registered rule | same | `ea01358` |
| 2026-09-23 | DFC2019 | Part B protocol prereg; D1 stale-claim sweep; result-file traceability | — | `05` §13; "(final close-out)" D1 | `5d91176`, `d3a397f`, `b168dca` |

(Rows for Part B results, C1–C3, D2–D4 and the final doc update are appended when those land.
See the bottom of this file.)

## Gaps (result ↔ commit mismatches), flagged

1. **Methods 1–4 and 6's summary, the stage-0 gates, and this file's stub** entered git only on
   2026-09-23 (`0638569`), bundled into an unrelated RDAH commit. The work predates that. Their
   content has no earlier commit to check against.
2. **The 2026-09-21 Sentinel-2 entries** (detector, calibration, datum fix, SRTM, Method 4 ports)
   entered git on 2026-09-22 inside `216ddb4`, whose message describes only the viewing-angle test.
   That's about 2,000 lines of earlier log committed together.
3. **All DFC2019 result files** (Methods 2–6, the oracle baseline, RDAH FT-1/FT-2, DAv2 baselines)
   were **untracked** until `b168dca` (2026-09-23). The only exceptions are the Method 6
   GSD-FiLM/height-balanced JSONs (`39e20f5`) and the FT-2 aggregate (`5bb45f7`). Numbers cited in
   docs before then couldn't be checked against a commit.
4. **RDAH scripts** (`run_rdah_probe.py`, `run_rdah_scale_sweep.py`, `train_rdah_*`) were untracked
   until `37b9122`. The **Track1 ×255 50-tile zero-shot result** existed only in a `/private/tmp`
   scratchpad, and the **Swiss zero-shot result** existed only in a session transcript (rescued and
   marked TRANSCRIBED).
5. **Sentinel-2 result CSVs** `frequency_fusion_results/` and `srtm_3way_comparison.csv` were
   untracked until `a15b361`; ICESat-2 20 m segments until `ce861af`.
6. **`PROJECT_STATUS_REPORT.md` and `COMPETITIVE_REPO_AUDIT.md`** were untracked until the D1/D4
   commits of 2026-09-23.
7. **Commits with no research doc entry:** only the frontend/demo commits, which is by design.
   `89c017d` is a CLAUDE.md framing fix.
8. **Large artifacts deliberately not committed** (arrays, checkpoints, ground-photon CSVs, the
   41 MB `calibration_samples.csv`) are listed with regeneration commands and SHA-256 in
   `data/REGENERATION.md`.
