# Regenerating large uncommitted artifacts

Created 2026-09-23. SHA-256 for every file below is in `data/REGENERATION_sha256.csv` (path, bytes,
sha256). **A regenerated file will not necessarily hash-match.** ICESat-2 and GEDI
server-side processing, and PROJ/EE versions, can change the output. The hashes identify the exact
versions behind the committed results. A mismatch means "different inputs", not "broken
pipeline".

All commands run from the repo root with the project venv (`.venv/bin/python`).

| Artifact | Committed? | Regenerate with | Notes |
|---|---|---|---|
| `data/sentinel2_benchmark/dav2_depth_1008/<tile>_depth.npy` (25 × 4 MB) | no | `python scripts/run_dav2_1008.py` | Frozen DAv2 via `backend/depth/depth_engine.py`, only `MODEL_INPUT_SIZE` = 1008. Deterministic up to MPS kernel variation. |
| `data/sentinel2_benchmark/eth_canopy_2020/<tile>_eth.npy` (25 × 4 MB) | no | `python scripts/fetch_eth_canopy.py` | Needs `EARTHENGINE_PROJECT` in `.env`. EE asset `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`, bilinear onto each RGB tile's grid. |
| `data/icesat2_photons/<tile>.csv` (ground photons, 747 MB total) | no | `python scripts/query_icesat2_photons.py` (32 benchmark tiles); Darjeeling via `python scripts/rdah_sentinel2_zeroshot.py darjeeling --rgb data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif` (its `fetch_photons`) | sliderule `atl03sp`, `atl08_class=['atl08_ground']`, 2019-01-01 to 2025-12-31. |
| `data/icesat2_segments20m/<tile>.csv` | **yes** (committed 2026-09-23, largest 6 MB) | `python scripts/fetch_icesat2_segments20m.py [tile ...]` | Hashes kept anyway. See below. |
| `data/gedi_l2a/<tile>.csv` | **yes** | `python scripts/fetch_gedi_l2a.py` | EE `LARSE/GEDI/GEDI02_A_002_MONTHLY`. |
| `data/dfc2019/experiments/method6_height_balanced_seed{43,44}/fold*.pt` | no (~100 MB each) | `bash scripts/run_method6_seeds.sh` (**training**) | C3 seed runs. |
| `data/sentinel2_benchmark/fabdem/<tile>_fabdem.npy` (32 × 4 MB) | no | `python scripts/fetch_fabdem.py` | EE `projects/sat-io/open-datasets/FABDEM`, mosaic with the **native projection restored** (see the script comment; the default 1° mosaic projection is a bug trap). Data licence CC BY-NC-SA 4.0. |
| ETH for the 7 sign-flip-excluded tiles | no | `python scripts/fetch_eth_canopy.py --all-tiles` | Same asset as above. |
| `data/elevation/darjeeling/Darjeeling_Copernicus_GLO30_DSM.tif` (85 MB, N26 + N27 full-tile mosaic) | no | `python scripts/fetch_glo30_darjeeling.py` | The cropped `..._cropped.tif` is committed and byte-identical to the demo's DEM. |
| ICESat-2 segments / GEDI for the 7 extra tiles | yes | `python scripts/fetch_icesat2_segments20m.py --all-tiles`, `python scripts/fetch_gedi_l2a.py --all-tiles` | |
| `data/dfc2019/experiments/dav2_calibration/calibration_samples.csv` (41 MB) | no | Method 1's calibration run (`data/dfc2019/experiments/dav2_calibration/config.json`) | Raw pixel samples; the summaries are committed. |
| `data/sentinel2_benchmark/dem_baselines_32/` inputs | — | `python scripts/dem_baselines_32.py` | **Sets `PROJ_NETWORK=ON` before importing pyproj**; asserts \|N\| > 1 m. |

## ICESat-2 20 m segments: what was actually used, and what failed

- **Tracks used:** `data/icesat2_segments20m_tracks_used.csv`, one row per (tile, RGT, cycle) that
  contributed segments; 654 rows. Sliderule's `atl08p` output carries RGT/cycle/spot, not granule
  filenames. An ATL03 granule is identified by its RGT + cycle + region, so the (RGT, cycle) pairs
  pin down the granules up to region.
- **Failed granules:** `data/icesat2_segments20m_failed_granules.csv`. The fetch log
  (`data/phase3a_stdout.txt`) has **99 `H5Coro::Future read failure` alerts**, which are **25
  unique ATL03 granules**, each failing on one or more beams, across 16 tiles. Those beams
  contributed no segments. The segment counts are therefore a lower bound on what exists, and a
  re-fetch on a day the granules read cleanly would return more.

## Environment and external code (added 2026-09-23, fresh-checkout audit)

A fresh clone of HEAD was statically audited: every tracked `.py` file was parsed and every
import resolved. **Every in-repo import and hard-coded script path resolves.** Two things are
needed from outside the repo.

**1. Python environment: use `requirements-research.txt`, not `pyproject.toml` alone.**
- The committed `pyproject.toml` declares only `pyproj`, `rasterio` and `sliderule` (plus an
  `ml` extra). The code also imports pandas, geopandas, earthengine-api, fastapi, opencv,
  matplotlib, huggingface-hub, safetensors, shapely, onnxruntime, albumentations, httpx,
  pydantic, python-dotenv, joblib, affine, torchvision and scikit-learn.
- `requirements-research.txt` is a `uv pip freeze` of the exact `.venv` (Python 3.11.16) that
  produced every result in this repo: 113 pinned packages, with all 26 third-party top-level
  imports verified covered.
- Install:
  `uv venv --python 3.11 && uv pip install -r requirements-research.txt`.
- The owner's uncommitted `pyproject.toml` edit (adds `earthengine-api`) was deliberately left
  untouched.

**2. External code under `external/` (untracked by the project's clone-on-demand policy).** Only
these three are *imported*:

| directory | imported by | source / pin |
|---|---|---|
| `external/RDAH-Net/` (`test.py`, `loaddata.py`, `nyu_transform.py`, checkpoints) | `scripts/{run_rdah_probe,run_rdah_scale_sweep,train_rdah_spatial_cv,train_rdah_quadrant_cv,evaluate_rdah_pooled_cv,diagnose_rdah_fold0}.py` | Not a git clone, and the upstream code URL is **not recorded** anywhere in this repo. Provenance: paper DOI 10.3390/rs18071024; checkpoints from Figshare DOI 10.6084/m9.figshare.31986864. MD5 `104best_model.pth` = `4fdd8769d2a05aee0ed40234aeceee09` (Track1), `swiss_best_model.pth` = `a7e8a7933d8190058d2e117e3573e3cb` (Swiss); both match Figshare. Every file's SHA-256 is in `data/EXTERNAL_CODE_sha256.csv`. |
| `external/dinov3/` | `scripts/lib_dinov3_sat493m_loader.py` (`dinov3.hub.*`, `dinov3.eval.depth.models`) | `github.com/facebookresearch/dinov3` (vendored unmodified). Not a git clone, so there's no commit pin; content hashes in `data/EXTERNAL_CODE_sha256.csv`. Weights via `HF_TOKEN` (gated SAT493M). |
| `external/SynRS3D/` | `scripts/rs3dada_{mps,sentinel2}_smoketest.py` (`models.dpt`) | `git clone https://github.com/JTRNEO/SynRS3D.git && git -C external/SynRS3D checkout ab5a4857825448e4c5cd1fe3aa3045c7348814c5` |

**3. Housekeeping flag.** `scripts 2/` is a Finder-style duplicate folder from 2026-09-20,
swept into git by `fd9fd35`.
- Its `evaluate_method4.py` is byte-identical to `scripts/evaluate_method4.py`.
- `evaluate_method4_v2_kaggle.py` exists only there, and nothing imports it.

It's left in place as the owner's file, and flagged rather than deleted.

## VHR DSM pipeline outputs (added 2026-09-23)

`python scripts/vhr_dsm_pipeline.py` regenerates everything below. It needs the Maxar scenes
under the path the script reads, the seed-43 Method 6 fold checkpoints plus
`method6_full_checkpoint/method6_full_dfc2019.pt`, and Earth Engine access for FABDEM and GEDI.
GLO-30 comes from the public AWS COGs.

- **Committed:** `data/vhr_dsm/summary.json`, `data/vhr_dsm/<crop>/stats.json`, and
  `data/vhr_dsm/<crop>/panel_rgb_agl_dsmhs_dtmhs.png`.
- **Not committed (large):**
  - Per crop, about 95 MB of 2048² float32 GeoTIFFs: `agl`, `agl_std`, `sigma`, `agl_fullckpt`,
    `dtm_fabdem`, `dsm`, `glo30`.
  - `zoom_*.png`, and `data/vhr_dsm/_overview/`.
  - The viewer assets in `frontend/public/data/vhr/<crop>/` (about 29 MB in total), which
    `frontend/vhr_preview.html` needs in order to render.

## Deleted to reclaim disk (2026-09-23): closed-route artifacts, all recoverable

**Scope.** These files were deleted **after** this entry was committed. All were untracked by git. None is read by any pending
script: Parts A–F of `docs/method-audit/07-gamus-generalization/` were checked. For Part F variant B, `chmv2_mean` comes from
the saved `data/sentinel2_benchmark/dinov3_depth/*.npy`, which is already in `terrain_rf_residual/samples.parquet`
(1,241,260/1,241,260 non-null), so no live CHMv2 inference is needed.

**Totals and checksums.**
- 20 inventoried items; 8,188,481,521 bytes by file size.
- Size and SHA-256 of each file are appended to `data/REGENERATION_sha256.csv` under the same paths.
- For the two external repos, the whole `.git/` directory was removed, not just the pack file, so that no repo is left corrupt.
  The working trees are kept as plain snapshots.

| # | Deleted path(s) | What it was | Closed route it belonged to | Recovery |
|---|---|---|---|---|
| 1 | `external/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth` (1.47 GB) | RS3DAda ViT-L DPT height checkpoint | **RS3DAda rejected:** 49/50 DFC2019 benchmark tiles are in its training split (contaminated), and it is degenerate on Sentinel-2 (HANDOFF §4) | https://huggingface.co/JTRNEO/RS3DAda/blob/main/RS3DAda_vitl_DPT_height.pth → back to `external/SynRS3D/pretrain/` |
| 2 | `data/gamus_dc/test_tiles.npz` (1.37 GB) and `data/gamus_dc/train_quadrants.npz` (0.55 GB) | Leakage-safe GAMUS-DC arrays for the Method 6 + GAMUS-DC pilot retrain | **GAMUS-DC pilot, closed and not adopted.** The tall-tree criterion failed because the imagery is leaf-off. The 07 C.0 pre-check then found every GAMUS city leaf-off or mixed, so no GAMUS retrain is planned (`07-.../log.md`) | `uv run --no-project --with h5py --with numpy python scripts/prepare_gamus_dc.py` (streams HF `earthflow/GAMUS`; deterministic split, seed 0; `split.json` is kept) |
| 3 | `models/hub/hf_dinov3_sat493m/model.safetensors` (1.21 GB) | DINOv3 ViT-L SAT-493M backbone weights | **DAv2-vs-DINOv3 comparison, superseded.** No backbone is carried forward, because no 10 m RGB detail source adds anything over the DEM (Phase 4) (`sentinel2/backbone-comparison.md`) | HF `facebook/dinov3-vitl16-pretrain-sat493m` (gated; `HF_TOKEN` has access). Conversion: `scripts/lib_dinov3_sat493m_loader.py` |
| 4 | `kaggle/gamus_b_kaggle_bundle.zip` (1.19 GB) | Part B Kaggle upload bundle | Part B **complete** (results merged and committed, `6e820c7`) | Rebuild: `kaggle/bundle/` (script, `read.md`, hashes, done list) plus hard-linked checkpoints; `zip -r -0 gamus_b_kaggle_bundle.zip bundle` |
| 5 | `kaggle_phase2.5_package.zip` (0.65 GB) | Kaggle export for the Method 4 v2 SID ordinal-constraint follow-up | **Method 4 superseded** by Method 6 | The unzipped `kaggle_phase2.5_package/` folder is still in the repo; re-zip it if needed |
| 6 | `models/semantic/hotosm_dinov3s_buildings/model.onnx` (0.22 GB) | HOT OSM DINOv3-S building-segmentation model | **Semantic prior closed** (Method 3; Sentinel-2 phase 2.3) | HF `hotosm/dinov3s-buildings` (`hf_hub_download` in `scripts/generate_dfc_building_prior.py`) |
| 7 | **Duplicate group (SHA-256 `b12e22b4…`, 180,352,229 B each):** `data/sentinel2/semantic_sources/globalml_building_footprints/raw/kolkata/part-00108-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` and `…/raw/bardhaman/part-00108-…` | MS Global ML Building Footprints, India partition | **Semantic prior / building-footprint route closed** (phase 2.3; Open-Buildings CNN failure) | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133321/part-00108-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz (index: `…/globalml_building_footprints/dataset-links.csv`, kept) |
| 8 | `data/sentinel2/…/raw/kolkata/part-00047-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` (169 MB) | Same dataset | Same | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133323/part-00047-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz |
| 9 | `models/hub/checkpoints/dinov3_vitl16_chmv2_dpt_head-3703d643.pth` (135 MB) | DINOv3 CHMv2 canopy-height DPT head | **CHMv2 closed** as a detail source (Phase 4) and as a DEM add-on (A3). Its outputs are saved in `sentinel2_benchmark/dinov3_depth/` | HF `facebook/dinov3-vitl16-chmv2-dpt-head`, or https://ai.meta.com/resources/models-and-libraries/chmv2-downloads/ |
| 10 | **Duplicate group (SHA-256 `e4a19ea2…`, 131,950,214 B each):** `data/sentinel2/…/raw/delhi/part-00170-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` and `data/sentinel2_benchmark/semantic_sources/globalml_building_footprints/raw/delhi/123121303_part-00170-…` | MS Global ML Building Footprints | Semantic prior closed | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123121303/part-00170-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz |
| 11 | `external/DepthWizard-SIH26175/.git/` (95 MB; the 99 MB pack was in it) | Git history of an audited competitor repo | Competitor audit done (GSD-FiLM rejected; the height-balanced idea was adopted into Method 6 and is reimplemented here). History isn't needed | `git clone https://github.com/devendrakushwah80/DepthWizard-SIH26175.git && git -C DepthWizard-SIH26175 checkout 558ddde63847a2d58426a802c628486ff6093bed` |
| 12 | **Duplicate group (SHA-256 `5e81306f…`, 93,708,060 B each):** `data/sentinel2/…/raw/bengaluru/part-00014-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` and `data/sentinel2_benchmark/…/raw/bengaluru/123303312_part-00014-…` | MS Global ML Building Footprints | Semantic prior closed | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123303312/part-00014-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz |
| 13 | `data/sentinel2/…/raw/bardhaman/part-00066-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` (90 MB) | Same dataset | Same | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133303/part-00066-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz |
| 14 | `data/elevation/darjeeling/Darjeeling_Copernicus_GLO30_DSM.tif` (85 MB) | The full N26 + N27 GLO-30 mosaic from the Darjeeling fetch-bug diagnosis | Diagnosis resolved (HANDOFF §2b). The committed `…_cropped.tif` is what's used, and is byte-identical to the demo DEM | `python scripts/fetch_glo30_darjeeling.py` (public AWS `copernicus-dem-30m`) |
| 15 | `data/sentinel2/…/raw/bardhaman/part-00180-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` (74 MB) | Same dataset | Same | https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133302/part-00180-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz |
| 16 | `external/ArnabTechiee-depthwizard/.git/` (67 MB; the 70 MB pack was in it) | Git history of an audited competitor repo | **Shadow-geometry photogrammetry rejected** at 10 m (HANDOFF §4) | `git clone https://github.com/ArnabTechiee/depthwizard.git ArnabTechiee-depthwizard && git -C ArnabTechiee-depthwizard checkout 664e92bf625b0bbedfcdb52abfede791d788c394` |

Rows 2, 7, 10 and 12 each cover two paths, so the 16 rows cover the 20 inventoried paths.

## Deleted to reclaim disk, part 2 (2026-09-23): `external/` reference repos

**Scope.**
- These are 11 folders under `external/`, plus the two duplicate RDAH-Net checkpoints. The `external/RDAH-Net/` folder itself is **kept**.
- **Checked before deletion:**
  - Nothing in the pending 07 work uses these folders: the Part F merge (`scripts/terrain_rf_residual.py --from-kaggle`,
    reads only `samples.parquet`) and the docs step.
  - The only references in `scripts/` are provenance docstrings and comments ("ported from…", "read, never executed").
  - Two closed-route scripts import code from the folders: `scripts/rs3dada_{mps,sentinel2}_smoketest.py` needs
    `external/SynRS3D`, and `scripts/lib_dinov3_sat493m_loader.py` needs `external/dinov3`. **Re-clone first if either is ever re-run.**
- **Policy:** clone on demand (HANDOFF §9). Restore with the command in the last column, run from `external/`.

**RDAH-Net duplicate checkpoints, replaced by symlinks (not lost).**
- `external/RDAH-Net/104best_model.pth` (SHA-256 4c940ef3a3f2cea5…) and `external/RDAH-Net/swiss_best_model.pth` (SHA-256 6efd4ac5655760e6…) were
  **SHA-256 byte-identical** to `data/checkpoints/rdah/checkpoints-track1/104best_model.pth` and `…/checkpoints-Swiss/best_model.pth`.
- Each is now a relative symlink to that copy, so every `RDAH_DIR / "104best_model.pth"` path in `scripts/` keeps working.
- Freed about 131 MB.
- Original provenance: Figshare DOI 10.6084/m9.figshare.31986864 (MD5 in the "External code" section above).

| Folder | Source | Commit | Contributed to this project | Restore (from `external/`) |
|---|---|---|---|---|
| `sih2026-depthwizard` | github.com/zaidnansari2011/sih2026-depthwizard | c5646628ebec22b3f66b590b20ad5deda4112525 | **Source of Method 6's idea:** a full DAv2 fine-tune with a twin head, reimplemented here | `git clone https://github.com/zaidnansari2011/sih2026-depthwizard.git sih2026-depthwizard && git -C sih2026-depthwizard checkout c5646628ebec22b3f66b590b20ad5deda4112525` |
| `depthwizard` | github.com/blakc-coffee/depthwizard | 55bc2c0d61f94f430205b4c5390643be3c2f961d | Source of frequency fusion, **retired**: it equals its DEM-only control | `git clone https://github.com/blakc-coffee/depthwizard.git depthwizard && git -C depthwizard checkout 55bc2c0d61f94f430205b4c5390643be3c2f961d` |
| `DepthWizard-SIH26175` (snapshot; `.git` removed earlier today) | github.com/devendrakushwah80/DepthWizard-SIH26175 | 558ddde63847a2d58426a802c628486ff6093bed | **Height-balanced loss and sampler adopted into Method 6** (reimplemented); GSD-FiLM rejected | `git clone https://github.com/devendrakushwah80/DepthWizard-SIH26175.git && git -C DepthWizard-SIH26175 checkout 558ddde63847a2d58426a802c628486ff6093bed` |
| `ArnabTechiee-depthwizard` (snapshot; `.git` removed earlier today) | github.com/ArnabTechiee/depthwizard | 664e92bf625b0bbedfcdb52abfede791d788c394 | Shadow-geometry photogrammetry, **rejected** at 10 m | `git clone https://github.com/ArnabTechiee/depthwizard.git ArnabTechiee-depthwizard && git -C ArnabTechiee-depthwizard checkout 664e92bf625b0bbedfcdb52abfede791d788c394` |
| `SynRS3D` | github.com/JTRNEO/SynRS3D | ab5a4857825448e4c5cd1fe3aa3045c7348814c5 | RS3DAda, **rejected** as contaminated (49/50 DFC2019 tiles in its training split). The only local changes were `__pycache__` | `git clone https://github.com/JTRNEO/SynRS3D.git && git -C SynRS3D checkout ab5a4857825448e4c5cd1fe3aa3045c7348814c5`; weights: part 1, row 1 |
| `madhu-mitha-e-depthwizard` | github.com/madhu-mitha-e/DepthWizard | 92cd97f62290cf0a8643af2699bc24a72fa8d225 | Audited, nothing adopted (`COMPETITIVE_REPO_AUDIT.md`) | `git clone https://github.com/madhu-mitha-e/DepthWizard.git madhu-mitha-e-depthwizard && git -C madhu-mitha-e-depthwizard checkout 92cd97f62290cf0a8643af2699bc24a72fa8d225` |
| `gowthamkrishna27-elevate3d` | github.com/gowthamkrishna27/Elevate3d | 7efa6433583cf36e116fef1c3ac7e8f463335a66 | Audited, nothing adopted | `git clone https://github.com/gowthamkrishna27/Elevate3d.git gowthamkrishna27-elevate3d && git -C gowthamkrishna27-elevate3d checkout 7efa6433583cf36e116fef1c3ac7e8f463335a66` |
| `amogh-hub-depthwizard` | github.com/amogh-hub/depthwizard | 2b2bd2653989403ddf5f468e09a8fdf10e697759 | Evidence gating and leave-one-out validation, adapted as LOBO on frequency fusion (retired with it) | `git clone https://github.com/amogh-hub/depthwizard.git amogh-hub-depthwizard && git -C amogh-hub-depthwizard checkout 2b2bd2653989403ddf5f468e09a8fdf10e697759` |
| `yats0x7-depthwizard` | github.com/yats0x7/DepthWizard | 116a205e2572c272cb924d9465b648163cf74868 | Ground-trend scale: compared, converges with blakc-coffee, not adopted | `git clone https://github.com/yats0x7/DepthWizard.git yats0x7-depthwizard && git -C yats0x7-depthwizard checkout 116a205e2572c272cb924d9465b648163cf74868` |
| `arpitparashar06-depthwizard` | github.com/arpitparashar06/depthwizard | 5eea1c3906408debd7b8d2bb57e5a8183a3fac62 | Non-regression scale derivation, **rejected** | `git clone https://github.com/arpitparashar06/depthwizard.git arpitparashar06-depthwizard && git -C arpitparashar06-depthwizard checkout 5eea1c3906408debd7b8d2bb57e5a8183a3fac62` |
| `dinov3` (vendored code, **was git-tracked**, 213 files) | github.com/facebookresearch/dinov3 (vendored unmodified; no upstream pin) | this repo's commit `ee30f7f` | The DAv2-vs-DINOv3 backbone comparison, **superseded**: no backbone is carried forward | **Exact copy from this repo's own history:** `git checkout ee30f7f -- external/dinov3` (run from the repo root). Removed with `git rm`, so the history keeps every file |
