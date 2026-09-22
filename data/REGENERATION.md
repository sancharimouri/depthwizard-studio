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
