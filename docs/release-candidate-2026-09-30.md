# Release candidate 2026-09-30: backend deploy-ready for Cloud Run (branch `release-candidate-2026-09-30`)

Local only: no deploys, no push, no GCP commands, and nothing merged into `main`. Dated entries per part. The runbook
is `docs/deploy-cloud-run.md`.

## Part 1: integration branch (2026-09-30)

**Ancestry** (`git merge-base --is-ancestor`):

```
main (b34064c, = origin/main)
 ├── curated-tiles-2026-09-29 (7ed9789, +6)
 │    └── slim-container-2026-09-30 (399de60, +12 from main)
 │         └── facts-research-2026-09-30 (da814cf)
 │              └── facts-v2-2026-09-30 (2fe572b)
 └── pack-rebuild-code (f771ac0, +1): desktop/tiles/build_dem_pack.py, scripts/dfc2019_build_packs.py
```

- `release-candidate-2026-09-30` was created from `facts-v2-2026-09-30`, which already contains curated-tiles,
  slim-container and facts-research. Merging those three reported "Already up to date".
- Then `pack-rebuild-code` was merged: merge commit `2d6e696`. The RC contains all five branches (verified).

**Conflicts:** none textual. The two files `pack-rebuild-code` changes were never touched on the facts/slim chain, so
git merged them automatically. The expected conflict in `desktop/tiles/build_dem_pack.py` (offset-only vs calibrated
FABDEM) is instead a layering, checked against the library:

- `build_dem_pack.py` (from `pack-rebuild-code`): the **calibrated** Sentinel-2 recipe, a per-tile OLS fit
  h = a + b·FABDEM against ICESat-2 ground. It is the default recipe: 29 tiles are `ols_a_plus_b_fabdem` in
  `tile_manifest.json`.
- `scripts/build_library_v2.py` (already on the chain): **offset-only** (b = 1, a = median(ICESat-2 − FABDEM)), used
  only for the 3 tiles marked `offset_only_b1` (bhitarkanika, amalapuram, kutch). Its own docstring describes it
  as "build_dem_pack.py's Sentinel-2 recipe with the OLS fit replaced by an offset".
- **Resolution:** keep both, exactly as merged. This is the combination the current `data/library_v2_2026-09-29`
  packs were built with. No code change, and no pack was rebuilt.
- `scripts/dfc2019_build_packs.py` (DFC2019 heights from lidar AGL only) matches the manifest's DFC2019 recipe.

**Checks on the merged RC:**
- Backend tests: 74 passed, 1 skipped. Frontend tests: 104 passed.
- **Desktop smoke test** (`desktop/freeze_trial/dw2_entry.py` from source with the sidecar's own ONNX and
  `library_bundle/`) **PASS**:
  - `--selftest`: HTTP 200, local ONNX depth 518 × 518, and a GeoTIFF upload reads EPSG:32645.
  - Serve mode:
    - `/health` ok;
    - 89 library items, 25 local;
    - `generate/library/{sentinel2-darjeeling, vhr-a_valley}` 200;
    - `/api/facts?bbox=` answered, and the WB landslide read timed out that time while the other 5 sources still
      answered: the fallback working as designed.
  - Tauri's `beforeBuildCommand` builds with no `library-static/`.
- The three real flows (PNG, GeoTIFF, CDSE) with parity are in Part 2, run on the final RC image, which also carries
  the Part 3/4 changes.
