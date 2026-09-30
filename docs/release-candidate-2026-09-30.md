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

## Part 2: re-measure on amd64 (2026-09-30)

- **Image:** `dw2-backend:rc-amd64`, built from this branch with `docker buildx build --platform linux/amd64 -f
  docker/Dockerfile`, which includes Parts 3–4.
- **Machine:** Colima (VZ + Rosetta) on Apple Silicon, so timings are indicative.
- **Tool:** `scripts/container_bench.py`, which gained a `cdse_facts` flow and a `concurrent` command. Results are in
  `build/slim/results/rc_amd64_*.json` (gitignored).

| | slim-container (final_amd64) | **RC** |
|---|---|---|
| Image, compressed | 116.7 MB | **119.1 MB** (+2.4 MB: the ThinkHazard bundle) |
| Image, unpacked | 331.5 MB | **335.9 MB** |
| Start → first `/health` 200 (median of 5) | — | **0.75 s** |
| Idle RSS / cgroup (median of 5) | 80.0 / 68.6 MiB | **79.5 / 65.5 MiB** |
| PNG flow: peak RSS / cgroup peak / /tmp | 157.7 / 129.2 / 4.5 MiB | **156.4 / 166.3 / 4.5 MiB**, 9.2 s |
| GeoTIFF flow | 233.9 / 181.8 / 5.2 MiB | **232.2 / 205.6 / 5.2 MiB**, 20.0 s |
| CDSE flow | — | **231.0 / 177.7 / 3.7 MiB**, 104.8 s (the upstream CDSE / EE / Space were slow this run; 47.8 s in the next run) |
| **CDSE + live Facts & Scenario** (`/api/facts?bbox=` of the scene) | — | **250.9 / 200.7 / 3.7 MiB**; the facts call took 5.0 s, 10 lines, sources ok (Wikidata empty in the Punjab plains) |
| **3 flows at once in one container** (PNG + GeoTIFF + CDSE+facts) | 287.2 / 240.1 / 12.3 MiB | **287.6 / 246.5 / 13.4 MiB**, 21.5 s |

- **Verdict: fits 512 MiB with margin.**
  - The worst measured peak (3 concurrent flows including the live facts) is 287.6 MiB RSS / 246.5 MiB cgroup. That
    is **~225 MiB (44%) below 512 MiB**.
  - A 4th concurrent request (concurrency 4) adds roughly one flow's increment, about 70–90 MiB, giving an estimated
    ~360–380 MiB.
  - Add the /tmp cap's worst case (64 MiB + in-flight entries, ~20 MiB): **an estimated ≈ 450 MiB at the very worst,
    still under 512**. That rests on the cap below, and it is an estimate: 4-way concurrency was not measured.
- **Parity vs slim-container: identical on every flow** (`scripts/flow_parity.py`; `build/slim/results/parity_slim_vs_rc.txt`):
  - PNG and GeoTIFF vs `final_amd64`;
  - CDSE and CDSE+facts vs `cdse_fix`. That run is the slim-container run *after* its own CDSE 8×8 → 375×324 fix
    `5647668`; the older `final_amd64` CDSE predates that fix, so it differs as expected;
  - the concurrent run vs the single runs.

  Checked: the depth payload, all PNGs pixel-identical, terrain heights max |diff| 0, FABDEM stats, and response meta.

## Part 3: the /tmp cap (2026-09-30)

- **Why:** on Cloud Run the whole writable filesystem, including /tmp, is instance memory.
- **Where:** `backend/storage/tmp_cap.py` manages `DW2_GENERATED_DIR` (one folder per generated job) and
  `DW2_UPLOADS_DIR` (one folder per upload or searched scene).
- **Rules:**
  - an entry's "last used" time is its folder mtime, refreshed on every access;
  - a sweep deletes entries **idle > 30 min**, then more of the **oldest while the total exceeds 64 MiB**;
  - the sweep is rate-limited to once per 30 s and is triggered by the writers (upload, scene, generate).
- **Never deleted:**
  1. entries held by an in-flight request (a reference count):
     - the generate request holds its input for the whole request, including a slow Space call;
     - `generate()` holds its new job;
     - the FABDEM fetch and a DEM upload hold their input;
  2. entries used in the **last 2 min**. That covers `FileResponse` downloads that stream after the handler returns,
     and the gaps between one input's requests (upload → FABDEM → generate).
  Under pressure it logs "over_cap_but_protected" rather than delete them.
- **On only in the image** (`DW2_TMP_CAP=1` in `docker/Dockerfile`). The desktop app uses the same folder variables
  for its per-user cache and keeps its files.
- **Values, from measured sizes** (/tmp after a flow: PNG 4.5, GeoTIFF 5.2, CDSE 3.7 MiB; 3 concurrent 13.4 MiB):
  - **64 MiB** holds about 12–17 recent jobs, far more than one instance serves in 30 min of demo traffic.
    Together with the measured worst peak (287.6 MiB), it leaves more than 150 MiB of the 512 MiB.
  - **30 min** is well beyond a session's upload → generate → view span; the browser fetches assets right after
    generation.
  - **2 min** of grace is far longer than any asset download (≤ 5 MB).
  - Overridable with `DW2_TMP_MAX_MB`, `DW2_TMP_MAX_AGE_S` and `DW2_TMP_MIN_IDLE_S`.
- **Tests:** `backend/tests/test_tmp_cap.py`, 8 tests:
  - age;
  - size, oldest first;
  - an in-flight entry that is old and over the cap is kept;
  - a fresh entry under pressure is kept;
  - nested holds;
  - off unless enabled, and rate-limited;
  - 4 concurrent holders while 2 threads sweep continuously (nothing lost);
  - `generate()` holds its job and input.
- `HF_HOME=/tmp/dw2/hf` puts any Hugging Face download in /tmp. It is used only by the remote-mode library route for
  private packs, which the static web build doesn't call. It is **not capped**: see the open items.

## Part 4: Cloud Run compatibility audit (2026-09-30)

| Check | Result |
|---|---|
| Listens on 0.0.0.0:$PORT | ✅ `CMD … uvicorn … --host 0.0.0.0 --port ${PORT}`, with `PORT=8080` as the default |
| Health endpoint | ✅ `GET /health` → `{"ok": true}`: no I/O and no imports. Start to first 200: 0.75 s |
| Clean shutdown on SIGTERM | ✅ **fixed and verified**: `exec` makes uvicorn PID 1; added `--timeout-graceful-shutdown 8` (Cloud Run allows 10 s). `docker stop` → "Shutting down … Finished server process" in 0.41 s, exit code 0 |
| Logs to stdout | ✅ **fixed and verified**: uvicorn logs to stderr by default. `backend/log_config.json` (`--log-config`) sends app, uvicorn and access logs to stdout: 11 lines on stdout, 0 on stderr |
| Longest flow vs request timeout | CDSE: 29–105 s measured, dominated by upstream CDSE / EE / Space latency; the Space's own timeout is 120 s (`DAV2_TIMEOUT_S`). **Deploy with `--timeout 300`** |
| Thread safety at concurrency 4 | ✅ facts, CDSE-token, depth-client, audit and ThinkHazard caches are locked; catalog / manifest / library memos swap whole values (a worst-case extra reload). **Fixed:** Earth Engine init now has a lock (it could double-initialise). Temp files: every upload and job has its own uuid folder; the cap is locked and in-use aware. `depth_routes` uses `TemporaryDirectory` per request |
| CORS allowlist, no wildcard | ✅ exact `CORS_ORIGINS` only; `CORS_ORIGIN_REGEX` left unset. **Fixed:** a `*` entry is now dropped with a warning (`test_cors.py`). Verified: the Vercel origin gets `Access-Control-Allow-Origin`, another origin gets none. Value: `https://depthwizard-studio.vercel.app,http://localhost:5173,http://127.0.0.1:5173` |
| Non-root user | ✅ `USER app` (uid 10001), verified with `id` in the container |
| Nothing reads files outside the container | ✅ the image contains `backend/` + venv only (`.dockerignore` whitelist). All `data/…` paths in code are dev-only (`DW2_LIBRARY=local`) or overridden to /tmp by the image env; no `.env` is read (`DW2_NO_DOTENV=1`). **Fixed:** `hf_checkpoints.token()` no longer falls back to a `.env` when `DW2_NO_DOTENV=1`. EE credentials come from the metadata server (ADC), not a file |

## Part 5: config and secrets inventory (2026-09-30)

Every variable the backend reads (`grep os.environ` over `backend/`, excluding tests), plus the libraries' own:

| Variable | Secret? | Cloud Run value |
|---|---|---|
| `HF_TOKEN` | **SECRET** | Secret Manager `hf-token`. Used for the Space calls (the ZeroGPU quota is billed to the token) and the private HF dataset / checkpoints |
| `CDSE_CLIENT_ID` | **SECRET** (treat as one) | Secret Manager `cdse-client-id` |
| `CDSE_CLIENT_SECRET` | **SECRET** | Secret Manager `cdse-client-secret` |
| `EARTHENGINE_PROJECT` | no | the EE-registered Cloud project id |
| `CORS_ORIGINS` | no | `https://depthwizard-studio.vercel.app,http://localhost:5173,http://127.0.0.1:5173` |
| `CORS_ORIGIN_REGEX` | no | unset (no wildcard) |
| `DAV2_INFERENCE_URL` / `DAV2_FALLBACK_URL` / `DAV2_TIMEOUT_S` | no | unset → the Space `sancharimouri/DepthWizard2`, no fallback, 120 s |
| `DW2_NO_DOTENV`, `DW2_UPLOADS_DIR`, `DW2_CACHE_DIR`, `DW2_GENERATED_DIR`, `DW2_TMP_CAP`, `HF_HOME`, `PORT` | no | baked into the image (`docker/Dockerfile`); Cloud Run sets `PORT` |
| `DW2_TMP_MAX_MB` / `DW2_TMP_MAX_AGE_S` / `DW2_TMP_MIN_IDLE_S` | no | unset (64 MiB / 30 min / 2 min) |
| `DW2_LIBRARY` (`remote`), `DW2_LIBRARY_DATASET`, `DW2_ASSETS_REPO`, `DW2_ASSETS_TAG`, `DW2_MANIFEST_TTL`, `DW2_CKPT_REPO` | no | unset (the defaults) |
| `DW2_LIBRARY_BUNDLE`, `DW2_LIBRARY_USER`, `DW2_TILE_MANIFEST` | no | unset (desktop / bake only) |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (secrets), `R2_BUCKET`, `R2_PUBLIC_BASE_URL`, `R2_URL_TTL` | — | unset (R2 is dormant) |
| `CURL_CA_BUNDLE`, `HF_HUB_DISABLE_TELEMETRY` | no | set internally |

**Earth Engine:**
- **Today:** `backend/dem/fabdem.py` calls `ee.Initialize(project=EARTHENGINE_PROJECT)`. Locally, credentials come
  from the owner's `~/.config/earthengine` (user OAuth; `earthengine authenticate`), and the bench mounts it read-only.
- **On Cloud Run:** there is no such file, and earthengine-api 1.7.43 then falls back to **Application Default
  Credentials**, the service's runtime service account, read from `ee.data.get_persistent_credentials`. **No code
  change and no key file.**
- **What's needed:**
  1. an EE-registered Cloud project (noncommercial registration);
  2. a service account `dw2-backend@<RUN_PROJECT>.iam.gserviceaccount.com`, set as the Cloud Run service's identity;
  3. on the EE project, `roles/earthengine.viewer` (FABDEM is read-only) and `roles/serviceusage.serviceUsageConsumer`
     for the service account;
  4. `EARTHENGINE_PROJECT=<EE project id>`.
- **Reuse or register?** **Recommendation: reuse the existing `EARTHENGINE_PROJECT` if it is a Cloud project you own**
  (EE projects are Cloud projects).
  - Best: deploy Cloud Run *in that same project*, so it is one project, one bill and one set of IAM.
  - If you create a new project for Cloud Run, either register it for EE too (and set `EARTHENGINE_PROJECT` to it),
    or grant the new service account the two roles on the existing EE project (cross-project).
  - Owner decision; the runbook covers both.

**HF token:** Secret Manager `hf-token`, mounted as the `HF_TOKEN` env var. A **read** token is enough (Space calls +
private dataset / model reads). The owner's write token must not be used.

**Backend URL in the web build:**
- `VITE_API_BASE` is baked at build time (`frontend/src/api-base.js`). It is a **Vercel Production env var**, not in
  the repo; today it is `https://depthwizard2-api.onrender.com`.
- **Cutover (not done now):** set Vercel `VITE_API_BASE` to the Cloud Run URL and redeploy the frontend, because Vite
  inlines it at build time. Rollback is the reverse.
- The desktop build is unaffected (`VITE_API_BASE=http://127.0.0.1:8765`, its sidecar).

## Part 7: model size inventory (read-only, 2026-09-30)

**HF Space** `sancharimouri/DepthWizard2` (read with `huggingface_hub`):
- **Hardware:** Gradio SDK 6.28.0, Python 3.12, **`zero-a10g` (ZeroGPU)**, RUNNING.
- **Files:** the repo is 4 small files (`app.py` 3 KB, `README.md`, `requirements.txt`, `.gitattributes`).
- **Model:** it serves **Depth-Anything-V2-Small** (`depth-anything/Depth-Anything-V2-Small-hf`, downloaded at run
  time: `model.safetensors` **99.2 MB**) through `@spaces.GPU(duration=30)`, returning u16-zlib relative depth
  at 518 × 518.
- **Requirements:** torch 2.13.0, transformers 5.17.0, safetensors 0.8.0, numpy 2.4.6, pillow 12.3.0,
  huggingface-hub 1.30.0.

**Local:**
- **DAv2-Small ONNX:** one file, `desktop/tauri/src-tauri/sidecar/dw2-backend/_internal/models/dav2_small.onnx`,
  **99.1 MB**. Fixed input 1 × 3 × 518 × 518, opset 17 (`desktop/freeze_trial/export_onnx.py`).
- **Method 6 checkpoints** (PyTorch `.pt`, **99.2 MB each**, the DAv2-Small backbone + neck + twin head):

  | Set | Files |
  |---|---|
  | `data/dfc2019/experiments/method6_height_balanced_seed42_ckpt/` | fold0–3 (4 × 99.2 MB) |
  | `data/dfc2019/experiments/method6_height_balanced_seed43/` | fold0–3 (4 × 99.2 MB) |
  | `data/dfc2019/experiments/method6_height_balanced_seed44/` | fold0–3 (4 × 99.2 MB) |
  | `data/dfc2019/experiments/method6_hb_gamusdc_seed43/` | fold0–3 (4 × 99.2 MB) |
  | `data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt` | 1 × 99.3 MB |

  17 files, ~1.69 GB. The private HF repo `sancharimouri/depthwizard2-method6` holds 5 of them: `full_dfc2019` +
  `height_balanced_seed43` fold0–3, 496.2 MB in total.
- **ONNX Runtime 1.30.0 CPU wheel** (cp311, from PyPI): **23.6 MB** for manylinux x86_64 (amd64), 21.3 MB for aarch64.

**Method 6 ONNX: none exists.** Exporting it would involve:
1. Build `TwinHeadDav2` (`scripts/evaluate_method6_finetune_twinhead.py:112`) and load a checkpoint's `state_dict`
   (e.g. `method6_full_dfc2019.pt`, the deployable all-tile model).
2. `torch.onnx.export` with input `1×3×518×518`, outputs `(mu, log_var)`, opset 17, like
   `desktop/freeze_trial/export_onnx.py`. Optionally make H and W dynamic (multiples of 14) if the VHR pipeline's
   512 px windows are exported as-is.
3. A parity check against PyTorch on real tiles, like `desktop/freeze_trial/parity_onnx.py`. Record the max |diff| in
   metres of AGL.
4. Keep the sliding window + blending (`scripts/vhr_dsm_pipeline.py`: 512 px windows, stride 448, ramp 64, margin
   mode) in Python around the session.

About 99 MB at fp32 (≈ 50 MB fp16; int8 would need its own accuracy check). Not exported now.

**Model service estimate (DAv2-Small + Method 6, ONNX, CPU, 1 vCPU).** DAv2-Small was **measured** in an amd64
`python:3.11-slim` container with only onnxruntime 1.30.0, numpy and pillow, the model mounted, one 1024 × 1024
input resized to 518:

| | DAv2-Small alone (**measured**) | + Method 6 (**estimate**) |
|---|---|---|
| Image, unpacked | runtime 105.0 MB (base 45.9 + ORT/numpy/pillow) + model 99.1 = **≈ 204 MB** | + 99 MB = **≈ 303 MB** (≈ 230 MB compressed; weights barely compress) |
| Idle, after the session(s) load | **214 MiB RSS** (68 MiB Python + libraries, +146 MiB per session) | **≈ 360 MiB** (+146 for the second session) |
| Peak, one inference each | **432 MiB** with ORT's default memory arena; **354 MiB** with the arena off (317 MiB after); 3.0 s per 518 pass (Rosetta, indicative) | **≈ 500 MiB** with the arena off (both sessions resident + one pass's activations). Method 6 on a 1024 × 1024 VHR image is ≈ 9 window passes, run sequentially (≈ 9× the time, not the memory) |

**Implication:** DAv2-Small alone fits a 512 MiB service with the arena disabled. **DAv2 + Method 6 together does
not fit 512 MiB with a safe margin (estimate ≈ 500 MiB): plan 1 GiB**, or load Method 6 on demand.

## Part 6: runbook (2026-09-30)

`docs/deploy-cloud-run.md` is rewritten as a copy-paste runbook. Every value to fill in is marked `<<…>>` and set
once in section 0. It covers:
- project and billing;
- the $250 budget with 20 / 60 / 100% thresholds = **$50 / $150 / $250**;
- the APIs;
- EE registration (reuse vs new) and the service account with `earthengine.viewer` + `serviceUsageConsumer`;
- Secret Manager (`hf-token`, `cdse-client-id`, `cdse-client-secret`, entered at `read -s` prompts);
- Artifact Registry with a **keep-last-2** cleanup policy;
- an amd64 buildx push;
- deploy flags: **512Mi, 1 vCPU, concurrency 4, min 0, max 3, CPU boost, timeout 300 (Part 4), gen2**, env with
  the `^@^` delimiter, and secrets;
- curl smoke tests: health, CORS allowed / denied, facts, PNG upload → generate, GeoTIFF → FABDEM (EE), CDSE search,
  logs;
- the cold-start measurement and its ≤ 5 s rule;
- the `VITE_API_BASE` cutover;
- min-instances 1 / 0;
- **ROLLBACK:** Render stays up; switch Vercel back, route Cloud Run traffic to the previous revision, or set
  `--max-instances 0`.

## Part 8: docs and verification (2026-09-30)

- **HANDOFF:** §5z+5 carries the build state, the deploy readiness checklist, the open items and the decisions
  waiting on the owner.
- **Protected data** (`data/library/`, `data/library_v2_2026-09-29/` except `_qa/`, `data/dfc2019/terrain_packs/`,
  `data/display_test_2026-09-29/`, `data/sentinel2/`): SHA-256 before vs after **identical, 1,865 of 1,865 files**.
  The file list is unchanged: 1,864 plus the owner-approved ratings file from facts-v2.
- The pre-commit hook is active (`core.hooksPath = scripts/git-hooks`) and passed on every commit.
- **Local Docker cleaned up:** the local registry container is removed, and no `dw2` containers are left. The images
  `dw2-backend:rc-amd64` and `dw2-modelsvc-probe` stay in Colima for re-runs.
- **Verdict: READY for a backend-only Cloud Run deploy**, pending the owner's GCP setup (runbook §0–5). There are no
  code blockers. The only unmeasured risks are the real cold start and 4-way concurrency, and both are within margin
  by estimate.

---

# Session 2026-10-01 (same branch; no merge, push, deploy or GCP)

- Owner decisions: no merge or push until cutover day, after Cloud Run is verified; region `asia-south1` and
  max-instances 3 confirmed; the Earth Engine project to be decided during GCP setup.

## Standing rule: the 5 protected links (2026-10-01)

- **Where:** added as a dated top-level rule to `docs/HANDOFF.md` (§0) and to `CLAUDE.md`, which is local and
  gitignored.
- **Check:** `scripts/check_protected_links.sh`:
  - the 5 URLs return HTTP 200 after redirects, with `/releases/latest` resolving to a release tag;
  - the live home page serves `id="demo-video"`;
  - the local web build keeps `id="demo-video"` and routes `page-explore → #/demo`;
  - `src/routes.js` keeps both.
- **Baseline run 2026-10-01: ALL PASS, exit 0.** `/releases/latest` → `.../releases/tag/v1.0.2`.
- It must run before any deploy, push, release or merge.

## Part 1 (2026-10-01): the private DFC2019 library is off on Cloud Run

- **Flag:** `DW2_PRIVATE_LIBRARY=off`, set **only in `docker/Dockerfile`**. The desktop sidecar never sets it: the
  default is on, and its bundle mode is unchanged.
- **Chokepoint:** `library_store._hub_file()` is the single function that downloads from the private HF dataset:
  - the remote catalog manifest behind `/api/library`;
  - DFC2019 thumbnails and previews;
  - the private DEM packs;
  - on-demand tiles.

  With the flag off it raises `PrivateLibraryDisabled` before `huggingface_hub` is imported or called.
- **Responses:** `catalog.load()` and `_image()` let that exception through instead of turning it into a 503.
  `main.py` maps it to **404** "The tile library is served statically by the web app…".
- **Effect on Cloud Run:** every `/api/library*` route and `POST /api/generate/library/*` answers 404. The web build's
  library is static (`library-static/`, 0 backend calls, verified in facts-v2), so no user-facing path changes.
  Uploads, CDSE and Facts are unaffected.
- **Tests:** `backend/tests/test_private_library_off.py`, 3 tests:
  - `_hub_file`, `private_file` and `catalog.load` all refuse, with `hf_hub_download` spied and **never called**, and
    the `HF_HOME` folder never created;
  - the library routes answer 404, not 5xx;
  - the default is on for the desktop.

  Suite: backend 87 passed, 1 skipped.
- **Confirmed in the real container** (`dw2-backend:rc2-amd64`, amd64): PNG, GeoTIFF and CDSE+facts flows, plus the
  library routes (all 404). Then `docker diff` against the image:
  - **no Hugging Face path anywhere**; `/tmp/dw2/hf` and `~/.cache/huggingface` were never created;
  - the app wrote only `/tmp/dw2/gen` and `/tmp/dw2/up` (both under the /tmp cap), plus an empty `/tmp/gradio`
    (4 KB, `gradio_client`);
  - the other two entries exist from start-up, before any request, and are local test artefacts that don't exist on
    Cloud Run: `/home/app/.cache/rosetta` (Colima's Rosetta amd64 emulation) and `/home/app/.config/earthengine` (the
    bench's read-only mount of local EE credentials).
- The HANDOFF open item "HF_HOME not capped" is resolved.
