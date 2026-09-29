# Deployment (Render backend + Vercel frontend) — status 2026-09-29

## LIVE (2026-09-29: the 2026-09-28/29 UI/UX sessions, backend fixes and DFC2019 packs are deployed)
- **Deployed 2026-09-29:** GitHub `main` up to `36789c0` (Render autodeploy); Vercel production `depthwizard2-62bvvq0xf`
  (both domains); Space `sancharimouri/DepthWizard2` at `f9c671a` (`ssr_mode=False`).
- **Verified live 2026-09-29** (headless Chrome with the Metal GPU, fresh profile per deck link, DevTools protocol):
  - Deck links: `/` → DW Studio, `#/demo` → Demo, `#demo-video` → Home scrolled to the video section, `#/docs` → Home;
    0 console errors on each. The two GitHub links answer 200; `…/releases/latest` → v1.0.1 until the desktop release.
  - Backend (also in a clean Python 3.11 venv from `requirements.txt`, production mode): `/openapi.json` 200,
    `/api/library` 89 items, `POST /api/input/{id}/gsd` listed; Darjeeling 200 with no pit deeper than 39.7 m below its
    5×5 median (was ~1,850 m); DFC2019 `JAX_004_006` → "curated elevation pack"; 400×300 PNG at 0.5 m → 200 × 150 m plane;
    CORS exactly the two Vercel origins. Peak backend RSS 192 MB over 15 generations (Render free: 512 MB).
  - Generations on the live API: 15/15 Almora (13–29 s) after the retry fixes; the UI flow on the live site: Almora 22.1 s
    end to end (depth request 16.4 s), every box readout 1.32–1.39 s from 0%, the log done before the DEM Elevation box.
  - UI: sidebar labels, D mark → Home, drag 208–360 px and collapse below 160 px, footer links; Terrain dropdown and the
    DFC2019 + Hilly notice (closes itself); PNG upload → GSD card, START waits for it; 12 background icons moving, big
    satellite top-left, earth + satellite bottom-right; a job opened mid-generation (the running job completes: 0→100%);
    viewer edits kept across job switches; a failed job shows "Failed" + Retry and the retry succeeds; low-relief warning
    on Vidisha; wireframe green (dark) and orange (bright), recoloured live; bright mode has no green except the T2 badges.
- **HF edge 502s (2026-09-29):** 8–23% of requests to any `*.hf.space` (also unrelated Spaces) got a 502 from HF's edge.
  The backend re-sends those (`EDGE_RETRY_DELAYS_S`) and retries a Space job cancelled by a broken event stream.

## Earlier state (2026-09-27)
- **Frontend:** https://depthwizard-studio.vercel.app (primary; also https://depthwizard2.vercel.app) (Vercel project `sherry-c508/depthwizard2`, root `frontend/`,
  deployed with `npx vercel deploy --prod` from `frontend/`; `VITE_API_BASE` is a Production env var).
- **Backend:** https://depthwizard2-api.onrender.com (Render `srv-dasjfqvpn0mc738mbqcg`, free plan, Singapore).
  - Source (corrected 2026-09-29): `sancharimouri/depthwizard-studio` is **this repo, PUBLIC** (it's the SIH deck's GitHub
    link). Render builds it on every push to `main`: build `pip install -r requirements.txt` (root file, added 2026-09-29;
    before that every build failed), start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
  - Env (Render only, never committed): `HF_TOKEN`, `CDSE_CLIENT_ID/SECRET`, `DW2_NO_DOTENV=1`, `PYTHON_VERSION=3.11.9`,
    `CORS_ORIGINS=https://depthwizard-studio.vercel.app,https://depthwizard2.vercel.app` (exact; no regex), `DW2_{CACHE,GENERATED,UPLOADS}_DIR` under `/tmp` (ephemeral).
  - Depth: ZeroGPU Space `sancharimouri/DepthWizard2` (default host).
  - Health check: `/openapi.json` (every ~5 s from Render; it was `/api/depth/status`, which queried the HF API on every probe).
  - **Cold start (measured 2026-09-27):** Free instances spin down 15 min after the last inbound request (log: last request
    16:17:39Z, "Shutting down" 16:32:37Z; health checks do not keep it awake). A fresh load of the Vercel site then waited
    **34.6 s** for the first `/api/library` (200); the Library cards appear only after that. Warm, the same call is ~0.8–2 s.
    The first attempt the same evening was invalid (real use inside the idle window).
- **Permanent links:** `https://depthwizard-studio.vercel.app/#/docs`, `…/#demo-video` (`frontend/src/routes.js`). Desktop:
  `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest` (v1.0.1).
- Verified live: `#demo-video` opens Docs and scrolls to it; Library (89) → Almora → START GENERATION → 200, ZeroGPU depth +
  live GLO-30 terrain (1,024–2,042 m), 0 console errors; CORS allows only the production origin.

## PENDING DEPLOY: nothing (emptied 2026-09-29)
- Everything committed up to 2026-09-29 is live: web (Render + Vercel + Space) and desktop v1.0.2. Record:
  `docs/CHANGES_2026-09-28_29.md` "Deployed".
- **SIH deck links** (from `145604_SIH26175.pdf`) must keep working: `https://depthwizard-studio.vercel.app/`,
  `…/#/demo`, `…/#demo-video`, `…/#/docs`, `https://github.com/sancharimouri/depthwizard-studio` (this repo, PUBLIC; Render
  builds its `main`), `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest`.
- **Shipping later changes:** push `main` (Render redeploys; builds from the root `requirements.txt`); `npx vercel deploy
  --prod` from `frontend/`; desktop: bump the version, re-freeze, `build-signed.sh`, `publish-release.sh`.

## Origins (done, commit 1277b8a)
- Backend `CORS_ORIGINS` (comma-separated exact origins) + optional `CORS_ORIGIN_REGEX` (Vercel previews). Nothing hard-coded.
  - Local `.env`: `http://localhost:5173`.
  - Render: add the Vercel URL once it is assigned.
- `Retry-After` is exposed cross-origin.
- Frontend `VITE_API_BASE` (build-time; `frontend/src/api-base.js`): every `/api` call and every backend-relative URL resolves against it.
  - Empty in dev, where the Vite proxy is same-origin.
  - The Render URL on Vercel.
- Verified cross-origin: a production build on `localhost:4173` calling a backend on `127.0.0.1:8011`; 88/88 thumbnails, preview, magnifier, POST /select, 0 CORS errors.

## CPU inference benchmark (`scripts/bench/cpu_inference_bench.py`)
- **Machine:** Apple M4, PyTorch 2.14 CPU. Each model and thread count runs in a separate process; median of 3 runs after a warm-up.
- **Input:** `vhr-a_forest` preview, 1024 × 1024.

| Model | Unit of work | 1 thread | 2 threads | 4 threads | Peak RSS |
|---|---|---|---|---|---|
| DAv2-Small (relative depth) | one 518² image | 0.27 s | 0.18 s | 0.16 s | ~1.3 GB |
| DAv2-Large (relative depth) | one 518² image | 1.66 s | 1.32 s | 1.22 s | ~4.2–4.6 GB |
| Method 6, 1 checkpoint | 1024² crop (9 × 512 tiles) | 2.28 s | 1.65 s | 1.49 s | ~1.9 GB |
| Method 6, 4-fold ensemble | 1024² crop (36 tile passes) | — | 6.62 s | — | ~1.9 GB |

Model load time is 3–4.5 s, once per process.

**Caveats:**
- An M4 core is much faster than a shared cloud vCPU. Treat these as optimistic; expect Render to be roughly 3–5× slower. That is an estimate, not measured, and must be re-measured on Render.
- RSS on macOS includes the torch runtime.

**Implication:**
- Render's free and Starter instances (512 MB) cannot load any model.
- Standard (2 GB) fits DAv2-Small or Method 6, but not DAv2-Large.
- DAv2-Large needs ≥ 8 GB (Pro Plus).

## Hugging Face inference Space: BLOCKED on HF PRO (2026-09-26)

- **Files ready in `space/`** (not live):
  - `app.py`: DAv2-Small on plain CPU; loading and preprocessing copied verbatim from `scripts/bench/cpu_inference_bench.py`; Gradio `api_name="/predict"`.
    Output: JSON with the raw 518×518 `predicted_depth` as base64 float32, plus shape and range.
  - `requirements.txt`: `torch==2.14.0+cpu` from the PyTorch CPU index, and the project's working versions.
  - `README.md`: Python 3.11, gradio 6.28.0.
- **What HF refused.** The HF_TOKEN in `.env` has `write` role, so it was reused and no new credential was made. The Hub then refused both routes with HTTP 402:
  - Changing the hardware to `cpu-basic`: "Without a PRO subscription, you can't downgrade this Space to cpu-basic."
  - Creating a new Gradio Space on `cpu-basic`: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
  - **Free accounts can no longer host Gradio or Docker Spaces.** The existing Space `sancharimouri/DepthWizard2` requests ZeroGPU, which needs PRO too.
- **Space state.** It was deleted and recreated during the attempt. It is now restored as it was (Gradio, `zero-a10g` requested, same README), and it is not running.
- **Sleep behaviour (if it ever runs on CPU Basic).** Free CPU Basic Spaces sleep after 48 h of inactivity (the runtime reports `gcTimeout: 172800`). The first request after an idle period includes a cold start: container boot plus model load, about 3–5 s on the M4 benchmark and longer on HF. The same caveat applies to Render's free tier.

## TEMPORARY DAv2-Small bridge: Colab GPU + Cloudflare quick tunnel (2026-09-26)

> **This is a stopgap, not hosting.**
> - The tunnel URL stops working the moment the Colab notebook disconnects, the runtime is recycled, or the session ends.
> - The notebook tab must stay open and connected. **Assume no background persistence.**
> - Every restart produces a **new** `*.trycloudflare.com` URL.
> - It is to be replaced by the planned Cloudflare R2 + permanent hosting setup once card details are available (about 2026-09-28).
> - A quick tunnel needs no Cloudflare account and no card, and neither does anything else on this path.

**Pieces:**
- **`bridge/dav2_server.py`:** FastAPI with `POST /predict` (multipart `file`) and `GET /health`.
  - Model loading and preprocessing are verbatim from `scripts/bench/cpu_inference_bench.py`; the model and input are moved to CUDA when there is a GPU.
  - It returns the same JSON shape as the HF Space attempt: raw 518×518 relative depth as base64 float32, plus shape, range and timing.
- **`bridge/dav2_colab_bridge.ipynb`:** generated by `bridge/build_notebook.py`, so its server cell is byte-identical to the file above.
  - It asserts a GPU, pins `transformers==5.17.0` and starts uvicorn.
  - It downloads the official `cloudflared` Linux binary, runs `cloudflared tunnel --url http://127.0.0.1:8000` and prints `DAV2_INFERENCE_URL = https://….trycloudflare.com`.
  - It self-tests a real Sentinel-2 preview through the public URL, then runs a health monitor every 5 min.
  - The monitor only reports; it does not keep Colab alive.
- **Backend `backend/api/depth_routes.py`:** `POST /api/depth/relative` forwards to `$DAV2_INFERENCE_URL/predict`; `GET /api/depth/status` checks `/health`.
  - The URL is read on every request, so swapping hosts is an env-var change, not a code change.
  - On Render, saving an env var restarts the service; no rebuild is needed.
  - Unset URL → 503. Unreachable host (a dead tunnel) → 502 with "the temporary Colab bridge may be disconnected".

**Operating it:**
1. Open the notebook in Colab and choose a GPU runtime. Run all.
2. Copy the printed URL into the backend's `DAV2_INFERENCE_URL`.
3. Check `GET /api/depth/status` → `"reachable": true`.

**Verified (2026-09-26, this Mac, not Colab):**
- **Setup:** `bridge/dav2_server.py` on CPU, a real quick tunnel (`cloudflared` 2026.9.3), and the backend as an isolated copy with no `.env`, with `DAV2_INFERENCE_URL` = the tunnel URL.
- **Route:** a real image (the `sentinel2-almora` preview, 1024²) was POSTed to `/api/depth/relative` → HTTP 200 in 1.5 s. The bridge logged the request arriving from a public address via Cloudflare.
- **Output:** 518×518 float32, finite, 265,942 distinct values. It is **bit-identical** (max |diff| 0.0) to an independent in-process DAv2-Small run.
- **Colab GPU run (2026-09-26, the user's notebook, URL `pregnant-visited-til-tested.trycloudflare.com`, now dead):**
  - The notebook self-test passed on `cuda`.
  - The backend (isolated production-mode copy, run on this Mac) with `DAV2_INFERENCE_URL` = that URL → `POST /api/depth/relative` with the Almora preview → HTTP 200 in 13.2 s round trip, 0.12 s GPU inference.
  - Output: 518×518, matching the CPU reference to 1e-5 (Pearson 1.0).
  - The next calls got Cloudflare **error 1033**, because the notebook had been stopped and restarted. That is the failure mode described above: a restart kills the tunnel, and the new run has a new URL.
- **Not yet run:** a real Render deployment (no Render account or API key yet).
- **Second Colab run (`awesome-hoping-selling-unsigned.trycloudflare.com`, 2026-09-26):**
  - Setup: backend (isolated production-mode copy on this Mac) → tunnel → Colab `cuda`.
  - Result: **10/10 requests HTTP 200** across 3 real images (Sentinel-2 Almora, DFC2019 JAX_004_006, Maxar forest).
- **Latency: about 13 s per call, and it's the tunnel, not the GPU.**
  - GPU inference is 0.11 s, and the first byte arrives at 0.9 s.
  - The 1.43 MB JSON response comes down the quick tunnel at about 108 KB/s.
  - The same Mac downloads at 4.6 MB/s directly from Cloudflare, so a Render backend would see the same tunnel limit.
  - Measured options: gzip 1.0 MB; float16 0.72 MB (error 1e-3); uint16 + zlib 0.68 MB (about 6 s, error 3e-5).
  - At most about 2×, so this is not changed mid-demo (it would need a notebook restart). Permanent hosting removes the limit.

## Compact depth format + real depth in the UI (2026-09-26)

**Wire format "u16-zlib"** (`encode_depth` in `bridge/dav2_server.py` and `space/app.py`):
- Depth is min-max quantised to uint16, then zlib, then base64, with `min`/`max`/`shape` alongside.
- Payload is 0.68 MB, down from 1.43 MB for raw float32. That is about 6 s instead of about 13 s through the quick tunnel.
- Precision against an independent float32 CPU reference, on 3 real images (Almora, DFC2019 JAX_004_006, Maxar forest):
  - max |err| 2.64e-5 / 1.96e-5 / 1.92e-5, all within the half-step + float32-rounding bound;
  - Pearson 1.00000000.
- The frontend also still decodes the older raw float32, so a notebook started before this change keeps working.

**UI:**
- The Workbench **Relative Depth** box now shows DAv2-Small run on the job's own input: library item, upload or CDSE scene.
- Path: `POST /api/depth/relative/{library|input}/{id}` → the backend sends that input's 1024 px preview to `$DAV2_INFERENCE_URL` → decoded in `frontend/src/depth-result.js`.
- It is shown in grayscale (brighter = nearer) with a real caption and log line. It is not elevation, and the later stages are still the Darjeeling reference.
- If the host is down, the box shows the Darjeeling reference, labelled "Inference host unavailable" with the reason.

**Verified locally (headless Chrome → `localhost:5173` → dev backend with `.env` `DAV2_INFERENCE_URL` = the live Colab tunnel → Colab `cuda`):**
- Library `sentinel2-almora`: a real 518×518 map in 17 s (0.10 s GPU; the old float32 format, since that notebook predates the change).
- An upload of a JPG: the same, 15.8 s.
- Host down (simulated 502): the labelled fallback.
- 0 page errors.

## ZeroGPU Space: WORKS without PRO (verified 2026-09-26)

`sancharimouri/DepthWizard2` (https://huggingface.co/spaces/sancharimouri/DepthWizard2), which already existed with `zero-a10g` requested.

**Code and build:**
- The code in `space/` has `@spaces.GPU(duration=30)` around the forward pass and the model moved to CUDA at import.
- `torch==2.13.0`. ZeroGPU rejected 2.14.0 at config time: "Supported versions: 2.13.0, 2.12.1, 2.11.0, 2.10.0, 2.9.1, 2.8.0".
- The hardware tier was not touched.
- Build → `RUNNING` on `zero-a10g` in about 2 min, with **no payment prompt**. (The earlier 402s were for *creating* a Gradio Space and for *downgrading* to cpu-basic.)

**Real calls** (`gradio_client` 2.7.1, `Client("sancharimouri/DepthWizard2").predict(handle_file(img), api_name="/predict")`):
- Anonymous, no token: 3/3 OK on `cuda (ZeroGPU)`. The first call took 18 s (cold GPU attach); later calls took 5–6 s round trip with 0.2–0.9 s inference.
- With `HF_TOKEN`: 3/3 OK, about 5–7 s.
- Output: 518×518 u16-zlib. Against the local float32 CPU reference (Almora): max |diff| 9.6e-4, Pearson 0.99999998. That is GPU vs CPU numerics plus torch 2.13 vs 2.14.

**Auth:** the public Space's API needs **no token**.

**Caveats (not verified here):**
- ZeroGPU has a daily GPU-time quota per caller. Anonymous callers are counted per IP; logged-in callers get a larger allowance.
- A backend on one server IP shares one quota, so passing `HF_TOKEN` is safer. Exact quota numbers were not measured.
- Idle Spaces sleep; the first call after a sleep is slow.

**Not done:** the backend's `/api/depth` routes still speak the bridge's plain `/predict` HTTP. Using the Space needs a `gradio_client` path in `backend/api/depth_routes.py`.

## Backend → ZeroGPU Space as the primary depth host (2026-09-26)

**Config** (`backend/api/depth_routes.py`):
- `DAV2_INFERENCE_URL` takes a Space id (`owner/name`, or its huggingface.co/spaces URL) → Gradio API via `gradio_client==2.7.1`, or a plain http(s) `/predict` URL (the Colab bridge).
  - Default when unset: `sancharimouri/DepthWizard2`.
- `DAV2_FALLBACK_URL` (optional, same forms) is tried only when set and the primary fails with a quota, 5xx or timeout error. It is meant for the Colab bridge. Both paths live in the code side by side.
- The UI and log mark a result from the fallback ("fallback host", "via … (fallback: primary host failed)").

**Token on every Space call:**
- `gradio_client`'s `token=` alone does **not** reach the Space. Measured: it authenticated the huggingface.co API calls, but all 6 requests to `*.hf.space` (config, info, upload, queue join/data, heartbeat) went without `Authorization`, so they were anonymous for ZeroGPU.
- The backend therefore also passes `headers={"Authorization": "Bearer $HF_TOKEN"}`, and refuses to call a Space without `HF_TOKEN` (503).
- An httpx-level audit counts every `*.hf.space` request, shown in `GET /api/depth/status` → `space_request_auth`. After the UI runs: **9 with auth, 0 without**.

**UI (headless Chrome → localhost:5173 → dev backend → Space):**
- Library `sentinel2-almora`: real depth, 8.7 s round trip (1.04 s GPU, first call).
- Upload JPG: 5.8 s (0.2 s GPU).
- Primary broken (a nonexistent Space id), no fallback: the labelled Darjeeling reference with the real 404 reason.
- Primary broken + `DAV2_FALLBACK_URL` = a plain-HTTP bridge: real depth, labelled fallback.
- 0 page errors.

`uv.lock` is not regenerated for `gradio_client` (the lockfile has unrelated uncommitted changes); `uv pip install gradio_client==2.7.1` adds it.

## ZeroGPU daily quota: MEASURED (2026-09-26)

**Setup:**
- Authenticated calls (HF_TOKEN as an explicit header; the audit showed every Space request carried it).
- One real 1024² image per call, back to back, through the real backend path (`POST /api/depth/relative` → gradio_client → Space).
- Per-call log: `docs/zerogpu_quota_log_2026-09-26.jsonl`.

**Result:**
- **585 successful calls in 55 min**, then the first quota refusal at call 587:
  > You have exceeded your free ZeroGPU quota (45s requested vs. 45s left). Try again in 22:55:31.
  > Subscribe to Hugging Face PRO to get 40 min of ZeroGPU quota a day.
- The run stopped itself on that 429 before the stop request arrived, so this is the **actual ceiling for the day**, not only a lower bound.
- Before it: 1 transient failure (call 541, "write operation timed out"). No other refusals.
- Summed model inference over the 585 calls was **452.9 s** (median 0.20 s per call, p95 4.7 s, max 10.0 s).
- ZeroGPU charges its own GPU-slot time, which includes attach/overhead and is larger than our inference timer. Its exact daily seconds figure isn't exposed.
  The message shows each call is admitted only with **45 s** left, so the last ~45 s of quota is unusable.
- The message says "free ZeroGPU quota", attributed to the logged-in free account (not an anonymous/IP quota). It refills on a ~24 h window.

**What it supports:**
- About **585 depth calls per day** for this workload.
- A demo session that runs 10–20 generations therefore gives **~30–60 full demo sessions a day**.
- That comfortably exceeds real demo use.

**⚠ As of this run the account's ZeroGPU quota is spent until about 2026-09-27 02:40 IST** (refusal at 03:44:32 IST + 22:55:31).
Until then, `/api/depth` needs `DAV2_FALLBACK_URL` (e.g. the Colab bridge); otherwise the UI shows its labelled Darjeeling fallback.
