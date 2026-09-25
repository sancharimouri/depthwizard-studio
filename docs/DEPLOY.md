# Deployment (Render backend + Vercel frontend) — status 2026-09-26

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
