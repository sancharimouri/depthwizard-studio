# Deploy report, 2026-09-29: sizes, load times, SIH deck links

Everything here was measured on 2026-09-29, not estimated. How: `docs/NEXT_SESSION_DEPLOY.md` Phase 7.
Raw data: one JSON line per session, from the measuring script (headless Chrome, Metal GPU, DevTools protocol,
a fresh profile per session).

## 1. Desktop app size (v1.0.2 vs v1.0.1)

| | v1.0.1 (`docs/DESKTOP_APP.md`) | **v1.0.2** | Method |
|---|---|---|---|
| `.dmg` installer | 340.4 MB | **355.7 MB** (355,727,127 B) | release asset size |
| Updater `DepthWizard.app.tar.gz` | 328.7 MB | **343.2 MB** (343,215,706 B) | release asset size |
| `DepthWizard.app` installed | 455.2 MB | **492.2 MB** | `du -sk`, a v1.0.1 install updated to 1.0.2 by the in-app updater |
| · frozen backend (ONNX Runtime; + scipy in 1.0.2) | 297 MB | **335 MB** | `du -sh Resources/dw2-backend` |
| · · of which the ONNX model | 95 MB | 95 MB | `du -sh …/_internal/models` |
| · bundled tile library + elevation packs | 99 MB | 99 MB | `du -sh Resources/library` |
| · shell + embedded frontend | 57 MB | 57 MB | `du -sh Contents/MacOS` |
| Per-user cache after on-demand downloads | — | 10 MB after 2 downloads (S2 Bengaluru 3.0 MB, DFC2019 OMA_364_043 2.2 MB) + generating both | `du -sh` of a fresh `DW2_USER_CACHE` |

- The +37 MB is scipy. The hole fill (`fill_nan_nearest`) needs it, and v1.0.1's frozen backend had none.
- Every asset is far below GitHub's 2 GB per-file limit.

## 2. Website load times: cold start vs normal

**Setup:**
- Site: `https://depthwizard-studio.vercel.app`, headless Chrome with the Metal GPU, foreground tab, a fresh profile
  each session.
- Times are ms from navigation start unless marked "after …".

**What counts as cold and normal:**
- **Cold:** no request to Render for at least 17 min (Render's free instance spins down 15 min after the last inbound
  request). Every cold session showed the site's cold-start message and a 25–35 s `/api/library`, which a warm
  instance never does. The "Shutting down" log line itself is visible only in the Render dashboard.
- **Normal:** right after a warm request.

**Sessions:**
- Cold: 3 sessions at 09:23, 09:42 and 10:00Z. A fourth at 09:03Z had two fields mis-measured; its other fields agree:
  library 35.5 s, cold message 3.4 s, generation to Studio 33.4 s.
- Normal: 3 sessions at 09:05–09:06Z.

| Step | Cold: median (range) | Normal: median (range) |
|---|---|---|
| Vercel page: time to first byte | 258 ms (217–288) | 208 ms (172–254) |
| Vercel page: DOMContentLoaded | 490 ms (446–499) | 495 ms (409–522) |
| `/api/library` → 200 (the Library cards appear right after) | **35.0 s** (24.9–35.2) | 0.75 s (0.71–0.91) |
| Cold-start message appears | 3.0 s (2.98–3.03), every cold session | never (as intended) |
| First thumbnail visible | 35.9 s (26.7–36.0) | 1.33 s (1.26–1.65) |
| All first-screen thumbnails visible | 40.6 s (29.5–41.4) | 1.54 s (1.38–1.82) |
| Select Almora → preview image loaded (after the click) | 0.58 s (0.54–0.59) | 0.36 s (0.26–0.41) |
| START GENERATION on Almora → all boxes done + Studio expanded (after the click) | **28.6 s** (28.1–29.4) | 20.2 s (20.2–21.8) |
| · of which the `/api/generate` request (depth on the Space + DEM + outputs) | 22.9 s (22.4–23.6) | 14.4 s (13.5–15.3) |
| Demo page (`#/demo`) → 3D terrain visible | 2.2 s (0.5–18.7) | 3.6 s (0.5–4.0) |
| Home (`#demo-video`) → rendered and scrolled to the video | 0.41 s (0.25–0.78) | 0.54 s (0.41–0.65) |

The cold-start message is due at 2.5 s. It shows at about 3.0 s because the check polls every 50 ms and the app
schedules it after its first request.

**Where the time goes:**
- **Render spin-up: 25–35 s, the whole cold penalty on first load.** The Vercel page arrives in about 0.25 s either way.
  Everything that needs the backend waits for the first `/api/library`.
- **The first generation on a new instance costs about 8 s more** (request 22.9 s vs 14.4 s). The backend builds its
  gradio client to the Space, and fetches the Space config and first DEM tiles, all per instance.
- **The generate request itself (about 14 s warm) is mostly the Space round trip:**
  - DAv2 on ZeroGPU is 0.2 s of GPU, but the Gradio upload, queue and result add seconds;
  - HF's edge answered 8–23% of requests with a 502 that day, and each one costs a 0.5–3 s retry;
  - on top of that come the DEM pack or live GLO-30, and the textures and terrain written out.
- **The boxes add their minimum readouts** (4 × 1.3 s after the depth box), plus the calculation log's pacing.
- **The Demo page is static** (Vercel CDN, about 130 MB of `public/data`). Its 18.7 s outlier was a cold CDN edge, not
  Render.
- **Thumbnails and previews:** public tiles 307-redirect to GitHub release downloads (78 of 178 URLs); DFC2019 ones are
  served by the backend from its HF cache.

**What could reduce it (rule 3: free options first):**
- **Keep Render warm with a scheduled ping every 10–14 min** (e.g. a GitHub Actions cron on the public repo, which is
  free). This removes the 25–35 s cold load and the extra 8 s on the first generation.
  - The risk (render.com/docs/free, read 2026-09-29): 750 free instance-hours a month **per workspace**. When they run
    out, Render **suspends all free services until the next month**.
  - Always-on uses about 720 h, so this is only safe if nothing else in the workspace uses free hours.
  - A safer variant: ping only during the judging window.
  - Not done: it needs your OK.
- Paid alternative: a paid Render instance never sleeps. The pricing page didn't state the price when read, so check
  render.com/pricing before deciding. Not done without your OK.
- Warm the backend's Space client at startup, so it's ready when the first request arrives (small code change).
- Serve the 39 public previews and thumbnails from Vercel's CDN instead of GitHub redirects (about 1 s saved warm).

## 3. SIH deck links (from `145604_SIH26175.pdf`)

| Link | Final URL | HTTP | What it shows |
|---|---|---|---|
| `https://depthwizard-studio.vercel.app/` | same | 200 | DW Studio (input page), Library of 89 scenes; 0 console errors |
| `https://depthwizard-studio.vercel.app/#/demo` | same | 200 | Demo page, 3D terrain of the demo regions; 0 console errors |
| `https://depthwizard-studio.vercel.app/#demo-video` | same | 200 | Home, scrolled to "Demo video". **The section shows a placeholder until the recording is added**; 0 console errors |
| `https://depthwizard-studio.vercel.app/#/docs` | same | 200 | Home; 0 console errors |
| `https://github.com/sancharimouri/depthwizard-studio` | same | 200 | the public repo (this project) |
| `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest` | `…/releases/tag/v1.0.2` | 200 | release v1.0.2: DMG, update archive, `.sig`, `latest.json` |
| `https://github.com/sancharimouri` | same | 200 | the GitHub profile |
| `https://www.linkedin.com/in/sancharimouri` | same | 999 | LinkedIn's standard reply to non-browser clients (not a broken link); open it in a browser to confirm |
| `mailto:sancharimouri@gmail.com` | — | — | not an HTTP link; the sidebar footer uses the same address |
