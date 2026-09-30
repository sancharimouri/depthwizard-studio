# Facts panel research + Maxar preset (2026-09-30, branch `facts-research-2026-09-30`)

Owner decisions this session:
- Maxar look: SUBTLE.
- Facts panel: diagnose, research and draft only. No live service is built; the owner decides after this report.
- /tmp cap, DFC2019 history cleanup, GCP and the data/ backup are deferred.

No deploys, no push, no API sign-ups.

## Part A: Maxar preset = subtle (2026-09-30)

- **Command:** `.venv/bin/python scripts/bake_static_library.py --maxar-preset subtle`, which ran in 27 s. Output: 76 tiles,
  457 files, **52.5 MB**, written to `frontend/public/library-static/` (gitignored).
- **Manifest:** `data/library_v2_2026-09-29/tile_manifest.json` already had `maxar_display_preset: "subtle"`, and so
  did `dem_maxar/ACTIVE`.
  - The placeholder flag was never stored in the manifest. The bake computes it (`scripts/bake_static_library.py:76`)
    as "`--maxar-preset` given and different from the manifest's value".
  - The previous bake ran with `--maxar-preset medium`, so it wrote `placeholder: true`.
  - With `subtle`, `index.json` and all 5 included Maxar entries now say `{"name": "subtle", "placeholder": false}`.
  - The manifest file was **not edited** (it stays byte-identical).
- **Vercel upload**, measured as before (`frontend/` minus `node_modules/`, `.vercel/`, `dist/` and the
  `.vercelignore`d `public/data/vhr/`): **539 files, 73.0 MB, so 27.0 MB of headroom** under Hobby's 100 MB. It was
  72.9 MB.
  - The local `dist/` is 160.7 MB only because Vite copies the 84 MB `public/data/vhr/`, which the upload excludes.
    Vercel builds from source.
- **Visual check:** headless Chrome (Metal, DevTools protocol) against `vite preview` of `npm run build:web`, clicking
  the library card, then Generate. The preview server was stopped afterwards.
  - `docs/screenshots/2026-09-30_maxar_subtle_a_valley.png`: relief 327 m, rendered at the manifest's default ×10.
    The canopy reads spiky at ×10, so a lower default may suit subtle (owner call).
  - `docs/screenshots/2026-09-30_maxar_subtle_c_town.png`: relief 305 m, at ×4.
  - Both render with the static source: no backend, no Space call.
