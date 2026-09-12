# DepthWizard2
SIH 2026 — Single-View Height Estimation and 3D Flythrough.

Clean rebuild scaffold. Implementation is intentionally deferred.
Planned architecture: Satellite RGB → frozen DAv2 relative-depth prior → RDAH-Net-style nDSM fusion → sparse-anchor/LoRA correction → CartoDEM terrain → absolute DSM → validation/failure detector → Three.js visualization.

Terrain and nDSM remain separate products. Demo mode is replaceable and must not masquerade as validated ML.

## Running locally

One-time setup: `npm install` (root) and `npm install --prefix frontend`, plus a
`.env` at the repo root (copy `.env.example`) with real `CDSE_CLIENT_ID` /
`CDSE_CLIENT_SECRET` values.

Then, from the repo root:

```
npm run dev
```

This starts the backend (FastAPI/uvicorn, port 8000) and frontend (Vite, port
5173) together, with each process's output labeled `[BACKEND]` / `[FRONTEND]`
in one terminal. It runs a preflight check first and exits with a clear error
— missing `.env` values, an occupied port, missing `node_modules` — instead of
starting halfway or failing silently. Open http://localhost:5173.
