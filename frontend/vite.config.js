import { defineConfig } from "vite";

// Dev only: proxies /api/* to the local FastAPI backend (backend/main.py), so
// local calls are same-origin. Deployed builds set VITE_API_BASE to the backend
// URL (src/api-base.js) and the backend allows the frontend's origin via its
// CORS_ORIGINS env var.
export default defineConfig({
    server: {
        proxy: {
            "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
