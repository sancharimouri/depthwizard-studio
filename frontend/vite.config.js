import { defineConfig } from "vite";

// Proxies /api/* to the local FastAPI backend (backend/main.py) so the
// frontend can call same-origin relative paths in both dev and any later
// same-host deploy, and so we don't need to manage CORS beyond the
// backend's own localhost:5173 allowlist.
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
