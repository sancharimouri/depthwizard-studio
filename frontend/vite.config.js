import { rmSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

// Dev only: proxies /api/* to the local FastAPI backend (backend/main.py), so
// local calls are same-origin. Deployed builds set VITE_API_BASE to the backend
// URL (src/api-base.js) and the backend allows the frontend's origin via its
// CORS_ORIGINS env var.
//
// VITE_LIBRARY_SOURCE (src/library-source.js): only the web build ("static", npm run build:web) ships the
// pre-baked library. Every other build (the desktop app's `vite build`) drops public/library-static/ from its
// output, so the desktop bundle never gains it.
const STATIC_LIBRARY = "library-static";

function staticLibraryOnlyForWeb() {
    let outDir;
    return {
        name: "dw2-static-library-only-for-web",
        apply: "build",
        configResolved(config) {
            outDir = resolve(config.root, config.build.outDir);
        },
        closeBundle() {
            if ((process.env.VITE_LIBRARY_SOURCE ?? "").trim() !== "static") {
                rmSync(resolve(outDir, STATIC_LIBRARY), { recursive: true, force: true });
            }
        },
    };
}

export default defineConfig({
    plugins: [staticLibraryOnlyForWeb()],
    server: {
        proxy: {
            "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
