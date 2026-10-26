import { rmSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { transform as lightningcss } from "lightningcss";

// Dev only: proxies /api/* to the local FastAPI backend (backend/main.py), so
// local calls are same-origin. Deployed builds set VITE_API_BASE to the backend
// URL (src/api-base.js) and the backend allows the frontend's origin via its
// CORS_ORIGINS env var.
//
// VITE_LIBRARY_SOURCE (src/library-source.js): only the web build ("static", npm run build:web) ships the
// pre-baked library. Every other build (the desktop app's `vite build`) drops public/library-static/ from its
// output, so the desktop bundle never gains it. The same goes for Home's public files (public/home/): the desktop app
// opens the Studio and never loads Home (src/boot.js).
const WEB_ONLY = ["library-static", "home"];

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
                WEB_ONLY.forEach(dir => rmSync(resolve(outDir, dir), { recursive: true, force: true }));
            }
        },
    };
}

// Home's dev-only ?edit layout mode (its src/edit-mode.js) stays in depthwizard-home, so its import resolves to nothing.
const homeEditModeStub = {
    name: "dw2-home-edit-mode-stub",
    resolveId: (id, importer) => (id === "./edit-mode.js" && importer?.includes("/src/home/") ? "\0home-edit-mode" : null),
    load: id => (id === "\0home-edit-mode" ? "export {};" : null),
};

// Vite 8 minifies CSS with Lightning CSS, which changes four of Home's rules: it moves `font` after
// `font-variant-numeric` (the shorthand then resets it) and drops `translate` beside `transform`. Home's own build
// (Vite 7, esbuild) keeps them. So CSS is minified here instead of by Vite (build.cssMinify: false): Home's stylesheet
// ships as written, every other one exactly as Vite would minify it (Lightning CSS, Vite's default CSS targets).
const VITE_CSS_TARGETS = { chrome: 111 << 16, edge: 111 << 16, firefox: 114 << 16, safari: (16 << 16) | (4 << 8), ios_saf: (16 << 16) | (4 << 8) };
const minifyCssExceptHome = {
    name: "dw2-minify-css-except-home",
    apply: "build",
    generateBundle(_, bundle) {
        const home = new Set(Object.values(bundle).flatMap(c => (c.type === "chunk" && c.facadeModuleId?.endsWith("/src/home/main.js")
            ? [...(c.viteMetadata?.importedCss ?? [])] : [])));
        if (!home.size) {
            this.error("Home's stylesheet not found in the bundle");
        }
        for (const file of Object.values(bundle)) {
            if (file.type === "asset" && file.fileName.endsWith(".css") && !home.has(file.fileName)) {
                const { code } = lightningcss({ filename: file.fileName, code: Buffer.from(file.source), minify: true, targets: VITE_CSS_TARGETS });
                file.source = `${code.toString()}\n`;
            }
        }
    },
};

export default defineConfig({
    plugins: [staticLibraryOnlyForWeb(), homeEditModeStub, minifyCssExceptHome],
    build: { cssMinify: false },
    // Home's Connaught Place 3D (src/home/demo-city.js) imports './demo/connaught-city.json', a byte-identical copy of
    // the Studio's cityscape data: it reads the Studio's file instead (scripts/import_home.sh does not copy it).
    resolve: {
        alias: [{ find: /^\.\/demo\/connaught-city\.json$/, replacement: resolve("public/cityscape/connaught_place.json") }],
    },
    server: {
        proxy: {
            "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
