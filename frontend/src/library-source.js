// Where library tiles come from, fixed at BUILD time (VITE_LIBRARY_SOURCE):
//   "static"  the web build (vercel.json -> npm run build:web): the pre-baked static library in
//             /library-static/ (scripts/bake_static_library.py). Listing, routing and generation results are files;
//             opening a library tile makes no backend and no Space call.
//   "backend" (default: the desktop app's Tauri build and local dev) the backend's /api/library and
//             /api/generate/library routes, unchanged; the desktop sidecar reads its local packs.
// vite.config.js drops public/library-static/ from any build that isn't "static", so the desktop bundle never
// carries it.

export const LIBRARY_SOURCE = (import.meta.env?.VITE_LIBRARY_SOURCE ?? "").trim() === "static" ? "static" : "backend";
export const STATIC_BASE = "library-static";

async function getJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} for ${url}`);
    }
    return response.json();
}

// The same shape as GET /api/library (generated_at, counts, tiers, total, items); item URLs are static paths.
export function staticLibraryListing() {
    return getJson(`${STATIC_BASE}/index.json`);
}

// The same shape as POST /api/generate/library/<id> ({ meta, depth, assets }), minus the base64 depth payload
// (relative_depth.png already renders it).
export async function staticTileResponse(itemId) {
    const tile = await getJson(`${STATIC_BASE}/${encodeURIComponent(itemId)}/tile.json`);
    return { meta: tile.meta, depth: tile.depth, assets: tile.assets };
}
