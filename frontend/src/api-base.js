// Backend base URL for every /api call and every backend-relative URL the API
// returns (DFC2019 thumbnails/previews, upload previews). VITE_API_BASE is baked
// in at build time: empty in local dev (the Vite proxy makes /api same-origin),
// the Render URL on Vercel (e.g. https://depthwizard2-api.onrender.com).
export const API_BASE = (import.meta.env?.VITE_API_BASE ?? "").trim().replace(/\/+$/, "");

// Root-relative paths ("/api/...") get the backend base; absolute URLs (the public
// GitHub Release assets) and null pass through unchanged.
export function apiUrl(path, base = API_BASE) {
    if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) {
        return path;
    }
    return base + path;
}
