// Real relative depth for a job's own input (DAv2-Small on the inference host set
// by the backend's DAV2_INFERENCE_URL; backend/api/depth_routes.py, docs/DEPLOY.md).
// Relative depth (brighter = nearer), NOT elevation.
import { apiUrl } from "./api-base.js";

function base64Bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) {
        out[i] = bin.charCodeAt(i);
    }
    return out;
}

async function inflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Host response -> { width, height, min, max, values: Float32Array (row-major, metres-free
// relative depth) }. "u16-zlib" is the compact format; a response with no "encoding" is
// the older raw little-endian float32.
export async function decodeDepth(resp) {
    const [height, width] = resp.shape;
    const n = width * height;
    const bytes = base64Bytes(resp.data_b64);
    const values = new Float32Array(n);
    if (resp.encoding === "u16-zlib") {
        const raw = await inflate(bytes);
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        if (raw.byteLength !== n * 2) {
            throw new Error(`depth payload is ${raw.byteLength} bytes, expected ${n * 2}`);
        }
        const span = resp.max - resp.min;
        for (let i = 0; i < n; i += 1) {
            values[i] = resp.min + (view.getUint16(i * 2, true) / 65535) * span;
        }
    } else if (!resp.encoding) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        if (bytes.byteLength !== n * 4) {
            throw new Error(`depth payload is ${bytes.byteLength} bytes, expected ${n * 4}`);
        }
        for (let i = 0; i < n; i += 1) {
            values[i] = view.getFloat32(i * 4, true);
        }
    } else {
        throw new Error(`unknown depth encoding "${resp.encoding}"`);
    }
    return { width, height, min: resp.min, max: resp.max, values };
}

// Min-max grayscale, the same mapping as backend/depth/depth_engine.save_depth_png.
export function depthToRgba({ width, height, values }) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
    }
    const span = hi > lo ? hi - lo : 1;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < values.length; i += 1) {
        const g = Math.round(((values[i] - lo) / span) * 255);
        rgba.set([g, g, g, 255], i * 4);
    }
    return rgba;
}

// { source: "library" | "upload" | "search", id } -> the backend route's source kind
export function depthRoute(inputRef) {
    const kind = inputRef?.source === "library" ? "library" : "input";
    return `/api/depth/relative/${kind}/${encodeURIComponent(inputRef.id)}`;
}

// POST -> { resp, depth, roundTripS }; throws with the backend's detail on failure
export async function requestRelativeDepth(inputRef) {
    const t0 = performance.now();
    const response = await fetch(apiUrl(depthRoute(inputRef)), { method: "POST" });
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            detail = (await response.json()).detail || detail;
        } catch {
            // not JSON
        }
        throw new Error(detail);
    }
    const resp = await response.json();
    const depth = await decodeDepth(resp);
    return { resp, depth, roundTripS: (performance.now() - t0) / 1000 };
}
