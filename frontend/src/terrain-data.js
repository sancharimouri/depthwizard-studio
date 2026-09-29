// Terrain data for the viewer: terrain.json (backend jobs, demo regions) or the compact binary the static library
// ships (scripts/bake_static_library.py, "terrain.u16.gz"). Both give the same object:
//   { width, height, bounds, elevationMin, elevationMax, heights, display?, limitOutliers? }
// with heights normalised to [0, 1] (row-major), exactly what terrain.js createTerrain() reads.
//
// Compact layout (little-endian), gzip-compressed as a whole:
//   u32 header length N | N bytes of JSON header | 0-1 pad bytes to an even offset |
//   width*height u16 heights (q / 65535) | [width*height u16 display heights, when header.display exists]
// Quantisation: 1/65535 of the tile's own range, so the worst height error is range / 131070 (half a step).

export const COMPACT_SUFFIX = ".u16.gz";

async function gunzip(buffer) {
    const bytes = new Uint8Array(buffer);
    // A host may already have decoded it (Content-Encoding): only gzip data starts 1f 8b.
    if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) {
        return buffer;
    }
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
}

export function decodeTerrainU16(buffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    const n = header.width * header.height;
    let offset = 4 + headerLength;
    offset += offset % 2;
    const readHeights = () => {
        const q = new Uint16Array(buffer.slice(offset, offset + n * 2));
        offset += n * 2;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            out[i] = q[i] / 65535;
        }
        return out;
    };
    const terrain = { ...header, heights: readHeights() };
    if (header.display) {
        terrain.display = { ...header.display, heights: readHeights() };
    }
    return terrain;
}

export async function fetchTerrainData(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load terrain: ${response.status}`);
    }
    if (String(url).split("?")[0].endsWith(COMPACT_SUFFIX)) {
        return decodeTerrainU16(await gunzip(await response.arrayBuffer()));
    }
    return response.json();
}
