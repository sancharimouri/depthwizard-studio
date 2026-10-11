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

// terrain.json / header `mask: { polygon: [[lon, lat], ...] }` (the circle and outline demo regions, see
// scripts/bake_showcase_tiles.py): `inside` gets 1 per grid cell inside it, 0 outside, on the mesh's vertex
// positions (column i at i / (width - 1) across the bounds, row 0 at the north edge). The mesh cuts the cells
// outside (terrain.js) and the statistics leave them out (side-panels.js).
export function maskInside(terrain) {
    const ring = terrain.mask?.polygon;
    if (!ring?.length) {
        return null;
    }
    const { width: w, height: h, bounds: b } = terrain;
    const inside = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        const lat = b.north - ((b.north - b.south) * y) / (h - 1);
        // even-odd rule along this row: the crossings of the outline, sorted
        const xs = [];
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const [x1, y1] = ring[i], [x2, y2] = ring[j];
            if ((y1 > lat) !== (y2 > lat)) {
                xs.push(x1 + ((lat - y1) * (x2 - x1)) / (y2 - y1));
            }
        }
        xs.sort((p, q) => p - q);
        for (let k = 0; k + 1 < xs.length; k += 2) {
            const c0 = Math.ceil(((xs[k] - b.west) / (b.east - b.west)) * (w - 1));
            const c1 = Math.floor(((xs[k + 1] - b.west) / (b.east - b.west)) * (w - 1));
            for (let x = Math.max(0, c0); x <= Math.min(w - 1, c1); x++) {
                inside[y * w + x] = 1;
            }
        }
    }
    return inside;
}

export async function fetchTerrainData(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load terrain: ${response.status}`);
    }
    const terrain = String(url).split("?")[0].endsWith(COMPACT_SUFFIX)
        ? decodeTerrainU16(await gunzip(await response.arrayBuffer()))
        : await response.json();
    const inside = maskInside(terrain);
    if (inside) {
        terrain.inside = inside;
    }
    return terrain;
}
