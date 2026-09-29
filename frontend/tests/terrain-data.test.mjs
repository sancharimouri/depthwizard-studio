import test from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import { decodeTerrainU16, fetchTerrainData } from "../src/terrain-data.js";
import { LIBRARY_SOURCE } from "../src/library-source.js";

// Encode like scripts/bake_static_library.py: u32 header length | JSON | pad to even | u16 heights [| u16 display]
function encode(header, ...arrays) {
    const hb = Buffer.from(JSON.stringify(header));
    const pad = (4 + hb.length) % 2;
    const parts = [Buffer.alloc(4), hb, Buffer.alloc(pad)];
    parts[0].writeUInt32LE(hb.length);
    for (const a of arrays) {
        const q = Buffer.alloc(a.length * 2);
        a.forEach((v, i) => q.writeUInt16LE(Math.round(v * 65535), i * 2));
        parts.push(q);
    }
    return Buffer.concat(parts);
}

const asArrayBuffer = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

test("compact terrain decodes within half a quantisation step, display heights included", () => {
    const heights = [0, 0.123456, 0.5, 0.999999, 1, 0.33];
    const display = [1, 0.9, 0.2, 0.1, 0, 0.75];
    const header = { width: 3, height: 2, bounds: { west: 0, south: 0, east: 1, north: 1 }, elevationMin: 10, elevationMax: 30,
        display: { elevationMin: 0, elevationMax: 5 } };
    const t = decodeTerrainU16(asArrayBuffer(encode(header, heights, display)));
    assert.equal(t.width, 3);
    assert.deepEqual(t.bounds, header.bounds);
    heights.forEach((h, i) => assert.ok(Math.abs(t.heights[i] - h) <= 0.5 / 65535 + 1e-12));
    display.forEach((h, i) => assert.ok(Math.abs(t.display.heights[i] - h) <= 0.5 / 65535 + 1e-12));
    assert.equal(t.display.elevationMax, 5);
});

test("odd header lengths are padded, and plain JSON terrain still loads", async () => {
    const t = decodeTerrainU16(asArrayBuffer(encode({ width: 1, height: 1, bounds: {}, elevationMin: 0, elevationMax: 1, x: "a" }, [0.25])));
    assert.ok(Math.abs(t.heights[0] - 0.25) < 1e-4);
    const realFetch = globalThis.fetch;
    const blob = gzipSync(encode({ width: 1, height: 1, bounds: {}, elevationMin: 0, elevationMax: 1 }, [0.5]));
    globalThis.fetch = async url => (String(url).endsWith(".u16.gz")
        ? new Response(blob) : new Response(JSON.stringify({ width: 1, height: 1, heights: [0.5] })));
    try {
        assert.ok(Math.abs((await fetchTerrainData("a/terrain.u16.gz")).heights[0] - 0.5) < 1e-4);
        assert.deepEqual((await fetchTerrainData("a/terrain.json")).heights, [0.5]);
    } finally {
        globalThis.fetch = realFetch;
    }
});

test("without VITE_LIBRARY_SOURCE=static the app uses the backend library (desktop, local dev)", () => {
    assert.equal(LIBRARY_SOURCE, "backend");
});
