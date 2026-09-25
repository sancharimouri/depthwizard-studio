import { test } from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import { decodeDepth, depthToRgba, depthRoute } from "../src/depth-result.js";

test("u16-zlib decodes to min + q/65535 * (max - min), little-endian", async () => {
    const q = new Uint16Array([0, 65535, 32768, 1]);
    const buf = Buffer.from(q.buffer); // little-endian on every supported platform
    const resp = { encoding: "u16-zlib", shape: [2, 2], min: 0.5, max: 3.5,
        data_b64: deflateSync(buf).toString("base64") };
    const d = await decodeDepth(resp);
    assert.equal(d.width, 2);
    assert.equal(d.height, 2);
    const want = [0.5, 3.5, 0.5 + (32768 / 65535) * 3, 0.5 + (1 / 65535) * 3];
    d.values.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-6, `${v} vs ${want[i]}`));
});

test("legacy raw float32 (no encoding field) still decodes", async () => {
    const f = new Float32Array([1, 2, 3, 4, 5, 6]);
    const d = await decodeDepth({ shape: [2, 3], data_b64: Buffer.from(f.buffer).toString("base64") });
    assert.deepEqual([...d.values], [1, 2, 3, 4, 5, 6]);
    assert.equal(d.width, 3);
});

test("wrong payload size and unknown encodings are errors, not garbage images", async () => {
    await assert.rejects(decodeDepth({ encoding: "u16-zlib", shape: [2, 2], min: 0, max: 1,
        data_b64: deflateSync(Buffer.alloc(6)).toString("base64") }));
    await assert.rejects(decodeDepth({ encoding: "png", shape: [1, 1], data_b64: "" }));
});

test("grayscale: nearest (max) is white, farthest is black", () => {
    const rgba = depthToRgba({ width: 2, height: 1, values: new Float32Array([0.5, 3.9]) });
    assert.deepEqual([...rgba], [0, 0, 0, 255, 255, 255, 255, 255]);
});

test("library items and inputs (upload / CDSE search) map to the right backend route", () => {
    assert.equal(depthRoute({ source: "library", id: "dfc2019-JAX_004_006" }), "/api/depth/relative/library/dfc2019-JAX_004_006");
    assert.equal(depthRoute({ source: "upload", id: "ab12" }), "/api/depth/relative/input/ab12");
    assert.equal(depthRoute({ source: "search", id: "cd34" }), "/api/depth/relative/input/cd34");
});
