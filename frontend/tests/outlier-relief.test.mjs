import test from "node:test";
import assert from "node:assert/strict";
import { outlierFences, limitOutliers } from "../src/outlier-relief.js";

// a gently varying plain (0–10) with one 60-unit spike and one −40 pit
function plainWithSpike() {
    const z = new Float32Array(1000).map((_, i) => (i % 100) / 10);
    z[500] = 60;
    z[501] = -40;
    return z;
}

test("spread-out terrain (no extreme outliers) is left exactly as is", () => {
    const z = new Float32Array(1000).map((_, i) => i);
    assert.equal(outlierFences(z), null);
    const out = limitOutliers(z, new Float32Array(z.length), null, 5);
    assert.deepEqual(out, z);
});

test("the spike's displayed excess never exceeds the cap, at any exaggeration", () => {
    const z = plainWithSpike();
    const f = outlierFences(z);
    assert.ok(f && f.hi < 60 && f.lo > -40);
    const out = new Float32Array(z.length);
    for (const s of [0.25, 1, 2, 5, 20, 100]) {
        limitOutliers(z, out, f, s);
        const excess = (out[500] - f.hi) * s;
        assert.ok(excess > 0 && excess <= f.cap * (1 + 1e-5), `s=${s} excess=${excess}`);
        const deficit = (f.lo - out[501]) * s;
        assert.ok(deficit > 0 && deficit <= f.cap * (1 + 1e-5), `s=${s} deficit=${deficit}`);
        // points inside the fences scale untouched
        assert.equal(out[10], z[10]);
    }
});

test("continuous at the fence and order-preserving above it", () => {
    const f = { lo: 0, hi: 10, cap: 5 };
    const z = Float32Array.from([10, 10.001, 11, 20, 40]);
    const out = limitOutliers(z, new Float32Array(z.length), f, 3);
    assert.ok(Math.abs(out[1] - 10.001) < 1e-4);
    for (let i = 1; i < out.length; i++) {
        assert.ok(out[i] > out[i - 1]);
    }
});
