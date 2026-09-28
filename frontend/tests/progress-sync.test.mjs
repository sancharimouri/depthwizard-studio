import test from "node:test";
import assert from "node:assert/strict";
import { syncedPercent, createDurationMemory, trackWork } from "../src/progress-sync.js";

test("percent follows elapsed time against the expected duration and never reaches 100 early", () => {
    assert.equal(syncedPercent(0, 10000), 0);
    assert.ok(Math.abs(syncedPercent(10000, 10000) - 82) <= 1); // ~86% of the 95 ceiling
    assert.ok(syncedPercent(30000, 10000) <= 95);
    // a slower stage crawls: the same elapsed time reads lower
    assert.ok(syncedPercent(5000, 30000) < syncedPercent(5000, 10000));
});

test("instant work still shows a readout that runs evenly to 100% over the minimum time", async () => {
    let shown = false;
    const seen = [];
    const t0 = Date.now();
    const r = await trackWork({ work: Promise.resolve(7), steps: ["a"], expectedMs: 1000, minMs: 300, tickMs: 20,
        onShow: () => { shown = true; }, onTick: p => seen.push(p) });
    assert.equal(r.result, 7);
    assert.equal(shown, true);
    assert.ok(Date.now() - t0 >= 300, "lasts at least minMs");
    assert.ok(seen.length > 5, "ticks through, not a jump");
    assert.equal(seen.at(-1), 100);
    assert.ok(seen.every((p, i) => i === 0 || p >= seen[i - 1]), "monotonic");
    assert.ok(r.ms < 100, "reports how long the work itself took");
});

test("slow work shows a readout that lasts as long as the work and ends at 100", async () => {
    const seen = [];
    const t0 = Date.now();
    const r = await trackWork({
        work: new Promise(resolve => setTimeout(() => resolve("ok"), 600)),
        steps: ["one", "two"],
        expectedMs: 5000,
        minMs: 200,
        onTick: p => seen.push(p),
        tickMs: 50,
    });
    assert.equal(r.shown, true);
    assert.equal(r.result, "ok");
    assert.ok(Date.now() - t0 >= 600);
    assert.ok(r.ms >= 590);
    assert.equal(seen.at(-1), 100);
    assert.ok(seen.every((p, i) => i === 0 || p >= seen[i - 1]), "monotonic");
});

test("a rejected work still settles", async () => {
    const r = await trackWork({ work: Promise.reject(new Error("x")), steps: [], expectedMs: 100, minMs: 50, onTick: () => {} });
    assert.equal(r.result, undefined);
});

test("durations are remembered as a moving average, with defaults", () => {
    const store = new Map();
    const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
    const m = createDurationMemory({ depth: 12000 }, storage);
    assert.equal(m.expected("depth"), 12000);
    m.record("depth", 20000);
    assert.equal(m.expected("depth"), 20000);
    m.record("depth", 10000);
    assert.equal(m.expected("depth"), 16000);
    assert.equal(createDurationMemory({}, storage).expected("depth"), 16000);
});

test("after the work settles, the readout ticks at its pace instead of spinning", async () => {
    let ticks = 0;
    await trackWork({ work: Promise.resolve(1), steps: ["a"], expectedMs: 100, minMs: 400, tickMs: 50, onTick: () => { ticks += 1; } });
    // ~400 / 50 = 8 ticks (+ first and last); a busy loop would make thousands
    assert.ok(ticks < 20, `ticks=${ticks}`);
});
