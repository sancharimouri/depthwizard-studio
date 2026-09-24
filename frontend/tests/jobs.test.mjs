import test from "node:test";
import assert from "node:assert/strict";

import { createJobStore, exportFilename, jobsExport, unsavedCopy } from "../src/jobs.js";

const input = (title, tier) => ({
    title, source: "library", meta: [["GSD", "10 m"]], routing: { tier, label: `Tier ${tier}` }, dem: null,
});

test("jobs: newest first in the panel, oldest first in the tab strip, newest active", () => {
    const store = createJobStore();
    let changes = 0;
    store.onChange(() => changes++);
    const a = store.add(input("A", 2));
    const b = store.add(input("B", 1));
    assert.deepEqual(store.panelOrder().map(j => j.input.title), ["B", "A"]);
    assert.deepEqual(store.creationOrder().map(j => j.input.title), ["A", "B"]);
    assert.equal(store.active(), b);
    store.setActive(a.id);
    assert.equal(store.active(), a);
    assert.equal(store.generating(), a); // both start "generating"
    store.update(a.id, { status: "complete" });
    assert.equal(store.generating(), b);
    assert.ok(changes >= 4);
});

test("jobs: unsaved until saved", () => {
    const store = createJobStore();
    const a = store.add(input("A", 2));
    const b = store.add(input("B", 1));
    assert.equal(store.unsaved().length, 2);
    store.markSaved([a.id]);
    assert.deepEqual(store.unsaved().map(j => j.id), [b.id]);
});

test("unsaved modal copy states the count and never offers 'don't show again'", () => {
    for (const action of ["new", "close", "quit"]) {
        const one = unsavedCopy(action, 1);
        const three = unsavedCopy(action, 3);
        assert.equal(one.title, "1 unsaved job");
        assert.equal(three.title, "3 unsaved jobs");
        assert.ok(!/show (this|again)/i.test(JSON.stringify([one, three])));
    }
    assert.match(unsavedCopy("new", 2).discard, /Continue without saving/);
    assert.match(unsavedCopy("close", 2).discard, /Discard & close/);
});

test("export: real job content, marked as a session export", () => {
    const store = createJobStore();
    const job = store.add(input("almora_RGB.tif", 1));
    job.log.push({ t: "00:00.20", text: "Input: almora_RGB.tif" });
    const now = new Date("2026-09-25T10:11:12Z");
    const out = jobsExport([job], now);
    assert.equal(out.format, "depthwizard2.jobs/v1");
    assert.equal(out.jobs[0].input.details.GSD, "10 m");
    assert.deepEqual(out.jobs[0].calculation_log, ["[00:00.20] Input: almora_RGB.tif"]);
    assert.match(out.storage_note, /no server-side job storage/);
    assert.equal(exportFilename([job], now), "depthwizard-job1-2026-09-25-10-11-12.json");
});
