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

import { createSavedStore, jobIconLabels, jobListLabel, jobPlace, savedRecord, splitListLabel } from "../src/jobs.js";

test("sidebar label: job number + short place; renaming replaces the whole label", () => {
    assert.equal(jobPlace({ source: "library", title: "Ooty (Nilgiris), Tamil Nadu" }), "Ooty (Nilgiris), TN");
    assert.equal(jobPlace({ source: "search", geo: { lat: 27.045, lon: -88.26 } }), "27.05°N 88.26°W");
    assert.equal(jobPlace({ source: "upload" }), "");
    const store = createJobStore();
    const j = store.add({ source: "library", title: "Darjeeling, West Bengal" });
    assert.equal(jobListLabel(j), "Job 1 (Darjeeling, WB)");
    assert.deepEqual(splitListLabel("Job 1 (Ooty (Nilgiris), TN)"), { head: "Job 1", place: "Ooty (Nilgiris), TN" });
    assert.deepEqual(splitListLabel("Hills"), { head: "Hills", place: "" });
    store.rename(j.id, "Job 1 (Darjeeling, WB)");
    assert.equal(j.name, null);
    store.rename(j.id, "Job 1");
    assert.equal(jobListLabel(j), "Job 1");
});

function memoryStorage() {
    const data = new Map();
    return { getItem: k => data.get(k) ?? null, setItem: (k, v) => data.set(k, String(v)) };
}

test("a pinned job shows only in the pinned section; remove re-activates the newest remaining job", () => {
    const store = createJobStore();
    const a = store.add(input("A", 2));
    const b = store.add(input("B", 1));
    const c = store.add(input("C", 1));
    store.togglePin(a.id);
    assert.deepEqual(store.pinnedOrder().map(j => j.input.title), ["A"]);
    assert.deepEqual(store.panelOrder().map(j => j.input.title), ["C", "B"]);
    assert.equal(store.remove(c.id), b);
    assert.equal(store.remove(b.id), a);
    assert.equal(store.remove(a.id), null);
    assert.equal(store.count(), 0);
});

test("saved store: survives a new store on the same storage, newest first, re-save replaces", () => {
    const storage = memoryStorage();
    const store = createJobStore();
    const a = store.add(input("A", 2));
    a.log.push({ t: "00:00.20", text: "x" });
    const b = store.add(input("B", 1));
    createSavedStore(storage).put(savedRecord(a, new Date("2026-09-25T01:00:00Z")));
    createSavedStore(storage).put(savedRecord(b, new Date("2026-09-25T02:00:00Z")));
    createSavedStore(storage).put(savedRecord(a, new Date("2026-09-25T03:00:00Z")));
    const saved = createSavedStore(storage).list();
    assert.deepEqual(saved.map(r => r.input.title), ["A", "B"]);
    // restoring gives a complete, saved job with the same uid and log
    const restored = createJobStore().add(saved[0].input, saved[0]);
    assert.equal(restored.status, "complete");
    assert.equal(restored.saved, true);
    assert.equal(restored.uid, a.uid);
    assert.deepEqual(restored.log, [{ t: "00:00.20", text: "x" }]);
    createSavedStore(storage).remove(a.uid);
    assert.deepEqual(createSavedStore(storage).list().map(r => r.input.title), ["B"]);
});

test("saved store: broken or blocked storage degrades to an empty list", () => {
    const blocked = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
    assert.deepEqual(createSavedStore(blocked).list(), []);
    assert.equal(createSavedStore(blocked).put({ uid: "x", savedAt: "" }), false);
    const junk = { getItem: () => "{not json", setItem() {} };
    assert.deepEqual(createSavedStore(junk).list(), []);
});

test("job icons: J<n> by default, two letters when renamed, first differing letter on a clash", () => {
    const j = (n, name = null) => ({ n, name });
    assert.deepEqual(jobIconLabels([j(1), j(2, "Darjeeling hills"), j(3, "Hills"), j(4, "Himalaya"), j(5, "hills"), j(6, "Ooty")]),
        ["J1", "Da", "H3", "Hm", "H5", "Oo"]);
    assert.deepEqual(jobIconLabels([j(1, "Hills"), j(2, "Himalaya")]), ["Hl", "Hm"]);
});
