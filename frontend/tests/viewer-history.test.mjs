import { test } from "node:test";
import assert from "node:assert/strict";
import { createViewerHistory } from "../src/viewer-history.js";

function setup() {
    const state = { a: 1 };
    let camera = { p: [0, 0, 0], t: [0, 0, 0] };
    const h = createViewerHistory({
        capture: () => ({ ...state }),
        apply: st => Object.assign(state, st),
        applyCamera: c => { camera = c; },
        sameCamera: (x, y) => JSON.stringify(x) === JSON.stringify(y),
    });
    h.reset();
    return { h, state, cam: () => camera, setCam: c => { camera = c; } };
}

test("commits only real changes; undo/redo restore state", () => {
    const { h, state } = setup();
    assert.equal(h.commit(), false);
    assert.equal(h.canUndo, false);
    state.a = 2;
    assert.equal(h.commit(), true);
    state.a = 3;
    h.commit();
    h.undo();
    assert.equal(state.a, 2);
    h.undo();
    assert.equal(state.a, 1);
    assert.equal(h.canUndo, false);
    assert.equal(h.canRedo, true);
    h.redo();
    h.redo();
    assert.equal(state.a, 3);
    assert.equal(h.canRedo, false);
});

test("a new action clears the redo stack", () => {
    const { h, state } = setup();
    state.a = 2; h.commit();
    h.undo();
    state.a = 5; h.commit();
    assert.equal(h.canRedo, false);
    h.undo();
    assert.equal(state.a, 1);
});

test("camera and state entries share one timeline", () => {
    const { h, state, cam } = setup();
    state.a = 2; h.commit();
    const before = { p: [1, 0, 0], t: [0, 0, 0] };
    const after = { p: [2, 0, 0], t: [0, 0, 0] };
    assert.equal(h.commitCamera(before, before), false); // no-op gesture
    h.commitCamera(before, after);
    h.undo();
    assert.deepEqual(cam(), before);
    assert.equal(state.a, 2);
    h.undo();
    assert.equal(state.a, 1);
    h.redo();
    h.redo();
    assert.deepEqual(cam(), after);
});

test("applying an entry never records a new one", () => {
    const { h, state } = setup();
    state.a = 2; h.commit();
    h.undo();
    assert.equal(h.commit(), false);
    assert.equal(h.canRedo, true);
});
