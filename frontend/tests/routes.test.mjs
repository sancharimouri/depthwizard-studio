import test from "node:test";
import assert from "node:assert/strict";
import { PAGE_ROUTES, DEMO_VIDEO_ANCHOR, resolveHash, hashMatchesPage } from "../src/routes.js";

const pageOf = id => ({ "demo-video": "page-docs", "docs-pages": "page-docs" })[id] ?? null;

test("permanent routes are fixed strings", () => {
    assert.equal(PAGE_ROUTES["page-docs"], "#/docs");
    assert.equal(PAGE_ROUTES["page-explore"], "#/demo");
    assert.equal(PAGE_ROUTES["page-workbench"], "");
    assert.equal(DEMO_VIDEO_ANCHOR, "demo-video");
    assert.ok(Object.isFrozen(PAGE_ROUTES));
});

test("route and anchor hashes resolve to their page", () => {
    assert.deepEqual(resolveHash("#/docs", pageOf), { pageId: "page-docs", anchorId: null });
    assert.deepEqual(resolveHash("#/demo", pageOf), { pageId: "page-explore", anchorId: null });
    assert.deepEqual(resolveHash("#demo-video", pageOf), { pageId: "page-docs", anchorId: "demo-video" });
    assert.deepEqual(resolveHash("", pageOf), { pageId: "page-workbench", anchorId: null });
    assert.deepEqual(resolveHash("#/nope", pageOf), { pageId: "page-workbench", anchorId: null });
    assert.deepEqual(resolveHash("#missing", pageOf), { pageId: "page-workbench", anchorId: null });
});

test("hashMatchesPage", () => {
    assert.ok(hashMatchesPage("#demo-video", "page-docs", pageOf));
    assert.ok(!hashMatchesPage("#/docs", "page-workbench", pageOf));
});
