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

// The links in the submitted SIH26175 deck (145604_SIH26175.pdf, checked 2026-09-29) and the
// permanent #/docs route. They must keep resolving whatever the pages are renamed to.
test("SIH deck links keep resolving", async () => {
    const { readFile } = await import("node:fs/promises");
    const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
    // the demo-video anchor exists, inside the Home (page-docs) page
    const docsStart = html.indexOf('id="page-docs"');
    const anchor = html.indexOf('id="demo-video"');
    assert.ok(docsStart > 0 && anchor > docsStart, "#demo-video lives on the Home page");
    const pageOf = id => (id === "demo-video" ? "page-docs" : null);
    assert.deepEqual(resolveHash("", pageOf), { pageId: "page-workbench", anchorId: null });           // https://depthwizard-studio.vercel.app/
    assert.deepEqual(resolveHash("#/demo", pageOf), { pageId: "page-explore", anchorId: null });     // …/#/demo
    assert.deepEqual(resolveHash("#demo-video", pageOf), { pageId: "page-docs", anchorId: "demo-video" }); // …/#demo-video
    assert.deepEqual(resolveHash("#/docs", pageOf), { pageId: "page-docs", anchorId: null });        // …/#/docs
});
