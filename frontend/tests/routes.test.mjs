import test from "node:test";
import assert from "node:assert/strict";
import { PAGE_ROUTES, DEMO_VIDEO_ANCHOR, resolveHash, hashMatchesPage, worldForHash, isRetiredDocsHash } from "../src/routes.js";

const pageOf = id => ({ "demo-rotate-toggle": "page-explore" })[id] ?? null;
const appHas = id => id === "demo-rotate-toggle";

test("permanent routes are fixed strings", () => {
    assert.equal(PAGE_ROUTES["page-explore"], "#/demo");
    assert.equal(PAGE_ROUTES["page-workbench"], "#/studio");
    assert.equal(PAGE_ROUTES["page-docs"], undefined); // Docs was removed (2026-10-26); #/docs opens Home
    assert.equal(DEMO_VIDEO_ANCHOR, "demo-video");
    assert.ok(Object.isFrozen(PAGE_ROUTES));
});

test("route and anchor hashes resolve to their page", () => {
    assert.deepEqual(resolveHash("#/demo", pageOf), { pageId: "page-explore", anchorId: null });
    assert.deepEqual(resolveHash("#/studio", pageOf), { pageId: "page-workbench", anchorId: null });
    assert.deepEqual(resolveHash("#demo-rotate-toggle", pageOf), { pageId: "page-explore", anchorId: "demo-rotate-toggle" });
    assert.deepEqual(resolveHash("", pageOf), { pageId: "page-workbench", anchorId: null }); // the desktop app's bare URL
    assert.deepEqual(resolveHash("#/nope", pageOf), { pageId: "page-workbench", anchorId: null });
    assert.deepEqual(resolveHash("#missing", pageOf), { pageId: "page-workbench", anchorId: null });
});

test("hashMatchesPage", () => {
    assert.ok(hashMatchesPage("#/demo", "page-explore", pageOf));
    assert.ok(hashMatchesPage("", "page-workbench", pageOf));
    assert.ok(!hashMatchesPage("#/demo", "page-workbench", pageOf));
});

test("worldForHash: Home owns the bare URL, its anchors, retired and unknown hashes; the app owns its routes and anchors", () => {
    for (const h of ["#/studio", "#/demo", "#demo-rotate-toggle"]) {
        assert.equal(worldForHash(h, appHas), "app", h);
    }
    for (const h of ["", "#", "#/", "#demo-video", "#credits", "#/docs", "#docs-credits", "#/nope", "#missing", "#%E0%A4%A"]) {
        assert.equal(worldForHash(h, appHas), "home", h);
    }
    assert.equal(worldForHash("", appHas, true), "app"); // desktop: no Home
    assert.equal(worldForHash("#/docs", appHas, true), "app");
});

test("the retired Docs addresses", () => {
    for (const h of ["#/docs", "#docs-credits", "#docs-pages", " #/docs "]) {
        assert.ok(isRetiredDocsHash(h), h);
    }
    for (const h of ["", "#/demo", "#/studio", "#demo-video", "#/docs/x", "#docsx"]) {
        assert.ok(!isRetiredDocsHash(h), h);
    }
});

// The links in the submitted SIH26175 deck (145604_SIH26175.pdf, checked 2026-09-29), including #/docs.
// They must keep resolving whatever the pages are renamed to.
test("SIH deck links keep resolving", async () => {
    const { readFile } = await import("node:fs/promises");
    const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
    // the demo-video anchor is served in the HTML, once, in Home's markup (the Home world: src/boot.js)
    const home = html.slice(html.indexOf('<template id="world-home">'), html.indexOf('<template id="world-app">'));
    assert.equal(html.split('id="demo-video"').length, 2, "exactly one id=\"demo-video\"");
    assert.ok(home.includes('id="demo-video"'), "#demo-video lives on Home");
    assert.ok(!html.includes('id="page-docs"'), "the Docs page is gone");
    assert.equal(worldForHash("", appHas), "home");                                                      // https://depthwizard-studio.vercel.app/
    assert.equal(worldForHash("#demo-video", appHas), "home");                                           // …/#demo-video
    assert.equal(worldForHash("#/demo", appHas), "app");                                                 // …/#/demo
    assert.deepEqual(resolveHash("#/demo", pageOf), { pageId: "page-explore", anchorId: null });
    assert.equal(worldForHash("#/docs", appHas), "home");                                                // …/#/docs
    assert.ok(isRetiredDocsHash("#/docs"));
});
