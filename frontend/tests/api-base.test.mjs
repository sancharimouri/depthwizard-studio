import { test } from "node:test";
import assert from "node:assert/strict";
import { API_BASE, apiUrl } from "../src/api-base.js";

const RENDER = "https://depthwizard2-api.onrender.com";

test("no VITE_API_BASE (local dev, Vite proxy): paths stay same-origin", () => {
    assert.equal(API_BASE, "");
    assert.equal(apiUrl("/api/library"), "/api/library");
});

test("backend-relative URLs resolve against the configured backend", () => {
    assert.equal(apiUrl("/api/library/dfc2019-JAX_004_006/thumbnail", RENDER),
        `${RENDER}/api/library/dfc2019-JAX_004_006/thumbnail`);
    assert.equal(apiUrl("/api/input/abc/dem/preview", RENDER), `${RENDER}/api/input/abc/dem/preview`);
});

test("absolute release URLs, protocol-relative URLs and null pass through", () => {
    const gh = "https://github.com/sancharimouri/depthwizard2-assets/releases/download/library-v1/x__thumb.jpg";
    assert.equal(apiUrl(gh, RENDER), gh);
    assert.equal(apiUrl("//cdn.example/x", RENDER), "//cdn.example/x");
    assert.equal(apiUrl(null, RENDER), null);
});
