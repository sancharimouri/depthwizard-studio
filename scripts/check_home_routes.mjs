#!/usr/bin/env node
// Home / app routing check (src/boot.js, src/routes.js): every URL lands in the right world and page, and neither world
// loads the other's files.
//
//   / and an unknown hash      Home, no sidebar
//   /#demo-video               Home, the demo video section in view (a PROTECTED link)
//   /#/demo                    the Demo page (a PROTECTED link)
//   /#/studio                  the Studio input page
//   /#/docs, /#docs-credits    Home, with the retired hash dropped from the URL (the Docs page was removed)
//   files                      what Home loads and what the app loads share only the boot code and the shared
//                              three.js chunks; nothing either world requests fails
//
// Run from the repo root against the dev server or a preview (local) or the live site:
//   node scripts/check_home_routes.mjs [--url http://localhost:5173] [--browser chromium|firefox]
// Browser: as scripts/check_layout_fit.mjs. Exit code 1 on any failure.

import { createRequire } from "node:module";
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(join(dirname(fileURLToPath(import.meta.url)), "../frontend/package.json"));
const playwright = require("playwright-core");

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
    (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const BASE = (args.url ?? "http://localhost:5173").replace(/\/$/, "");
const BROWSER = args.browser ?? "chromium";
// Loaded by both worlds on purpose: the boot script and its route table, Vite's runtime helpers, three.js, the icon.
const SHARED = /\/(@vite\/|node_modules\/|src\/boot\.js|src\/routes\.js|assets\/(index|routes|preload-helper|rolldown-runtime|three\.[a-z]+|BufferGeometryUtils|OrbitControls)-)|\/brand-d\.png$/;

function executable(kind) {
    try {
        if (existsSync(playwright[kind].executablePath())) return undefined;
    } catch { /* not installed */ }
    if (process.env.PW_EXECUTABLE) return process.env.PW_EXECUTABLE;
    const root = join(homedir(), ".cache", "ms-playwright");
    const builds = existsSync(root) ? readdirSync(root).filter(d => d.startsWith(`${kind}-`)).sort().reverse() : [];
    for (const b of builds) {
        const paths = kind === "firefox" ? [join(root, b, "firefox", "firefox")]
            : ["chrome-linux64", "chrome-linux"].map(d => join(root, b, d, "chrome"));
        const p = paths.find(existsSync);
        if (p) return p;
    }
    return undefined;
}

const browser = await playwright[BROWSER].launch({ executablePath: executable(BROWSER) });
let failed = 0;
const say = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failed = 1; };

// Opens `hash` in a fresh page; returns its state and every same-origin file it asked for.
async function visit(hash) {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const files = new Set(), errors = [];
    page.on("request", r => { if (r.url().startsWith(BASE)) files.add(new URL(r.url()).pathname); });
    page.on("response", r => { if (r.url().startsWith(BASE) && r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    page.on("requestfailed", r => { if (r.url().startsWith(BASE)) errors.push(`failed ${r.url()}`); });
    await page.goto(BASE + "/" + hash, { waitUntil: "load" });
    await page.waitForFunction(() => !document.documentElement.classList.contains("dw-boot"), null, { timeout: 30000 });
    await page.waitForTimeout(2500);
    const state = await page.evaluate(() => {
        const inView = id => {
            const r = document.getElementById(id)?.getBoundingClientRect();
            return Boolean(r) && r.top > -r.height && r.top < innerHeight / 3; // scrolled to (scroll margins allowed)
        };
        return {
            world: document.documentElement.dataset.world,
            sidebar: Boolean(document.getElementById("app-sidebar")),
            home: Boolean(document.getElementById("site")),
            activePage: document.querySelector(".page.active")?.id ?? null,
            demoVideoInView: inView("demo-video"),
            hash: location.hash,
            historyLength: history.length,
        };
    });
    await page.close();
    return { state, files, errors };
}

const home = await visit("");
say(home.state.world === "home" && home.state.home && !home.state.sidebar, `/ is Home, no sidebar (${JSON.stringify(home.state)})`);
const video = await visit("#demo-video");
say(video.state.world === "home" && video.state.demoVideoInView, `/#demo-video is Home with the demo video in view`);
const demo = await visit("#/demo");
say(demo.state.world === "app" && demo.state.activePage === "page-explore", `/#/demo is the Demo page (${demo.state.activePage})`);
const studio = await visit("#/studio");
say(studio.state.world === "app" && studio.state.activePage === "page-workbench" && studio.state.hash === "#/studio",
    `/#/studio is the Studio input page (${studio.state.activePage}, ${studio.state.hash})`);
const docs = await visit("#/docs");
say(docs.state.world === "home" && !docs.state.sidebar && docs.state.hash === "" && docs.state.historyLength === home.state.historyLength,
    `/#/docs (retired) is Home, hash replaced (${JSON.stringify(docs.state)})`);
const anchor = await visit("#docs-credits");
say(anchor.state.world === "home" && !anchor.state.sidebar && anchor.state.hash === "" && anchor.state.historyLength === home.state.historyLength,
    `/#docs-credits (retired) is Home, hash replaced`);
const unknown = await visit("#/no-such-page");
say(unknown.state.world === "home" && !unknown.state.sidebar, `an unknown hash is Home`);

// Crossing worlds: Home's ENTER STUDIO (a same-document hash change) and the sidebar's Home item each reload into
// the other world.
{
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const settled = () => page.waitForFunction(() => document.documentElement.dataset.world
        && !document.documentElement.classList.contains("dw-boot"), null, { timeout: 30000 });
    await page.goto(BASE + "/");
    await settled();
    await page.evaluate(() => document.querySelector("button.hero__cta").click());
    await page.waitForFunction(() => document.documentElement.dataset.world === "app", null, { timeout: 30000 });
    await settled();
    say(await page.evaluate(() => document.querySelector(".page.active")?.id === "page-workbench"),
        "Home's ENTER STUDIO opens the Studio");
    await page.evaluate(() => document.querySelector('.sb-page-link[data-page="home"]').click());
    await page.waitForFunction(() => document.documentElement.dataset.world === "home", null, { timeout: 30000 });
    say(await page.evaluate(() => location.hash === "" && !document.getElementById("app-sidebar")), "the sidebar's Home item opens Home");
    await page.goto(BASE + "/#/studio");
    await settled();
    await page.evaluate(() => { location.hash = "#/docs"; });
    await page.waitForFunction(() => document.documentElement.dataset.world === "home", null, { timeout: 30000 });
    say(await page.evaluate(() => location.hash === ""), "#/docs typed in the app opens Home, hash replaced");
    await page.close();
}

const homeFiles = new Set([...home.files, ...video.files, ...unknown.files, ...docs.files, ...anchor.files]);
const appFiles = new Set([...demo.files, ...studio.files]);
const both = [...homeFiles].filter(f => appFiles.has(f) && f !== "/" && !SHARED.test(f));
say(both.length === 0, `Home and the app share no files beyond the boot code${both.length ? `: ${both.join(", ")}` : ""}`);
say(![...appFiles].some(f => /\/home\//.test(f)), "no Home files (/home/, /src/home/) load on app routes");
const errors = [home, video, demo, studio, docs, anchor, unknown].flatMap(v => v.errors);
say(errors.length === 0, `no failed requests${errors.length ? `: ${[...new Set(errors)].join(", ")}` : ""}`);

await browser.close();
console.log(failed ? "HOME ROUTES CHECK FAILED" : "ALL HOME ROUTES OK");
process.exit(failed);
