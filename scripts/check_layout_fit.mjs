#!/usr/bin/env node
// Layout-fit check: every app screen, in its default open/closed state, at a set of laptop / tablet viewports.
// For each panel (a box drawn over or beside the 3D view) it records the rectangle and reports
//   - panels overlapping each other,
//   - panels extending outside the window,
//   - panels / columns that need scrolling (content taller than the box),
// plus the UI scale in use (--ui-scale, src/ui-scale.js), and saves a screenshot per screen and viewport.
//
// Needs the dev server in static mode:   (cd frontend && VITE_LIBRARY_SOURCE=static npx vite --host)
// Run from the repo root:                node scripts/check_layout_fit.mjs [--url http://localhost:5173]
//                                        [--out DIR] [--browser firefox|chromium] [--only demo,studio-viewer]
// Browser: playwright-core (frontend devDependency, pinned). Its own browser if installed, else the newest
// Playwright Firefox / Chromium build under ~/.cache/ms-playwright, else PW_EXECUTABLE.
// Exit code 1 when any landscape viewport has an overlap or a panel outside the window.

import { createRequire } from "node:module";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(join(dirname(fileURLToPath(import.meta.url)), "../frontend/package.json"));
const playwright = require("playwright-core");

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
    (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const BASE = (args.url ?? "http://localhost:5173").replace(/\/$/, "");
const OUT = args.out ?? join(tmpdir(), "dw-layout-fit");
const BROWSER = args.browser ?? "firefox";
const ONLY = args.only ? args.only.split(",") : null;

export const LANDSCAPE = [[1280, 600], [1316, 645], [1366, 650], [1536, 730], [1440, 790], [1470, 830], [1920, 950],
    [2560, 1300], [1133, 670], [1180, 740], [1280, 700], [1366, 950]];
export const PORTRAIT = [[768, 950], [820, 1100]];

// Panels per screen: [label, selector]. Every visible match is one panel.
const COMMON = [["nav", "#app-sidebar"], ["theme", "#global-theme-toggle"]];
const SCREENS = {
    demo: {
        url: "/#/demo",
        panels: [...COMMON,
            ["left", "#page-explore .left-rail > *"], ["right", "#page-explore .right-rail > *"],
            ["hint", "#page-explore .bottom-right"], ["rotate", "#demo-rotate-toggle"], ["credit", "#region-credit"]],
        columns: ["#page-explore .left-rail", "#page-explore .right-rail"],
        async open(page) {
            await page.goto(BASE + this.url);
            await page.waitForTimeout(2500);
        },
    },
    "studio-input": {
        url: "/",
        panels: [...COMMON, ["card", "#input-view > .iv-pane"]],
        columns: [],
        scrollByDesign: ".iv-input-pane", // the scene list: 76 cards, scrolls at every size
        async open(page) {
            await page.goto(BASE + this.url);
            await page.waitForTimeout(2500);
        },
    },
    "studio-viewer": {
        url: "/",
        panels: [...COMMON, ["bar", "#final-demo-expanded-bar"], ["jobs", ".final-demo-box.is-expanded .job-tabs-wrap"],
            ["left", ".final-demo-box.is-expanded .final-demo-left-rail > .xp-box"],
            ["right", ".final-demo-box.is-expanded .final-demo-right-rail > .xp-box"],
            ["navbar", ".final-demo-box.is-expanded .xv-navbar"], ["toolbar", ".final-demo-box.is-expanded .xv-toolbar"]],
        columns: [".final-demo-box.is-expanded .final-demo-left-rail", ".final-demo-box.is-expanded .final-demo-right-rail"],
        async open(page) {
            await page.goto(BASE + this.url);
            await page.waitForTimeout(2500);
            await page.click(".iv-card >> nth=0"); // first library tile (Darjeeling)
            await page.waitForTimeout(800);
            await page.locator("button:visible", { hasText: "START GENERATION" }).first().click();
            await page.waitForSelector("#final-demo-back", { state: "visible", timeout: 180000 }); // software GL is slow
            await page.waitForTimeout(1500);
        },
    },
    "studio-grid": {
        url: "/",
        panels: [...COMMON, ["box", ".workbench-grid .wb-main > *, .workbench-grid .wb-side > *"]],
        columns: [],
        async open(page) {
            await SCREENS["studio-viewer"].open(page);
            await page.$eval("#final-demo-back", b => b.click()); // no actionability wait: slow software-GL frames stall it
            await page.waitForTimeout(1500);
        },
    },
};

function executable(kind) {
    try {
        const p = playwright[kind].executablePath();
        if (p && existsSync(p)) return undefined; // Playwright's own build is installed
    } catch { /* not installed */ }
    if (process.env.PW_EXECUTABLE) return process.env.PW_EXECUTABLE;
    const root = join(homedir(), ".cache", "ms-playwright");
    const builds = existsSync(root) ? readdirSync(root).filter(d => d.startsWith(`${kind}-`)).sort().reverse() : [];
    for (const b of builds) {
        const p = kind === "firefox" ? join(root, b, "firefox", "firefox") : join(root, b, "chrome-linux", "chrome");
        if (existsSync(p)) return p;
    }
    throw new Error(`no ${kind} build found: set PW_EXECUTABLE`);
}

// In-page measurement: rectangles, overlaps, outside-window, scrolling.
function measure({ panels, columns, scrollByDesign }) {
    const visible = el => {
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        return cs.display !== "none" && cs.visibility !== "hidden" && el.closest("[hidden]") === null
            && r.width > 1 && r.height > 1;
    };
    const rect = el => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; };
    // the part of a panel actually on screen: clipped by every scrolling ancestor (a panel scrolled out of its
    // column counts under "scrolls", not as outside the window); overflow: hidden ancestors are not applied,
    // so a panel cut off by the window edge still counts as outside
    const shownRect = el => {
        let { x, y, w, h } = rect(el);
        let x1 = x + w, y1 = y + h;
        for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
            const cs = getComputedStyle(a);
            if (/(auto|scroll)/.test(cs.overflowX + cs.overflowY)) {
                const b = a.getBoundingClientRect();
                x = Math.max(x, b.left); y = Math.max(y, b.top);
                x1 = Math.min(x1, b.right); y1 = Math.min(y1, b.bottom);
            }
        }
        return { x, y, w: Math.max(0, x1 - x), h: Math.max(0, y1 - y) };
    };
    const items = [];
    for (const [label, sel] of panels) {
        document.querySelectorAll(sel).forEach((el, i) => {
            if (visible(el)) items.push({ label: `${label}${i ? i : ""}:${el.dataset.box ?? el.id ?? ""}`.replace(/:$/, ""), el, r: shownRect(el) });
        });
    }
    const W = innerWidth, H = innerHeight, eps = 0.5;
    const canvases = [...document.querySelectorAll("canvas")].filter(visible).map(rect);
    const area = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
        * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const overlaps = [];
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const A = items[i], B = items[j];
            if (A.el.contains(B.el) || B.el.contains(A.el)) continue;
            const x0 = Math.max(A.r.x, B.r.x), y0 = Math.max(A.r.y, B.r.y);
            const x1 = Math.min(A.r.x + A.r.w, B.r.x + B.r.w), y1 = Math.min(A.r.y + A.r.h, B.r.y + B.r.h);
            if (x1 - x0 > 1 && y1 - y0 > 1) {
                // a small control drawn over a 3D canvas inside a box (the theme toggle on the grid's 3D preview) is fine
                const hit = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
                if ([A, B].some(p => p.label.startsWith("theme")) && canvases.some(c => area(hit, c) >= 0.9 * hit.w * hit.h)) continue;
                overlaps.push(`${A.label} x ${B.label} (${Math.round(hit.w)}x${Math.round(hit.h)})`);
            }
        }
    }
    const outside = items.filter(p => p.r.w > 0 && p.r.h > 0).filter(p => p.r.x < -eps || p.r.y < -eps || p.r.x + p.r.w > W + eps || p.r.y + p.r.h > H + eps)
        .map(p => `${p.label} (${Math.round(p.r.x)},${Math.round(p.r.y)} ${Math.round(p.r.w)}x${Math.round(p.r.h)})`);
    const scrolling = [...columns.flatMap(s => [...document.querySelectorAll(s)]), ...items.map(p => p.el)]
        .filter(visible)
        .filter(el => !(scrollByDesign && el.matches(scrollByDesign)))
        .filter(el => /(auto|scroll)/.test(getComputedStyle(el).overflowY) && el.scrollHeight > el.clientHeight + 1)
        .map(el => `${el.className.split(" ")[0] || el.id} (+${el.scrollHeight - el.clientHeight}px)`);
    const scale = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ui-scale")) || 1;
    return {
        scale,
        overlaps, outside, scrolling: [...new Set(scrolling)],
        panels: items.map(p => ({ label: p.label, ...Object.fromEntries(Object.entries(p.r).map(([k, v]) => [k, Math.round(v)])) })),
    };
}

async function main() {
    mkdirSync(OUT, { recursive: true });
    const browser = await playwright[BROWSER].launch({
        executablePath: executable(BROWSER),
        ...(BROWSER === "firefox" ? { firefoxUserPrefs: { "webgl.force-enabled": true } } : { args: ["--use-gl=swiftshader", "--ignore-gpu-blocklist"] }),
    });
    const report = {};
    let failed = 0;
    for (const [name, screen] of Object.entries(SCREENS)) {
        if (ONLY && !ONLY.includes(name)) continue;
        const page = await browser.newPage({ viewport: { width: 1470, height: 830 } });
        page.on("pageerror", e => console.log(`  [${name}] pageerror: ${e.message}`));
        await screen.open(page);
        for (const [w, h] of [...LANDSCAPE, ...PORTRAIT]) {
            await page.setViewportSize({ width: w, height: h });
            await page.mouse.move(w / 2, h / 2); // off the left nav, which opens on hover
            // wait until src/ui-scale.js has computed the scale for this window size (slow software-GL frames can
            // delay its timer), then let the resulting relayout settle
            await page.waitForFunction(size => document.documentElement.dataset.uiScaleFor === size, `${w}x${h}`,
                { timeout: 15000 }).catch(() => console.log(`  [${name}] ${w}x${h}: scale not recomputed in 15 s`));
            await page.waitForTimeout(500);
            const m = await page.evaluate(measure, { panels: screen.panels, columns: screen.columns, scrollByDesign: screen.scrollByDesign ?? null });
            const portrait = h > w;
            await page.screenshot({ path: join(OUT, `${name}-${w}x${h}.png`) });
            report[`${name} ${w}x${h}`] = m;
            const bad = m.overlaps.length + m.outside.length;
            if (bad && !portrait) failed += 1;
            const tag = bad ? (portrait ? "WARN" : "FAIL") : "ok  ";
            console.log(`${tag} ${name.padEnd(13)} ${`${w}x${h}`.padEnd(10)} scale ${m.scale.toFixed(2)}`
                + (m.overlaps.length ? ` | overlap: ${m.overlaps.join("; ")}` : "")
                + (m.outside.length ? ` | outside: ${m.outside.join("; ")}` : "")
                + (m.scrolling.length ? ` | scrolls: ${m.scrolling.join("; ")}` : ""));
        }
        await page.close();
    }
    await browser.close();
    writeFileSync(join(OUT, "report.json"), JSON.stringify(report, null, 1));
    console.log(`\n${failed ? `${failed} landscape failure(s)` : "all landscape viewports fit"}; screenshots + report.json in ${OUT}`);
    process.exit(failed ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
