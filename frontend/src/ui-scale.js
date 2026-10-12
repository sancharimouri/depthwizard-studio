// UI scale: when the window is smaller than a screen's design size, the UI layer (left nav, panels, toolbars,
// top bar) shrinks proportionally; the 3D canvas never does. Sets --ui-scale on <html>; styles.css applies it
// with CSS `zoom` to the UI-layer elements ("UI SCALE" block). zoom (not transform) because panels keep
// reflowing (columns, wrapping, scroll caps) and, in current Chromium and Firefox, hit-testing, range inputs,
// getBoundingClientRect and fixed / absolute anchors all follow it; a transform would leave layout boxes at full
// size and re-anchor position: fixed descendants (tooltips, menus).
//
// Design size per screen, in the screen's own (unscaled) CSS px, from its DEFAULT open/closed state:
//   width  = left nav + each column with its margin + the toolbars or a minimum 3D area between them
//   height = the tallest column's content + that column's reserved top / bottom areas (--col-reserve)
// It is re-measured only while every box is in its default state, so expanding a box makes its column scroll
// instead of shrinking the whole UI. scale = min(MAX_SCALE, window / design), never below MIN_SCALE; below
// the floor the columns scroll internally. The Studio input page is not scaled: picking a tile filled its preview
// card and shrank every text and icon to make it fit (2026-10-12); the preview card scrolls instead.

export const MAX_SCALE = 1; // the UI never grows past its designed size; raise to let big screens scale up
export const MIN_SCALE = 0.68;
const MIN_3D_WIDTH = 480; // px of 3D view kept clear between the side columns
const DEBOUNCE_MS = 80;

// Effective zoom of an element (visual px per own CSS px): for pointer maths inside scaled panels.
export function zoomOf(el) {
    let z = 1;
    for (let a = el; a; a = a.parentElement) {
        z *= parseFloat(getComputedStyle(a).zoom) || 1; // computed zoom is the plain factor in Chromium and Firefox
    }
    return z;
}

const px = (el, prop) => parseFloat(getComputedStyle(el)[prop]) || 0;
const reserve = el => parseFloat(getComputedStyle(el).getPropertyValue("--col-reserve")) || 0;
const shown = el => el && el.getClientRects().length > 0;
const navWidth = () => {
    const nav = document.getElementById("app-sidebar");
    return nav ? parseFloat(getComputedStyle(nav).getPropertyValue("--sb-rail-w")) || nav.offsetWidth : 0;
};

// A box is in its default state when its open/closed state matches data-default-open.
function boxesAtDefault(root) {
    return [...root.querySelectorAll(".xp-box")].every(box => {
        const head = box.querySelector(":scope > .xp-head");
        return !head || (head.getAttribute("aria-expanded") === "true") === box.hasAttribute("data-default-open");
    });
}

// Two side columns over a 3D view: design width and height.
function columnsDesign(left, right, middleMin) {
    const width = navWidth()
        + px(left, "left") + left.offsetWidth
        + middleMin
        + right.offsetWidth + px(right, "right");
    const height = Math.max(left.scrollHeight + reserve(left), right.scrollHeight + reserve(right));
    return { width, height };
}

const SCREENS = [
    {
        key: "demo",
        active: () => document.getElementById("page-explore")?.classList.contains("active"),
        design() {
            const left = document.querySelector("#page-explore .left-rail");
            const right = document.querySelector("#page-explore .right-rail");
            if (!left || !right) {
                return null;
            }
            const design = columnsDesign(left, right, MIN_3D_WIDTH);
            // the reconstruction's progress bar (right column) only shows during a run: its room is always
            // counted, so the scale doesn't change when it appears
            const bar = document.getElementById("pipeline-progress");
            if (bar?.hidden && right.contains(bar)) {
                design.height = Math.max(design.height, right.scrollHeight + reserve(right)
                    + px(bar, "height") + px(bar, "marginTop"));
            }
            return design;
        },
    },
    {
        key: "studio-viewer",
        active: () => document.getElementById("page-workbench")?.classList.contains("active")
            && shown(document.querySelector(".final-demo-box.is-expanded")),
        design() {
            const box = document.querySelector(".final-demo-box.is-expanded");
            const left = box.querySelector(".final-demo-left-rail");
            const right = box.querySelector(".final-demo-right-rail");
            if (!shown(left) || !shown(right) || !boxesAtDefault(box)) {
                return null; // keep the last default-state measurement
            }
            const toolbar = box.querySelector(".xv-toolbar");
            const middle = Math.max(MIN_3D_WIDTH, (toolbar?.offsetWidth ?? 0) + 32);
            return columnsDesign(left, right, middle);
        },
    },
];

// screen key -> { size: the window size it was measured at, design }. Within one window size the design only
// grows: content that changes height while the page runs (the reconstruction's progress bar, a line re-wrapping
// at another scale) then settles the scale once instead of making it flicker back and forth. A new window size
// measures afresh.
const cache = new Map();
let currentScale = 1;

function compute() {
    const screen = SCREENS.find(s => s.active());
    if (!screen) {
        return 1;
    }
    const size = `${window.innerWidth}x${window.innerHeight}`;
    const measured = screen.design();
    const prev = cache.get(screen.key);
    if (measured) {
        const keep = prev?.size === size ? prev.design : { width: 0, height: 0 };
        cache.set(screen.key, {
            size,
            design: { width: Math.max(keep.width, measured.width), height: Math.max(keep.height, measured.height) },
        });
    }
    const design = cache.get(screen.key)?.design;
    if (design) {
        document.documentElement.dataset.uiDesign = `${Math.round(design.width)}x${Math.round(design.height)}`; // for checks
    }
    if (!design) {
        return 1;
    }
    const fit = Math.min(
        design.width ? window.innerWidth / design.width : Infinity,
        design.height ? window.innerHeight / design.height : Infinity,
    );
    return Math.max(MIN_SCALE, Math.min(MAX_SCALE, Math.floor(fit * 100) / 100));
}

function apply() {
    // measure at the current scale (design() reads own px, so the scale doesn't feed back), then set the new one
    const scale = compute();
    document.documentElement.dataset.uiScaleFor = `${window.innerWidth}x${window.innerHeight}`; // for checks
    if (scale !== currentScale) {
        currentScale = scale;
        document.documentElement.style.setProperty("--ui-scale", String(scale));
        window.dispatchEvent(new Event("resize")); // canvases and observers re-measure the freed / taken space
    }
}

// Coalesces bursts (resize drags, a run toggling classes every frame) into one update DEBOUNCE_MS after the
// first event; the timer isn't restarted, so a steady stream of changes can't starve it.
let timer = 0;
export function scheduleUiScale() {
    if (!timer) {
        timer = setTimeout(() => {
            timer = 0;
            apply();
        }, DEBOUNCE_MS);
    }
}

export function initUiScale() {
    document.documentElement.style.setProperty("--ui-scale", "1");
    window.addEventListener("resize", scheduleUiScale);
    window.addEventListener("orientationchange", scheduleUiScale);
    window.visualViewport?.addEventListener("resize", scheduleUiScale); // browser zoom
    // page switches, the Studio viewer opening / closing, boxes opening / closing, panels filling in
    new MutationObserver(scheduleUiScale).observe(document.body, {
        subtree: true, attributes: true, attributeFilter: ["class", "hidden", "aria-expanded"],
    });
    apply();
}
