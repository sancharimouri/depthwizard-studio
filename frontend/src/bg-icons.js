// Drives the input page's background icons (index.html .input-bg-icons): each
// floats in its own neighbourhood and reacts to the pointer (src/bg-float.js).
// Runs only while the icons are on screen; still under prefers-reduced-motion.

import { CM_PX, createFloater, kick, stepPush, wander } from "./bg-float.js";

export function startBackgroundIcons(layer, page) {
    if (!layer || !page) {
        return;
    }
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const icons = [...layer.querySelectorAll(".bgi")].map((el, i) => ({
        el,
        rot: Number(el.dataset.rot ?? 0),
        floater: createFloater(i + 1, Number(el.dataset.radiusCm ?? 5) * CM_PX),
    }));
    let pointer = null;
    let raf = 0;
    let last = 0;

    // home centre of each icon in page coordinates (its CSS left/top), refreshed per frame
    const homeOf = (icon, box) => ({
        x: box.left + (parseFloat(icon.el.style.left) / 100) * box.width,
        y: box.top + (parseFloat(icon.el.style.top) / 100) * box.height,
    });

    page.addEventListener("pointermove", event => {
        pointer = { x: event.clientX, y: event.clientY };
    }, { passive: true });
    page.addEventListener("pointerleave", () => {
        pointer = null;
    });
    page.addEventListener("pointerdown", event => {
        if (!visible()) {
            return;
        }
        const box = layer.getBoundingClientRect();
        const t = performance.now() / 1000;
        icons.forEach(icon => {
            const home = homeOf(icon, box);
            const w = wander(icon.floater, t);
            kick(icon.floater, { x: home.x + w.x + icon.floater.push.x, y: home.y + w.y + icon.floater.push.y },
                { x: event.clientX, y: event.clientY });
        });
    }, { passive: true });

    function visible() {
        return layer.offsetParent !== null && document.visibilityState === "visible";
    }

    function frame(now) {
        raf = 0;
        if (!visible()) {
            return;
        }
        const t = now / 1000;
        const dt = Math.min(0.05, last ? (now - last) / 1000 : 1 / 60);
        last = now;
        const box = layer.getBoundingClientRect();
        icons.forEach(icon => {
            const w = reduced.matches ? { x: 0, y: 0, rot: 0 } : wander(icon.floater, t);
            const home = homeOf(icon, box);
            const at = { x: home.x + w.x + icon.floater.push.x, y: home.y + w.y + icon.floater.push.y };
            const p = reduced.matches ? icon.floater.push : stepPush(icon.floater, dt, at, pointer);
            icon.el.style.transform = `translate(-50%, -50%) translate(${(w.x + p.x).toFixed(1)}px, ${(w.y + p.y).toFixed(1)}px) `
                + `rotate(${(icon.rot + w.rot).toFixed(2)}deg)`;
        });
        schedule();
    }

    function schedule() {
        if (!raf) {
            raf = requestAnimationFrame(frame);
        }
    }

    // restart whenever the icons may have come back on screen
    new MutationObserver(() => {
        last = 0;
        schedule();
    }).observe(page, { attributes: true, subtree: true, attributeFilter: ["hidden", "class"] });
    document.addEventListener("visibilitychange", () => {
        last = 0;
        schedule();
    });
    schedule();
}
