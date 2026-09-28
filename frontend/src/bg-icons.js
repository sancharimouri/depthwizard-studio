// Drives the input page's background icons (index.html .input-bg-icons): each
// floats in its own neighbourhood, reacts to the pointer, keeps clear of the others
// (they repel: never closer than 1 cm edge to edge), and is never more than 60% hidden
// behind the Scene Input / Preview boxes or past the page edge (src/bg-float.js).
// Their home spots are spread evenly over the margins on all four sides, recomputed
// whenever the page or the boxes change size. Behind them, a canvas of very small
// star dots (a few twinkle), never on the boxes.

const ICON_GAP_PX = CM_PX; // 1 cm between icons, edge to edge
const STAR_RGB = "84, 171, 115"; // the background green (--bg-green-rgb in styles.css)
// Runs only while the icons are on screen; still under prefers-reduced-motion.

import { CM_PX, constrainVisible, createFloater, kick, layoutHomes, separate, starField, stepPush, visibleAnchor, wander } from "./bg-float.js";

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

    // Home spots, relative to the layer: laid out evenly over the margins (layoutHomes),
    // redone when the layout changes. Until then, the CSS left/top from index.html.
    let homes = null;
    let layoutKey = "";
    const boxesNow = () => [...document.querySelectorAll("#input-view .iv-pane")].map(el => el.getBoundingClientRect());
    function relayout(box, boxes) {
        // the icons' own sizes are part of the key: on the first frames (layer just shown) they can
        // still measure 0, and a layout made then would put homes anywhere, even under the boxes
        const sizes = icons.map(icon => icon.el.offsetWidth);
        const key = [box.width, box.height, ...sizes, ...boxes.flatMap(b => [b.left - box.left, b.top - box.top, b.width, b.height])]
            .map(v => Math.round(v)).join(",");
        if (key === layoutKey || !boxes.length || !boxes.every(b => b.width > 0) || sizes.some(v => v <= 0)) {
            return;
        }
        layoutKey = key;
        const page = { left: 0, top: 0, right: box.width, bottom: box.height };
        const rel = boxes.map(b => ({ left: b.left - box.left, top: b.top - box.top, right: b.right - box.left, bottom: b.bottom - box.top }));
        homes = layoutHomes(sizes, page, rel, { gap: ICON_GAP_PX });
        // the element's own CSS anchor moves to its home: the per-frame transform is relative to it
        icons.forEach((icon, i) => {
            icon.el.style.left = `${homes[i].x.toFixed(1)}px`;
            icon.el.style.top = `${homes[i].y.toFixed(1)}px`;
        });
        drawStars.stars = starField(page, rel);
        sizeStarCanvas(box);
    }
    const homeOf = (icon, box) => {
        const h = homes?.[icons.indexOf(icon)];
        return h
            ? { x: box.left + h.x, y: box.top + h.y }
            // before the first layout: the CSS left/top (percentages) from index.html
            : { x: box.left + (parseFloat(icon.el.style.left) / 100) * box.width, y: box.top + (parseFloat(icon.el.style.top) / 100) * box.height };
    };

    // Star dots: one canvas under the icons, redrawn ~12 times a second for the twinkle.
    const starCanvas = document.createElement("canvas");
    starCanvas.className = "bgi-stars";
    starCanvas.setAttribute("aria-hidden", "true");
    layer.prepend(starCanvas);
    const ctx = starCanvas.getContext("2d");
    function sizeStarCanvas(box) {
        const dpr = window.devicePixelRatio || 1;
        starCanvas.width = Math.round(box.width * dpr);
        starCanvas.height = Math.round(box.height * dpr);
        starCanvas.style.width = `${box.width}px`;
        starCanvas.style.height = `${box.height}px`;
        drawStars.lastT = -1;
    }
    function drawStars(t) {
        if (!drawStars.stars || (drawStars.lastT >= 0 && t - drawStars.lastT < 0.08)) {
            return;
        }
        drawStars.lastT = t;
        const dpr = window.devicePixelRatio || 1;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, starCanvas.width, starCanvas.height);
        for (const star of drawStars.stars) {
            const tw = star.tw == null || reduced.matches ? 1 : 0.55 + 0.45 * Math.sin(t * 1.3 + star.tw);
            ctx.fillStyle = `rgba(${STAR_RGB}, ${(star.a * tw).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

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
        relayout(box, boxesNow());
        drawStars(t);
        const bodies = icons.map(icon => {
            const w = reduced.matches ? { x: 0, y: 0, rot: 0 } : wander(icon.floater, t);
            const home = homeOf(icon, box);
            const at = { x: home.x + w.x + icon.floater.push.x, y: home.y + w.y + icon.floater.push.y };
            const p = reduced.matches ? icon.floater.push : stepPush(icon.floater, dt, at, pointer);
            // footprint: a circle around the icon's (unrotated) square; the artwork sits inside it
            const r = icon.el.offsetWidth / 2;
            return { icon, home, w, x: home.x + w.x + p.x, y: home.y + w.y + p.y, r, mass: r * r, push: p };
        });
        // Icons repel each other (no overlap) and stay at least 40% visible: the part behind the
        // boxes or past the page edge is <= 60%. Two rounds let the two rules settle together;
        // visibility is applied last, so it always holds.
        const page = { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
        const boxes = boxesNow();
        bodies.forEach(b => {
            b.size = b.icon.el.offsetWidth;
            b.anchor = visibleAnchor(b.home.x, b.home.y, b.size, page, boxes);
        });
        for (let round = 0; round < 2; round++) {
            separate(bodies, { gap: ICON_GAP_PX, iterations: 8 });
            bodies.forEach(b => {
                const c = constrainVisible(b.x, b.y, b.anchor, b.size, page, boxes);
                if (!c.moved) {
                    return;
                }
                const dx = c.x - b.x;
                const dy = c.y - b.y;
                b.x = c.x;
                b.y = c.y;
                b.push.x += dx;
                b.push.y += dy;
                // stop the part of its motion that heads back behind the box / off the page
                const d = Math.hypot(dx, dy) || 1;
                const vn = (b.push.vx * dx + b.push.vy * dy) / d;
                if (vn < 0) {
                    b.push.vx -= (dx / d) * vn;
                    b.push.vy -= (dy / d) * vn;
                }
            });
        }
        bodies.forEach(({ icon, home, w, x, y }) => {
            icon.el.style.transform = `translate(-50%, -50%) translate(${(x - home.x).toFixed(1)}px, ${(y - home.y).toFixed(1)}px) `
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
