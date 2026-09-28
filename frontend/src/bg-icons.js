// Drives the input page's background icons (index.html .input-bg-icons) and the star dots
// behind them. All the motion maths is in src/bg-float.js (unit-tested); this file measures the
// page, runs the frame loop and draws.
//
//   - Each icon cruises at its own steady speed around its home spot, steering back when it
//     strays, so it never slows to a halt (the old fixed wander path parked icons against their
//     limits). The cursor pushes icons and clicks kick them; they ease back to cruise speed.
//   - They bounce off each other (never closer than 1 cm edge to edge) and off their limits:
//     the boxes, the page edges and a fixed line at the collapsed sidebar's width. At most 30% of
//     an icon may be hidden behind a box or past an edge.
//   - The big satellite is locked in the top-left quadrant (II) of the floating space and the
//     earth + satellite in the bottom-right (IV); the rest are spread evenly on all four sides.
//     Two icons of the same artwork keep at least 5 cm apart, so they're never beside each other.
//   - Home spots are recomputed whenever the page, the boxes or the icon sizes change. The very
//     small icons are 1.4x on the left side.
// Runs only while the icons are on screen; still under prefers-reduced-motion.

import {
    CM_PX, constrainVisible, createCruiser, clampToZone, hiddenFraction, kickCruiser, layoutHomes,
    MAX_HIDDEN, reflect, separate, starField, steerCruiser, visibleAnchor,
} from "./bg-float.js";

const ICON_GAP_PX = CM_PX; // 1 cm between icons, edge to edge
const SAME_ARTWORK_GAP_PX = 5 * CM_PX; // two icons of the same artwork are never beside each other
const LEFT_SMALL_SCALE = 1.4; // the very small icons are this much bigger on the left side
// Star colour: the layer's --bg-green-rgb (green in dark mode, orange in bright mode), read on
// each redraw so a theme switch recolours them; bright mode makes them a little stronger.
const STAR_LIGHT_BOOST = 1.3;
const WALL_RESTITUTION = 0.85;
// Cruise speeds (px/s). With the bounces they average ~11 px/s on screen, the 1.4x drift asked
// for on 2026-09-29; the big icons a little slower.
const CRUISE = { big: 8, normal: 9.5, small: 11 };
// Quadrant locks (by artwork): II = top-left, IV = bottom-right of the floating space.
const QUADRANT = { satellite: 2, earthsat: 4 };

// The page's left limit is a fixed line at the collapsed sidebar's width (from the window's left
// edge), whatever the sidebar's actual width: icons may hide at most 30% past it. When the
// sidebar is expanded it can cover some of them; that's accepted, and the line doesn't move.
function collapsedSidebarWidth() {
    const bar = document.getElementById("app-sidebar");
    const v = bar ? parseFloat(getComputedStyle(bar).getPropertyValue("--sb-rail-w")) : NaN;
    return Number.isFinite(v) ? v : 56;
}

function sizeClass(el) {
    return el.classList.contains("bgi-big") ? "big" : el.classList.contains("bgi-small-normal") ? "small" : "normal";
}

export function startBackgroundIcons(layer, page) {
    if (!layer || !page) {
        return;
    }
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const leftLimit = collapsedSidebarWidth(); // viewport x, fixed
    const icons = [...layer.querySelectorAll(".bgi")].map((el, i) => {
        const size = sizeClass(el);
        return {
            el,
            size,
            type: el.querySelector("use")?.getAttribute("href")?.replace("#bgi-", "") ?? null,
            small: size === "small",
            scale: 1, // LEFT_SMALL_SCALE when a very small icon's home is on the left
            rot: Number(el.dataset.rot ?? 0),
            cruiser: createCruiser(i + 1, { radius: Number(el.dataset.radiusCm ?? 5) * CM_PX, speed: CRUISE[size] }),
            pos: null, // layer-relative centre
            wallContact: false,
        };
    });
    let pointer = null;
    let raf = 0;
    let last = 0;
    const contacts = { prev: new Set(), next: new Set() };
    let bounces = 0; // exposed as data-bounces on the layer (for checks)

    // ---- layout: home spots (layer-relative), redone when the layout changes
    let homes = null;
    let zones = null; // per icon: a layer-relative rect or null
    let layoutKey = "";
    const boxesNow = () => [...document.querySelectorAll("#input-view .iv-pane")].map(el => el.getBoundingClientRect());
    function relayout(box, boxes) {
        // base (unscaled) sizes are part of the key: on the first frames they can still measure 0,
        // and scaling an icon must never re-trigger a layout
        const sizes = icons.map(icon => icon.el.offsetWidth / icon.scale);
        const key = [box.left, box.width, box.height, ...sizes, ...boxes.flatMap(b => [b.left - box.left, b.top - box.top, b.width, b.height])]
            .map(v => Math.round(v)).join(",");
        if (key === layoutKey || !boxes.length || !boxes.every(b => b.width > 0) || sizes.some(v => v <= 0)) {
            return;
        }
        layoutKey = key;
        const layerPage = { left: 0, top: 0, right: box.width, bottom: box.height };
        // Homes are planned in the visible page (right of the sidebar, whatever its width), so the
        // spread you see stays even; the fixed left line only limits how far they may float.
        const planPage = { ...layerPage, left: Math.max(leftLimit - box.left, 0) };
        const rel = boxes.map(b => ({ left: b.left - box.left, top: b.top - box.top, right: b.right - box.left, bottom: b.bottom - box.top }));
        const midX = (planPage.left + planPage.right) / 2;
        const midY = (planPage.top + planPage.bottom) / 2;
        const quadrant = q => (q === 2
            ? { left: Math.min(leftLimit - box.left, 0), top: 0, right: midX, bottom: midY }
            : { left: midX, top: midY, right: box.width, bottom: box.height });
        zones = icons.map(icon => (QUADRANT[icon.type] ? quadrant(QUADRANT[icon.type]) : null));
        homes = layoutHomes(sizes, planPage, rel, {
            gap: ICON_GAP_PX,
            sameTypeGap: SAME_ARTWORK_GAP_PX,
            scaleFor: (i, side) => (icons[i].small && side === "left" ? LEFT_SMALL_SCALE : 1),
            zoneFor: i => zones[i],
            typeOf: i => icons[i].type,
        });
        icons.forEach((icon, i) => {
            icon.scale = homes[i].size / sizes[i];
            icon.el.style.setProperty("--k", icon.scale.toFixed(3));
            // the element's CSS anchor is its home; the per-frame transform is relative to it
            icon.el.style.left = `${homes[i].x.toFixed(1)}px`;
            icon.el.style.top = `${homes[i].y.toFixed(1)}px`;
            icon.pos = icon.pos ?? { x: homes[i].x, y: homes[i].y };
        });
        drawStars.stars = starField(layerPage, rel);
        sizeStarCanvas(box);
    }

    // ---- star dots: one canvas under the icons, redrawn ~12 times a second for the twinkle
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
        const rgb = getComputedStyle(layer).getPropertyValue("--bg-green-rgb").trim() || "84, 171, 115";
        const boost = document.documentElement.dataset.theme === "light" ? STAR_LIGHT_BOOST : 1;
        const dpr = window.devicePixelRatio || 1;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, starCanvas.width, starCanvas.height);
        for (const star of drawStars.stars) {
            const tw = star.tw == null || reduced.matches ? 1 : 0.55 + 0.45 * Math.sin(t * 1.3 + star.tw);
            ctx.fillStyle = `rgba(${rgb}, ${Math.min(1, star.a * tw * boost).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ---- pointer
    page.addEventListener("pointermove", event => {
        pointer = { x: event.clientX, y: event.clientY };
    }, { passive: true });
    page.addEventListener("pointerleave", () => {
        pointer = null;
    });
    page.addEventListener("pointerdown", event => {
        if (!visible() || reduced.matches) {
            return;
        }
        const box = layer.getBoundingClientRect();
        icons.forEach(icon => {
            if (icon.pos) {
                kickCruiser(icon.cruiser, { x: box.left + icon.pos.x, y: box.top + icon.pos.y }, { x: event.clientX, y: event.clientY });
            }
        });
    }, { passive: true });

    function visible() {
        return layer.offsetParent !== null && document.visibilityState === "visible";
    }

    // ---- frame
    function frame(now) {
        raf = 0;
        if (!visible()) {
            return;
        }
        const t = now / 1000;
        const dt = Math.min(0.05, last ? (now - last) / 1000 : 1 / 60);
        last = now;
        const box = layer.getBoundingClientRect();
        const boxesVp = boxesNow();
        relayout(box, boxesVp);
        drawStars(t);
        if (!homes) {
            schedule();
            return;
        }
        // everything below in layer-relative px
        const pageRel = { left: leftLimit - box.left, top: 0, right: box.width, bottom: box.height };
        const boxes = boxesVp.map(b => ({ left: b.left - box.left, top: b.top - box.top, right: b.right - box.left, bottom: b.bottom - box.top }));
        const pointerRel = pointer ? { x: pointer.x - box.left, y: pointer.y - box.top } : null;

        // 1. steer and move
        const bodies = icons.map((icon, i) => {
            const home = homes[i];
            const c = icon.cruiser;
            let sway = 0;
            if (reduced.matches) {
                icon.pos = { x: home.x, y: home.y };
                c.vx = 0;
                c.vy = 0;
            } else {
                sway = steerCruiser(c, dt, t, { x: icon.pos.x - home.x, y: icon.pos.y - home.y },
                    pointerRel ? { x: pointerRel.x - icon.pos.x, y: pointerRel.y - icon.pos.y } : null);
                icon.pos = { x: icon.pos.x + c.vx * dt, y: icon.pos.y + c.vy * dt };
            }
            const size = icon.el.offsetWidth;
            const r = size / 2;
            return {
                icon, i, home, sway, size, r, mass: r * r, x: icon.pos.x, y: icon.pos.y,
                // separate() moves x/y and push.x/y together, and bounces push.vx/vy
                push: { x: 0, y: 0, vx: c.vx, vy: c.vy }, vx: c.vx, vy: c.vy,
                anchor: visibleAnchor(home.x, home.y, size, pageRel, boxes),
            };
        });

        // 2. limits: other icons (1 cm apart), the quadrant locks, and <= 30% hidden. Five rounds let
        //    them settle together; visibility is applied last, so it always holds.
        const bounce = !reduced.matches;
        contacts.next = new Set();
        contacts.onBounce = () => {
            bounces += 1;
        };
        const atWall = new Set();
        const wallHit = (b, nx, ny) => {
            atWall.add(b.i);
            if (reflect(b.push, { x: nx, y: ny }, WALL_RESTITUTION) && !b.icon.wallContact) {
                b.icon.wallContact = true;
                bounces += 1;
            }
        };
        for (let round = 0; round < 5; round++) {
            separate(bodies, {
                gap: ICON_GAP_PX, iterations: 8, contacts: bounce && round === 0 ? contacts : null,
                gapFor: (a, b) => (a.icon.type === b.icon.type ? SAME_ARTWORK_GAP_PX : ICON_GAP_PX),
            });
            bodies.forEach(b => {
                const zone = zones[b.i];
                if (zone) {
                    const z = clampToZone(b.x, b.y, zone);
                    if (z.nx || z.ny) {
                        b.x = z.x;
                        b.y = z.y;
                        if (z.nx) {
                            wallHit(b, z.nx, 0);
                        }
                        if (z.ny) {
                            wallHit(b, 0, z.ny);
                        }
                    }
                }
                const c = constrainVisible(b.x, b.y, b.anchor, b.size, pageRel, boxes);
                if (c.moved) {
                    const dx = c.x - b.x;
                    const dy = c.y - b.y;
                    const d = Math.hypot(dx, dy) || 1;
                    b.x = c.x;
                    b.y = c.y;
                    wallHit(b, dx / d, dy / d);
                }
            });
        }
        contacts.prev = contacts.next;

        // 3. write back, end wall contacts that are clearly over (hysteresis), draw
        bodies.forEach(b => {
            const icon = b.icon;
            icon.pos = { x: b.x, y: b.y };
            icon.cruiser.vx = b.push.vx;
            icon.cruiser.vy = b.push.vy;
            if (icon.wallContact && !atWall.has(b.i)
                && hiddenFraction(b.x, b.y, b.size, pageRel, boxes) < MAX_HIDDEN - 0.05) {
                icon.wallContact = false;
            }
            icon.el.style.transform = `translate(-50%, -50%) translate(${(b.x - b.home.x).toFixed(1)}px, ${(b.y - b.home.y).toFixed(1)}px) `
                + `rotate(${(icon.rot + b.sway).toFixed(2)}deg)`;
        });
        layer.dataset.bounces = String(bounces);
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
