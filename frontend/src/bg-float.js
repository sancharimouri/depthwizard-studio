// Input-page background icons: each floats in a circular neighbourhood of its home
// position and reacts to the pointer. Pure motion maths (no DOM), unit-tested.
//
//   wander   a slow, smooth path inside a circle of `radius` px around home: a sum
//            of two sines per axis with per-icon speeds and phases, scaled so the
//            path never leaves the circle. Rotation sways a few degrees.
//   pointer  a spring-damped offset on top of the wander. The cursor pushes icons
//            away while it's near; a click gives nearby icons an outward kick. The
//            spring brings them back, so they settle into their wander again.

export const CM_PX = 37.8; // CSS px per cm

const HOVER_RANGE_PX = 200;
// Motion was halved on 2026-09-29 (wander speed, hover push, click kick and the spring's pull
// back): the icons drift and scatter at half the old speed, with more inertia.
const HOVER_FORCE = 1300;     // px/s² at the cursor, falling off to 0 at HOVER_RANGE_PX
const CLICK_RANGE_PX = 520;
const CLICK_IMPULSE = 450;    // px/s at the click point, falling off with distance
const SPRING = 0.8;           // 1/s², pulls the pointer offset back to 0 (half the old natural speed)
const DAMPING = 1.2;          // 1/s  (same damping ratio as before)
const MAX_PUSH_PX = 170;

// Deterministic per-icon randomness (the layout looks the same on every load).
function rand(seed) {
    const x = Math.sin(seed * 9301 + 49297) * 233280;
    return x - Math.floor(x);
}

export function createFloater(index, radiusPx) {
    const r = k => rand(index * 17 + k);
    return {
        radius: radiusPx,
        // angular speeds (rad/s): one loop every ~70–160 s (half the old speed)
        w: [0.04 + r(1) * 0.05, 0.025 + r(2) * 0.035, 0.035 + r(3) * 0.045, 0.02 + r(4) * 0.03, 0.025 + r(5) * 0.025],
        p: [r(6), r(7), r(8), r(9), r(10)].map(v => v * Math.PI * 2),
        push: { x: 0, y: 0, vx: 0, vy: 0 },
    };
}

// Wander offset (px) and sway (deg) at time t (s). |offset| <= radius.
export function wander(f, t) {
    const x = 0.62 * Math.sin(t * f.w[0] + f.p[0]) + 0.38 * Math.sin(t * f.w[1] + f.p[1]);
    const y = 0.62 * Math.cos(t * f.w[2] + f.p[2]) + 0.38 * Math.sin(t * f.w[3] + f.p[3]);
    const len = Math.hypot(x, y);
    const k = len > 1 ? 1 / len : 1;
    return { x: x * k * f.radius, y: y * k * f.radius, rot: 7 * Math.sin(t * f.w[4] + f.p[4]) };
}

// One physics step for the pointer offset. `at` is the icon's current centre,
// `pointer` {x, y} or null (not over the page).
export function stepPush(f, dt, at, pointer) {
    const s = f.push;
    let ax = -SPRING * s.x - DAMPING * s.vx;
    let ay = -SPRING * s.y - DAMPING * s.vy;
    if (pointer) {
        const dx = at.x - pointer.x;
        const dy = at.y - pointer.y;
        const d = Math.hypot(dx, dy) || 1;
        if (d < HOVER_RANGE_PX) {
            const a = HOVER_FORCE * (1 - d / HOVER_RANGE_PX);
            ax += (dx / d) * a;
            ay += (dy / d) * a;
        }
    }
    s.vx += ax * dt;
    s.vy += ay * dt;
    s.x += s.vx * dt;
    s.y += s.vy * dt;
    const len = Math.hypot(s.x, s.y);
    if (len > MAX_PUSH_PX) {
        s.x *= MAX_PUSH_PX / len;
        s.y *= MAX_PUSH_PX / len;
        // don't keep pushing outward against the limit
        const out = (s.vx * s.x + s.vy * s.y) / MAX_PUSH_PX;
        if (out > 0) {
            s.vx -= (out * s.x) / MAX_PUSH_PX;
            s.vy -= (out * s.y) / MAX_PUSH_PX;
        }
    }
    return s;
}

// A click at `click` kicks the icon at `at` outward.
export function kick(f, at, click) {
    const dx = at.x - click.x;
    const dy = at.y - click.y;
    const d = Math.hypot(dx, dy) || 1;
    if (d > CLICK_RANGE_PX) {
        return;
    }
    const v = CLICK_IMPULSE * (1 - d / CLICK_RANGE_PX);
    f.push.vx += (dx / d) * v;
    f.push.vy += (dy / d) * v;
}

// Keeps icons from overlapping. `bodies`: [{ x, y, r, mass, push }] where (x, y) is the icon's current
// centre, r the radius of its footprint circle and push its floater's push state. Each overlapping
// pair is moved apart along the line between them until `gap` px separate their circles (the
// lighter icon moves more), and the part of their velocity that closes the gap is cancelled.
// A few relaxation passes resolve chains of contacts. Mutates x, y and push; returns the number
// of pairs that still overlap by more than 0.5 px.
export function separate(bodies, { gap = 8, iterations = 4 } = {}) {
    for (let pass = 0; pass < iterations; pass++) {
        for (let i = 0; i < bodies.length; i++) {
            for (let j = i + 1; j < bodies.length; j++) {
                const a = bodies[i];
                const b = bodies[j];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let d = Math.hypot(dx, dy);
                const min = a.r + b.r + gap;
                if (d >= min) {
                    continue;
                }
                if (d < 1e-6) { // exactly on top of each other: pick a direction
                    dx = 1;
                    dy = 0;
                    d = 1;
                }
                const nx = dx / d;
                const ny = dy / d;
                const overlap = min - d;
                const wa = b.mass / (a.mass + b.mass); // the lighter one moves more
                const wb = a.mass / (a.mass + b.mass);
                a.x -= nx * overlap * wa;
                a.y -= ny * overlap * wa;
                b.x += nx * overlap * wb;
                b.y += ny * overlap * wb;
                a.push.x -= nx * overlap * wa;
                a.push.y -= ny * overlap * wa;
                b.push.x += nx * overlap * wb;
                b.push.y += ny * overlap * wb;
                // cancel the closing part of their relative velocity
                const closing = (b.push.vx - a.push.vx) * nx + (b.push.vy - a.push.vy) * ny;
                if (closing < 0) {
                    a.push.vx += nx * closing * wa;
                    a.push.vy += ny * closing * wa;
                    b.push.vx -= nx * closing * wb;
                    b.push.vy -= ny * closing * wb;
                }
            }
        }
    }
    let left = 0;
    for (let i = 0; i < bodies.length; i++) {
        for (let j = i + 1; j < bodies.length; j++) {
            const a = bodies[i];
            const b = bodies[j];
            if (Math.hypot(b.x - a.x, b.y - a.y) < a.r + b.r + gap - 0.5) {
                left += 1;
            }
        }
    }
    return left;
}

// ---- Visibility: no icon may be more than MAX_HIDDEN of its area off the page or
// behind a box (exact rectangle areas of its unrotated square).

export const MAX_HIDDEN = 0.6;

function overlapArea(ax0, ay0, ax1, ay1, r) {
    const w = Math.min(ax1, r.right) - Math.max(ax0, r.left);
    const h = Math.min(ay1, r.bottom) - Math.max(ay0, r.top);
    return w > 0 && h > 0 ? w * h : 0;
}

// Exact fraction (0–1) of the square of side `size` centred at (x, y) that is outside
// `page` or inside any of `obstacles` (DOMRect-likes; the boxes, which don't overlap each
// other). Rotation is ignored: the square is the icon's layout box.
export function hiddenFraction(x, y, size, page, obstacles) {
    const x0 = x - size / 2;
    const y0 = y - size / 2;
    const x1 = x + size / 2;
    const y1 = y + size / 2;
    const total = size * size;
    const onPage = overlapArea(x0, y0, x1, y1, page);
    let behind = 0;
    for (const o of obstacles) {
        const clipped = {
            left: Math.max(o.left, page.left), top: Math.max(o.top, page.top),
            right: Math.min(o.right, page.right), bottom: Math.min(o.bottom, page.bottom),
        };
        behind += overlapArea(x0, y0, x1, y1, clipped);
    }
    return total > 0 ? Math.min(1, (total - onPage + behind) / total) : 0;
}

// A visible spot near (x, y): (x, y) itself when it's allowed; otherwise the nearest point
// found on rings around it. Used to give every icon a valid anchor.
export function visibleAnchor(x, y, size, page, obstacles, limit = MAX_HIDDEN) {
    if (hiddenFraction(x, y, size, page, obstacles) <= limit) {
        return { x, y };
    }
    for (let ring = 1; ring <= 60; ring++) {
        const d = ring * 8;
        for (let k = 0; k < 16; k++) {
            const a = (k / 16) * Math.PI * 2;
            const cx = x + Math.cos(a) * d;
            const cy = y + Math.sin(a) * d;
            if (hiddenFraction(cx, cy, size, page, obstacles) <= limit) {
                return { x: cx, y: cy };
            }
        }
    }
    return { x, y };
}

// Keeps (x, y) where at most `limit` of the icon is hidden: if it isn't, moves it back
// along the line toward `anchor` (a visible spot) to the last allowed point (binary search).
export function constrainVisible(x, y, anchor, size, page, obstacles, limit = MAX_HIDDEN) {
    if (hiddenFraction(x, y, size, page, obstacles) <= limit) {
        return { x, y, moved: false };
    }
    let lo = 0; // fraction of the way from the anchor: 0 = anchor (allowed)
    let hi = 1; // 1 = the requested spot (not allowed)
    for (let step = 0; step < 12; step++) {
        const mid = (lo + hi) / 2;
        const mx = anchor.x + (x - anchor.x) * mid;
        const my = anchor.y + (y - anchor.y) * mid;
        if (hiddenFraction(mx, my, size, page, obstacles) <= limit) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return { x: anchor.x + (x - anchor.x) * lo, y: anchor.y + (y - anchor.y) * lo, moved: true };
}
