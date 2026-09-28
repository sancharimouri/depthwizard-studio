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

// Wander speed relative to the original (1.0). 2026-09-29: halved (0.5), then the user asked for
// 1.4x the resulting on-screen drift. Measured with the 1 cm spacing and the 60%-visibility limit in
// force (which hold icons back), that needs about 0.85: mean drift 7.9 -> ~11 px/s.
export const WANDER_SPEED = 0.85;

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
        // angular speeds (rad/s), scaled by WANDER_SPEED
        w: [0.08 + r(1) * 0.1, 0.05 + r(2) * 0.07, 0.07 + r(3) * 0.09, 0.04 + r(4) * 0.06, 0.05 + r(5) * 0.05]
            .map(v => v * WANDER_SPEED),
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
//
// Bounce (optional): pass `contacts` = { prev: Set, next: Set } (pair keys "i:j") to have a pair
// that has just come into contact bounce apart. Its closing speed (from each body's total
// velocity vx/vy, default its push velocity) is reflected with `restitution`, at least `minBounce`
// px/s and at most MAX_BOUNCE, split by mass. A pair within CONTACT_SLACK px of touching stays
// "in contact", so resting against each other doesn't bounce every frame.
export const RESTITUTION = 0.75;
export const MIN_BOUNCE = 28;   // px/s: even a slow drift gives a visible bounce
export const MAX_BOUNCE = 220;  // px/s
export const CONTACT_SLACK = 4;  // px

// Impulse size for a contact with closing speed `closing` (px/s, > 0 when approaching).
export function bounceImpulse(closing, restitution = RESTITUTION, minBounce = MIN_BOUNCE) {
    return Math.min(MAX_BOUNCE, Math.max(minBounce, (1 + restitution) * Math.max(0, closing)));
}

export function separate(bodies, { gap = 8, iterations = 4, contacts = null, restitution = RESTITUTION, minBounce = MIN_BOUNCE } = {}) {
    for (let pass = 0; pass < iterations; pass++) {
        for (let i = 0; i < bodies.length; i++) {
            for (let j = i + 1; j < bodies.length; j++) {
                const a = bodies[i];
                const b = bodies[j];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let d = Math.hypot(dx, dy);
                const min = a.r + b.r + gap;
                const key = `${i}:${j}`;
                if (contacts && pass === 0 && d < min + CONTACT_SLACK) {
                    contacts.next.add(key);
                }
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
                // a fresh contact bounces them apart
                if (contacts && pass === 0 && !contacts.prev.has(key)) {
                    const vax = a.vx ?? a.push.vx;
                    const vay = a.vy ?? a.push.vy;
                    const vbx = b.vx ?? b.push.vx;
                    const vby = b.vy ?? b.push.vy;
                    const approach = -((vbx - vax) * nx + (vby - vay) * ny);
                    const J = bounceImpulse(approach, restitution, minBounce);
                    a.push.vx -= nx * J * wa;
                    a.push.vy -= ny * J * wa;
                    b.push.vx += nx * J * wb;
                    b.push.vy += ny * J * wb;
                    if (contacts.onBounce) {
                        contacts.onBounce(i, j, J);
                    }
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
// behind a box (exact rectangle areas of its unrotated square). The page's left edge is a
// fixed line at the collapsed sidebar's width (see src/bg-icons.js).

export const MAX_HIDDEN = 0.3; // was 0.6 until 2026-09-29

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
    if (!(total > 0)) { // a point: hidden if off the page or on a box
        const on = (r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
        return !on(page) || obstacles.some(on) ? 1 : 0;
    }
    const onPage = overlapArea(x0, y0, x1, y1, page);
    let behind = 0;
    for (const o of obstacles) {
        const clipped = {
            left: Math.max(o.left, page.left), top: Math.max(o.top, page.top),
            right: Math.min(o.right, page.right), bottom: Math.min(o.bottom, page.bottom),
        };
        behind += overlapArea(x0, y0, x1, y1, clipped);
    }
    return Math.min(1, (total - onPage + behind) / total);
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

// ---- Even layout: home spots spread evenly over the free space around the boxes ----

// Small deterministic PRNG (mulberry32), so the layout is the same on every load.
export function prng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// The four margin strips around the boxes. The narrow left/right strips run the full page
// height (they own the corners); top/bottom span between them.
export function marginStrips(page, boxes) {
    const top = Math.min(...boxes.map(b => b.top));
    const bottom = Math.max(...boxes.map(b => b.bottom));
    const left = Math.min(...boxes.map(b => b.left));
    const right = Math.max(...boxes.map(b => b.right));
    return {
        top: { left, top: page.top, right, bottom: top },
        bottom: { left, top: bottom, right, bottom: page.bottom },
        left: { left: page.left, top: page.top, right: left, bottom: page.bottom },
        right: { left: right, top: page.top, right: page.right, bottom: page.bottom },
    };
}

// Home centres for icons of the given sizes (px), spread evenly around all four sides:
//   - each side gets a share of the icons in proportion to its area (largest remainder), so
//     the density is the same on every side;
//   - within its side, each icon takes the best of many random candidates: the spot farthest
//     (edge to edge) from every icon placed so far, which spreads them without clumps;
//   - every home is at most `maxHidden` hidden and keeps `gap` px from the others where the
//     space allows.
// Largest icons are placed first. Returns [{x, y}] in the order of `sizes`.
//
// `scaleFor(i, side)` (optional) makes icon i that much bigger if it lands on `side` (the very
// small icons are 1.4x on the left); spacing and visibility use the scaled size. Each result
// also carries its `side` and final `size`.
export function layoutHomes(sizes, page, boxes, { gap = 38, maxHidden = 0.15, seed = 7, candidates = 300, scaleFor = () => 1 } = {}) {
    const rand = prng(seed);
    const strips = marginStrips(page, boxes);
    const sides = Object.keys(strips);
    const area = r => Math.max(0, r.right - r.left) * Math.max(0, r.bottom - r.top);
    const totalArea = sides.reduce((a, k) => a + area(strips[k]), 0) || 1;
    // quotas by area, largest remainder
    const exact = Object.fromEntries(sides.map(k => [k, (area(strips[k]) / totalArea) * sizes.length]));
    const quota = Object.fromEntries(sides.map(k => [k, Math.floor(exact[k])]));
    let left = sizes.length - sides.reduce((a, k) => a + quota[k], 0);
    sides.slice().sort((a, b) => (exact[b] - quota[b]) - (exact[a] - quota[a])).forEach(k => {
        if (left > 0) {
            quota[k] += 1;
            left -= 1;
        }
    });
    const used = Object.fromEntries(sides.map(k => [k, 0]));

    const order = sizes.map((size, i) => ({ size, i })).sort((a, b) => b.size - a.size);
    const placed = [];
    const out = new Array(sizes.length);
    const search = (side, size, limit) => {
        const r = strips[side];
        let best = null;
        let bestScore = -Infinity;
        for (let k = 0; k < candidates; k++) {
            const c = { x: r.left + rand() * (r.right - r.left), y: r.top + rand() * (r.bottom - r.top) };
            if (hiddenFraction(c.x, c.y, size, page, boxes) > limit) {
                continue;
            }
            let score = Infinity;
            for (const p of placed) {
                score = Math.min(score, Math.hypot(c.x - p.x, c.y - p.y) - (size + p.size) / 2);
            }
            if (score > bestScore) {
                bestScore = score;
                best = c;
            }
        }
        return best;
    };
    for (const { size: baseSize, i } of order) {
        // sides in order of need: furthest below quota first (ties: the larger side)
        const byNeed = sides.slice().sort((a, b) => ((used[a] - quota[a]) / Math.max(1, quota[a]))
            - ((used[b] - quota[b]) / Math.max(1, quota[b])) || area(strips[b]) - area(strips[a]));
        // A spot that works: the neediest side at the strict limit, then any side at the strict
        // limit, then any side at the runtime limit. (The big satellites only fit in corners.)
        let found = null;
        for (const limit of [maxHidden, MAX_HIDDEN]) {
            for (const side of byNeed) {
                const size = baseSize * scaleFor(i, side);
                const c = search(side, size, limit);
                if (c) {
                    found = { ...c, side, size };
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        if (!found) { // nowhere fits: the least hidden spot of any side
            let least = Infinity;
            for (const side of sides) {
                const r = strips[side];
                const size = baseSize * scaleFor(i, side);
                for (let k = 0; k < candidates; k++) {
                    const c = { x: r.left + rand() * (r.right - r.left), y: r.top + rand() * (r.bottom - r.top) };
                    const h = hiddenFraction(c.x, c.y, size, page, boxes);
                    if (h < least) {
                        least = h;
                        found = { ...c, side, size };
                    }
                }
            }
        }
        used[found.side] += 1;
        placed.push({ x: found.x, y: found.y, size: found.size });
        out[i] = found;
    }
    return out;
}

// ---- Star field: very small dots spread evenly (jittered grid) over the page, never on
// the boxes. Deterministic. Each star: {x, y, r (px), a (alpha), tw (twinkle phase or null)}.
export function starField(page, boxes, { cell = 46, seed = 11 } = {}) {
    const rand = prng(seed);
    const stars = [];
    for (let y = page.top; y < page.bottom; y += cell) {
        for (let x = page.left; x < page.right; x += cell) {
            if (rand() < 0.22) { // leave some cells empty, so it doesn't read as a grid
                rand(); rand(); rand(); rand(); rand();
                continue;
            }
            const sx = x + rand() * cell;
            const sy = y + rand() * cell;
            const size = rand();
            const alpha = rand();
            const twinkle = rand();
            const pad = 4;
            if (sx > page.right || sy > page.bottom
                || boxes.some(b => sx > b.left - pad && sx < b.right + pad && sy > b.top - pad && sy < b.bottom + pad)) {
                continue;
            }
            stars.push({
                x: sx,
                y: sy,
                // cubed: the extra 30% density (cell 52 -> 46 px) goes mostly to small stars;
                // the bigger ones (up to ~1.2 px) stay, just rarer
                r: 0.45 + size * size * size * 0.75,
                a: 0.4 + alpha * 0.45,
                tw: twinkle < 0.25 ? twinkle * 40 : null, // a quarter of them twinkle
            });
        }
    }
    return stars;
}
