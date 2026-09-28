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
const HOVER_FORCE = 2600;     // px/s² at the cursor, falling off to 0 at HOVER_RANGE_PX
const CLICK_RANGE_PX = 520;
const CLICK_IMPULSE = 900;    // px/s at the click point, falling off with distance
const SPRING = 3.2;           // 1/s², pulls the pointer offset back to 0
const DAMPING = 2.4;          // 1/s
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
        // angular speeds (rad/s): one loop every ~35–80 s
        w: [0.08 + r(1) * 0.1, 0.05 + r(2) * 0.07, 0.07 + r(3) * 0.09, 0.04 + r(4) * 0.06, 0.05 + r(5) * 0.05],
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
