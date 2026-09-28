import test from "node:test";
import assert from "node:assert/strict";
import { createFloater, wander, stepPush, kick, CM_PX } from "../src/bg-float.js";

test("the wander path stays inside its circle", () => {
    const f = createFloater(3, 5 * CM_PX);
    for (let t = 0; t < 600; t += 0.37) {
        const w = wander(f, t);
        assert.ok(Math.hypot(w.x, w.y) <= f.radius + 1e-9);
        assert.ok(Math.abs(w.rot) <= 7);
    }
});

test("the wander path actually covers its neighbourhood (not a tiny jitter)", () => {
    const f = createFloater(1, 4 * CM_PX);
    let far = 0;
    for (let t = 0; t < 300; t += 0.5) {
        far = Math.max(far, Math.hypot(wander(f, t).x, wander(f, t).y));
    }
    assert.ok(far > 0.6 * f.radius, `max reach ${far}`);
});

test("a nearby cursor pushes the icon away, and it springs back when the cursor leaves", () => {
    const f = createFloater(2, 150);
    const at = { x: 500, y: 300 };
    for (let i = 0; i < 60; i++) {
        stepPush(f, 1 / 60, { x: at.x + f.push.x, y: at.y + f.push.y }, { x: 450, y: 300 });
    }
    assert.ok(f.push.x > 10, `pushed right by ${f.push.x}`);
    for (let i = 0; i < 60 * 8; i++) {
        stepPush(f, 1 / 60, at, null);
    }
    assert.ok(Math.hypot(f.push.x, f.push.y) < 2, "settles back");
});

test("a click kicks nearby icons outward and leaves far ones alone", () => {
    const near = createFloater(4, 150);
    const far = createFloater(5, 150);
    kick(near, { x: 100, y: 0 }, { x: 0, y: 0 });
    kick(far, { x: 2000, y: 0 }, { x: 0, y: 0 });
    assert.ok(near.push.vx > 0 && Math.abs(near.push.vy) < 1e-9);
    assert.equal(far.push.vx, 0);
});

test("the push never exceeds its limit", () => {
    const f = createFloater(6, 150);
    for (let i = 0; i < 30; i++) {
        kick(f, { x: 1, y: 0 }, { x: 0, y: 0 });
        stepPush(f, 1 / 60, { x: 1, y: 0 }, { x: 0, y: 0 });
    }
    assert.ok(Math.hypot(f.push.x, f.push.y) <= 170 + 1e-6);
});

import { separate } from "../src/bg-float.js";

const body = (x, y, r, mass = r * r) => ({ x, y, r, mass, push: { x: 0, y: 0, vx: 0, vy: 0 } });

test("overlapping icons are pushed apart until their footprints don't touch", () => {
    const bodies = [body(100, 100, 35), body(120, 105, 35), body(110, 130, 26)];
    const left = separate(bodies, { gap: 8, iterations: 8 });
    assert.equal(left, 0);
    for (let i = 0; i < bodies.length; i++) {
        for (let j = i + 1; j < bodies.length; j++) {
            const d = Math.hypot(bodies[j].x - bodies[i].x, bodies[j].y - bodies[i].y);
            assert.ok(d >= bodies[i].r + bodies[j].r + 8 - 0.5, `pair ${i},${j} at ${d}`);
        }
    }
});

test("icons that don't touch are left alone; the lighter one moves more", () => {
    const far = [body(0, 0, 30), body(500, 0, 30)];
    separate(far);
    assert.deepEqual([far[0].x, far[1].x], [0, 500]);
    const pair = [body(0, 0, 112), body(130, 0, 26)];
    separate(pair, { gap: 8 });
    assert.ok(Math.abs(pair[1].push.x) > 5 * Math.abs(pair[0].push.x), "the small icon gives way");
});

test("icons exactly on top of each other still separate", () => {
    const pair = [body(50, 50, 20), body(50, 50, 20)];
    assert.equal(separate(pair, { gap: 4, iterations: 4 }), 0);
});

import { hiddenFraction, constrainVisible, visibleAnchor, MAX_HIDDEN } from "../src/bg-float.js";

const PAGE = { left: 0, top: 0, right: 1000, bottom: 700 };
const BOXES = [{ left: 150, top: 90, right: 480, bottom: 610 }, { left: 520, top: 90, right: 850, bottom: 610 }];

test("hidden fraction: fully visible, half behind a box, fully off the page", () => {
    assert.equal(hiddenFraction(80, 300, 60, PAGE, BOXES), 0);
    assert.ok(Math.abs(hiddenFraction(150, 300, 60, PAGE, BOXES) - 0.5) < 0.07);
    assert.equal(hiddenFraction(-100, 300, 60, PAGE, BOXES), 1);
    assert.equal(hiddenFraction(300, 300, 60, PAGE, BOXES), 1);
});

test("an icon pushed behind a box or off the page is held at <= 30% hidden", () => {
    const anchor = { x: 80, y: 300 };
    for (const [x, y] of [[300, 300], [-200, 300], [80, -150], [200, 740]]) {
        const p = constrainVisible(x, y, anchor, 70, PAGE, BOXES);
        assert.ok(p.moved);
        assert.ok(hiddenFraction(p.x, p.y, 70, PAGE, BOXES) <= MAX_HIDDEN + 1e-9, `${x},${y}`);
        // it stops at the limit, not back at the anchor
        assert.ok(hiddenFraction(p.x, p.y, 70, PAGE, BOXES) > MAX_HIDDEN - 0.15, `${x},${y} pulled too far back`);
    }
});

test("an allowed position is left exactly where it is", () => {
    const p = constrainVisible(100, 300, { x: 80, y: 300 }, 70, PAGE, BOXES);
    assert.deepEqual(p, { x: 100, y: 300, moved: false });
});

test("a home that is mostly hidden gets a visible anchor nearby", () => {
    const a = visibleAnchor(320, 300, 70, PAGE, BOXES); // deep inside the left box
    assert.ok(hiddenFraction(a.x, a.y, 70, PAGE, BOXES) <= MAX_HIDDEN);
    assert.ok(Math.hypot(a.x - 320, a.y - 300) < 300);
});

import { layoutHomes, marginStrips, starField } from "../src/bg-float.js";

const PG = { left: 0, top: 0, right: 1214, bottom: 713 };
const BX = [{ left: 152, top: 89, right: 594, bottom: 624 }, { left: 620, top: 89, right: 1062, bottom: 624 }];
const SIZES = [224, 224, 70, 70, 70, 70, 70, 70, 52, 52, 52, 52, 52, 52, 52, 52, 52];

test("home layout: every icon visible enough, and at least 1 cm (38 px) apart edge to edge", () => {
    const homes = layoutHomes(SIZES, PG, BX, { gap: 38 });
    // small/normal icons: strict 15%; the big satellites fit only in corners, within the 30% limit
    homes.forEach((h, i) => assert.ok(hiddenFraction(h.x, h.y, SIZES[i], PG, BX) <= (SIZES[i] > 100 ? MAX_HIDDEN : 0.15),
        `icon ${i} hidden ${hiddenFraction(h.x, h.y, SIZES[i], PG, BX).toFixed(2)}`));
    let minGap = Infinity;
    for (let i = 0; i < homes.length; i++) {
        for (let j = i + 1; j < homes.length; j++) {
            minGap = Math.min(minGap, Math.hypot(homes[i].x - homes[j].x, homes[i].y - homes[j].y) - (SIZES[i] + SIZES[j]) / 2);
        }
    }
    assert.ok(minGap >= 0, `closest pair overlaps (${minGap})`);
});

test("home layout: icons spread over all four sides roughly by each side's area", () => {
    const homes = layoutHomes(SIZES, PG, BX);
    const strips = marginStrips(PG, BX);
    const side = h => (h.x < strips.left.right ? "left" : h.x > strips.right.left ? "right" : h.y < strips.top.bottom ? "top" : "bottom");
    const counts = { top: 0, bottom: 0, left: 0, right: 0 };
    homes.forEach(h => { counts[side(h)] += 1; });
    const area = r => (r.right - r.left) * (r.bottom - r.top);
    const total = Object.values(strips).reduce((a, r) => a + area(r), 0);
    Object.entries(counts).forEach(([k, n]) => {
        const expected = (area(strips[k]) / total) * SIZES.length;
        assert.ok(Math.abs(n - expected) <= 1, `${k}: ${n} vs ${expected.toFixed(1)} (${JSON.stringify(counts)})`);
    });
});

test("star field: tiny, evenly spread, none on the boxes", () => {
    const stars = starField(PG, BX);
    assert.ok(stars.length > 60 && stars.length < 250, `${stars.length} stars`);
    stars.forEach(s => {
        assert.ok(s.r <= 1.25);
        assert.ok(!BX.some(b => s.x > b.left && s.x < b.right && s.y > b.top && s.y < b.bottom));
    });
    // no big empty holes: every 150x150 px cell of the margins has a star
    const strips = Object.values(marginStrips(PG, BX));
    let empty = 0; let cells = 0;
    strips.forEach(r => {
        for (let y = r.top; y + 150 <= r.bottom; y += 150) {
            for (let x = r.left; x + 150 <= r.right; x += 150) {
                cells += 1;
                if (!stars.some(s => s.x >= x && s.x < x + 150 && s.y >= y && s.y < y + 150)) empty += 1;
            }
        }
    });
    assert.ok(empty <= cells * 0.1, `${empty}/${cells} empty cells`);
});

test("home layout: very small icons that land on the left are 1.4x, and spacing uses that size", () => {
    const small = i => SIZES[i] === 52;
    const homes = layoutHomes(SIZES, PG, BX, { gap: 38, scaleFor: (i, side) => (small(i) && side === "left" ? 1.4 : 1) });
    const leftSmall = homes.filter((h, i) => small(i) && h.side === "left");
    assert.ok(leftSmall.length > 0 && leftSmall.every(h => Math.abs(h.size - 72.8) < 1e-9));
    assert.ok(homes.every((h, i) => h.side === "left" || h.size === SIZES[i]));
    for (let i = 0; i < homes.length; i++) {
        for (let j = i + 1; j < homes.length; j++) {
            const d = Math.hypot(homes[i].x - homes[j].x, homes[i].y - homes[j].y) - (homes[i].size + homes[j].size) / 2;
            assert.ok(d >= 0, `pair ${i},${j} overlaps by ${-d}`);
        }
    }
});

test("stars: 30% denser than before, and most of them small", () => {
    const now = starField(PG, BX);
    const before = starField(PG, BX, { cell: 52 });
    const ratio = now.length / before.length;
    assert.ok(ratio > 1.15 && ratio < 1.45, `density ratio ${ratio.toFixed(2)}`);
    const small = now.filter(s => s.r < 0.7).length / now.length;
    assert.ok(small > 0.6, `small share ${small.toFixed(2)}`);
    assert.ok(now.some(s => s.r > 1.0), "some big stars remain");
});

import { bounceImpulse, MIN_BOUNCE, MAX_BOUNCE } from "../src/bg-float.js";

test("bounce: a fresh contact sends the pair apart; resting contact doesn't bounce again", () => {
    const a = body(100, 100, 30);
    const b = body(160, 100, 30);          // 60 apart, needs 60 + 10 gap: touching
    a.vx = 40; a.vy = 0; b.vx = -40; b.vy = 0; // closing at 80 px/s
    const contacts = { prev: new Set(), next: new Set() };
    separate([a, b], { gap: 10, iterations: 2, contacts });
    assert.ok(a.push.vx < 0 && b.push.vx > 0, "they move apart");
    const kick = b.push.vx - a.push.vx;
    assert.ok(Math.abs(kick - bounceImpulse(80)) < 1e-6, `relative kick ${kick}`);
    // next frame, still touching: no second bounce
    contacts.prev = contacts.next; contacts.next = new Set();
    const before = b.push.vx - a.push.vx;
    a.x = 100; b.x = 169; // just inside the gap again
    separate([a, b], { gap: 10, iterations: 2, contacts });
    assert.ok(Math.abs((b.push.vx - a.push.vx) - before) < 1e-6, "no repeat bounce while in contact");
});

test("bounce impulse: at least the minimum, reflected with restitution, capped", () => {
    assert.equal(bounceImpulse(0), MIN_BOUNCE);
    assert.equal(bounceImpulse(-50), MIN_BOUNCE);
    assert.ok(Math.abs(bounceImpulse(100) - 175) < 1e-9);
    assert.equal(bounceImpulse(10000), MAX_BOUNCE);
});
