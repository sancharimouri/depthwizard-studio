import test from "node:test";
import assert from "node:assert/strict";
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

test("stars: half the previous density, with the big ones cut most (averaged over 100 layouts)", () => {
    let now = 0; let nowBig = 0; let before = 0; let beforeBig = 0;
    for (let seed = 1; seed <= 100; seed++) {
        const a = starField(PG, BX, { seed });
        const b = starField(PG, BX, { seed, cell: 46, bigShare: 0.15 }); // the old density and big share
        now += a.length; before += b.length;
        nowBig += a.filter(s => s.r > 0.8).length; beforeBig += b.filter(s => s.r > 0.8).length;
    }
    const ratio = now / before;
    const bigRatio = nowBig / beforeBig;
    assert.ok(ratio > 0.45 && ratio < 0.55, `density ratio ${ratio.toFixed(2)}`);
    assert.ok(bigRatio < 0.3, `big stars ratio ${bigRatio.toFixed(2)} (all stars ${ratio.toFixed(2)})`);
    assert.ok(starField(PG, BX).some(s => s.r > 0.8), "a few bigger stars remain in the default layout");
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

import { createCruiser, steerCruiser, kickCruiser, reflect, clampToZone } from "../src/bg-float.js";

test("cruise: an icon never slows to a halt, even boxed in by walls", () => {
    const c = createCruiser(3, { radius: 150, speed: 11 });
    const home = { x: 0, y: 0 };
    let pos = { x: 0, y: 0 };
    const wall = { left: -60, top: -40, right: 60, bottom: 40 }; // tighter than its radius
    let minSpeed = Infinity;
    for (let f = 0; f < 60 * 120; f++) {
        const t = f / 60;
        steerCruiser(c, 1 / 60, t, { x: pos.x - home.x, y: pos.y - home.y });
        pos = { x: pos.x + c.vx / 60, y: pos.y + c.vy / 60 };
        const z = clampToZone(pos.x, pos.y, wall);
        if (z.nx || z.ny) {
            pos = { x: z.x, y: z.y };
            if (z.nx) reflect(c, { x: z.nx, y: 0 });
            if (z.ny) reflect(c, { x: 0, y: z.ny });
        }
        if (t > 1) minSpeed = Math.min(minSpeed, Math.hypot(c.vx, c.vy));
    }
    assert.ok(minSpeed > 0.8 * 11, `min speed ${minSpeed.toFixed(2)} px/s`);
});

test("cruise: it roams its own neighbourhood and returns to cruise speed after a kick", () => {
    const c = createCruiser(5, { radius: 150, speed: 11 });
    let pos = { x: 0, y: 0 };
    let far = 0;
    for (let f = 0; f < 60 * 120; f++) {
        steerCruiser(c, 1 / 60, f / 60, pos);
        pos = { x: pos.x + c.vx / 60, y: pos.y + c.vy / 60 };
        far = Math.max(far, Math.hypot(pos.x, pos.y));
    }
    assert.ok(far > 60 && far < 1.4 * 150, `max distance from home ${far.toFixed(0)}`);
    kickCruiser(c, { x: 10, y: 0 }, { x: 0, y: 0 });
    assert.ok(Math.hypot(c.vx, c.vy) > 100);
    for (let f = 0; f < 60 * 4; f++) {
        steerCruiser(c, 1 / 60, 200 + f / 60, pos);
    }
    assert.ok(Math.abs(Math.hypot(c.vx, c.vy) - 11) < 1, "back to cruise within 4 s");
});

test("reflect: only the part going into the wall is bounced back", () => {
    const c = { vx: -10, vy: 5 };
    assert.equal(reflect(c, { x: 1, y: 0 }, 1), true);
    assert.deepEqual([c.vx, c.vy], [10, 5]);
    assert.equal(reflect(c, { x: 1, y: 0 }, 1), false);
});

test("layout: the big satellites stay in their quadrants; no icon's nearest neighbour is the same artwork", () => {
    const TYPES = ["satellite", "earthsat", "dish", "image", "terrain", "terrain", "globe", "image", "dish", "globe", "terrain", "image"];
    const S2 = [224, 224, 70, 70, 70, 52, 52, 52, 52, 52, 52, 52];
    const mid = { x: (PG.left + PG.right) / 2, y: (PG.top + PG.bottom) / 2 };
    const Q2 = { left: PG.left, top: PG.top, right: mid.x, bottom: mid.y };
    const Q4 = { left: mid.x, top: mid.y, right: PG.right, bottom: PG.bottom };
    const homes = layoutHomes(S2, PG, BX, { gap: 38, typeOf: i => TYPES[i], zoneFor: i => (i === 0 ? Q2 : i === 1 ? Q4 : null) });
    const inside = (h, z) => h.x >= z.left && h.x <= z.right && h.y >= z.top && h.y <= z.bottom;
    assert.ok(inside(homes[0], Q2), `satellite at ${homes[0].x},${homes[0].y}`);
    assert.ok(inside(homes[1], Q4), `earth-satellite at ${homes[1].x},${homes[1].y}`);
    homes.forEach((h, i) => {
        let nearest = -1;
        let nd = Infinity;
        homes.forEach((o, j) => {
            if (j !== i) {
                const d = Math.hypot(h.x - o.x, h.y - o.y) - (h.size + o.size) / 2;
                if (d < nd) { nd = d; nearest = j; }
            }
        });
        assert.notEqual(TYPES[nearest], TYPES[i], `icon ${i} (${TYPES[i]}) is beside another ${TYPES[i]}`);
    });
});

test("per-pair gap: same-artwork icons are held further apart than others", () => {
    const a = body(0, 0, 30); a.type = "globe";
    const b = body(100, 0, 30); b.type = "globe";
    const c = body(0, 100, 30); c.type = "dish";
    const gapFor = (p, q) => (p.type === q.type ? 190 : 38);
    assert.equal(separate([a, b, c], { gapFor, iterations: 12 }), 0);
    assert.ok(Math.hypot(b.x - a.x, b.y - a.y) - 60 >= 189.5);
    assert.ok(Math.hypot(c.x - a.x, c.y - a.y) - 60 >= 37.5);
});
