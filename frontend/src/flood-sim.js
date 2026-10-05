// Flood scenario for the expanded 3D view: a semi-transparent water plane at
// a chosen water level, a snappy tween when the level slider moves, a
// "Simulate" playback from the lowest ground up to the slider's maximum (a
// couple of metres above the highest point), and inundation stats for the
// water level actually shown at that moment. The level starts, and Reset
// returns it, half way up the slider.
//
// ILLUSTRATIVE "bathtub" fill, not a hydrological model: every DEM cell whose
// real elevation is below the level counts as inundated, whether or not
// water could reach it. The UI says so.
//
// The water plane is a child of the terrain mesh, placed at the level's
// local Z, so it follows the display-only vertical exaggeration (mesh
// scale) automatically. The stats read the raw DEM grid (real metres), never
// the displayed mesh.

import * as THREE from "three";

const TWEEN_TAU_MS = 70; // exponential approach: ~95% settled after ~210 ms
const PLAY_MS = 4000;
const TOP_MARGIN_M = 2; // the slider tops out this far above the highest ground

// The slider's range for a terrain: lowest ground → highest ground + TOP_MARGIN_M; default half way.
export function floodRange(elevationMin, elevationMax) {
    const min = Math.floor(elevationMin);
    const max = Math.ceil(elevationMax) + TOP_MARGIN_M;
    return { min, max, half: (min + max) / 2 };
}
const STATS_INTERVAL_MS = 100;

const easeInOut = t => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// Sorted real elevations → exact "cells below level" by binary search.
export function inundationIndex(grid) {
    const { heights, elevationMin, elevationMax } = grid;
    const range = elevationMax - elevationMin;
    const sorted = new Float64Array(heights.length);
    for (let i = 0; i < heights.length; i += 1) {
        sorted[i] = elevationMin + heights[i] * range;
    }
    sorted.sort();
    return {
        count: sorted.length,
        // fraction of cells strictly below `level` metres
        fractionBelow(level) {
            let lo = 0;
            let hi = sorted.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (sorted[mid] < level) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo / sorted.length;
        },
        percentile(p) {
            return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))];
        },
    };
}

export function createFloodSim({ getTerrain, onChange = () => {}, els }) {
    const { slider, levelLabel, playBtn, resetBtn, pctEl, areaEl, panel } = els;
    const reduceMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    let terrain = null;
    let index = null;
    let water = null;
    let walls = null; // the water body's side walls around the tile's edge
    let perimeter = null; // grid vertex indices around the edge, in order
    let active = false;
    let shown = null; // level currently drawn (m)
    let target = null; // level the slider asks for (m)
    let play = null; // { from, to, start } while playing
    let frame = 0;
    let lastT = 0;
    let lastStats = 0;
    let range = null; // floodRange() of the attached terrain

    function attach() {
        const t = getTerrain();
        if (t === terrain) {
            return t;
        }
        water?.parent?.remove(water);
        water?.geometry.dispose();
        water?.material.dispose();
        water = null;
        walls?.parent?.remove(walls);
        walls?.geometry.dispose();
        walls?.material.dispose();
        walls = null;
        terrain = t;
        if (!t) {
            return null;
        }
        index = inundationIndex(t.grid);
        const { elevationMin: lo, elevationMax: hi } = t.grid;
        range = floodRange(lo, hi);
        slider.min = String(range.min);
        slider.max = String(range.max);
        slider.step = String(Math.max(0.1, Number(((hi - lo) / 400).toPrecision(1))));
        target = shown = range.half;
        slider.value = String(target);

        water = new THREE.Mesh(
            new THREE.PlaneGeometry(t.terrainWidth, t.terrainHeight),
            new THREE.MeshStandardMaterial({
                color: 0x2a78c8,
                transparent: true,
                opacity: 0.55,
                roughness: 0.25,
                metalness: 0.1,
                depthWrite: false,
                side: THREE.DoubleSide,
            }),
        );
        water.renderOrder = 2;
        water.visible = false;
        t.mesh.add(water);

        // Fill, not just a sheet: translucent walls around the tile's edge,
        // from the terrain's edge up to the water level wherever the ground
        // there is below the water, so the flooded volume reads from the sides.
        const { width: W, height: H } = t.grid;
        perimeter = [];
        for (let x = 0; x < W; x++) perimeter.push(x); // north edge (row 0)
        for (let y = 1; y < H; y++) perimeter.push(y * W + W - 1); // east edge
        for (let x = W - 2; x >= 0; x--) perimeter.push((H - 1) * W + x); // south edge
        for (let y = H - 2; y >= 1; y--) perimeter.push(y * W); // west edge
        const n = perimeter.length;
        const wallPos = new Float32Array(n * 2 * 3);
        const pos = t.mesh.geometry.attributes.position;
        perimeter.forEach((vi, k) => {
            for (const j of [0, 1]) {
                wallPos[(k * 2 + j) * 3] = pos.getX(vi);
                wallPos[(k * 2 + j) * 3 + 1] = pos.getY(vi);
            }
        });
        const wallIdx = [];
        for (let k = 0; k < n; k++) {
            const k2 = (k + 1) % n;
            const b0 = k * 2;
            const t0 = k * 2 + 1;
            const b1 = k2 * 2;
            const t1 = k2 * 2 + 1;
            wallIdx.push(b0, b1, t0, t0, b1, t1);
        }
        const wallGeom = new THREE.BufferGeometry();
        wallGeom.setAttribute("position", new THREE.BufferAttribute(wallPos, 3));
        wallGeom.setIndex(wallIdx);
        walls = new THREE.Mesh(wallGeom, new THREE.MeshStandardMaterial({
            color: 0x1f5f9e,
            transparent: true,
            opacity: 0.6,
            roughness: 0.3,
            metalness: 0.05,
            depthWrite: false,
            side: THREE.DoubleSide,
        }));
        walls.renderOrder = 2;
        walls.visible = false;
        walls.frustumCulled = false;
        t.mesh.add(walls);
        return t;
    }

    function draw(level, forceStats = false) {
        shown = level;
        if (water && terrain) {
            // a flat (2D) layer has z = 0 everywhere; the plane would hide it
            const show = active && terrain.isExtruded();
            const zWater = terrain.localZForElevation(level) + 0.02;
            water.visible = show;
            water.position.z = zWater;
            walls.visible = show;
            if (show) {
                // wall = ground edge → water level where the ground is lower
                const pos = terrain.mesh.geometry.attributes.position;
                const wp = walls.geometry.attributes.position;
                perimeter.forEach((vi, k) => {
                    const zg = pos.getZ(vi);
                    wp.setZ(k * 2, zg);
                    wp.setZ(k * 2 + 1, Math.max(zg, zWater));
                });
                wp.needsUpdate = true;
                walls.geometry.computeVertexNormals();
            }
        }
        levelLabel.textContent = `${Math.round(level).toLocaleString()} m`;
        const now = performance.now();
        if (forceStats || now - lastStats >= STATS_INTERVAL_MS) {
            lastStats = now;
            const frac = index ? index.fractionBelow(level) : 0;
            const areaKm2 = terrain ? (frac * terrain.footprintWidthMeters * terrain.footprintHeightMeters) / 1e6 : 0;
            pctEl.textContent = `${(frac * 100).toFixed(1)}%`;
            areaEl.textContent = `${areaKm2.toFixed(areaKm2 < 10 ? 2 : 1)} km²`;
        }
        onChange();
    }

    function tick(t) {
        frame = 0;
        const dt = lastT ? Math.min(t - lastT, 100) : 16;
        lastT = t;
        let level;
        let done = false;
        if (play) {
            const k = Math.min(1, (t - play.start) / PLAY_MS);
            level = play.from + (play.to - play.from) * easeInOut(k);
            slider.value = String(level); // the slider follows the rising water
            if (k >= 1) {
                play = null;
                syncPlayButton();
                done = level === target;
            }
        } else {
            const a = 1 - Math.exp(-dt / TWEEN_TAU_MS);
            level = shown + (target - shown) * a;
            if (Math.abs(target - level) < 0.05) {
                level = target;
                done = true;
            }
        }
        draw(level, done);
        if (!done || play) {
            frame = requestAnimationFrame(tick);
        } else {
            lastT = 0;
        }
    }

    function kick() {
        if (!frame) {
            lastT = 0;
            frame = requestAnimationFrame(tick);
        }
    }

    function syncPlayButton() {
        playBtn.textContent = play ? "■ Stop" : "▶ Simulate";
        playBtn.setAttribute("aria-pressed", String(Boolean(play)));
    }

    slider.addEventListener("input", () => {
        attach();
        target = Number(slider.value);
        if (play) {
            play = null; // dragging takes over from playback
            syncPlayButton();
        }
        if (reduceMotion) {
            draw(target, true);
        } else {
            kick();
        }
    });

    playBtn.addEventListener("click", () => {
        if (!attach()) {
            return;
        }
        if (play) {
            // Stop: freeze where the water is now; stats settle on that level.
            play = null;
            cancelAnimationFrame(frame);
            frame = 0;
            target = shown;
            slider.value = String(shown);
            syncPlayButton();
            draw(shown, true);
            return;
        }
        play = { from: range.min, to: range.max, start: performance.now() };
        target = play.to;
        syncPlayButton();
        draw(play.from, true);
        kick();
    });

    resetBtn.addEventListener("click", () => {
        if (!attach()) {
            return;
        }
        // back to the default: half way up the slider
        play = null;
        syncPlayButton();
        target = range.half;
        slider.value = String(target);
        cancelAnimationFrame(frame);
        frame = 0;
        draw(target, true);
    });

    return {
        setActive(on) {
            attach();
            active = on;
            panel.hidden = !on;
            if (!on) {
                play = null;
                cancelAnimationFrame(frame);
                frame = 0;
                syncPlayButton();
            }
            if (terrain) {
                draw(on ? target : shown, true);
            }
        },
        // after a layer switch (flat ↔ 3D) or a new terrain
        refresh() {
            attach();
            if (terrain && shown != null) {
                draw(shown, true);
            }
        },
        level: () => shown,
    };
}
