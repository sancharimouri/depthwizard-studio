// Flood scenario for the expanded 3D view: a semi-transparent water plane at
// a chosen water level, a snappy tween when the level slider moves, a "Play
// flood simulation" playback from the lowest ground up to the slider level,
// and inundation stats for the water level actually shown at that moment.
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
    let active = false;
    let shown = null; // level currently drawn (m)
    let target = null; // level the slider asks for (m)
    let play = null; // { from, to, start } while playing
    let frame = 0;
    let lastT = 0;
    let lastStats = 0;

    function attach() {
        const t = getTerrain();
        if (t === terrain) {
            return t;
        }
        water?.parent?.remove(water);
        water?.geometry.dispose();
        water?.material.dispose();
        water = null;
        terrain = t;
        if (!t) {
            return null;
        }
        index = inundationIndex(t.grid);
        const { elevationMin: lo, elevationMax: hi } = t.grid;
        slider.min = String(Math.floor(lo));
        slider.max = String(Math.ceil(hi));
        slider.step = String(Math.max(0.1, Number(((hi - lo) / 400).toPrecision(1))));
        // Default: the 30th-percentile elevation (what the old flood tint marked).
        target = shown = index.percentile(0.3);
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
        return t;
    }

    function draw(level, forceStats = false) {
        shown = level;
        if (water && terrain) {
            // a flat (2D) layer has z = 0 everywhere; the plane would hide it
            water.visible = active && terrain.isExtruded();
            water.position.z = terrain.localZForElevation(level) + 0.02;
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
        playBtn.textContent = play ? "■ Stop" : "▶ Play flood simulation";
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
            syncPlayButton();
            draw(shown, true);
            return;
        }
        play = { from: terrain.grid.elevationMin, to: Number(slider.value), start: performance.now() };
        target = play.to;
        syncPlayButton();
        draw(play.from, true);
        kick();
    });

    resetBtn.addEventListener("click", () => {
        if (!attach()) {
            return;
        }
        play = null;
        syncPlayButton();
        target = Number(slider.value);
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
