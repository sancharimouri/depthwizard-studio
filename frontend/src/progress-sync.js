// Progress readouts that follow the real work instead of a fixed timer.
//
// A box's work (one HTTP request, a texture fetch, building a mesh) reports no
// progress of its own, so the percent is an estimate that tracks elapsed time
// against how long that stage took before (remembered per browser):
//   - work that finishes within IMMEDIATE_MS shows no readout at all;
//   - while it runs, the percent eases toward 95% (86% at the expected time),
//     so a stage that takes 30 s crawls for 30 s instead of sitting at 100%;
//   - when it finishes, the readout sweeps to 100% in FINISH_MS and closes,
//     so a fast stage is a fast animation.

export const IMMEDIATE_MS = 250;
export const FINISH_MS = 280;
const CEILING = 95;

export function syncedPercent(elapsedMs, expectedMs) {
    const tau = Math.max(1, expectedMs) / 2;
    return Math.min(CEILING, Math.round(CEILING * (1 - Math.exp(-Math.max(0, elapsedMs) / tau))));
}

// Remembered durations per stage (exponential moving average), in localStorage.
// Falls back to `defaults` when storage is blocked or has nothing yet.
export function createDurationMemory(defaults, storage = globalThis.localStorage, key = "dw2.stageMs.v1") {
    let saved = {};
    try {
        saved = JSON.parse(storage?.getItem(key) ?? "{}") ?? {};
    } catch {
        saved = {};
    }
    return {
        expected(id) {
            return saved[id] ?? defaults[id] ?? 2000;
        },
        record(id, ms) {
            const prev = saved[id];
            saved[id] = Math.round(prev == null ? ms : 0.6 * prev + 0.4 * ms);
            try {
                storage?.setItem(key, JSON.stringify(saved));
            } catch {
                // not remembered; the default stays in use
            }
        },
    };
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

// Runs the readout for `work` (a promise or value). onShow() is called once, only
// if the work outlasts IMMEDIATE_MS; onTick(percent, stepText) while it runs.
// Resolves to { result, ms, shown } after the work settles (a rejection counts as
// settled, with result undefined).
export async function trackWork({ work, steps, expectedMs, onShow, onTick, tickMs = 100 }) {
    const t0 = performance.now();
    let settled = false;
    let result;
    const done = Promise.resolve(work).then(value => {
        result = value;
    }, () => {}).finally(() => {
        settled = true;
    });

    await Promise.race([done, sleep(IMMEDIATE_MS)]);
    if (settled) {
        return { result, ms: performance.now() - t0, shown: false };
    }

    const stepFor = percent => steps[Math.min(steps.length - 1, Math.floor((percent / 100) * steps.length))] ?? "";
    onShow?.();
    let percent = 0;
    onTick(percent, stepFor(percent));
    while (!settled) {
        await Promise.race([done, sleep(tickMs)]);
        percent = syncedPercent(performance.now() - t0, expectedMs);
        onTick(percent, stepFor(percent));
    }
    const ms = performance.now() - t0;

    // sweep the rest of the way to 100% quickly, then close
    const from = percent;
    const s0 = performance.now();
    for (let t = 0; t < FINISH_MS; t = performance.now() - s0) {
        const p = Math.round(from + (100 - from) * (t / FINISH_MS));
        onTick(p, stepFor(p));
        await sleep(40);
    }
    onTick(100, steps[steps.length - 1] ?? "");
    return { result, ms, shown: true };
}
