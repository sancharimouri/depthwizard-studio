// Progress readouts that follow the real work instead of a fixed timer.
//
// A box's work (one HTTP request, a texture fetch, building a mesh) reports no
// progress of its own, so the percent is an estimate that tracks elapsed time
// against how long that stage took before (remembered per browser):
//   - every readout lasts at least MIN_VISIBLE_MS. Work that is done sooner
//     (even instantly) still gets a smooth 0 -> 100% over those 1.3 s, so the
//     boxes never pop in too fast to read;
//   - while longer work runs, the percent eases toward 95% (86% at the expected
//     time), so a stage that takes 30 s crawls for 30 s instead of sitting at 100%;
//   - when longer work finishes, the readout sweeps to 100% in FINISH_MS and closes.

export const MIN_VISIBLE_MS = 1300;
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

// Runs the readout for `work` (a promise or value). onShow() is called once, at
// the start; onTick(percent, stepText) while it runs. Resolves to
// { result, ms, shown } once the work has settled AND the readout has lasted
// minMs; `ms` is how long the work itself took. A rejection counts as settled,
// with result undefined.
export async function trackWork({ work, steps, expectedMs, onShow, onTick, tickMs = 100, minMs = MIN_VISIBLE_MS }) {
    const t0 = performance.now();
    let settledAt = null;
    let result;
    const done = Promise.resolve(work).then(value => {
        result = value;
    }, () => {}).finally(() => {
        settledAt = performance.now();
    });

    const stepFor = percent => steps[Math.min(steps.length - 1, Math.floor((percent / 100) * steps.length))] ?? "";
    onShow?.();
    let percent = 0;
    onTick(percent, stepFor(percent));
    for (;;) {
        // race the work only while it's pending: a settled promise would win every
        // race at once and turn this into a busy loop that blocks the page
        await (settledAt === null ? Promise.race([done, sleep(tickMs)]) : sleep(tickMs));
        const elapsed = performance.now() - t0;
        if (settledAt !== null && elapsed >= minMs) {
            break;
        }
        const target = settledAt !== null
            // done early: run evenly to 100% at minMs
            ? Math.round(Math.min(1, elapsed / minMs) * 100)
            : syncedPercent(elapsed, Math.max(expectedMs, minMs));
        percent = Math.min(99, Math.max(percent, target));
        onTick(percent, stepFor(percent));
    }
    const ms = settledAt - t0;

    // work that outlasted minMs: sweep the rest of the way to 100% quickly, then close.
    // (Work done sooner has already run evenly to ~100% over minMs: no extra sweep.)
    if (ms >= minMs && percent < 99) {
        const from = percent;
        const s0 = performance.now();
        for (let t = 0; t < FINISH_MS; t = performance.now() - s0) {
            const p = Math.round(from + (100 - from) * (t / FINISH_MS));
            onTick(p, stepFor(p));
            await sleep(40);
        }
    }
    onTick(100, steps[steps.length - 1] ?? "");
    return { result, ms, shown: true };
}
