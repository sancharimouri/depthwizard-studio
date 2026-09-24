// ============================================================
// JOBS — every START GENERATION is a separate job, held in this tab's
// memory only. There is no server-side job storage yet: a job is unsaved
// until the user downloads it (a JSON export of its input, routing, DEM
// and calculation log), and reloading or closing the tab loses unsaved
// jobs. Pure state + export; the DOM lives in main.js.
//
// Order (flagged for the user to confirm): the side panel lists the newest
// job at the TOP; the 3D view's tab strip runs oldest → newest left to
// right, the way browser tabs append.
// ============================================================

export const STORAGE_NOTE =
    "Session only: jobs live in this browser tab's memory. There is no server-side storage yet, "
    + "so reloading or closing the tab loses any job you haven't saved (downloaded).";

export function createJobStore() {
    const jobs = [];
    const listeners = new Set();
    let activeId = null;
    let seq = 0;

    function emit() {
        listeners.forEach(fn => fn());
    }

    return {
        add(input) {
            seq += 1;
            const job = {
                id: `job-${seq}`,
                n: seq,
                createdAt: new Date().toISOString(),
                status: "generating",
                progress: 0,
                saved: false,
                input,
                log: [],
            };
            jobs.push(job);
            activeId = job.id;
            emit();
            return job;
        },
        get(id) {
            return jobs.find(job => job.id === id) ?? null;
        },
        active() {
            return this.get(activeId);
        },
        setActive(id) {
            if (this.get(id) && id !== activeId) {
                activeId = id;
                emit();
            }
        },
        update(id, patch) {
            const job = this.get(id);
            if (job) {
                Object.assign(job, patch);
                emit();
            }
        },
        markSaved(ids) {
            ids.forEach(id => {
                const job = this.get(id);
                if (job) {
                    job.saved = true;
                }
            });
            emit();
        },
        // oldest → newest (tab strip)
        creationOrder() {
            return jobs.slice();
        },
        // newest first (side panel)
        panelOrder() {
            return jobs.slice().reverse();
        },
        unsaved() {
            return jobs.filter(job => !job.saved);
        },
        generating() {
            return jobs.find(job => job.status === "generating") ?? null;
        },
        count() {
            return jobs.length;
        },
        onChange(fn) {
            listeners.add(fn);
            return () => listeners.delete(fn);
        },
    };
}

export function jobLabel(job) {
    return `Job ${job.n}`;
}

// The unsaved-work modal's copy, per action. Always states the count; never
// offers "don't show again".
export function unsavedCopy(action, count) {
    const noun = count === 1 ? "job" : "jobs";
    const title = `${count} unsaved ${noun}`;
    if (action === "new") {
        return {
            title,
            body: `Starting a new generation keeps ${count === 1 ? "this job" : `these ${count} jobs`} open in the jobs panel, `
                + "but only in this tab's memory. Save the ones you want to keep before you go on.",
            discard: "Continue without saving",
            save: "Save selected & continue",
        };
    }
    if (action === "quit") {
        return {
            title,
            body: `Quitting Depth Wizard discards ${count === 1 ? "this job" : `all ${count} jobs`}. `
                + "Save the ones you want to keep first.",
            discard: "Discard & quit",
            save: "Save selected & quit",
        };
    }
    return {
        title,
        body: `Closing clears every job on this page, ${count === 1 ? "including this one" : `including these ${count}`}. `
            + "Save the ones you want to keep first.",
        discard: "Discard & close",
        save: "Save selected & close",
    };
}

// A saved job = a JSON download of what the job actually is.
export function jobsExport(jobs, now = new Date()) {
    return {
        format: "depthwizard2.jobs/v1",
        exported_at: now.toISOString(),
        storage_note: "Exported from a browser session; Depth Wizard has no server-side job storage yet.",
        placeholder_note: "The generation stages still show the Darjeeling reference outputs for every job; "
            + "the input, its ground resolution, tier routing and DEM below are real.",
        jobs: jobs.map(job => ({
            job: job.n,
            created_at: job.createdAt,
            status: job.status,
            input: {
                title: job.input.title,
                source: job.input.source,
                details: Object.fromEntries(job.input.meta ?? []),
                routing: job.input.routing,
                dem: job.input.dem ?? null,
            },
            calculation_log: job.log.map(line => `[${line.t}] ${line.text}`),
        })),
    };
}

export function exportFilename(jobs, now = new Date()) {
    const stamp = now.toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const which = jobs.length === 1 ? `job${jobs[0].n}` : `${jobs.length}-jobs`;
    return `depthwizard-${which}-${stamp}.json`;
}
