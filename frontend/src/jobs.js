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
    "Jobs live in this browser tab's memory, and there is no server-side storage yet: reloading or closing the tab "
    + "loses any job you haven't saved. Saving downloads the job as JSON and keeps a copy in this browser's Saved list.";

export function createJobStore() {
    const jobs = [];
    const listeners = new Set();
    let activeId = null;
    let seq = 0;

    function emit() {
        listeners.forEach(fn => fn());
    }

    return {
        add(input, restored = null) {
            seq += 1;
            const createdAt = restored?.createdAt ?? new Date().toISOString();
            const job = {
                id: `job-${seq}`,
                n: seq,
                // stable across sessions: identifies the job in the Saved list
                uid: restored?.uid ?? `${Date.parse(createdAt).toString(36)}-${seq}-${Math.random().toString(36).slice(2, 7)}`,
                createdAt,
                status: restored ? "complete" : "generating",
                progress: restored ? 100 : 0,
                saved: Boolean(restored),
                pinned: false,
                // user-editable display name (null → "Job N"); see jobLabel()
                name: restored?.name ?? null,
                input,
                log: restored?.log ? restored.log.map(line => ({ ...line })) : [],
            };
            jobs.push(job);
            activeId = job.id;
            emit();
            return job;
        },
        // Removes a job; if it was active, the most recent remaining job becomes active.
        remove(id) {
            const index = jobs.findIndex(job => job.id === id);
            if (index < 0) {
                return null;
            }
            jobs.splice(index, 1);
            if (activeId === id) {
                activeId = jobs.length ? jobs[jobs.length - 1].id : null;
            }
            emit();
            return this.active();
        },
        // Inline rename (pages-panel Jobs list and the 3D view's tab strip).
        // Blank or whitespace-only names fall back to the default "Job N".
        rename(id, name) {
            const job = this.get(id);
            if (!job) {
                return;
            }
            const clean = String(name ?? "").replace(/\s+/g, " ").trim().slice(0, 60);
            const next = clean && clean !== `Job ${job.n}` ? clean : null;
            if (next !== job.name) {
                job.name = next;
                emit();
            }
        },
        togglePin(id) {
            const job = this.get(id);
            if (job) {
                job.pinned = !job.pinned;
                emit();
            }
        },
        findByUid(uid) {
            return jobs.find(job => job.uid === uid) ?? null;
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
        // newest first (side panel); pinned jobs have their own section
        panelOrder() {
            return jobs.slice().reverse().filter(job => !job.pinned);
        },
        pinnedOrder() {
            return jobs.slice().reverse().filter(job => job.pinned);
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
    return job.name || `Job ${job.n}`;
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
            name: jobLabel(job),
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

// ---------------------------------------------------------------- Saved works
// "Save" downloads the job and also keeps a copy here, in this browser's
// localStorage, so the JOBS tab's SAVED section can list (and reopen) earlier work after a
// reload. Per-browser only; storage failures (private window, blocked site
// data) leave the list empty rather than breaking anything.

export const SAVED_KEY = "dw2.savedJobs.v1";

export function savedRecord(job, now = new Date()) {
    return {
        uid: job.uid,
        savedAt: now.toISOString(),
        createdAt: job.createdAt,
        name: job.name,
        label: jobLabel(job), // display name at save time ("Job 2" or the user's rename)
        input: job.input,
        log: job.log,
    };
}

export function createSavedStore(storage = globalThis.localStorage) {
    function read() {
        try {
            const list = JSON.parse(storage?.getItem(SAVED_KEY) ?? "[]");
            return Array.isArray(list) ? list : [];
        } catch {
            return [];
        }
    }

    function write(list) {
        try {
            storage?.setItem(SAVED_KEY, JSON.stringify(list));
            return true;
        } catch {
            return false;
        }
    }

    return {
        // newest save first
        list() {
            return read().sort((a, b) => b.savedAt.localeCompare(a.savedAt));
        },
        put(record) {
            const list = read().filter(item => item.uid !== record.uid);
            list.push(record);
            return write(list);
        },
        remove(uid) {
            return write(read().filter(item => item.uid !== uid));
        },
        get(uid) {
            return read().find(item => item.uid === uid) ?? null;
        },
    };
}
