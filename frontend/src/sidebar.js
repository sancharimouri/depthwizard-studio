// Global sidebar: the one navigation surface on every page (Workbench and its
// processing grid, Explore, Docs, the expanded 3D viewer window). Structured
// after shadcn/ui's sidebar-07 block:
//
//   header  the "D" mark (opens Home) + "Depth Wizard" + the lock button
//           (locked: stays open and pushes the page; unlocked: closed, and
//           hovering it opens it over the page until the pointer leaves)
//   tabs    PAGES | JOBS (equal width, muted green underline on the active one)
//   PAGES   DW Studio (page-workbench) / Demo / Home (page-docs) with icons + "N jobs running"
//   JOBS    the jobs list (rendered by main.js into the same element ids)
//   footer  GitHub / email / LinkedIn / X links
//   rail    drag to resize (SIDEBAR_MIN_W–SIDEBAR_MAX_W); dragging well below the
//           minimum collapses it, dragging back out opens it; click to collapse
//
// Collapsed, it is an icon rail: the D, the page icons (inside the 3D viewer:
// one square per job instead) and the link icons.
//
// Closed by default on every page (2026-10-05); the lock is remembered for the
// session (sessionStorage). Default tab: PAGES, except JOBS inside the 3D
// viewer window — applied when entering/leaving the viewer or changing page;
// the user can switch tabs any time in between.
//
// Locked, the sidebar is a flex sibling of #app-main that pushes the content
// (width animates over 200 ms); hover-opened, it overlays it instead (the
// layout keeps the rail's width), so the page never reflows under the pointer.

const LOCKED_KEY = "dw2.sidebarLocked";
const PEEK_OPEN_MS = 90;
const PEEK_CLOSE_MS = 220;
const WIDTH_KEY = "dw2.sidebarWidth";
export const SIDEBAR_MIN_W = 208;
export const SIDEBAR_MAX_W = 360;
// Dragging the rail this far below the minimum width collapses the bar to its
// icon rail (and dragging back out past it opens it again).
export const SIDEBAR_COLLAPSE_BELOW_W = SIDEBAR_MIN_W - 48;
const DRAG_THRESHOLD_PX = 4;

export function clampSidebarWidth(px) {
    return Math.round(Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, px)));
}

// What a rail drag to `px` (the pointer's distance from the bar's left edge) means:
// collapse, or open at the clamped width.
export function railDragTarget(px) {
    return px < SIDEBAR_COLLAPSE_BELOW_W ? { collapsed: true } : { collapsed: false, width: clampSidebarWidth(px) };
}

export function createSidebar({ onNavigate }) {
    const root = document.getElementById("app-sidebar");
    const toggle = document.getElementById("sb-toggle"); // the lock button
    const tabs = [...root.querySelectorAll(".sb-tab")];
    const panels = {
        pages: document.getElementById("sb-panel-pages"),
        jobs: document.getElementById("sb-panel-jobs"),
    };
    const links = [...root.querySelectorAll(".sb-page-link")];
    const runningEl = document.getElementById("sb-jobs-running");

    let locked = false;
    let peek = false;
    let peekTimer = 0;

    // The tab the closed bar shows (and opens on): JOBS inside the 3D viewer, PAGES elsewhere. An unlocked
    // bar goes back to it whenever it closes, whatever tab was picked while it was open.
    let inViewer = false;
    const defaultTab = () => (inViewer ? "jobs" : "pages");

    function applyState() {
        const open = locked || peek;
        if (!open) {
            setTab(defaultTab());
        }
        root.classList.toggle("is-collapsed", !open);
        root.classList.toggle("is-peek", peek && !locked);
        root.classList.toggle("is-locked", locked);
        toggle.setAttribute("aria-pressed", String(locked));
        const label = locked ? "Unlock sidebar (closes when the pointer leaves)" : "Lock sidebar open";
        toggle.setAttribute("aria-label", label);
        toggle.title = label;
    }

    function setLocked(value) {
        locked = Boolean(value);
        peek = !locked && root.matches(":hover");
        clearTimeout(peekTimer);
        applyState();
        try {
            sessionStorage.setItem(LOCKED_KEY, locked ? "1" : "0");
        } catch {
            // storage blocked: the lock just isn't remembered
        }
    }

    // kept for callers: "collapsed" = unlocked and closed
    function setCollapsed(collapsed) {
        setLocked(!collapsed);
        if (collapsed) {
            peek = false;
            applyState();
        }
    }

    function setPeek(value, delay) {
        clearTimeout(peekTimer);
        peekTimer = setTimeout(() => {
            // stays open while something inside it has the keyboard (renaming a job)
            if (!value && root.contains(document.activeElement) && document.activeElement.matches("input, textarea")) {
                return;
            }
            peek = value;
            applyState();
        }, delay);
    }

    root.addEventListener("pointerenter", event => {
        if (event.pointerType !== "touch") {
            setPeek(true, PEEK_OPEN_MS);
        }
    });
    root.addEventListener("pointerleave", () => setPeek(false, PEEK_CLOSE_MS));
    // keyboard focus opens it too (not a focus set from code: a re-render can remove that element
    // without any focusout, which would leave the bar stuck open)
    root.addEventListener("focusin", event => {
        if (event.target.matches?.(":focus-visible")) {
            setPeek(true, 0);
        }
    });
    root.addEventListener("focusout", event => {
        if (!root.contains(event.relatedTarget)) {
            setPeek(false, PEEK_CLOSE_MS);
        }
    });
    // a click anywhere else always closes a hover-opened bar
    document.addEventListener("pointerdown", event => {
        if (peek && !root.contains(event.target)) {
            setPeek(false, 0);
        }
    });

    // the green underline slides between PAGES and JOBS (one element under the tabs)
    const underline = document.createElement("span");
    underline.className = "sb-tab-underline";
    underline.setAttribute("aria-hidden", "true");
    root.querySelector(".sb-tabs").append(underline);

    function setTab(name) {
        tabs.forEach(tab => {
            const on = tab.dataset.tab === name;
            tab.setAttribute("aria-selected", String(on));
            tab.tabIndex = on ? 0 : -1;
        });
        underline.classList.toggle("is-jobs", name === "jobs");
        Object.entries(panels).forEach(([key, panel]) => {
            panel.hidden = key !== name;
        });
    }

    function setActivePage(pageId) {
        links.forEach(link => {
            const on = link.dataset.page === pageId;
            link.classList.toggle("active", on);
            if (on) {
                link.setAttribute("aria-current", "page");
            } else {
                link.removeAttribute("aria-current");
            }
        });
    }

    function setJobsRunning(count) {
        if (runningEl) {
            runningEl.textContent = count ? `${count} job${count === 1 ? "" : "s"} running` : "No jobs running";
        }
    }

    function setWidth(px, remember = true) {
        const width = clampSidebarWidth(px);
        root.style.setProperty("--sb-open-w", `${width}px`);
        rail?.setAttribute("aria-valuenow", String(width));
        if (remember) {
            try {
                sessionStorage.setItem(WIDTH_KEY, String(width));
            } catch {
                // storage blocked: the width just isn't remembered
            }
        }
        return width;
    }

    toggle.addEventListener("click", () => setLocked(!locked));

    // Rail: drag to resize (within the limits), click without dragging to collapse/expand.
    // Dragging a collapsed bar opens it at the dragged width.
    const rail = document.getElementById("sb-rail");
    if (rail) {
        rail.setAttribute("aria-valuemin", String(SIDEBAR_MIN_W));
        rail.setAttribute("aria-valuemax", String(SIDEBAR_MAX_W));
        let drag = null;
        rail.addEventListener("pointerdown", event => {
            if (event.button !== 0) {
                return;
            }
            event.preventDefault();
            rail.setPointerCapture(event.pointerId);
            drag = { x: event.clientX, left: root.getBoundingClientRect().left, moved: false };
        });
        rail.addEventListener("pointermove", event => {
            if (!drag) {
                return;
            }
            if (!drag.moved && Math.abs(event.clientX - drag.x) < DRAG_THRESHOLD_PX) {
                return;
            }
            if (!drag.moved) {
                drag.moved = true;
                root.classList.add("is-resizing");
            }
            // past the minimum by a margin -> close (unlock); back out -> locked open at that width
            const target = railDragTarget(event.clientX - drag.left);
            if (target.collapsed === locked) {
                setCollapsed(target.collapsed);
            }
            if (!target.collapsed) {
                setWidth(target.width);
            }
        });
        const endDrag = event => {
            if (!drag) {
                return;
            }
            const { moved } = drag;
            drag = null;
            root.classList.remove("is-resizing");
            if (rail.hasPointerCapture?.(event.pointerId)) {
                rail.releasePointerCapture(event.pointerId);
            }
            if (!moved && event.type === "pointerup") {
                setLocked(!locked);
            }
        };
        rail.addEventListener("pointerup", endDrag);
        rail.addEventListener("pointercancel", endDrag);
        // keyboard: arrows resize, Enter/Space collapse
        rail.addEventListener("keydown", event => {
            const current = root.getBoundingClientRect().width;
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                event.preventDefault();
                const collapsed = !locked;
                if (event.key === "ArrowLeft" && (collapsed || current <= SIDEBAR_MIN_W)) {
                    setCollapsed(true); // already at the minimum: one more step collapses
                    return;
                }
                setCollapsed(false);
                setWidth(collapsed ? SIDEBAR_MIN_W : current + (event.key === "ArrowRight" ? 16 : -16));
            } else if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setLocked(!locked);
            }
        });
    }

    // X has no profile link yet: the icon stays, but doesn't navigate.
    root.querySelectorAll(".sb-social[data-placeholder]").forEach(link => {
        link.addEventListener("click", event => event.preventDefault());
    });
    tabs.forEach(tab => tab.addEventListener("click", () => setTab(tab.dataset.tab)));
    // arrow keys move between the two tabs (standard tablist pattern)
    root.querySelector(".sb-tabs").addEventListener("keydown", event => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
            return;
        }
        const i = tabs.findIndex(tab => tab.getAttribute("aria-selected") === "true");
        const next = tabs[(i + 1) % tabs.length];
        setTab(next.dataset.tab);
        next.focus();
    });
    links.forEach(link => link.addEventListener("click", () => onNavigate(link.dataset.page)));
    // the "D" mark opens the Home page
    root.querySelector(".sb-logo")?.addEventListener("click", event => onNavigate(event.currentTarget.dataset.page));

    try {
        locked = sessionStorage.getItem(LOCKED_KEY) === "1";
    } catch {
        locked = false;
    }
    applyState();
    let savedWidth = null;
    try {
        savedWidth = Number(sessionStorage.getItem(WIDTH_KEY)) || null;
    } catch {
        savedWidth = null;
    }
    if (savedWidth) {
        setWidth(savedWidth, false);
    }
    setTab("pages");

    // Inside the 3D viewer the closed rail shows the jobs (renderJobIcons) instead of the page icons.
    const jobIcons = document.getElementById("sb-job-icons");
    function setInViewer(on) {
        inViewer = Boolean(on);
        root.classList.toggle("in-viewer", inViewer);
    }

    // jobs: [{ id, label, title, active }] in creation order; onPick(id) opens one
    function renderJobIcons(jobs, onPick) {
        if (!jobIcons) {
            return;
        }
        jobIcons.replaceChildren(...jobs.map(job => {
            const b = document.createElement("button");
            b.type = "button";
            b.className = `sb-job-icon${job.active ? " is-active" : ""}`;
            b.textContent = job.label;
            b.title = job.title;
            b.setAttribute("aria-label", `Open ${job.title}`);
            if (job.active) {
                b.setAttribute("aria-current", "true");
            }
            b.addEventListener("click", () => onPick(job.id));
            return b;
        }));
        jobIcons.querySelector(".is-active")?.scrollIntoView({ block: "nearest" });
    }

    return { setTab, setActivePage, setJobsRunning, setCollapsed, setLocked, setInViewer, renderJobIcons, element: root };
}
