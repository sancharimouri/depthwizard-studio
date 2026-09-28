// Global sidebar: the one navigation surface on every page (Workbench and its
// processing grid, Explore, Docs, the expanded 3D viewer window). Structured
// after shadcn/ui's sidebar-07 block:
//
//   toggle  PanelLeft button just outside the bar, riding its right edge
//   header  the "D" mark (faint white glow) + "Depth Wizard"
//   tabs    PAGES | JOBS (equal width, muted green underline on the active one)
//   PAGES   DW Studio (page-workbench) / Demo / Home (page-docs) with icons + "N jobs running"
//   JOBS    the jobs list (rendered by main.js into the same element ids)
//   footer  GitHub / email / LinkedIn / X links
//   rail    drag to resize (SIDEBAR_MIN_W–SIDEBAR_MAX_W), click to collapse
//
// Collapsed, it is an icon rail: the D, the page icons and the link icons.
//
// Expanded by default; the collapsed/expanded choice is remembered for the
// session (sessionStorage). Default tab: PAGES, except JOBS inside the 3D
// viewer window — applied when entering/leaving the viewer or changing page;
// the user can switch tabs any time in between.
//
// The sidebar is a flex sibling of #app-main, so it pushes the content
// (width animates over 200 ms) rather than overlaying it.

const COLLAPSED_KEY = "dw2.sidebarCollapsed";
const WIDTH_KEY = "dw2.sidebarWidth";
export const SIDEBAR_MIN_W = 208;
export const SIDEBAR_MAX_W = 360;
const DRAG_THRESHOLD_PX = 4;

export function clampSidebarWidth(px) {
    return Math.round(Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, px)));
}

export function createSidebar({ onNavigate }) {
    const root = document.getElementById("app-sidebar");
    const toggle = document.getElementById("sb-toggle");
    const tabs = [...root.querySelectorAll(".sb-tab")];
    const panels = {
        pages: document.getElementById("sb-panel-pages"),
        jobs: document.getElementById("sb-panel-jobs"),
    };
    const links = [...root.querySelectorAll(".sb-page-link")];
    const runningEl = document.getElementById("sb-jobs-running");

    function setCollapsed(collapsed) {
        root.classList.toggle("is-collapsed", collapsed);
        toggle.setAttribute("aria-expanded", String(!collapsed));
        const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
        toggle.setAttribute("aria-label", label);
        toggle.title = label;
        try {
            sessionStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
        } catch {
            // storage blocked: the choice just isn't remembered
        }
    }

    function setTab(name) {
        tabs.forEach(tab => {
            const on = tab.dataset.tab === name;
            tab.setAttribute("aria-selected", String(on));
            tab.tabIndex = on ? 0 : -1;
        });
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

    toggle.addEventListener("click", () => setCollapsed(!root.classList.contains("is-collapsed")));

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
                if (root.classList.contains("is-collapsed")) {
                    setCollapsed(false);
                }
            }
            setWidth(event.clientX - drag.left);
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
                setCollapsed(!root.classList.contains("is-collapsed"));
            }
        };
        rail.addEventListener("pointerup", endDrag);
        rail.addEventListener("pointercancel", endDrag);
        // keyboard: arrows resize, Enter/Space collapse
        rail.addEventListener("keydown", event => {
            const current = root.getBoundingClientRect().width;
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                event.preventDefault();
                setCollapsed(false);
                setWidth(current + (event.key === "ArrowRight" ? 16 : -16));
            } else if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setCollapsed(!root.classList.contains("is-collapsed"));
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

    let collapsed = false;
    try {
        collapsed = sessionStorage.getItem(COLLAPSED_KEY) === "1";
    } catch {
        collapsed = false;
    }
    setCollapsed(collapsed);
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

    return { setTab, setActivePage, setJobsRunning, setCollapsed, element: root };
}
