// Global sidebar: the one navigation surface on every page (Workbench and its
// processing grid, Explore, Docs, the expanded 3D viewer window).
//
//   header  PanelLeft toggle + "DEPTH WIZARD"; collapses to a slim rail
//   tabs    PAGES | JOBS (equal width, muted green underline on the active one)
//   PAGES   Workbench / Explore / Docs + "N jobs running"
//   JOBS    the jobs list (rendered by main.js into the same element ids)
//
// Expanded by default; the collapsed/expanded choice is remembered for the
// session (sessionStorage). Default tab: PAGES, except JOBS inside the 3D
// viewer window — applied when entering/leaving the viewer or changing page;
// the user can switch tabs any time in between.
//
// The sidebar is a flex sibling of #app-main, so it pushes the content
// (width animates over 200 ms) rather than overlaying it.

const COLLAPSED_KEY = "dw2.sidebarCollapsed";

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

    toggle.addEventListener("click", () => setCollapsed(!root.classList.contains("is-collapsed")));
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
    setTab("pages");

    return { setTab, setActivePage, setJobsRunning, setCollapsed, element: root };
}
