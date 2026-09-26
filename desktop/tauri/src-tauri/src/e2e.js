// TEST ONLY (DW2_E2E=1): drive the real UI inside the desktop shell and report.
// 1. Full UI run on a bundled tile: Library -> sentinel2-almora -> START GENERATION -> Studio.
// 2. Real generation via the same API for Maxar and DFC2019.
(async () => {
    const API = "http://127.0.0.1:8765";
    const out = { origin: location.origin, page_errors: [] };
    window.addEventListener("error", e => out.page_errors.push(`error: ${e.message ?? e.target?.src ?? e.type}`), true);
    window.addEventListener("unhandledrejection", e => out.page_errors.push(`rejection: ${e.reason?.message ?? e.reason}`));
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const until = async (fn, ms, what) => {
        const t = Date.now();
        while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(150); }
        throw new Error(`timed out: ${what}`);
    };
    const gen = async (kind, id) => {
        const t = performance.now();
        const r = await fetch(`${API}/api/generate/${kind}/${id}`, { method: "POST" });
        const j = await r.json();
        const m = j.meta ?? {};
        return { id, http: r.status, seconds: +((performance.now() - t) / 1000).toFixed(2), has_elevation: m.has_elevation,
                 terrain: m.terrain_source, surface: m.surface_source, how: m.how, terrain_range_m: m.terrain_range_m,
                 grid: m.grid, crs: m.crs, note: m.note, detail: j.detail };
    };
    try {
        const tab = await until(() => {
            const t = document.querySelector("#iv-tab-library");
            if (t && t.offsetParent) return t;
            if (document.querySelector("#input-view")?.hidden) {
                [...document.querySelectorAll("button")].find(b => /generate new/i.test(b.textContent))?.click();
            }
            return null;
        }, 60000, "library tab visible");
        tab.click();
        await until(() => document.querySelectorAll(".iv-card").length, 60000, "library cards");
        const cards = [...document.querySelectorAll(".iv-card")];
        out.first_cards_available = cards.slice(0, 26).every(c => !c.classList.contains("is-remote"));

        // 1. full UI run
        document.querySelector('.iv-card[data-id="sentinel2-almora"]').click();
        await until(() => !document.querySelector(".iv-start").disabled, 60000, "START enabled");
        const t1 = performance.now();
        document.querySelector(".iv-start").click();
        await until(() => !document.getElementById("final-demo-controls")?.hidden, 240000, "Studio ready");
        out.ui_seconds_to_studio = +((performance.now() - t1) / 1000).toFixed(1);
        await sleep(3000);
        const txt = id => document.getElementById(id)?.textContent.trim();
        const cap = id => document.querySelector(`#${id} .staged-caption`)?.innerText.replace(/\n+/g, " | ");
        out.ui = {
            depth: cap("depth-preview-box"), elevation: cap("elevation-preview-box"),
            dsm3d: cap("dsm-3d-box"), dem3d: cap("metric-elevation-3d-box"),
            studio: { elevation: txt("final-demo-elevation-value"), source: txt("final-demo-terrain-source-value"),
                      grid: txt("final-demo-stat-grid"), epsg: txt("final-demo-stat-epsg"), note: txt("xp-terrain-note") },
            assets_from_generation: [...document.querySelectorAll("#depth-preview-box img, #elevation-preview-box img")]
                .every(i => i.src.includes("/api/generated/")),
        };

        // 2. other input types through the same generation API
        // (uploads + live GLO-30 are tested against the same bundled backend with a real multipart upload;
        // a webview fetch of a GitHub release file is blocked by CORS, so it can't stand in for a file picker)
        out.generate = [await gen("library", "vhr-a_forest"), await gen("library", "dfc2019-OMA_315_020")];
        out.ok = true;
    } catch (e) {
        out.ok = false; out.error = String(e);
        const cap = id => document.querySelector(`#${id} .staged-caption`)?.innerText.replace(/\n+/g, " | ");
        out.state_at_failure = {
            depth: cap("depth-preview-box"), elevation: cap("elevation-preview-box"),
            dsm3d: cap("dsm-3d-box"), dem3d: cap("metric-elevation-3d-box"),
            studio_generating_hidden: document.getElementById("final-demo-generating")?.hidden,
            log: [...document.querySelectorAll("#calc-log-scroll .calc-log-line")].map(l => l.innerText).slice(-6),
        };
        // probe: can this webview load a generated PNG as a CORS image (what THREE.TextureLoader does)?
        const src = document.querySelector("#elevation-preview-box img")?.src;
        if (src && src.startsWith("http")) {
            out.texture_probe = await new Promise(res => {
                const im = new Image(); im.crossOrigin = "anonymous";
                im.onload = () => res(`loaded ${im.naturalWidth}x${im.naturalHeight}`); im.onerror = () => res("FAILED");
                im.src = src + (src.includes("?") ? "&" : "?") + "probe=1";
            });
            out.plain_image_probe = await new Promise(res => {
                const im = new Image(); im.onload = () => res("loaded"); im.onerror = () => res("FAILED"); im.src = src;
            });
        }
    }
    await window.__TAURI__.core.invoke("e2e_report", { result: JSON.stringify(out) });
})();
