// TEST ONLY (DW2_E2E=1): drive the real UI inside the desktop shell and report.
(async () => {
    const API = "http://127.0.0.1:8765";
    const out = { origin: location.origin, steps: [] };
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const until = async (fn, ms, what) => {
        const t = Date.now();
        while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(100); }
        throw new Error(`timed out: ${what}`);
    };
    const card = id => document.querySelector(`.iv-card[data-id="${id}"]`);
    const timeDepth = async id => {
        const t = performance.now();
        const r = await fetch(`${API}/api/depth/relative/library/${id}`, { method: "POST" });
        const j = await r.json();
        return { id, http: r.status, seconds: +((performance.now() - t) / 1000).toFixed(3), infer_s: j.infer_s, shape: j.shape };
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
        out.cards = { total: cards.length, on_demand: cards.filter(c => c.classList.contains("is-remote")).length };
        out.on_demand_by_collection = {};
        for (const c of cards.filter(c => c.classList.contains("is-remote"))) {
            const k = c.dataset.id.split("-")[0]; out.on_demand_by_collection[k] = (out.on_demand_by_collection[k] || 0) + 1;
        }

        // 1. on-demand DFC2019 without a token
        card("dfc2019-JAX_004_006").click();
        await until(() => {
            const c = card("dfc2019-JAX_004_006");
            return c && (c.classList.contains("is-error") || !c.classList.contains("is-remote"));
        }, 120000, "dfc2019 download result");
        const dc = card("dfc2019-JAX_004_006");
        out.dfc2019_on_demand = dc.classList.contains("is-error")
            ? `refused: "${dc.querySelector(".iv-card-download-label").textContent}" (${dc.title})`
            : "downloaded";

        // 2. on-demand Sentinel-2 through the real overlay (public GitHub Release)
        const t0 = performance.now();
        card("sentinel2-bengaluru").click();
        await until(() => card("sentinel2-bengaluru") && !card("sentinel2-bengaluru").classList.contains("is-remote"), 180000, "bengaluru download");
        out.sentinel2_on_demand_download_s = +((performance.now() - t0) / 1000).toFixed(2);

        // 3. full UI flow on a bundled tile
        card("sentinel2-almora").click();
        await until(() => !document.querySelector(".iv-start").disabled, 60000, "START enabled");
        const t1 = performance.now();
        document.querySelector(".iv-start").click();
        await until(() => {
            const c = document.querySelector("#depth-preview-box .preview-content");
            const i = document.querySelector("#depth-preview-box .preview-image");
            return c && !c.hidden && i.complete && i.naturalWidth > 0 && i.src.startsWith("blob:");
        }, 180000, "depth box");
        out.ui_bundled_seconds_to_depth = +((performance.now() - t1) / 1000).toFixed(2);
        await sleep(2500);
        const img = document.querySelector("#depth-preview-box .preview-image");
        out.ui_depth = { size: [img.naturalWidth, img.naturalHeight],
                         caption: document.querySelector("#depth-preview-box .staged-caption").innerText.replace(/\n+/g, " | ") };

        // 4. depth-by-id timing: bundled vs the tile just downloaded on demand (warm backend, 3 runs each)
        out.depth_timing = [];
        for (const id of ["sentinel2-almora", "sentinel2-bengaluru", "dfc2019-OMA_315_020", "vhr-a_forest"]) {
            for (let k = 0; k < 3; k += 1) out.depth_timing.push(await timeDepth(id));
        }
        out.ok = true;
    } catch (e) {
        out.ok = false; out.error = String(e);
    }
    await window.__TAURI__.core.invoke("e2e_report", { result: JSON.stringify(out) });
})();
