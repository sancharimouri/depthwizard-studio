// TEST ONLY (DW2_E2E=1): drive the real UI inside the desktop shell and report.
(async () => {
    const out = { origin: location.origin, steps: [] };
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const until = async (fn, ms, what) => {
        const t = Date.now();
        while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(200); }
        throw new Error(`timed out: ${what}`);
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
        out.steps.push(`library tab after ${Math.round(performance.now())} ms`);
        tab.click();
        await until(() => document.querySelectorAll(".iv-card").length, 60000, "library cards");
        out.library_cards = document.querySelectorAll(".iv-card").length;
        const card = await until(() => document.querySelector('.iv-card[data-id="sentinel2-almora"]'), 10000, "almora card");
        card.click();
        await until(() => !document.querySelector(".iv-start").disabled, 60000, "START enabled");
        const t0 = performance.now();
        document.querySelector(".iv-start").click();
        await until(() => {
            const c = document.querySelector("#depth-preview-box .preview-content");
            const i = document.querySelector("#depth-preview-box .preview-image");
            return c && !c.hidden && i.complete && i.naturalWidth > 0;
        }, 180000, "depth box");
        out.seconds_to_depth = +((performance.now() - t0) / 1000).toFixed(1);
        await sleep(2500);
        const box = document.querySelector("#depth-preview-box");
        const img = box.querySelector(".preview-image");
        const cv = document.createElement("canvas"); cv.width = img.naturalWidth; cv.height = img.naturalHeight;
        const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, cv.width, cv.height).data; let lo = 255, hi = 0; const vals = new Set();
        for (let k = 0; k < d.length; k += 4 * 97) { lo = Math.min(lo, d[k]); hi = Math.max(hi, d[k]); vals.add(d[k]); }
        Object.assign(out, {
            depth_img: img.src.slice(0, 24), size: [img.naturalWidth, img.naturalHeight], gray: [lo, hi], distinct_sampled: vals.size,
            caption: box.querySelector(".staged-caption").innerText.replace(/\n+/g, " | "),
            log: [...document.querySelectorAll("#calc-log-scroll .calc-log-line")].map(l => l.innerText).filter(t => /Relative depth/.test(t)),
            depth_requests: performance.getEntriesByType("resource").map(e => e.name).filter(n => n.includes("/api/depth/")),
        });
        out.ok = true;
    } catch (e) {
        out.ok = false; out.error = String(e);
    }
    await window.__TAURI__.core.invoke("e2e_report", { result: JSON.stringify(out) });
})();
