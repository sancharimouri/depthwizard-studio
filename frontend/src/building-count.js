// Building count box (Demo page + Studio viewer) for the urban tiles: the buildings in the tile's own data.
// Precise where the source is an authoritative city model (one record per building); rough where the
// footprints are machine-detected from imagery (Google Open Buildings v3), shown rounded with "~".
// Counts: frontend/public/cityscape/connaught_place.json and library-static/cityscape-<key>/model.json
// (`buildings`), data/cityscape_src/<key>/buildings.json for the GLB models (2026-10-12).
const COUNTS = {
    connaught: { n: 1134, rough: true },
    "cityscape-bengaluru": { n: 393, rough: true },
    "cityscape-kolkata": { n: 1504, rough: true },
    "cityscape-hyderabad": { n: 4563, rough: true },
    "cityscape-mumbai": { n: 779, rough: true },
    "cityscape-new_york": { n: 626, rough: false }, // NYC 3-D Building Model (one BIN per building)
    "cityscape-shinjuku": { n: 2399, rough: false }, // Project PLATEAU
    "cityscape-amsterdam": { n: 2724, rough: false }, // 3D BAG (the national building register)
};

export function buildingCountText(key) {
    const c = COUNTS[key];
    if (!c) {
        return null;
    }
    return { value: c.rough ? `~${(Math.round(c.n / 10) * 10).toLocaleString("en-US")}` : c.n.toLocaleString("en-US") };
}

// Shows the box for an urban tile with a count, hides it otherwise.
export function showBuildingCount(boxId, key) {
    const box = document.getElementById(boxId);
    if (!box) {
        return;
    }
    const text = buildingCountText(key);
    box.hidden = !text;
    if (text) {
        box.querySelector(".building-count-value").textContent = text.value;
    }
}
