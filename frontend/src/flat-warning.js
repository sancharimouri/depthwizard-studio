// Flat-terrain warning: a 10 m Sentinel-2 tile of farmland, coast or city is
// almost flat, so its 3D structure can look like a plane at 1x. The warning says
// so, with the tile's real footprint, and points at the two things that help
// (zoom, vertical exaggeration). Hilly tiles, VHR and DFC2019 never get it.

const LAND = {
    agricultural: "mostly flat farmland",
    coastal: "mostly flat coastal land",
    urban: "a mostly flat urban area",
    // a searched scene has no terrain class; it is judged by its generated relief
    lowRelief: "mostly flat, low-relief land",
};

// Relief (surface max − min) under which a searched Sentinel-2 scene counts as flat.
export const FLAT_RELIEF_M = 100;

// Approximate footprint in km from a WGS84 bbox [w, s, e, n].
export function footprintKmFromBbox([w, s, e, n]) {
    const lat = ((s + n) / 2) * Math.PI / 180;
    return [Math.abs(e - w) * 111.32 * Math.cos(lat), Math.abs(n - s) * 110.57];
}

function km(value) {
    return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

// landscape: { collection, terrain, footprintKm: [x, y], gsdM } from the input.
// reliefM: the generated surface relief, when known (null before generation).
// Returns { title, body } or null.
export function flatTerrainWarning(landscape, reliefM = null) {
    if (!landscape || landscape.collection !== "sentinel2" || !landscape.footprintKm) {
        return null;
    }
    let land = LAND[landscape.terrain];
    if (!landscape.terrain && reliefM != null && reliefM < FLAT_RELIEF_M) {
        land = LAND.lowRelief;
    }
    if (!land) {
        return null;
    }
    const [x, y] = landscape.footprintKm;
    const gsd = landscape.gsdM ? `${Math.round(landscape.gsdM)} m` : "10 m";
    return {
        title: "Low-relief scene",
        body: `${gsd} Sentinel-2 over ${km(x)} × ${km(y)} km of ${land}. `
            + "Zoom in or raise vertical exaggeration to see any relief.",
    };
}
