// Real-world measurements for the 3D measurement tool (deliverables-audit
// item 6: slope assessment / structural height analysis).
//
// Pure math, no THREE/DOM, so it is unit-testable in Node. Every value comes
// from the RAW DEM grid in terrain.json (`heights` normalized 0-1, scaled by
// elevationMin/Max) — never from the display mesh, which is median-filtered,
// slope-capped and vertically exaggerated for viewing.
//
// Coordinates are fractional grid pixels (col = x, row = y) of that grid.
// Vertex (0,0) is the north-west corner; (width-1, height-1) the south-east.
// Metres use the same local equirectangular conversion as terrain.js
// (111,320 m per degree of latitude, scaled by cos(mid-latitude) for
// longitude) — accurate to well under 0.1 % over a 10 km tile.

const METERS_PER_DEG_LAT = 111320;

export function createGeoGrid(grid) {
    const { width, height, heights, bounds, elevationMin, elevationMax } = grid;
    const latMid = ((bounds.north + bounds.south) / 2) * (Math.PI / 180);
    const metersPerDegLon = METERS_PER_DEG_LAT * Math.cos(latMid);
    const dLon = (bounds.east - bounds.west) / (width - 1);
    const dLat = (bounds.north - bounds.south) / (height - 1);
    const range = elevationMax - elevationMin;

    const geo = {
        width,
        height,
        // metres per grid pixel (x = east-west, y = north-south)
        pixelSizeX: dLon * metersPerDegLon,
        pixelSizeY: dLat * METERS_PER_DEG_LAT,

        inGrid(col, row) {
            return col >= 0 && row >= 0 && col <= width - 1 && row <= height - 1;
        },

        lonLat(col, row) {
            return { lon: bounds.west + col * dLon, lat: bounds.north - row * dLat };
        },

        // Bilinear DEM elevation (m) at a fractional grid position.
        elevation(col, row) {
            const c = Math.min(Math.max(col, 0), width - 1);
            const r = Math.min(Math.max(row, 0), height - 1);
            const ix = Math.min(Math.floor(c), width - 2);
            const iy = Math.min(Math.floor(r), height - 2);
            const fx = c - ix;
            const fy = r - iy;
            const h00 = heights[iy * width + ix];
            const h10 = heights[iy * width + ix + 1];
            const h01 = heights[(iy + 1) * width + ix];
            const h11 = heights[(iy + 1) * width + ix + 1];
            const h = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy;
            return elevationMin + h * range;
        },

        // Local metric position (east, north) in metres from the NW corner.
        toMeters(col, row) {
            return { e: col * geo.pixelSizeX, n: -row * geo.pixelSizeY };
        },
    };
    return geo;
}

// One straight segment between two grid points.
//   horizontal: ground distance, ignoring height (m)
//   rise:       end elevation − start elevation (m; negative = descent)
//   gradient:   rise / horizontal, as a percent (null for a zero-length segment)
//   slopeDeg:   the same, as an angle in degrees
//   chord:      straight 3D distance between the two points (m)
//   surface:    length following the DEM surface along the segment (m),
//               sampled at ≤ 0.5 grid pixel — what a walker would cover
//   profile:    [{ d (m from start), elev (m) }] along that path
export function measureSegment(geo, p0, p1) {
    const a = geo.toMeters(p0.x, p0.y);
    const b = geo.toMeters(p1.x, p1.y);
    const horizontal = Math.hypot(b.e - a.e, b.n - a.n);
    const e0 = geo.elevation(p0.x, p0.y);
    const e1 = geo.elevation(p1.x, p1.y);
    const rise = e1 - e0;

    const pixels = Math.hypot(p1.x - p0.x, p1.y - p0.y);
    const n = Math.max(1, Math.ceil(pixels * 2));
    const profile = [];
    let surface = 0;
    let prev = null;
    for (let i = 0; i <= n; i++) {
        const t = i / n;
        const elev = geo.elevation(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t);
        const d = horizontal * t;
        if (prev) {
            surface += Math.hypot(d - prev.d, elev - prev.elev);
        }
        prev = { d, elev };
        profile.push(prev);
    }

    return {
        horizontal,
        rise,
        gradient: horizontal > 0 ? (rise / horizontal) * 100 : null,
        slopeDeg: horizontal > 0 ? (Math.atan2(rise, horizontal) * 180) / Math.PI : null,
        chord: Math.hypot(horizontal, rise),
        surface,
        startElevation: e0,
        endElevation: e1,
        profile,
    };
}

// A whole chain: open polyline (≥ 2 points) or closed polygon (≥ 3 points).
export function measureChain(geo, points, closed) {
    const segs = [];
    const count = closed ? points.length : points.length - 1;
    for (let i = 0; i < count; i++) {
        segs.push(measureSegment(geo, points[i], points[(i + 1) % points.length]));
    }
    const sum = key => segs.reduce((acc, s) => acc + s[key], 0);
    const result = {
        type: closed ? "polygon" : points.length === 2 ? "segment" : "polyline",
        points: points.map(p => ({
            x: p.x,
            y: p.y,
            ...geo.lonLat(p.x, p.y),
            elevation: geo.elevation(p.x, p.y),
        })),
        segments: segs,
        horizontal: sum("horizontal"),
        surface: sum("surface"),
        ascent: segs.reduce((acc, s) => acc + Math.max(0, s.rise), 0),
        descent: segs.reduce((acc, s) => acc + Math.max(0, -s.rise), 0),
    };
    if (points.length >= 2) {
        const first = geo.elevation(points[0].x, points[0].y);
        const last = geo.elevation(points[points.length - 1].x, points[points.length - 1].y);
        result.netRise = closed ? 0 : last - first;
        result.gradient = !closed && result.horizontal > 0 ? (result.netRise / result.horizontal) * 100 : null;
    }
    if (closed && points.length >= 3) {
        // Shoelace formula on the metric (east, north) coordinates.
        let twice = 0;
        for (let i = 0; i < points.length; i++) {
            const a = geo.toMeters(points[i].x, points[i].y);
            const b = geo.toMeters(points[(i + 1) % points.length].x, points[(i + 1) % points.length].y);
            twice += a.e * b.n - b.e * a.n;
        }
        result.area = Math.abs(twice) / 2;
        result.perimeter = result.horizontal;
        const elevs = result.points.map(p => p.elevation);
        result.minElevation = Math.min(...elevs);
        result.maxElevation = Math.max(...elevs);
    }
    return result;
}
