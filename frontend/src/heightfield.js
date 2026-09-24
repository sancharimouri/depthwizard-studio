// Exact ray → terrain hit for the PlaneGeometry heightfield built by
// terrain.js, without THREE's brute-force per-triangle raycast (~233k
// triangles for a 361×325 grid — too slow to run on every mouse move).
//
// Pure math, no THREE import, so it is unit-testable in Node. Works in the
// terrain mesh's LOCAL frame (before its -90° X rotation):
//   x ∈ [-terrainWidth/2, +terrainWidth/2]   (west → east)
//   y ∈ [+terrainHeight/2, -terrainHeight/2] (north → south; grid row 0 is +y)
//   z = current vertex height (0 when a flat layer is shown)
//
// Surface interpolation matches THREE.PlaneGeometry's triangulation
// exactly, so a hit here is the same point THREE.Raycaster would return:
// quad (ix, iy) has corners a=(ix,iy) b=(ix,iy+1) c=(ix+1,iy+1) d=(ix+1,iy)
// and triangles (a,b,d) [fx+fy ≤ 1] and (b,c,d) [fx+fy > 1], where
// fx/fy are the fractional column/row inside the quad.

// field = {
//   width, height,                 // grid vertices per row / rows
//   terrainWidth, terrainHeight,   // local extent (terrainHeight is 100)
//   getZ(index),                   // current local z of vertex `index`
//   zMin, zMax,                    // bounds of getZ over the grid
//   triangleValid(i0, i1, i2)?     // optional: false if the mesh has no such triangle (eroded edge)
// }

export function gridToLocalXY(field, col, row) {
    return {
        x: (col / (field.width - 1) - 0.5) * field.terrainWidth,
        y: (0.5 - row / (field.height - 1)) * field.terrainHeight,
    };
}

export function localXYToGrid(field, x, y) {
    return {
        col: (x / field.terrainWidth + 0.5) * (field.width - 1),
        row: (0.5 - y / field.terrainHeight) * (field.height - 1),
    };
}

// Triangle the point (col,row) falls in + its interpolated z, or null when
// outside the grid or on a triangle the (eroded) mesh doesn't contain.
export function surfaceAt(field, col, row) {
    const { width, height } = field;
    if (!(col >= 0 && row >= 0 && col <= width - 1 && row <= height - 1)) {
        return null;
    }
    const ix = Math.min(Math.floor(col), width - 2);
    const iy = Math.min(Math.floor(row), height - 2);
    const fx = col - ix;
    const fy = row - iy;

    const a = ix + width * iy;
    const b = ix + width * (iy + 1);
    const c = ix + 1 + width * (iy + 1);
    const d = ix + 1 + width * iy;

    if (fx + fy <= 1) {
        if (field.triangleValid && !field.triangleValid(a, b, d)) {
            return null;
        }
        const za = field.getZ(a);
        return { z: za + fx * (field.getZ(d) - za) + fy * (field.getZ(b) - za), triangle: [a, b, d] };
    }
    if (field.triangleValid && !field.triangleValid(b, c, d)) {
        return null;
    }
    const zc = field.getZ(c);
    return { z: zc + (1 - fx) * (field.getZ(b) - zc) + (1 - fy) * (field.getZ(d) - zc), triangle: [b, c, d] };
}

// Möller–Trumbore ray/triangle test (double-sided, like the terrain's
// DoubleSide material). Returns the ray parameter t or null.
function intersectTriangle(o, d, p0, p1, p2) {
    const e1x = p1.x - p0.x, e1y = p1.y - p0.y, e1z = p1.z - p0.z;
    const e2x = p2.x - p0.x, e2y = p2.y - p0.y, e2z = p2.z - p0.z;
    const px = d.y * e2z - d.z * e2y;
    const py = d.z * e2x - d.x * e2z;
    const pz = d.x * e2y - d.y * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < 1e-12) {
        return null;
    }
    const inv = 1 / det;
    const tx = o.x - p0.x, ty = o.y - p0.y, tz = o.z - p0.z;
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < -1e-9 || u > 1 + 1e-9) {
        return null;
    }
    const qx = ty * e1z - tz * e1y;
    const qy = tz * e1x - tx * e1z;
    const qz = tx * e1y - ty * e1x;
    const v = (d.x * qx + d.y * qy + d.z * qz) * inv;
    if (v < -1e-9 || u + v > 1 + 1e-9) {
        return null;
    }
    const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    return t >= 0 ? t : null;
}

// origin/dir: local-frame ray ({x,y,z}; dir need not be normalized).
// Returns { col, row, x, y, z, t } for the FIRST surface intersection, or null.
//
// Exact: walks the grid cells the ray's xy-projection crosses, in ray order
// (Amanatides–Woo DDA), and intersects each cell's two real triangles
// analytically. (A fixed-step march can step over a grazing ridge crossing.)
export function raycastHeightfield(field, origin, dir) {
    const { width, height } = field;
    const halfW = field.terrainWidth / 2;
    const halfH = field.terrainHeight / 2;
    const eps = 1e-12;

    // Clip to the box [x] × [y] × [zMin, zMax] (padded) to bound the walk.
    let t0 = 0;
    let t1 = Infinity;
    const slabs = [
        [origin.x, dir.x, -halfW, halfW],
        [origin.y, dir.y, -halfH, halfH],
        [origin.z, dir.z, field.zMin - 1e-3, field.zMax + 1e-3],
    ];
    for (const [o, d, lo, hi] of slabs) {
        if (Math.abs(d) < eps) {
            if (o < lo || o > hi) {
                return null;
            }
            continue;
        }
        let ta = (lo - o) / d;
        let tb = (hi - o) / d;
        if (ta > tb) {
            [ta, tb] = [tb, ta];
        }
        t0 = Math.max(t0, ta);
        t1 = Math.min(t1, tb);
        if (t0 > t1) {
            return null;
        }
    }

    // Ray in grid (col,row) space.
    const g0 = localXYToGrid(field, origin.x + dir.x * t0, origin.y + dir.y * t0);
    const dc = (dir.x / field.terrainWidth) * (width - 1);
    const dr = (-dir.y / field.terrainHeight) * (height - 1);

    let ix = Math.min(Math.max(Math.floor(g0.col), 0), width - 2);
    let iy = Math.min(Math.max(Math.floor(g0.row), 0), height - 2);
    const stepX = dc > 0 ? 1 : -1;
    const stepY = dr > 0 ? 1 : -1;
    // t (in ray units, from t0) at which the walk crosses the next column/row line
    const nextLine = (g, i, step, dg) => {
        if (Math.abs(dg) < eps) {
            return Infinity;
        }
        const boundary = step > 0 ? i + 1 : i;
        return t0 + (boundary - g) / dg;
    };
    let tMaxX = nextLine(g0.col, ix, stepX, dc);
    let tMaxY = nextLine(g0.row, iy, stepY, dr);
    const tDeltaX = Math.abs(dc) < eps ? Infinity : 1 / Math.abs(dc);
    const tDeltaY = Math.abs(dr) < eps ? Infinity : 1 / Math.abs(dr);

    const vertex = index => {
        const col = index % width;
        const row = (index - col) / width;
        const xy = gridToLocalXY(field, col, row);
        return { x: xy.x, y: xy.y, z: field.getZ(index) };
    };

    for (let guard = 0; guard < 4 * (width + height); guard++) {
        const a = ix + width * iy;
        const b = ix + width * (iy + 1);
        const c = ix + 1 + width * (iy + 1);
        const d = ix + 1 + width * iy;
        let best = null;
        for (const [i0, i1, i2] of [[a, b, d], [b, c, d]]) {
            if (field.triangleValid && !field.triangleValid(i0, i1, i2)) {
                continue;
            }
            const t = intersectTriangle(origin, dir, vertex(i0), vertex(i1), vertex(i2));
            if (t !== null && (best === null || t < best)) {
                best = t;
            }
        }
        if (best !== null) {
            const x = origin.x + dir.x * best;
            const y = origin.y + dir.y * best;
            const g = localXYToGrid(field, x, y);
            return { col: g.col, row: g.row, x, y, z: origin.z + dir.z * best, t: best };
        }

        // advance to the next cell along the ray
        if (tMaxX < tMaxY) {
            if (tMaxX > t1) {
                return null;
            }
            ix += stepX;
            tMaxX += tDeltaX;
        } else {
            if (tMaxY > t1) {
                return null;
            }
            iy += stepY;
            tMaxY += tDeltaY;
        }
        if (ix < 0 || iy < 0 || ix > width - 2 || iy > height - 2) {
            return null;
        }
    }
    return null;
}
