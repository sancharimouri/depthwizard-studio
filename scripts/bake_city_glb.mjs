#!/usr/bin/env node
// Bake a city tile's buildings from a 3D city model into one compressed GLB (meshopt), in the city model's local
// metres: x east, y up, z = -north, origin = the box centre on its UTM grid (as scripts/bake_cityscape.py), y measured
// from a reference height written next to it. Buildings whose centre is inside the box are kept, clipped to it (as the
// footprint models are; the cut leaves the wall open on that side).
//
//   plateau  PLATEAU (MLIT) 3D Tiles: LoD2 with textures where a building has it, otherwise LoD1 (same gml_id).
//   3dbag    3D BAG (TU Delft 3D geoinformation / 3DGI), LoD2.2 solids from its OGC API (CityJSON), NAP heights.
//   nyc      NYC 3-D Building Model (OTI, CityGML LoD2, State Plane Long Island ft, NAVD88 ft), streamed from the
//            delivery-area zip (data/cityscape_src/<key>/DA_WISE_GML.zip, downloaded by hand: see city_tiles.json).
// Untextured models carry COLOR_0 = height within the building (0 foot .. 1 roof) for the viewer's palette.
//
// Writes frontend/public/cityscape/<key>.glb, and data/cityscape_src/<key>/buildings.json (per building: id, LoD,
// base and top heights, roof triangles) for scripts/bake_cityscape.py. Downloads are cached in data/cityscape_src/<key>/.
//
//   node scripts/bake_city_glb.mjs shinjuku [--tex 576] [--quality 70]

import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "tools", "package.json")); // tools/package.json: cd tools && npm ci
const { Document, NodeIO } = require("@gltf-transform/core");
const { ALL_EXTENSIONS } = require("@gltf-transform/extensions");
const { dedup, meshopt, prune, textureCompress, weld } = require("@gltf-transform/functions");
const draco3d = require("draco3dgltf");
const { MeshoptEncoder } = require("meshoptimizer");
const sharp = require("sharp");
const proj4 = require("proj4");

const args = process.argv.slice(2);
const KEY = args[0];
const opt = (name, dflt) => (args.includes(`--${name}`) ? Number(args[args.indexOf(`--${name}`) + 1]) : dflt);
const TEX = opt("tex", 576);       // px, longest side of each texture atlas after resizing
const QUALITY = opt("quality", 70); // JPEG quality of the resized atlases
const cfg = JSON.parse(readFileSync(join(ROOT, "scripts", "city_tiles.json"), "utf8"))[KEY];
if (!["plateau", "3dbag", "nyc"].includes(cfg?.buildings)) {
    console.error(`usage: node scripts/bake_city_glb.mjs <key>  (a city_tiles.json entry with a GLB source)`);
    process.exit(2);
}
const CACHE = join(ROOT, "data", "cityscape_src", KEY);
const OUT = join(ROOT, "frontend", "public", "cityscape", `${KEY}.glb`);
mkdirSync(CACHE, { recursive: true });

proj4.defs("EPSG:4978", "+proj=geocent +datum=WGS84 +units=m +no_defs");
const zone = Number(cfg.utm.split(":")[1]) - 32600;
proj4.defs(cfg.utm, `+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs`);
const toUtm = proj4("EPSG:4326", cfg.utm);
const ecefToLl = proj4("EPSG:4978", "EPSG:4326");
const [clat, clon] = cfg.centre;
const [cx, cy] = toUtm.forward([clon, clat]).map(v => Math.round(v / 10) * 10); // 10 m grid, as bake_cityscape.py
const HALF = cfg.half;
// lon/lat envelope of the box (radians), for picking tiles; padded so buildings straddling the edge are seen
const toLl = proj4(cfg.utm, "EPSG:4326");
const pad = 150;
const env = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => toLl.forward([cx + sx * (HALF + pad), cy + sy * (HALF + pad)]));
const rad = d => (d * Math.PI) / 180;
const [W, E] = [Math.min(...env.map(p => p[0])), Math.max(...env.map(p => p[0]))].map(rad);
const [S, N] = [Math.min(...env.map(p => p[1])), Math.max(...env.map(p => p[1]))].map(rad);

async function fetchCached(url) {
    const file = join(CACHE, createHash("sha1").update(url).digest("hex").slice(0, 16) + "_" + url.split("/").pop());
    if (existsSync(file) && statSync(file).size > 0) return readFileSync(file);
    for (let attempt = 0; ; attempt++) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
            const buf = Buffer.from(await res.arrayBuffer());
            writeFileSync(file, buf);
            return buf;
        } catch (err) {
            if (attempt === 3) throw err;
            await new Promise(r => setTimeout(r, 5000 * (attempt + 1)));
        }
    }
}

// Leaf tiles (REPLACE refinement: the leaves hold the full detail) whose region meets the box envelope.
async function leafTiles(tilesetUrl) {
    const ts = JSON.parse(await fetchCached(tilesetUrl));
    const base = tilesetUrl.slice(0, tilesetUrl.lastIndexOf("/") + 1);
    const out = [];
    const walk = node => {
        const r = node.boundingVolume.region;
        if (!r || r[2] < W || r[0] > E || r[3] < S || r[1] > N) return;
        if (!node.children?.length) {
            if (node.content?.uri) out.push(base + node.content.uri);
        } else node.children.forEach(walk);
    };
    walk(ts.root);
    return out;
}

// b3dm -> { batch table, glb bytes without CESIUM_RTC, rtc centre }
function parseB3dm(buf) {
    const [ftj, ftb, btj, btb] = [12, 16, 20, 24].map(o => buf.readUInt32LE(o));
    let o = 28;
    const ft = JSON.parse(buf.subarray(o, o + ftj).toString());
    o += ftj + ftb;
    const bt = btj ? JSON.parse(buf.subarray(o, o + btj).toString()) : {};
    o += btj + btb;
    const glb = buf.subarray(o);
    const jsonLen = glb.readUInt32LE(12);
    const json = JSON.parse(glb.subarray(20, 20 + jsonLen).toString());
    const rtc = json.extensions?.CESIUM_RTC?.center ?? ft.RTC_CENTER ?? [0, 0, 0];
    delete json.extensions?.CESIUM_RTC;
    for (const k of ["extensionsUsed", "extensionsRequired"]) {
        if (json[k]) json[k] = json[k].filter(e => e !== "CESIUM_RTC");
    }
    let text = Buffer.from(JSON.stringify(json));
    text = Buffer.concat([text, Buffer.alloc((4 - (text.length % 4)) % 4, 0x20)]);
    const rest = glb.subarray(20 + jsonLen);
    const head = Buffer.alloc(20);
    head.writeUInt32LE(0x46546c67, 0);
    head.writeUInt32LE(2, 4);
    head.writeUInt32LE(20 + text.length + rest.length, 8);
    head.writeUInt32LE(text.length, 12);
    head.writeUInt32LE(0x4e4f534a, 16);
    return { bt, rtc, glb: new Uint8Array(Buffer.concat([head, text, rest])), n: ft.BATCH_LENGTH };
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    "draco3d.decoder": await draco3d.createDecoderModule(),
    "meshopt.encoder": MeshoptEncoder,
});

// ECEF -> local (x east, y north, h ellipsoidal) through UTM; normals through the box centre's ENU frame
const [sl, cl, sp, cp] = [Math.sin(rad(clon)), Math.cos(rad(clon)), Math.sin(rad(clat)), Math.cos(rad(clat))];
const enu = (x, y, z) => [-sl * x + cl * y, -sp * cl * x - sp * sl * y + cp * z, cp * cl * x + cp * sl * y + sp * z];

const buildings = new Map(); // gml_id -> { lod, parts: [{ material key, pos[], nrm[], uv[] }], meta }
const materials = new Map(); // key -> { image, mime } | { colour }

async function readTile(url, lod) {
    const { bt, rtc, glb, n } = parseB3dm(await fetchCached(url));
    const doc = await io.readBinary(glb);
    const ids = bt.gml_id ?? [];
    const found = new Map(); // batch id -> part accumulators, only for gml_ids not already taken by a better LoD
    for (const node of doc.getRoot().listNodes()) {
        const mesh = node.getMesh();
        if (!mesh) continue;
        const m = node.getWorldMatrix();
        for (const prim of mesh.listPrimitives()) {
            const P = prim.getAttribute("POSITION").getArray();
            const Nn = prim.getAttribute("NORMAL")?.getArray();
            const B = prim.getAttribute("_BATCHID").getArray();
            const T = prim.getAttribute("TEXCOORD_0")?.getArray();
            const I = prim.getIndices()?.getArray() ?? Uint32Array.from({ length: P.length / 3 }, (_, i) => i);
            const mat = prim.getMaterial();
            const tex = mat?.getBaseColorTexture();
            const matKey = tex ? `${url}#${doc.getRoot().listTextures().indexOf(tex)}` : `colour:${(mat?.getBaseColorFactor() ?? [1, 1, 1, 1]).slice(0, 3).map(v => v.toFixed(3)).join(",")}`;
            if (tex && !materials.has(matKey)) materials.set(matKey, { image: tex.getImage(), mime: tex.getMimeType() });
            if (!tex && !materials.has(matKey)) materials.set(matKey, { colour: true });
            // positions: glTF (y up) -> 3D Tiles (z up), + RTC, -> lon/lat/h -> UTM local
            const local = new Float64Array(P.length);
            for (let i = 0; i < P.length; i += 3) {
                const [x, y, z] = [P[i], P[i + 1], P[i + 2]];
                const gx = m[0] * x + m[4] * y + m[8] * z + m[12], gy = m[1] * x + m[5] * y + m[9] * z + m[13];
                const gz = m[2] * x + m[6] * y + m[10] * z + m[14];
                const [lon, lat, h] = ecefToLl.forward([gx + rtc[0], -gz + rtc[1], gy + rtc[2]]);
                const [ux, uy] = toUtm.forward([lon, lat]);
                local[i] = ux - cx;
                local[i + 1] = uy - cy;
                local[i + 2] = h;
            }
            for (let t = 0; t < I.length; t += 3) {
                const b = B[I[t]];
                const id = ids[b];
                if (id === undefined || (buildings.has(id) && buildings.get(id).lod !== lod)) continue;
                let acc = found.get(b);
                if (!acc) found.set(b, (acc = { id, parts: new Map() }));
                let part = acc.parts.get(matKey);
                if (!part) acc.parts.set(matKey, (part = { pos: [], nrm: [], uv: [] }));
                for (const v of [I[t], I[t + 1], I[t + 2]]) {
                    part.pos.push(local[v * 3], local[v * 3 + 1], local[v * 3 + 2]);
                    if (Nn) {
                        const [nx, ny, nz] = [Nn[v * 3], Nn[v * 3 + 1], Nn[v * 3 + 2]];
                        const [e, nn, u] = enu(m[0] * nx + m[4] * ny + m[8] * nz, -(m[2] * nx + m[6] * ny + m[10] * nz),
                            m[1] * nx + m[5] * ny + m[9] * nz);
                        part.nrm.push(e, nn, u);
                    }
                    if (T) part.uv.push(T[v * 2], T[v * 2 + 1]);
                }
            }
        }
    }
    for (const [b, acc] of found) {
        let sx = 0, sy = 0, k = 0;
        for (const part of acc.parts.values()) for (let i = 0; i < part.pos.length; i += 3, k++) { sx += part.pos[i]; sy += part.pos[i + 1]; }
        if (Math.abs(sx / k) > HALF || Math.abs(sy / k) > HALF) continue; // centre outside the box
        const meta = { measuredHeight: bt["bldg:measuredHeight"]?.[b] ?? null, surveyYear: bt["uro:BuildingDetailAttribute_uro:surveyYear"]?.[b] ?? null,
                       name: bt["gml:name"]?.[b] ?? null, city: bt.city_name?.[b] ?? null };
        const prev = buildings.get(acc.id);
        if (prev) for (const [mk, p] of acc.parts) { // the same building split over several tiles
            const q = prev.parts.get(mk);
            if (q) { q.pos.push(...p.pos); q.nrm.push(...p.nrm); q.uv.push(...p.uv); } else prev.parts.set(mk, p);
        } else buildings.set(acc.id, { lod, parts: acc.parts, meta });
    }
    return n;
}

async function pool(items, size, fn) {
    let next = 0;
    await Promise.all(Array.from({ length: size }, async () => { while (next < items.length) await fn(items[next++]); }));
}

async function readPlateau() {
    for (const lod of ["lod2", "lod1"]) {
        const tiles = (await Promise.all(cfg.plateau[lod].map(leafTiles))).flat();
        let done = 0;
        await pool(tiles, 6, async url => {
            await readTile(url, lod);
            if (++done % 20 === 0 || done === tiles.length) process.stdout.write(`\r${lod}: ${done}/${tiles.length} tiles, ${buildings.size} buildings`);
        });
        process.stdout.write("\n");
    }
}

// ---------------- untextured models: planar polygons (outer ring + holes, [x, y, h] local metres) -> flat triangles
const THREE = require("three");
const PLAIN = "colour:plain";
materials.set(PLAIN, { colour: true });

function addPolygon(part, rings) {
    rings = rings.map(r => (r.length > 1 && r[0].every((v, i) => v === r[r.length - 1][i]) ? r.slice(0, -1) : r))
        .filter(r => r.length >= 3);
    if (!rings.length) return;
    const outer = rings[0];
    let nx = 0, ny = 0, nz = 0; // Newell normal: the outer ring runs counter-clockwise seen from outside
    for (let i = 0; i < outer.length; i++) {
        const a = outer[i], b = outer[(i + 1) % outer.length];
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
    }
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-9) return;
    [nx, ny, nz] = [nx / len, ny / len, nz / len];
    const [i0, i1] = Math.abs(nz) >= Math.abs(nx) && Math.abs(nz) >= Math.abs(ny) ? [0, 1] : Math.abs(ny) >= Math.abs(nx) ? [0, 2] : [1, 2];
    const flat = r => r.map(p => new THREE.Vector2(p[i0], p[i1]));
    const all = rings.flat();
    for (let [a, b, c] of THREE.ShapeUtils.triangulateShape(flat(outer), rings.slice(1).map(flat))) {
        const [A, B, C] = [all[a], all[b], all[c]];
        const tx = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]);
        const ty = (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]);
        const tz = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
        const tri = tx * nx + ty * ny + tz * nz >= 0 ? [A, B, C] : [A, C, B];
        for (const p of tri) { part.pos.push(...p); part.nrm.push(nx, ny, nz); }
    }
}

function keepBuilding(id, lod, part, meta) {
    let sx = 0, sy = 0;
    const n = part.pos.length / 3;
    if (!n) return;
    for (let i = 0; i < part.pos.length; i += 3) { sx += part.pos[i]; sy += part.pos[i + 1]; }
    if (Math.abs(sx / n) > HALF || Math.abs(sy / n) > HALF) return; // centre outside the box
    buildings.set(id, { lod, parts: new Map([[PLAIN, part]]), meta });
}

// 3D BAG: every building in the padded box from the OGC API (100 per page), LoD 2.2 solids, RD New + NAP.
async function read3dbag() {
    proj4.defs("EPSG:28992", "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 "
        + "+y_0=463000 +ellps=bessel +towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 +units=m +no_defs");
    const toRd = proj4(cfg.utm, "EPSG:28992");
    const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => toRd.forward([cx + sx * (HALF + pad), cy + sy * (HALF + pad)]));
    const bbox = [Math.min(...c.map(p => p[0])), Math.min(...c.map(p => p[1])), Math.max(...c.map(p => p[0])), Math.max(...c.map(p => p[1]))];
    let url = `https://api.3dbag.nl/collections/pand/items?bbox=${bbox.map(v => v.toFixed(1)).join(",")}&limit=100`;
    let pages = 0, version = null;
    while (url) {
        const page = JSON.parse(await fetchCached(url));
        const { scale, translate } = page.metadata.transform;
        for (const f of page.features) {
            const verts = f.vertices.map(([x, y, z]) => {
                const [ux, uy] = toRd.inverse([x * scale[0] + translate[0], y * scale[1] + translate[1]]);
                return [ux - cx, uy - cy, z * scale[2] + translate[2]];
            });
            const building = f.CityObjects[f.id];
            const part = { pos: [], nrm: [], uv: [] };
            for (const childId of building.children ?? []) {
                const g = f.CityObjects[childId].geometry?.find(x => x.lod === "2.2");
                for (const shell of g?.boundaries ?? []) for (const surface of shell) addPolygon(part, surface.map(r => r.map(i => verts[i])));
            }
            const a = building.attributes ?? {};
            keepBuilding(f.id, "lod2.2", part, { ground: a.b3_h_maaiveld ?? null, surveyYear: a.b3_pw_datum ?? null,
                name: null, city: null, measuredHeight: null });
        }
        url = page.links.find(l => l.rel === "next")?.href ?? null;
        process.stdout.write(`\r3D BAG: ${++pages} pages, ${buildings.size} buildings`);
    }
    process.stdout.write("\n");
}

// NYC 3-D Building Model: CityGML streamed out of the zip; one <bldg:Building> at a time, its surfaces' polygons.
async function readNyc() {
    const FT = 0.3048006096012192;
    proj4.defs("EPSG:2263", "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 "
        + "+lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs");
    const toSp = proj4(cfg.utm, "EPSG:2263");
    const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => toSp.forward([cx + sx * (HALF + pad), cy + sy * (HALF + pad)]));
    const [x0, y0, x1, y1] = [Math.min(...c.map(p => p[0])), Math.min(...c.map(p => p[1])), Math.max(...c.map(p => p[0])), Math.max(...c.map(p => p[1]))];
    const zip = join(CACHE, "DA_WISE_GML.zip");
    if (!existsSync(zip)) throw new Error(`${zip} missing: download ${cfg.nyc.zip}`);
    const ring = text => {
        const v = text.trim().split(/\s+/).map(Number), out = [];
        for (let i = 0; i + 2 < v.length; i += 3) {
            const [ux, uy] = toSp.inverse([v[i], v[i + 1]]);
            out.push([ux - cx, uy - cy, v[i + 2] * FT]);
        }
        return out;
    };
    let seen = 0;
    for (const member of cfg.nyc.members) {
        await new Promise((resolve, reject) => {
            const proc = spawn("unzip", ["-p", zip, member]);
            let buf = "";
            proc.stdout.setEncoding("utf8");
            proc.stdout.on("data", chunk => {
                buf += chunk;
                let end;
                while ((end = buf.indexOf("</bldg:Building>")) >= 0) {
                    const start = buf.lastIndexOf("<bldg:Building ", end);
                    const text = buf.slice(start, end);
                    buf = buf.slice(end + 16);
                    seen++;
                    const first = /<gml:posList[^>]*>\s*([-\d.]+)\s+([-\d.]+)/.exec(text);
                    if (!first || +first[1] < x0 || +first[1] > x1 || +first[2] < y0 || +first[2] > y1) continue;
                    const id = /gml:id="([^"]+)"/.exec(text)[1];
                    const bin = /name="BIN">\s*<gen:value>([^<]*)</.exec(text)?.[1] ?? null;
                    const part = { pos: [], nrm: [], uv: [] };
                    for (const poly of text.matchAll(/<gml:Polygon[^>]*>([\s\S]*?)<\/gml:Polygon>/g)) {
                        const rings = [...poly[1].matchAll(/<gml:posList[^>]*>([^<]*)<\/gml:posList>/g)].map(m => ring(m[1]));
                        addPolygon(part, rings);
                    }
                    keepBuilding(id, "lod2", part, { bin, name: /<gml:name>([^<]*)</.exec(text)?.[1] ?? null,
                        surveyYear: null, city: member.match(/DA\d+/)[0], measuredHeight: null });
                }
            });
            proc.on("close", code => (code === 0 ? resolve() : reject(new Error(`unzip ${member}: exit ${code}`))));
            proc.on("error", reject);
        });
        process.stdout.write(`\rNYC: ${seen} buildings read, ${buildings.size} in the box\n`);
    }
}

await { plateau: readPlateau, "3dbag": read3dbag, nyc: readNyc }[cfg.buildings]();

// Clip a part's triangles to the box (|x|, |y| <= HALF): Sutherland-Hodgman per triangle, attributes interpolated.
function clipPart(part) {
    const hasN = part.nrm.length > 0, hasT = part.uv.length > 0;
    const vert = i => ({ p: part.pos.slice(i * 3, i * 3 + 3), n: hasN ? part.nrm.slice(i * 3, i * 3 + 3) : null,
                         t: hasT ? part.uv.slice(i * 2, i * 2 + 2) : null });
    const lerp = (a, b, k) => a.map((v, i) => v + (b[i] - v) * k);
    const mix = (a, b, k) => ({ p: lerp(a.p, b.p, k), n: hasN ? lerp(a.n, b.n, k) : null, t: hasT ? lerp(a.t, b.t, k) : null });
    const planes = [[0, 1], [0, -1], [1, 1], [1, -1]]; // keep axis * sign <= HALF
    const out = { pos: [], nrm: [], uv: [] };
    for (let i = 0; i < part.pos.length / 3; i += 3) {
        let poly = [vert(i), vert(i + 1), vert(i + 2)];
        for (const [ax, sg] of planes) {
            const next = [];
            for (let j = 0; j < poly.length; j++) {
                const a = poly[j], b = poly[(j + 1) % poly.length];
                const da = HALF - sg * a.p[ax], db = HALF - sg * b.p[ax];
                if (da >= 0) next.push(a);
                if ((da >= 0) !== (db >= 0)) next.push(mix(a, b, da / (da - db)));
            }
            poly = next;
            if (poly.length < 3) break;
        }
        for (let j = 1; j + 1 < poly.length; j++) {
            for (const v of [poly[0], poly[j], poly[j + 1]]) {
                out.pos.push(...v.p);
                if (hasN) out.nrm.push(...v.n);
                if (hasT) out.uv.push(...v.t);
            }
        }
    }
    return out;
}
let clipped = 0;
for (const b of buildings.values()) {
    for (const [mk, p] of b.parts) {
        if (!p.pos.every((v, i) => i % 3 === 2 || Math.abs(v) <= HALF)) {
            b.parts.set(mk, clipPart(p));
            clipped++;
        }
    }
}

// ---------------- per-building heights; the GLB's y origin (REF) = the median building base, ellipsoidal
const sidecar = [];
for (const [id, b] of buildings) {
    let lo = Infinity, hi = -Infinity;
    const roof = [];
    for (const part of b.parts.values()) {
        for (let i = 0; i < part.pos.length; i += 3) { lo = Math.min(lo, part.pos[i + 2]); hi = Math.max(hi, part.pos[i + 2]); }
        for (let t = 0; t < part.pos.length; t += 9) { // upward-facing triangles: the roof, for the footprint
            const p = part.pos;
            const ux = p[t + 3] - p[t], uy = p[t + 4] - p[t + 1], vx = p[t + 6] - p[t], vy = p[t + 7] - p[t + 1];
            const area2 = ux * vy - uy * vx;
            if (Math.abs(area2) > 0.02) roof.push([p[t], p[t + 1], p[t + 3], p[t + 4], p[t + 6], p[t + 7]].map(v => Math.round(v * 10) / 10));
        }
    }
    b.base = lo;
    sidecar.push({ id, lod: b.lod, base: Math.round(lo * 100) / 100, top: Math.round(hi * 100) / 100, ...b.meta, tris: roof });
}
const bases = sidecar.map(s => s.base).sort((a, b) => a - b);
const REF = Math.round(bases[Math.floor(bases.length / 2)] * 100) / 100;

// ---------------- one GLB: a primitive per texture atlas, one for all untextured walls/roofs (COLOR_0 = height t)
const doc = new Document();
const buffer = doc.createBuffer();
const scene = doc.createScene();
const mesh = doc.createMesh("buildings");
scene.addChild(doc.createNode("buildings").setMesh(mesh));
const groups = new Map(); // material key -> { pos, nrm, uv, t }
for (const b of buildings.values()) {
    let top = -Infinity;
    for (const p of b.parts.values()) for (let i = 2; i < p.pos.length; i += 3) top = Math.max(top, p.pos[i]);
    for (const [mk, p] of b.parts) {
        let g = groups.get(mk);
        if (!g) groups.set(mk, (g = { pos: [], nrm: [], uv: [], t: [] }));
        for (let i = 0; i < p.pos.length; i += 3) {
            g.pos.push(p.pos[i], p.pos[i + 2] - REF, -p.pos[i + 1]);
            const t = Math.min(Math.max((p.pos[i + 2] - b.base) / Math.max(top - b.base, 1), 0), 1);
            g.t.push(t, t, t);
        }
        for (let i = 0; i < p.nrm.length; i += 3) g.nrm.push(p.nrm[i], p.nrm[i + 2], -p.nrm[i + 1]);
        g.uv.push(...p.uv);
    }
}
const acc = (type, arr) => doc.createAccessor().setType(type).setArray(arr).setBuffer(buffer);
let nTextured = 0, nTris = 0;
const plain = doc.createMaterial("untextured").setBaseColorFactor([1, 1, 1, 1]).setRoughnessFactor(0.85).setMetallicFactor(0);
for (const [mk, g] of groups) {
    const m = materials.get(mk);
    const textured = Boolean(m.image) && g.uv.length;
    const prim = doc.createPrimitive().setAttribute("POSITION", acc("VEC3", new Float32Array(g.pos)));
    // untextured parts: no normals stored (the viewer flat-shades them), so welding can share far more vertices
    if (m.image && g.nrm.length === g.pos.length) {
        const n = new Float32Array(g.nrm);
        for (let i = 0; i < n.length; i += 3) { const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1; n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; }
        prim.setAttribute("NORMAL", acc("VEC3", n));
    }
    if (textured) {
        nTextured++;
        prim.setAttribute("TEXCOORD_0", acc("VEC2", new Float32Array(g.uv)));
        const tex = doc.createTexture().setImage(m.image).setMimeType(m.mime);
        prim.setMaterial(doc.createMaterial().setBaseColorTexture(tex).setRoughnessFactor(0.9).setMetallicFactor(0));
    } else {
        prim.setAttribute("COLOR_0", acc("VEC3", new Float32Array(g.t)));
        prim.setMaterial(plain);
    }
    nTris += g.pos.length / 9;
    mesh.addPrimitive(prim);
}
doc.getRoot().getAsset().extras = { ref_height_m: REF, frame: "x east, y up, z = -north; metres from the box centre", key: KEY };
await doc.transform(
    dedup(),
    weld(),
    prune(),
    textureCompress({ encoder: sharp, targetFormat: "jpeg", resize: [TEX, TEX], quality: QUALITY }),
    meshopt({ encoder: MeshoptEncoder, level: "high" }),
);
mkdirSync(dirname(OUT), { recursive: true });
await io.write(OUT, doc);
writeFileSync(join(CACHE, "buildings.json"), JSON.stringify({ ref: REF, centre_utm: [cx, cy], half: HALF, buildings: sidecar }));

const lods = sidecar.reduce((a, s) => ((a[s.lod] = (a[s.lod] ?? 0) + 1), a), {});
const years = sidecar.reduce((a, s) => ((a[s.surveyYear] = (a[s.surveyYear] ?? 0) + 1), a), {});
const cities = sidecar.reduce((a, s) => ((a[s.city] = (a[s.city] ?? 0) + 1), a), {});
console.log(`buildings in box: ${sidecar.length} ${JSON.stringify(lods)} (${clipped} parts clipped at the edge); survey years ${JSON.stringify(years)}; by city ${JSON.stringify(cities)}`);
console.log(`primitives: ${groups.size} (${nTextured} textured); triangles ${Math.round(nTris)}; ref height ${REF} m (source vertical datum)`);
console.log(`wrote ${OUT.slice(ROOT.length + 1)}  ${(statSync(OUT).size / 1e6).toFixed(2)} MB`);
