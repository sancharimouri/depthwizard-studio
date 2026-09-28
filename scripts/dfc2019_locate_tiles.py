#!/usr/bin/env python3
"""Locate the 50 DFC2019 Track-1 tiles inside the US3D point clouds (08-dfc2019-terrain-packs, Prompt 2).

The earlier spatial join (docs/method-audit/stage0-gates/sparse-lidar-feasibility.md §8) rasterised
at 0.5 m/px, about 1.6x the true scale (native GSD is ~0.3 m), so its 0/50 was not conclusive.
This redoes it:

  parse     stream {JAX,OMA}_PointClouds.zip -> per cloud tile, 0.6 m grids of
            DSM (max Up, all returns) and ground (min Up of class-2 returns).
            The clouds DO carry per-point classes (README; the brief assumed none), so
            ground comes from ground returns, not a high-pass filter.
  mosaic    city mosaic of cloud AGL = DSM - interpolated ground, at 0.6 / 1.2 / 2.4 m.
  selftest  synthetic crops (random GSD 0.20-0.35, random dihedral transform, height scale,
            noise, blur, dropout) must be recovered.
  match     FFT normalised cross-correlation of each tile's AGL against a city mosaic, scanning
            GSD 0.20-0.35 and all 8 flips/rotations; coarse 2.4 m, then refinement at 0.6 m.
            --pool picks the city (a JAX tile against the OMA pool = negative control).

The confidence rule is pre-registered in docs/method-audit/08-dfc2019-terrain-packs/locate.md
before any real tile is matched. Outputs (with coordinates) go only to the gitignored
data/dfc2019/terrain_packs/locate/; nothing here writes coordinates anywhere else.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/dfc2019/terrain_packs/locate"
ZIPS = {"JAX": Path.home() / "Downloads/JAX_PointClouds.zip",
        "OMA": Path.home() / "Downloads/OMA_PointClouds.zip"}
AGL_DIR = ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
CELL = 0.6          # finest cloud raster (m); ~1.8 pts/m^2 leaves 0.3 m rasters ~85 % empty
TILE_M = 512.0      # US3D point-cloud tiles are 512 m squares
GSD_LO, GSD_HI = 0.20, 0.35


# ----------------------------------------------------------------------------- parse
def _parse_one(args):
    city, name = args
    import pandas as pd
    zf = zipfile.ZipFile(ZIPS[city])
    pts = pd.read_csv(io.BytesIO(zf.read(name)), header=None, usecols=[0, 1, 2],
                      dtype=np.float64, engine="c").to_numpy()
    cls = pd.read_csv(io.BytesIO(zf.read(name.replace("-reduced", "-classification"))), header=None,
                      dtype=np.int16, engine="c").to_numpy()[:, 0]
    assert len(cls) == len(pts), (name, len(cls), len(pts))
    e0 = float(np.floor(pts[:, 0].min()))   # tile origin = its own extent (not on a 512 m UTM grid)
    n0 = float(np.floor(pts[:, 1].min()))
    n = int(round(TILE_M / CELL))                         # 853 cells (511.8 m)
    c = np.clip(((pts[:, 0] - e0) / CELL).astype(np.int64), 0, n - 1)
    r = np.clip(((n0 + TILE_M - pts[:, 1]) / CELL).astype(np.int64), 0, n - 1)  # row 0 = north
    idx = r * n + c
    dsm = np.full(n * n, -np.inf)
    np.maximum.at(dsm, idx, pts[:, 2])
    gnd = np.full(n * n, np.inf)
    g = cls == 2
    np.minimum.at(gnd, idx[g], pts[g, 2])
    dsm[~np.isfinite(dsm)] = np.nan
    gnd[~np.isfinite(gnd)] = np.nan
    tid = name.split("/")[-1].replace("_PC-reduced.txt", "")
    np.savez_compressed(OUT / "clouds" / f"{tid}.npz", dsm=dsm.reshape(n, n).astype(np.float32),
                        gnd=gnd.reshape(n, n).astype(np.float32), e0=e0, n0=n0)
    return tid, float(e0), float(n0), len(pts), float(np.bincount(cls.clip(0, 20), minlength=21)[2] / len(cls)), \
        float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def cmd_parse(a):
    (OUT / "clouds").mkdir(parents=True, exist_ok=True)
    jobs = []
    for city in a.cities:
        names = [n for n in zipfile.ZipFile(ZIPS[city]).namelist() if n.endswith("-reduced.txt")]
        jobs += [(city, n) for n in names
                 if not (OUT / "clouds" / (n.split("/")[-1].replace("_PC-reduced.txt", "") + ".npz")).exists()]
    print(f"parse: {len(jobs)} cloud tiles to do", flush=True)
    t0 = time.time()
    index = {}
    idx_path = OUT / "cloud_index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
    with ProcessPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(_parse_one, jobs, chunksize=1)):
            tid, e0, n0, npts, gfrac, *ext = row
            index[tid] = {"e0": e0, "n0": n0, "points": npts, "ground_frac": round(gfrac, 3), "extent": ext}
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)}  {el:.0f}s  eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s", flush=True)
                idx_path.write_text(json.dumps(index))
    idx_path.write_text(json.dumps(index))
    print(f"parse done in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- mosaic
def _fill_nearest(a):
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    _, (ri, ci) = distance_transform_edt(bad, return_indices=True)
    return a[ri, ci]


def block(a, f):
    """Mean over f x f blocks, NaN-aware."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    x = a[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=(1, 3))


def cmd_mosaic(a):
    """Memory-lean: DSM holes are filled per cloud tile (853^2) before placement; the city
    mosaic is then DSM and ground only, and AGL is computed in place."""
    from scipy.ndimage import gaussian_filter, median_filter, minimum_filter
    index = json.loads((OUT / "cloud_index.json").read_text())
    for city in a.cities:
        tiles = {k: v for k, v in index.items() if k.startswith(city)}
        n = int(round(TILE_M / CELL))
        E0 = min(v["e0"] for v in tiles.values())
        N1 = max(v["n0"] for v in tiles.values()) + TILE_M
        W = int(np.ceil((max(v["e0"] for v in tiles.values()) + TILE_M - E0) / CELL)) + 1
        H = int(np.ceil((N1 - min(v["n0"] for v in tiles.values())) / CELL)) + 1
        print(f"{city}: {len(tiles)} cloud tiles, mosaic {H}x{W} at {CELL} m", flush=True)
        dsm = np.full((H, W), np.nan, np.float32)
        gnd = np.full((H, W), np.nan, np.float32)
        for tid, v in tiles.items():
            z = np.load(OUT / "clouds" / f"{tid}.npz")
            d = z["dsm"]
            covered = np.isfinite(d)
            if covered.mean() < 0.05:
                continue
            d = median_filter(_fill_nearest(d), 3)   # empty 0.6 m cells from neighbours
            r0 = int(round((N1 - v["n0"] - TILE_M) / CELL))
            c0 = int(round((v["e0"] - E0) / CELL))
            for dst, src in ((dsm, d), (gnd, z["gnd"])):
                sl = dst[r0:r0 + n, c0:c0 + n]
                np.copyto(sl, src, where=np.isfinite(src) & ~np.isfinite(sl))
        # ground surface: min of ground returns at 4.8 m, nearest-filled, min-filtered, smoothed
        f = 8
        g8 = block(gnd, f)
        del gnd
        g8 = gaussian_filter(minimum_filter(_fill_nearest(g8), 3), 1.5).astype(np.float32)
        hb, wb = g8.shape
        for r in range(hb):   # dsm -= upsampled ground, row block by row block (no full-size temp)
            dsm[r * f:(r + 1) * f, :wb * f] -= np.repeat(g8[r], f)[None, :]
        dsm[:, wb * f:] = np.nan
        dsm[hb * f:, :] = np.nan
        np.savez(OUT / f"mosaic_{city}.npz", agl=dsm, ground8=g8, E0=E0, N1=N1, cell=CELL)
        print(f"{city}: coverage {np.isfinite(dsm).mean():.1%}; AGL p50 {np.nanpercentile(dsm[::7, ::7], 50):.2f} "
              f"p95 {np.nanpercentile(dsm[::7, ::7], 95):.2f} m", flush=True)
        del dsm


# ----------------------------------------------------------------------------- matching core
def dihedral(a, k):
    """k in 0..7: rotations by 90*k%4, flipped (left-right) first when k >= 4."""
    if k >= 4:
        a = a[:, ::-1]
    return np.rot90(a, k % 4)


def resample(a, out_hw):
    """Area resample (anti-aliased) of a 2-D array to out_hw."""
    from PIL import Image
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((out_hw[1], out_hw[0]), Image.BOX),
                      dtype=np.float32)


class Pool:
    """A city mosaic at one resolution with precomputed FFT and box-sum helpers."""

    def __init__(self, agl: np.ndarray, res: float, E0: float, N1: float):
        import scipy.fft as sf
        self.res, self.E0, self.N1 = res, E0, N1
        self.valid = np.isfinite(agl)
        self.img = np.where(self.valid, agl, 0.0).astype(np.float64)
        self.H, self.W = agl.shape
        self.shape = (sf.next_fast_len(self.H + 700), sf.next_fast_len(self.W + 700))
        self.F = sf.rfft2(self.img, self.shape, workers=-1)
        self.I1 = np.pad(self.img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.I2 = np.pad(self.img ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.IV = np.pad(self.valid.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(self, I, h, w):
        return I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]

    def ncc(self, t: np.ndarray):
        """Normalised cross-correlation of template t at every valid top-left offset.
        Returns (ncc map, coverage map); both (H-h+1, W-w+1)."""
        import scipy.fft as sf
        h, w = t.shape
        tz = t - t.mean()
        tn = np.sqrt((tz ** 2).sum())
        c = sf.irfft2(self.F * np.conj(sf.rfft2(tz, self.shape, workers=-1)), self.shape, workers=-1)
        # conj-product gives circular cross-correlation: c[y, x] = sum img[y+i, x+j] * tz[i, j]
        c = c[:self.H - h + 1, :self.W - w + 1]
        n = h * w
        s1 = self.box(self.I1, h, w)
        s2 = self.box(self.I2, h, w)
        var = np.maximum(s2 - s1 ** 2 / n, 1e-9)
        cov = self.box(self.IV, h, w) / n
        out = c / (np.sqrt(var) * tn + 1e-9)
        out[cov < 0.9] = -1.0
        # a flat window (no structure) must not score: require some variance
        out[var / n < 0.25] = -1.0
        return out, cov


def gsd_grid(step=0.04):
    g, out = GSD_LO, []
    while g <= GSD_HI * 1.0001:
        out.append(round(g, 4))
        g *= 1 + step
    return out


def search(agl_tile: np.ndarray, pool: Pool, gsds, orients=range(8), exclude_m=100.0, topk=5):
    """Exhaustive (gsd, orientation, position) search; returns sorted candidate peaks, where
    every candidate is the best of its own location (other peaks within exclude_m suppressed)."""
    cands = []
    H0, W0 = agl_tile.shape
    for g in gsds:
        hw = (max(8, int(round(H0 * g / pool.res))), max(8, int(round(W0 * g / pool.res))))
        base = resample(np.nan_to_num(agl_tile, nan=0.0), hw)
        for k in orients:
            t = dihedral(base, k).astype(np.float64)
            m, _ = pool.ncc(t)
            # keep a handful of separated local peaks per map
            mm = m.copy()
            rad = int(exclude_m / pool.res)
            for _ in range(topk):
                i = int(np.argmax(mm))
                y, x = divmod(i, mm.shape[1])
                v = float(mm[y, x])
                if v <= -1:
                    break
                cands.append({"ncc": v, "gsd": g, "orient": k, "row": y, "col": x,
                              "h": t.shape[0], "w": t.shape[1]})
                mm[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = -1
    cands.sort(key=lambda d: -d["ncc"])
    return cands


def centre_utm(c, pool: Pool):
    cy = (c["row"] + c["h"] / 2) * pool.res
    cx = (c["col"] + c["w"] / 2) * pool.res
    return pool.E0 + cx, pool.N1 - cy


def best_and_runner(cands, pool: Pool, exclude_m=100.0):
    best = cands[0]
    be, bn = centre_utm(best, pool)
    runner = None
    for c in cands[1:]:
        e, n = centre_utm(c, pool)
        if np.hypot(e - be, n - bn) > exclude_m:
            runner = c
            break
    return best, runner


def refine(agl_tile, fine: Pool, coarse_best, coarse_pool: Pool, span_m=60.0):
    """Around the coarse peak: finer GSD steps (1 %), same orientation, on the 0.6 m pool,
    restricted to a window of +-span_m around the coarse centre (cheap: crop the pool)."""
    e, n = centre_utm(coarse_best, coarse_pool)
    g0 = coarse_best["gsd"]
    size_m = max(agl_tile.shape) * GSD_HI + 2 * span_m
    cx = int((e - fine.E0) / fine.res)
    cy = int((fine.N1 - n) / fine.res)
    half = int(size_m / 2 / fine.res)
    r0, c0 = max(0, cy - half), max(0, cx - half)
    sub = np.where(fine.valid, fine.img, np.nan)[r0:cy + half, c0:cx + half]
    sp = Pool(sub, fine.res, fine.E0 + c0 * fine.res, fine.N1 - r0 * fine.res)
    gs = [round(g0 * (1 + d), 4) for d in np.arange(-0.06, 0.0601, 0.01)]
    cands = search(agl_tile, sp, gs, orients=[coarse_best["orient"]], topk=1)
    return cands[0], sp


def height_scale(agl_tile, pool: Pool, c):
    """Fitted height scale: through-origin LS slope of cloud AGL on tile AGL, on the matched
    window, at the pool resolution, over pixels where either is > 2 m."""
    t = dihedral(resample(np.nan_to_num(agl_tile, nan=0.0), (c["h"], c["w"])) if c["orient"] % 2 == 0
                 else resample(np.nan_to_num(agl_tile, nan=0.0), (c["w"], c["h"])), c["orient"])
    win = np.where(pool.valid, pool.img, np.nan)[c["row"]:c["row"] + t.shape[0], c["col"]:c["col"] + t.shape[1]]
    m = np.isfinite(win) & ((t > 2) | (win > 2))
    if m.sum() < 50:
        return float("nan")
    return float((win[m] * t[m]).sum() / (t[m] ** 2).sum())


def load_pool(city, res):
    """res > CELL: a full FFT Pool of the block-averaged mosaic. res == CELL: a light holder
    (refine() FFTs only a local crop of it)."""
    from types import SimpleNamespace
    z = np.load(OUT / f"mosaic_{city}.npz")
    agl = z["agl"]
    f = int(round(res / float(z["cell"])))
    if f == 1:
        v = np.isfinite(agl)
        return SimpleNamespace(img=np.where(v, agl, 0.0).astype(np.float32), valid=v, res=float(z["cell"]),
                               E0=float(z["E0"]), N1=float(z["N1"]))
    return Pool(block(agl, f), float(z["cell"]) * f, float(z["E0"]), float(z["N1"]))


def tile_agl(tid):
    import rasterio
    with rasterio.open(AGL_DIR / f"{tid}_AGL.tif") as r:
        a = r.read(1).astype(np.float32)
    a[~np.isfinite(a)] = 0.0
    return np.clip(a, 0, None)


def locate(agl, coarse: Pool, fine: Pool, gsds):
    cands = search(agl, coarse, gsds)
    best, runner = best_and_runner(cands, coarse)
    rb, sp = refine(agl, fine, best, coarse)
    e, n = centre_utm(rb, sp)
    return {"coarse_ncc": best["ncc"], "runner_ncc": runner["ncc"] if runner else None,
            "runner_centre": centre_utm(runner, coarse) if runner else None,
            "fine_ncc": rb["ncc"], "gsd": rb["gsd"], "orient": rb["orient"],
            "centre_e": e, "centre_n": n, "width_m": rb["w"] * sp.res, "height_m": rb["h"] * sp.res,
            "height_scale": height_scale(agl, sp, rb)}


# ----------------------------------------------------------------------------- self-test
def cmd_selftest(a):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(a.seed)
    res = []
    for city in a.cities:
        coarse, fine = load_pool(city, 2.4), load_pool(city, CELL)
        mz = np.load(OUT / f"mosaic_{city}.npz")
        full = mz["agl"]
        gsds = gsd_grid()
        done = 0
        while done < a.n:
            g = float(rng.uniform(GSD_LO, GSD_HI))
            size = int(round(1024 * g / CELL))
            r0 = int(rng.integers(0, full.shape[0] - size))
            c0 = int(rng.integers(0, full.shape[1] - size))
            win = full[r0:r0 + size, c0:c0 + size]
            if np.isfinite(win).mean() < 0.98 or np.nanstd(win) < 1.0:
                continue
            k = int(rng.integers(0, 8))
            s = float(rng.uniform(0.9, 1.1))
            # degrade: to a 1024 px "tile", undo the orientation the matcher must find, scale heights,
            # blur, noise, 10 % dropout blocks set to ground
            t = resample(np.nan_to_num(win), (1024, 1024))
            t = dihedral(t, k)
            k_inv = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}[k]  # dihedral(dihedral(x,k),k_inv) == x
            t = gaussian_filter(t * s, 2.0) + rng.normal(0, 0.5, t.shape)
            for _ in range(10):
                y, x = rng.integers(0, 1024 - 100, 2)
                t[y:y + 100, x:x + 100] = 0
            t = np.clip(t, 0, None).astype(np.float32)
            true_e = float(mz["E0"]) + (c0 + size / 2) * CELL
            true_n = float(mz["N1"]) - (r0 + size / 2) * CELL
            t0 = time.time()
            r = locate(t, coarse, fine, gsds)
            err = float(np.hypot(r["centre_e"] - true_e, r["centre_n"] - true_n))
            ok_orient = r["orient"] == k_inv
            row = {"city": city, "true_gsd": round(g, 4), "found_gsd": r["gsd"], "gsd_err_pct": round(100 * (r["gsd"] / g - 1), 2),
                   "true_orient_to_recover": k_inv, "found_orient": r["orient"], "pos_err_m": round(err, 1),
                   "fine_ncc": round(r["fine_ncc"], 3), "coarse_ncc": round(r["coarse_ncc"], 3),
                   "runner_ncc": round(r["runner_ncc"], 3) if r["runner_ncc"] is not None else None,
                   "height_scale": round(r["height_scale"], 3), "true_height_scale": round(1 / s, 3), "sec": round(time.time() - t0, 1)}
            print(row, flush=True)
            res.append(row)
            done += 1
    (OUT / "selftest.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- match
def cmd_match(a):
    tiles = sorted(p.name[:-8] for p in AGL_DIR.glob("*_AGL.tif"))
    if a.tiles:
        tiles = [t for t in tiles if t in a.tiles]
    out_path = OUT / f"match_{a.label}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["tile"] for l in out_path.read_text().splitlines() if l.strip()}
    pools = {}
    gsds = gsd_grid()
    with out_path.open("a") as f:
        for tid in tiles:
            if tid in done:
                continue
            city = a.pool or tid[:3]
            if city not in pools:
                pools[city] = (load_pool(city, 2.4), load_pool(city, CELL))
            coarse, fine = pools[city]
            t0 = time.time()
            r = locate(tile_agl(tid), coarse, fine, gsds)
            r.update(tile=tid, pool=city, sec=round(time.time() - t0, 1))
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{tid} vs {city}: coarse {r['coarse_ncc']:.3f} runner {r['runner_ncc']:.3f} fine {r['fine_ncc']:.3f} "
                  f"gsd {r['gsd']:.3f} orient {r['orient']} hscale {r['height_scale']:.2f} ({r['sec']}s)", flush=True)


# ----------------------------------------------------------------------------- report
RULE = {"C1_fine_ncc_min": 0.50, "C2_gap_min": 0.15, "C2_ratio_max": 0.8, "C3_scale": (0.80, 1.25)}


def judge(r):
    """The pre-registered rule (locate.md). Returns (confident, [failed criteria])."""
    fails = []
    if not r["fine_ncc"] >= RULE["C1_fine_ncc_min"]:
        fails.append("C1 peak")
    run = r["runner_ncc"] if r["runner_ncc"] is not None else -1.0
    if not (r["coarse_ncc"] - run >= RULE["C2_gap_min"] and run <= RULE["C2_ratio_max"] * r["coarse_ncc"]):
        fails.append("C2 separation")
    lo, hi = RULE["C3_scale"]
    if not (np.isfinite(r["height_scale"]) and lo <= r["height_scale"] <= hi):
        fails.append("C3 height scale")
    return not fails, fails


def cloud_tile_at(e, n, city, index):
    for tid, v in index.items():
        if tid.startswith(city) and v["e0"] <= e < v["e0"] + TILE_M and v["n0"] <= n < v["n0"] + TILE_M:
            return tid
    return None


def cmd_report(a):
    index = json.loads((OUT / "cloud_index.json").read_text())
    st = json.loads((OUT / "selftest.json").read_text())
    ok = [c for c in st if c["pos_err_m"] <= 30 and abs(c["gsd_err_pct"]) <= 3 and c["found_orient"] == c["true_orient_to_recover"]]
    v1 = len(ok) / len(st) >= 0.9 and len(st) >= 12
    print(f"V1 self-test: {len(ok)}/{len(st)} pass -> {'PASS' if v1 else 'FAIL'}")
    neg = []
    for lab in ("negctrl_vs_OMA", "negctrl_vs_JAX"):
        p = OUT / f"match_{lab}.jsonl"
        neg += list({r["tile"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}.values())
    nconf = sum(judge(r)[0] for r in neg)
    v2 = nconf == 0 and len(neg) == 50
    print(f"V2 negative control: {nconf}/{len(neg)} confident -> {'PASS' if v2 else 'FAIL'}")
    print("   neg-ctrl max fine NCC %.3f, max coarse NCC %.3f" % (max(r["fine_ncc"] for r in neg), max(r["coarse_ncc"] for r in neg)))
    real = list({r["tile"]: r for r in (json.loads(l) for l in (OUT / "match_real.jsonl").read_text().splitlines()
                                         if l.strip())}.values())   # dedupe (a parallel run may repeat a tile)
    locs, rows = {}, []
    for r in sorted(real, key=lambda r: r["tile"]):
        conf, fails = judge(r)
        ct = cloud_tile_at(r["centre_e"], r["centre_n"], r["tile"][:3], index)
        named = f"{r['tile'][:3]}_Tile_{r['tile'][4:7]}"
        # which cloud tiles does the matched window touch
        half = max(r["width_m"], r["height_m"]) / 2
        touched = sorted({cloud_tile_at(r["centre_e"] + dx, r["centre_n"] + dy, r["tile"][:3], index)
                          for dx in (-half, 0, half) for dy in (-half, 0, half)} - {None})
        row = {"tile": r["tile"], "confident": conf, "fails": fails, "fine_ncc": round(r["fine_ncc"], 3),
               "coarse_ncc": round(r["coarse_ncc"], 3), "runner_ncc": round(r["runner_ncc"], 3) if r["runner_ncc"] is not None else None,
               "gsd": r["gsd"], "orient": r["orient"], "height_scale": round(r["height_scale"], 3),
               "cloud_tile_centre": ct, "cloud_tiles_touched": touched, "name_index_tile": named,
               "centre_in_named_tile": ct == named}
        rows.append(row)
        if conf and v1 and v2:
            locs[r["tile"]] = {"city": r["tile"][:3], "centre_e": r["centre_e"], "centre_n": r["centre_n"], "gsd": r["gsd"],
                               "orient": r["orient"], "width_m": r["width_m"], "height_m": r["height_m"],
                               "cloud_tiles": touched, "utm_zone_note": "US3D cloud UTM (JAX 17N EPSG:32617, OMA 14N EPSG:32614)"}
    (OUT / "report_rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "locations.json").write_text(json.dumps({"validity": {"V1": v1, "V2": v2}, "rule": RULE, "tiles": locs}, indent=1))
    for r in rows:
        print(f"{r['tile']}  {'CONF' if r['confident'] else 'no  '}  fine {r['fine_ncc']:.3f} coarse {r['coarse_ncc']:.3f} "
              f"run {r['runner_ncc']}  gsd {r['gsd']:.3f} or {r['orient']} hs {r['height_scale']:.2f}  "
              f"named-tile {'Y' if r['centre_in_named_tile'] else 'n'}  {','.join(r['fails'])}")
    print(f"confident: JAX {sum(r['confident'] for r in rows if r['tile'][:3]=='JAX')}/26, "
          f"OMA {sum(r['confident'] for r in rows if r['tile'][:3]=='OMA')}/24; locations written: {len(locs)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse"); p.add_argument("--cities", nargs="+", default=["JAX", "OMA"]); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("mosaic"); p.add_argument("--cities", nargs="+", default=["JAX", "OMA"])
    p = sub.add_parser("selftest"); p.add_argument("--cities", nargs="+", default=["JAX", "OMA"]); p.add_argument("--n", type=int, default=6); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("match"); p.add_argument("--pool", default=None); p.add_argument("--label", required=True); p.add_argument("--tiles", nargs="*")
    sub.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"parse": cmd_parse, "mosaic": cmd_mosaic, "selftest": cmd_selftest, "match": cmd_match, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
