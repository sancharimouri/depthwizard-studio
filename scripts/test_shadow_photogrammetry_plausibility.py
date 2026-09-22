"""Plausibility-only check: is ArnabTechiee/depthwizard's shadow-length
photogrammetry even usable at Sentinel-2's 10m GSD, on frequency fusion's
4 losing tiles (bathinda, amalapuram, kutch, chennai)?

Not a full validation or pipeline integration -- the task explicitly
scoped this as "report plausibility only... a quick check ... before
investing further." Ported (read directly, never executed) from
external/ArnabTechiee-depthwizard/pipeline/calibrate.py:
shadow_mask, shadow_direction, building_mask, measure_building -- the
core shadow-length-to-height geometry, run against this project's own
real Sentinel-2 RGB + DAv2 raw relative-depth (as the "nDSM" building_mask
expects -- their function only needs a relative height field to threshold,
any relative signal works per its own percentile-based design) + real
sun-angle metadata for these 4 tiles.

A genuine bug found while reading their code, not reproduced here: their
own docstring states the physics as `h = L_pixels x GSD x tan(sun_elevation)`,
and their own `max_len` cap is derived consistently with that formula
(`max_len = 200.0 / tan_elev / gsd`, i.e. solving h=200 for L). But their
actual height computation, `height_m = length_px * gsd / tan_elev`,
DIVIDES by tan(elevation) instead of multiplying -- inconsistent with
both their stated physics and their own max_len derivation. This
adaptation uses the physically correct formula (h = L * GSD * tan(elev)),
flagged here rather than silently reproduced or silently "fixed" without
comment.

Sun elevation/azimuth: fetched fresh from Microsoft Planetary Computer's
public Sentinel-2 STAC (`s2:mean_solar_zenith`/`s2:mean_solar_azimuth`,
elevation = 90 - zenith) -- same free, keyless, no-new-credentials path
already established and used for the viewing-angle sign-flip investigation
earlier in this project's audit trail. Real per-tile values, not assumed.
"""
from __future__ import annotations

import math
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
GRID_RES_M = 10.0
TARGETS = ["bathinda", "amalapuram", "kutch", "chennai"]

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def fetch_sun_angles(lat: float, lon: float, date: str) -> dict:
    r = httpx.get(STAC_SEARCH_URL, params={
        "collections": "sentinel-2-l2a",
        "bbox": f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}",
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 10,
    }, timeout=30)
    r.raise_for_status()
    feats = r.json().get("features", [])
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 999))
    p = feats[0]["properties"]
    return {
        "sun_zenith_deg": p["s2:mean_solar_zenith"],
        "sun_azimuth_deg": p["s2:mean_solar_azimuth"],
        "sun_elevation_deg": 90.0 - p["s2:mean_solar_zenith"],
    }


# ============================================================================
# Ported from external/ArnabTechiee-depthwizard/pipeline/calibrate.py
# ============================================================================

def shadow_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    luma = arr.mean(axis=2)
    dark = luma < np.percentile(luma, 22)
    blue_bias = arr[:, :, 2] - arr[:, :, 0]
    bluish = blue_bias > np.percentile(blue_bias, 45)
    mask = dark & bluish
    if mask.mean() < 0.02:
        mask = dark
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def building_mask(ndsm: np.ndarray, min_area_px: int) -> np.ndarray:
    positive = ndsm[ndsm > 1e-6]
    if positive.size == 0:
        return np.zeros_like(ndsm, dtype=bool)
    thresh = np.percentile(positive, 55)
    mask = ndsm > thresh
    mask = ndimage.binary_opening(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = np.isin(labels, np.nonzero(sizes >= min_area_px)[0] + 1)
    return keep


def shadow_direction(sun_azimuth_deg: float) -> tuple[float, float]:
    az = math.radians((sun_azimuth_deg + 180.0) % 360.0)
    d_col = math.sin(az)
    d_row = -math.cos(az)
    return d_row, d_col


def measure_building(labels: np.ndarray, label_id: int, shadows: np.ndarray,
                     buildings: np.ndarray, d_row: float, d_col: float,
                     max_len: int, samples: int = 40) -> float | None:
    h, w = labels.shape
    ys, xs = np.nonzero(labels == label_id)
    if ys.size == 0:
        return None

    edge = []
    for y, x in zip(ys, xs):
        ny, nx = int(round(y + d_row)), int(round(x + d_col))
        if not (0 <= ny < h and 0 <= nx < w) or not buildings[ny, nx]:
            edge.append((y, x))
    if len(edge) < 4:
        return None

    if len(edge) > samples:
        idx = np.linspace(0, len(edge) - 1, samples).astype(int)
        edge = [edge[i] for i in idx]

    lengths = []
    for y, x in edge:
        start = 0
        for step in range(1, 25):
            ny = int(round(y + d_row * step)); nx = int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                break
            if not buildings[ny, nx]:
                start = step
                break
        if start == 0:
            continue

        length = 0
        hits = 0
        blocked = False
        for step in range(start, max_len + 1):
            ny = int(round(y + d_row * step)); nx = int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                blocked = True; break
            if buildings[ny, nx]:
                blocked = True; break
            if shadows[ny, nx]:
                length = step - start + 1; hits += 1
            elif step - start + 1 > length + 1:
                break
        if not blocked and length > 0 and hits / float(length) >= 0.85:
            lengths.append(length)

    if len(lengths) < 4:
        return None

    lengths = np.array(lengths, dtype=np.float32)
    median = float(np.median(lengths))
    if median < 3:
        return None
    if float(np.std(lengths)) / median > 0.45:
        return None
    return median


# ============================================================================
# Plausibility probe (this project's own code, not ArnabTechiee's)
# ============================================================================

def probe_tile(tile_id: str, rgb_path: Path, sun: dict, min_area_m2: float = 50.0) -> dict:
    with rasterio.open(rgb_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # (H, W, 3)
    ndsm_proxy = np.load(DAV2_DIR / f"{tile_id}_depth.npy").astype(np.float32)

    gsd = GRID_RES_M
    sun_elev, sun_azim = sun["sun_elevation_deg"], sun["sun_azimuth_deg"]
    tan_elev = math.tan(math.radians(sun_elev))

    # Physically correct formula (h = L * GSD * tan(elev)) -- see module
    # docstring for the bug found in ArnabTechiee's own height_m line.
    min_height_for_3px_shadow_m = 3 * gsd * tan_elev

    min_area_px = int(min_area_m2 / (gsd ** 2))
    shadows = shadow_mask(rgb)
    buildings = building_mask(ndsm_proxy, min_area_px)
    labels, n_candidates = ndimage.label(buildings)

    d_row, d_col = shadow_direction(sun_azim)
    max_len = max(1, int(200.0 / tan_elev / gsd))

    accepted, rejected = 0, 0
    lengths_found = []
    for label_id in range(1, n_candidates + 1):
        length_px = measure_building(labels, label_id, shadows, buildings, d_row, d_col, max_len)
        if length_px is None:
            rejected += 1
            continue
        accepted += 1
        lengths_found.append(length_px)

    return {
        "tile_id": tile_id,
        "sun_elevation_deg": sun_elev, "sun_azimuth_deg": sun_azim,
        "min_height_for_3px_shadow_m": min_height_for_3px_shadow_m,
        "shadow_pixel_pct": 100.0 * shadows.mean(),
        "building_pixel_pct": 100.0 * buildings.mean(),
        "n_candidate_buildings": n_candidates,
        "n_accepted_anchors": accepted,
        "n_rejected_anchors": rejected,
        "median_shadow_length_px": float(np.median(lengths_found)) if lengths_found else None,
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    results = []
    for tile_id in TARGETS:
        m = manifest.loc[tile_id]
        sun = fetch_sun_angles(float(m["lat"]), float(m["lon"]), m["date_acquired"])
        r = probe_tile(tile_id, ROOT / m["rgb_path"], sun)
        results.append(r)
        print(f"{tile_id:12s} sun_elev={sun['sun_elevation_deg']:.1f} deg  "
              f"min_height_for_3px_shadow={r['min_height_for_3px_shadow_m']:.1f}m  "
              f"shadow_px={r['shadow_pixel_pct']:.1f}%  building_px={r['building_pixel_pct']:.1f}%  "
              f"candidates={r['n_candidate_buildings']}  accepted_anchors={r['n_accepted_anchors']}")

    df = pd.DataFrame(results)
    out_csv = ROOT / "data/sentinel2_benchmark/shadow_photogrammetry_plausibility.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
