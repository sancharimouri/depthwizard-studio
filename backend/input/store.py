"""Page 1 input sources (Upload / Search Online): inspection, tier routing, DEM.

Each input gets a folder data/uploads/<id>/ holding the original file (or,
for a Search Online scene, just its footprint), a preview JPEG, meta.json and,
once sourced, dem.tif + dem_preview.png. Session storage only: nothing here is
tied to a user account, and the folder is not cleaned up automatically.

Tier routing (the product's two processing tiers):
  GSD <= 2.4 m  -> Tier 2 candidate (height prediction). The user supplies a DEM
                   or has FABDEM fetched for them.
  GSD  > 2.4 m  -> Tier 1 (DEM only). FABDEM is fetched automatically; no height
                   prediction is attempted.
  no georeference (PNG/JPG, or a GeoTIFF without a CRS/transform)
                -> GSD unknown and no location to fetch a DEM for: relative-depth
                   preview only (placeholder — no metric output).
2.4 m is the project's tested VHR boundary for real imagery. (A synthetic
degradation test on one sensor held up to 5 m; real sensors weren't tested
between 2.4 and 10 m, so 2.4 m is the conservative, evidence-backed limit.)
"""

from __future__ import annotations

import io
import json
import math
import re
import uuid
import os
from pathlib import Path

import numpy as np

# rasterio / PIL are imported inside the functions that use them (not at module import): the web backend's idle
# memory and start-up stay low until an input actually needs GDAL (docs/container-measurements.md).

from backend.dem import fabdem
from backend.storage import tmp_cap

ROOT = Path(__file__).resolve().parents[2]
# DW2_UPLOADS_DIR: a writable per-user folder when the backend runs from a read-only
# install (the desktop app, desktop/freeze_trial/dw2_entry.py); default for dev.
UPLOADS = Path(os.environ["DW2_UPLOADS_DIR"]) if os.environ.get("DW2_UPLOADS_DIR") else ROOT / "data" / "uploads"
TIER2_MAX_GSD_M = 2.4
MANUAL_GSD_RANGE_M = (0.01, 1000.0)  # a GSD typed in by the user (images with no geotransform)
MIN_DEM_OVERLAP = 0.9
ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class InputError(ValueError):
    """Bad input (unsupported file, no overlap, unknown id…) — a 4xx for the client."""


# --------------------------------------------------------------------------- paths / meta
def _dir(input_id: str) -> Path:
    if not _ID_RE.match(input_id):
        raise InputError("unknown input id")
    d = UPLOADS / input_id
    if not d.is_dir():
        raise InputError("unknown input id")
    tmp_cap.touch(d)  # "last used" for the /tmp cap (backend/storage/tmp_cap.py)
    return d


def input_dir(input_id: str) -> Path:
    """The input's folder (validated id; raises InputError when unknown)."""
    return _dir(input_id)


def load_meta(input_id: str) -> dict:
    return json.loads((_dir(input_id) / "meta.json").read_text())


def _save_meta(d: Path, meta: dict) -> dict:
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def preview_path(input_id: str, kind: str = "image") -> Path:
    d = _dir(input_id)
    p = d / ("preview.jpg" if kind == "image" else "dem_preview.png")
    if not p.exists():
        raise InputError(f"no {kind} preview for this input")
    return p


# --------------------------------------------------------------------------- helpers
def _stretch(arr: np.ndarray) -> np.ndarray:
    bands = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
    if bands.dtype == np.uint8:
        return np.moveaxis(bands, 0, -1)
    out = np.zeros(bands.shape[1:] + (3,), np.uint8)
    for i in range(3):
        b = bands[i].astype(np.float32)
        finite = b[np.isfinite(b)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [2, 98])
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def _gsd_m(src) -> tuple[float, float]:
    """Pixel size in metres along x and y (degrees converted at the image centre)."""
    t = src.transform
    px = math.hypot(t.a, t.d)
    py = math.hypot(t.b, t.e)
    if src.crs.is_geographic:
        cx, cy = t @ (src.width / 2, src.height / 2)
        lat = math.radians(cy)
        return px * 111_320 * math.cos(lat), py * 110_574
    unit = src.crs.linear_units_factor[1] if src.crs.linear_units_factor else 1.0
    return px * unit, py * unit


def _footprint_wgs84(crs, bounds) -> list[float]:
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]


def routing_for(meta: dict) -> dict:
    if not meta.get("georeferenced") and meta.get("gsd_manual"):
        return {
            "tier": None,
            "label": "Relative preview only",
            "placeholder": True,
            "dem_required": False,
            "summary": f"No georeference. {meta['gsd_m']:.2f} m/pixel (entered manually) sets the image's scale, but "
                       "there is no location to fetch a DEM for, so only relative depth can run, on a flat plane at "
                       "that scale. Upload a GeoTIFF for DEM-based or height-predicted output.",
        }
    if not meta.get("georeferenced"):
        return {
            "tier": None,
            "label": "Relative preview only",
            "placeholder": True,
            "dem_required": False,
            "summary": "No georeference, so the ground resolution is unknown and there is no location to fetch a DEM "
                       "for. Only the relative-depth preview can run (a placeholder: no metric heights). "
                       "Upload a GeoTIFF for DEM-based or height-predicted output.",
        }
    if meta["gsd_m"] <= TIER2_MAX_GSD_M:
        return {
            "tier": 2,
            "label": "Tier 2 — height prediction",
            "placeholder": False,
            "dem_required": True,
            "summary": f"{meta['gsd_m']:.2f} m/pixel is within the tested height-prediction range (≤ {TIER2_MAX_GSD_M} m). "
                       "Choose a DEM: upload your own, or have FABDEM fetched for this footprint.",
        }
    return {
        "tier": 1,
        "label": "Tier 1 — DEM only",
        "placeholder": False,
        "dem_required": True,
        "summary": f"{meta['gsd_m']:.2f} m/pixel is coarser than {TIER2_MAX_GSD_M} m, so no height prediction is attempted. "
                   "Terrain comes from FABDEM, fetched automatically.",
    }


def _hillshade_png(elev: np.ndarray, px_m: float) -> bytes:
    z = np.nan_to_num(elev, nan=float(np.nanmin(elev)) if np.isfinite(elev).any() else 0.0)
    gy, gx = np.gradient(z, px_m)
    az, alt = math.radians(315), math.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    img = (np.clip(hs, 0, 1) * 255).astype(np.uint8)
    img[~np.isfinite(elev)] = 0
    from PIL import Image
    pil = Image.fromarray(img)
    pil.thumbnail((1024, 1024))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _dem_stats(elev: np.ndarray) -> dict:
    finite = elev[np.isfinite(elev)]
    if finite.size == 0:
        return {"valid_fraction": 0.0}
    return {
        "min_m": round(float(finite.min()), 1),
        "max_m": round(float(finite.max()), 1),
        "mean_m": round(float(finite.mean()), 1),
        "valid_fraction": round(float(finite.size / elev.size), 4),
    }


# --------------------------------------------------------------------------- upload
def create_upload(filename: str, stream, max_bytes: int = 1024 ** 3) -> dict:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise InputError("Unsupported file type — upload a GeoTIFF (.tif/.tiff), PNG or JPG.")
    tmp_cap.maybe_sweep()
    input_id = uuid.uuid4().hex
    d = UPLOADS / input_id
    d.mkdir(parents=True)
    dest = d / f"original{ext}"
    written = 0
    with dest.open("wb") as fh:
        while chunk := stream.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                dest.unlink()
                raise InputError(f"File too large (limit {max_bytes // 1024 ** 2} MB).")
            fh.write(chunk)
    try:
        return _inspect(d, dest, filename)
    except InputError:
        raise
    except Exception as exc:  # noqa: BLE001 - rasterio/PIL failures = unreadable file
        raise InputError(f"Couldn't read that file as an image: {exc}") from exc


def _inspect(d: Path, path: Path, filename: str) -> dict:
    meta = {"id": d.name, "kind": "upload", "filename": filename, "format": path.suffix.lower().lstrip(".")}
    import rasterio
    from PIL import Image
    with rasterio.open(path) as src:
        georef = src.crs is not None and not src.transform.is_identity
        scale = max(src.width, src.height) / 1024
        h, w = max(1, round(src.height / max(scale, 1))), max(1, round(src.width / max(scale, 1)))
        arr = src.read(out_shape=(src.count, h, w))
        meta.update(size_px=[src.width, src.height], bands=src.count, dtype=str(src.dtypes[0]), georeferenced=georef)
        if georef:
            gx, gy = _gsd_m(src)
            meta.update(
                crs=src.crs.to_string(),
                gsd_m=round(max(gx, gy), 4),
                gsd_xy_m=[round(gx, 4), round(gy, 4)],
                gsd_source="geotransform",
                bounds=list(src.bounds),
                footprint_wgs84=_footprint_wgs84(src.crs, src.bounds),
            )
        else:
            # the UI asks the user to type the GSD in (set_manual_gsd)
            meta.update(gsd_m=None, gsd_source="unknown — no geotransform in this file", gsd_required=True)
    Image.fromarray(_stretch(arr)).save(d / "preview.jpg", quality=88)
    meta["routing"] = routing_for(meta)
    meta["dem"] = None
    return _save_meta(d, meta)


def set_manual_gsd(input_id: str, gsd_m) -> dict:
    """The GSD (m/pixel) of an uploaded image that has no geotransform, typed in by the user."""
    d = _dir(input_id)
    meta = load_meta(input_id)
    if meta.get("georeferenced"):
        raise InputError("This image has a geotransform, so its GSD is read from it and can't be entered manually.")
    try:
        gsd = float(gsd_m)
    except (TypeError, ValueError) as exc:
        raise InputError("The GSD must be a number of metres per pixel.") from exc
    lo, hi = MANUAL_GSD_RANGE_M
    if not (math.isfinite(gsd) and lo <= gsd <= hi):
        raise InputError(f"The GSD must be between {lo} and {hi:g} m per pixel.")
    meta.update(gsd_m=round(gsd, 4), gsd_source="entered manually; no geotransform in this file",
                gsd_manual=True, gsd_required=False)
    meta["routing"] = routing_for(meta)
    return _save_meta(d, meta)


# --------------------------------------------------------------------------- search-online scene
def create_scene(scene: dict) -> dict:
    """A Sentinel-2 scene picked in Search Online. Sentinel-2 is 10 m, so it is
    always Tier 1 (DEM only) — same rule as the curated library."""
    bbox = [float(v) for v in scene["bbox"]]
    tmp_cap.maybe_sweep()
    input_id = uuid.uuid4().hex
    d = UPLOADS / input_id
    d.mkdir(parents=True)
    meta = {
        "id": input_id,
        "kind": "sentinel2-scene",
        "scene_id": scene.get("id"),
        "date": scene.get("date"),
        "cloud": scene.get("cloud"),
        "georeferenced": True,
        "crs": "EPSG:4326",
        "gsd_m": 10.0,
        "gsd_source": "Sentinel-2 L2A band spec (B02/B03/B04)",
        "bounds": bbox,
        "footprint_wgs84": bbox,
        "dem": None,
    }
    meta["routing"] = {
        "tier": 1,
        "label": "Tier 1 — DEM only",
        "placeholder": False,
        "dem_required": True,
        "locked": True,
        "summary": "Sentinel-2 is 10 m: height prediction was shown not to work at that resolution, so terrain comes "
                   "from FABDEM only (fetched automatically). This can't be switched to Tier 2.",
    }
    return _save_meta(d, meta)


def save_scene_preview(input_id: str, png: bytes) -> None:
    d = _dir(input_id)
    from PIL import Image
    Image.open(io.BytesIO(png)).convert("RGB").save(d / "preview.jpg", quality=88)


# --------------------------------------------------------------------------- DEM sourcing
def _write_dem(d: Path, elev: np.ndarray, transform: Affine, crs: str, source: str, px_m: float) -> dict:
    import rasterio
    with rasterio.open(d / "dem.tif", "w", driver="GTiff", width=elev.shape[1], height=elev.shape[0], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(elev.astype(np.float32), 1)
    (d / "dem_preview.png").write_bytes(_hillshade_png(elev, px_m))
    return {"source": source, "crs": crs, "size_px": [int(elev.shape[1]), int(elev.shape[0])], **_dem_stats(elev)}


def fetch_fabdem(input_id: str) -> dict:
    with tmp_cap.in_use(_dir(input_id)):  # the Earth Engine fetch can take a while: keep the input
        return _fetch_fabdem(input_id)


def _fetch_fabdem(input_id: str) -> dict:
    d = _dir(input_id)
    meta = load_meta(input_id)
    if not meta.get("georeferenced"):
        raise InputError("This input has no georeference, so there is no footprint to fetch FABDEM for.")
    from rasterio.crs import CRS  # was pyproj.CRS: only .is_geographic is used, identical in rasterio
    from rasterio.transform import Affine
    crs = CRS.from_user_input(meta["crs"])
    res = fabdem.native_res_for(crs.is_geographic)
    left, bottom, right, top = meta["bounds"]
    grid = fabdem.fetch_grid(meta["crs"], left, bottom, right, top, res)
    px_m = fabdem.FABDEM_RES_M
    dem = _write_dem(d, grid["elevation"], Affine(*grid["transform"]), meta["crs"],
                     "FABDEM (Hawker et al. 2022) via Google Earth Engine", px_m)
    meta["dem"] = dem
    return _save_meta(d, meta)


def attach_user_dem(input_id: str, filename: str, stream, max_bytes: int = 512 * 1024 ** 2) -> dict:
    with tmp_cap.in_use(_dir(input_id)):
        return _attach_user_dem(input_id, filename, stream, max_bytes)


def _attach_user_dem(input_id: str, filename: str, stream, max_bytes: int = 512 * 1024 ** 2) -> dict:
    d = _dir(input_id)
    meta = load_meta(input_id)
    if not meta.get("georeferenced"):
        raise InputError("The image has no georeference, so a DEM can't be aligned to it.")
    if meta["routing"].get("tier") != 2:
        raise InputError("A user DEM is only used for Tier 2 (≤ 2.4 m) images; this one uses FABDEM.")
    if Path(filename or "").suffix.lower() not in {".tif", ".tiff"}:
        raise InputError("The DEM must be a GeoTIFF (.tif/.tiff).")
    import rasterio
    tmp = d / "user_dem_upload.tif"
    written = 0
    with tmp.open("wb") as fh:
        while chunk := stream.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                tmp.unlink()
                raise InputError(f"DEM file too large (limit {max_bytes // 1024 ** 2} MB).")
            fh.write(chunk)
    try:
        with rasterio.open(tmp) as src:
            if src.crs is None or src.transform.is_identity:
                raise InputError("That DEM has no georeference (CRS + geotransform), so it can't be aligned.")
            if src.count != 1:
                raise InputError(f"A DEM must have one band (this file has {src.count}).")
            dem_fp = _footprint_wgs84(src.crs, src.bounds)
            elev = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            gx, gy = _gsd_m(src)
            crs, transform = src.crs.to_string(), src.transform
    except InputError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise InputError(f"Couldn't read that DEM: {exc}") from exc
    img = meta["footprint_wgs84"]
    ix = max(0.0, min(img[2], dem_fp[2]) - max(img[0], dem_fp[0]))
    iy = max(0.0, min(img[3], dem_fp[3]) - max(img[1], dem_fp[1]))
    area = (img[2] - img[0]) * (img[3] - img[1])
    overlap = (ix * iy / area) if area > 0 else 0.0
    if overlap < MIN_DEM_OVERLAP:
        tmp.unlink(missing_ok=True)
        raise InputError(
            f"That DEM covers only {overlap:.0%} of the image footprint (at least {MIN_DEM_OVERLAP:.0%} is needed)."
        )
    dem = _write_dem(d, elev, transform, crs, f"User upload: {filename}", max(gx, gy))
    dem["overlap_fraction"] = round(overlap, 4)
    tmp.unlink(missing_ok=True)
    meta["dem"] = dem
    return _save_meta(d, meta)
