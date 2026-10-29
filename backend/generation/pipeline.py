"""Real per-job generation: the assets the Workbench boxes and the Depth Wizard Studio load,
made for THIS job's input instead of the Darjeeling demo files (docs/DESKTOP_APP.md).

Writes into <DW2_GENERATED_DIR or data/generated>/<job id>/ the viewer's exact contract
(frontend/src/viewer.js loadRegion):
  satellite.png       the input's own preview image
  relative_depth.png  DAv2-Small relative depth (backend/api/depth_routes.py host), grayscale
  elevation.png       the TERRAIN DEM, colour-ramped (the ramp of scripts/prepare_elevation_texture.py)
  terrain.json        mesh heights from the SURFACE model (DSM) when there is one, else the terrain
  meta.json           which sources were used, and why
  predicted_height.png / .json   Tier 2 uploads only: Method 6's predicted height above ground (a separate layer;
                      the terrain mesh still comes from the DEM)
Elevation always comes from a real elevation model, never from the image (CLAUDE.md framing):
  library item with a bundled pack (desktop/tiles/build_dem_pack.py): its FABDEM + GLO-30 / VHR DSM
  upload / Search Online scene: its attached DEM (user DEM or FABDEM) + live GLO-30 surface;
                                no attached DEM -> live GLO-30 for both
  DFC2019 library tile: its curated private pack (docs/method-audit/08-dfc2019-terrain-packs/)
  no georeference and no pack (plain PNG/JPG): no elevation model. An upload's 3D surface is then
                                flat ground (0 m) + Method 6 height above ground when Method 6 ran (<= 0.6 m),
                                else relative relief from DAv2 depth (not to scale); never elevation
"""
from __future__ import annotations

import base64
import io
import json
import os
import uuid
import zlib
from pathlib import Path

import numpy as np

from backend.storage import tmp_cap
from backend.terrain.mesh_export import write_terrain_json

ROOT = Path(__file__).resolve().parents[2]
MESH_MAX = 400  # mesh cells per side (Darjeeling's demo mesh is 361 x 325)
DFC2019_CREDIT = "Data: IEEE GRSS Data Fusion Contest 2019 (Track 1), from the JHU/APL Urban Semantic 3D (US3D) dataset."
RAMP = np.array([  # scripts/prepare_elevation_texture.py
    [0.00, 0.08, 0.18, 0.55],
    [0.25, 0.00, 0.55, 0.75],
    [0.50, 0.05, 0.65, 0.35],
    [0.70, 0.75, 0.80, 0.10],
    [0.85, 0.75, 0.35, 0.05],
    [1.00, 0.95, 0.95, 0.95],
])


class GenerationError(RuntimeError):
    pass


def generated_dir() -> Path:
    return Path(os.environ.get("DW2_GENERATED_DIR") or ROOT / "data" / "generated")


def job_path(job_id: str, name: str) -> Path:
    if not job_id.isalnum() or name not in {"satellite.png", "relative_depth.png", "elevation.png",
                                            "terrain.json", "meta.json", "predicted_height.png",
                                            "predicted_height.json"}:
        raise GenerationError("unknown generated asset")
    path = generated_dir() / job_id / name
    tmp_cap.touch(path.parent)  # "last used" for the /tmp cap
    return path


# ----------------------------------------------------------------------------- elevation
M_PER_DEG = 111_320.0  # metres per degree of latitude


def _grid_from_bounds(crs, bounds, res_m: float):
    """A north-up grid over `bounds` (in `crs` units) with cells of about `res_m` metres. In a geographic CRS the
    bounds are degrees, so the cell size is converted to degrees first. (It used to be applied as-is: a Search Online
    scene, EPSG:4326, got cells of 1 or 30 degrees, i.e. the 8 x 8 minimum mesh; fixed 2026-09-30.)"""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    step = res_m / M_PER_DEG if CRS.from_user_input(crs).is_geographic else res_m
    left, bottom, right, top = bounds
    w = max(8, round((right - left) / step))
    h = max(8, round((top - bottom) / step))
    return (h, w), from_bounds(left, bottom, right, top, w, h)


def _library_elevation(item: dict) -> dict | None:
    from backend.storage import library_store
    import rasterio
    pack = library_store.local_asset(item, "dem")
    how = "bundled elevation pack"
    if pack is None:
        pack = library_store.private_pack(item)  # DFC2019: curated pack in the private dataset (web)
        how = "curated elevation pack"
    if pack is not None:
        with rasterio.open(pack) as r:
            tags = r.tags()
            # optional tags (DFC2019 packs): CRS_LABEL for a local frame with no real location,
            # RAMP_BAND=SURFACE when the terrain band is a constant, NOTE = provenance for the UI
            out = {"terrain": r.read(1), "surface": r.read(2), "crs": r.crs, "transform": r.transform,
                   "bounds": tuple(r.bounds), "res_m": abs(r.transform.a),
                   "terrain_source": tags.get("TERRAIN_SOURCE"), "surface_source": tags.get("SURFACE_SOURCE"),
                   "how": tags.get("HOW", how), "crs_label": tags.get("CRS_LABEL"),
                   "ramp": tags.get("RAMP_BAND", "TERRAIN"), "note": tags.get("NOTE")}
            # optional band 3 DISPLAY (Maxar library_v2 packs, scripts/build_maxar_display_packs.py): a cosmetic
            # surface that shapes the 3D mesh only; the real bands still feed statistics and readouts
            if r.count >= 3 and (r.descriptions[2] or "").upper() == "DISPLAY":
                out["display"] = r.read(3)
                out["display_note"] = tags.get("DISPLAY_NOTE")
            return out
    geo = item.get("geo")
    if not geo:
        return None
    # no pack available (e.g. the web backend): live GLO-30 on the item's footprint
    from rasterio.warp import transform  # was pyproj's Transformer (always_xy); parity in docs/container-measurements.md
    (x,), (y,) = transform("EPSG:4326", geo["crs"], [geo["lon"]], [geo["lat"]])
    hw, hh = geo["footprint_km"][0] * 500, geo["footprint_km"][1] * 500
    return _glo30_only(geo["crs"], (x - hw, y - hh, x + hw, y + hh), "live GLO-30 (no bundled pack)")


def _glo30_only(crs, bounds, how: str) -> dict:
    from backend.dem import glo30
    shape, tf = _grid_from_bounds(crs, bounds, 30.0)
    surf = glo30.fetch_grid(crs, bounds, shape, tf)
    return {"terrain": surf, "surface": surf, "crs": crs, "transform": tf, "bounds": tuple(bounds), "res_m": 30.0,
            "terrain_source": "Copernicus GLO-30 DSM (30 m; no bare-earth DEM for this input)",
            "surface_source": "Copernicus GLO-30 DSM (30 m)", "how": how}


def _input_elevation(meta: dict, input_dir: Path) -> dict | None:
    import rasterio
    from rasterio.warp import Resampling, reproject
    if not meta.get("crs") or not meta.get("bounds"):
        return None
    crs, bounds = meta["crs"], tuple(meta["bounds"])
    dem = meta.get("dem")
    if not dem or not (input_dir / "dem.tif").is_file():
        return _glo30_only(crs, bounds, "live GLO-30 (no DEM attached to this input)")
    with rasterio.open(input_dir / "dem.tif") as r:
        # the DEM's own pixel size in metres (a geographic DEM's transform is in degrees), floored at 1 m
        res = max(abs(r.transform.a) * (M_PER_DEG if r.crs.is_geographic else 1.0), 1.0)
        shape, tf = _grid_from_bounds(crs, bounds, res)
        terrain = np.full(shape, np.nan, np.float32)
        reproject(r.read(1).astype(np.float32), terrain, src_transform=r.transform, src_crs=r.crs,
                  dst_transform=tf, dst_crs=crs, resampling=Resampling.bilinear,
                  src_nodata=r.nodata if r.nodata is not None else np.nan, dst_nodata=np.nan)
    out = {"terrain": terrain, "crs": crs, "transform": tf, "bounds": bounds, "res_m": res,
           "terrain_source": f"{dem.get('source', 'attached DEM')} (attached to this input)"}
    try:
        from backend.dem import glo30
        out["surface"] = glo30.fetch_grid(crs, bounds, shape, tf)
        out["surface_source"] = "Copernicus GLO-30 DSM (30 m, live)"
        out["how"] = "attached DEM + live GLO-30 surface"
    except Exception as exc:  # noqa: BLE001 - offline: the attached DEM still gives real terrain
        out["surface"] = terrain
        out["surface_source"] = out["terrain_source"]
        out["how"] = f"attached DEM only (GLO-30 surface unavailable: {type(exc).__name__})"
    return out


# ----------------------------------------------------------------------------- baked depth
DEPTH_NS = "DAV2_DEPTH"  # pack metadata domain: one "SHA256_<preview sha256>" entry per exact preview image


def baked_depth(item: dict | None, preview: bytes) -> dict | None:
    """The DAv2 response baked into the item's pack for exactly these preview bytes (same u16+zlib payload the
    Space returned), or None: no pack, no entry for this preview (e.g. the colours changed), or unreadable."""
    import hashlib
    import logging
    if not item:
        return None
    from backend.storage import library_store
    try:
        pack = library_store.local_asset(item, "dem") or library_store.private_pack(item)
        if pack is None:
            return None
        import rasterio
        sha = hashlib.sha256(preview).hexdigest()
        with rasterio.open(pack) as r:
            raw = r.tags(ns=DEPTH_NS).get(f"SHA256_{sha}")
        if raw is None:
            return None
        entry = json.loads(raw)
        if entry.get("preview_sha256") != sha:
            return None
        return {**entry, "baked": True}
    except Exception as exc:  # noqa: BLE001 - a bad entry must never break generation: the Space answers instead
        logging.getLogger(__name__).warning("baked depth unreadable for %s (%s)", item.get("id"), type(exc).__name__)
        return None


# ----------------------------------------------------------------------------- textures
def _colour_ramp(z: np.ndarray) -> Image.Image:
    from PIL import Image
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    n = np.clip((z - lo) / (hi - lo) if hi > lo else np.zeros_like(z), 0, 1)
    n = np.where(np.isfinite(n), n, 0)
    rgb = np.stack([np.interp(n, RAMP[:, 0], RAMP[:, c + 1]) for c in range(3)], -1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def _decode(resp: dict) -> np.ndarray:
    """A "u16-zlib" grid (or older raw float32) back to float32 values."""
    h, w = resp["shape"]
    raw = base64.b64decode(resp["data_b64"])
    if resp.get("encoding") == "u16-zlib":
        q = np.frombuffer(zlib.decompress(raw), "<u2").astype(np.float32).reshape(h, w)
        return resp["min"] + q / 65535 * (resp["max"] - resp["min"])
    return np.frombuffer(raw, "<f4").reshape(h, w)


# Predicted height (Method 6): viridis, 0 m to the tile's top; the legend in frontend/src/main.js uses the same stops.
AGL_RAMP = np.array([[0.00, 0.267, 0.005, 0.329], [0.25, 0.231, 0.322, 0.545], [0.50, 0.129, 0.569, 0.549],
                     [0.75, 0.369, 0.788, 0.384], [1.00, 0.992, 0.906, 0.145]])
AGL_GRID_MAX = 256  # side of the probe grid in predicted_height.json


def _write_agl(out: Path, agl: dict) -> dict:
    from PIL import Image
    h = np.maximum(_decode(agl["agl"]), 0)  # below-ground predictions are shown as ground
    top = max(float(h.max()), 1.0)
    n = h / top
    rgb = np.stack([np.interp(n, AGL_RAMP[:, 0], AGL_RAMP[:, c + 1]) for c in range(3)], -1)
    Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(out / "predicted_height.png", optimize=True)
    f = max(1, -(-max(h.shape) // AGL_GRID_MAX))
    g = np.asarray(Image.fromarray(h).resize((max(1, h.shape[1] // f), max(1, h.shape[0] // f)), Image.BILINEAR))
    q = np.round(g / top * 65535).astype("<u2")
    (out / "predicted_height.json").write_text(json.dumps({
        "width": g.shape[1], "height": g.shape[0], "min": 0.0, "max": top, "encoding": "u16-zlib",
        "data_b64": base64.b64encode(zlib.compress(q.tobytes(), 6)).decode("ascii")}))
    sigma = _decode(agl["sigma"])
    return {"ran": True, "model": agl.get("model"), "note": agl.get("note"), "gsd_m": agl.get("gsd_m"),
            "tiles": agl.get("tiles"), "infer_s": agl.get("infer_s"), "max_m": round(top, 1),
            "median_m": round(float(np.median(h)), 1), "median_sigma_m": round(float(np.median(sigma)), 1)}


def _depth_png(resp: dict) -> Image.Image:
    from PIL import Image
    d = _decode(resp)
    lo, hi = float(d.min()), float(d.max())
    g = ((d - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(d)).astype(np.uint8)
    return Image.fromarray(g, "L")  # same min-max grayscale as backend/depth/depth_engine.save_depth_png


def _mesh_hw(shape) -> tuple[int, int]:
    f = max(1, -(-max(shape) // MESH_MAX))  # ceil
    return shape[0] // f, shape[1] // f


# ----------------------------------------------------------------------------- job
def generate(kind: str, item_id: str, preview: bytes, depth_resp: dict, *, item: dict | None = None,
             input_meta: dict | None = None, input_dir: Path | None = None, agl: dict | None = None) -> dict:
    tmp_cap.maybe_sweep()
    job_id = uuid.uuid4().hex
    out = generated_dir() / job_id
    out.mkdir(parents=True, exist_ok=True)
    # the new job and its input are held until written (the /tmp cap never deletes them mid-request)
    with tmp_cap.in_use(out, input_dir):
        meta = _generate(kind, item_id, preview, depth_resp, job_id, out, item=item, input_meta=input_meta,
                         input_dir=input_dir, agl=agl)
        if agl is not None:
            meta["agl"] = _write_agl(out, agl) if agl.get("ran") else agl
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta


# Relative relief (an upload with no georeference and no Method 6): DAv2 depth (larger = nearer = higher), scaled so
# its full range is this fraction of the plane's width. A display choice, not a measurement: the UI shows it as 0-1, no metres.
RELATIVE_RELIEF_FRACTION = 0.08


def agl_exaggeration(h) -> float:
    """Default slider for a flat-ground + predicted-height surface. The library's DFC2019 rule (docs/library_v2.md,
    "Exaggeration": exag ~ (p98-p2)^-0.521), anchored on the owner's pick for the upload of JAX_004_006 (p98-p2
    15.1 m -> x22), clamped to x15-35 in 0.5 steps."""
    r = float(np.percentile(h, 98) - np.percentile(h, 2))
    e = 22 * (max(r, 0.1) / 15.1) ** -0.521
    return float(min(35.0, max(15.0, round(e * 2) / 2)))


def _no_georef_surface(kind: str, img, depth_resp: dict, agl: dict | None, bounds, gsd) -> tuple:
    """(surface grid, mesh hw, extra meta) for a no-georeference job; None surface keeps the flat plane."""
    from PIL import Image
    if kind != "input":
        return None, None, {}
    if agl and agl.get("ran"):
        h = np.maximum(_decode(agl["agl"]), 0)  # below-ground predictions are ground, as on the layer
        return h, _mesh_hw(h.shape), {"surface_kind": "agl", "default_exaggeration": agl_exaggeration(h),
                                      "surface_range_m": [0.0, round(float(h.max()), 1)]}
    w, h = img.size
    mh, mw = _mesh_hw((h, w)) if max(w, h) > 256 else (h, w)
    d = np.asarray(Image.fromarray(_decode(depth_resp)).resize((mw, mh), Image.BILINEAR), np.float32)
    lo, hi = float(d.min()), float(d.max())
    width_m = (bounds[2] - bounds[0]) * 111_320
    rel = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    return rel * RELATIVE_RELIEF_FRACTION * width_m, (mh, mw), {"surface_kind": "relative"}


def _generate(kind: str, item_id: str, preview: bytes, depth_resp: dict, job_id: str, out: Path, *,
              item: dict | None = None, input_meta: dict | None = None, input_dir: Path | None = None,
              agl: dict | None = None) -> dict:
    from PIL import Image  # imported here, not at module import (idle memory; docs/container-measurements.md)
    img = Image.open(io.BytesIO(preview)).convert("RGB")
    img.save(out / "satellite.png", optimize=True)
    _depth_png(depth_resp).save(out / "relative_depth.png", optimize=True)

    elev = _library_elevation(item) if kind == "library" else _input_elevation(input_meta, input_dir)
    meta = {"job": job_id, "input": {"kind": kind, "id": item_id}, "has_elevation": elev is not None,
            "depth": {k: depth_resp.get(k) for k in ("model", "device", "infer_s", "encoding", "host", "baked", "fallback")}}
    if elev is None:
        # no georeference: no elevation model exists for this input; a flat plane carries the textures
        w, h = img.size
        aspect = w / h
        flat = np.zeros((64, max(8, round(64 * aspect))), np.float32)
        gsd = (input_meta or {}).get("gsd_m") if (input_meta or {}).get("gsd_manual") else None
        if gsd:
            # a GSD the user typed in: the plane gets the image's real size (pixels x GSD), laid out
            # at the equator in degrees (the viewer's native unit), so measurements are in real metres
            pw, ph = (input_meta.get("size_px") or [w, h])
            bounds = (0.0, 0.0, pw * gsd / 111_320, ph * gsd / 110_574)
            note = (f"No georeference, so there is no DEM for this input: relative depth only, on a flat plane "
                    f"sized from the {gsd:g} m/pixel GSD entered for it.")
        else:
            bounds = (0.0, 0.0, 0.01 * aspect, 0.01)
            note = "No georeference, so there is no DEM for this input: relative depth only, on a flat plane."
        surface, hw, extra = _no_georef_surface(kind, img, depth_resp, agl, bounds, gsd)
        if surface is None:
            write_terrain_json(out / "terrain.json", flat, bounds, flat.shape)
        else:
            mesh = write_terrain_json(out / "terrain.json", surface, bounds, hw, spike_smoothing=True)
            extra["grid"] = [mesh["width"], mesh["height"]]
            note = "No georeference, so there is no DEM for this input" + (
                f": plane sized from the {gsd:g} m/pixel GSD entered for it." if gsd else ".")
        Image.new("RGB", (64, 64), (46, 46, 46)).save(out / "elevation.png")
        meta.update({"terrain_source": None, "surface_source": None, "crs": None, "resolution_m": gsd, "note": note, **extra})
    else:
        from rasterio.warp import transform_bounds
        terrain = np.asarray(elev["terrain"], np.float32)
        surface = np.asarray(elev["surface"], np.float32)
        _colour_ramp(surface if elev.get("ramp") == "SURFACE" else terrain).save(out / "elevation.png", optimize=True)
        lonlat = transform_bounds(elev["crs"], "EPSG:4326", *elev["bounds"], densify_pts=21)
        display = elev.get("display")
        # DFC2019: the surface is reference lidar, so its tall cells are real buildings, not outliers. The viewer's
        # outlier limiter capped them near the ground on tiles that are mostly bare ground (IQR of a few cm).
        lidar = kind == "library" and bool(item) and item.get("collection") == "dfc2019"
        mesh = write_terrain_json(out / "terrain.json", surface, lonlat, _mesh_hw(surface.shape), display=display,
                                  limit_outliers=not lidar, spike_smoothing=kind == "input")
        crs = elev.get("crs_label") or str(elev["crs"])
        if elev.get("note"):
            meta["note"] = elev["note"]
        if display is not None:
            meta["display_note"] = elev.get("display_note") or "3D shape is a cosmetic display surface; statistics show real elevations."
        meta.update(terrain_source=elev["terrain_source"], surface_source=elev["surface_source"], how=elev["how"],
                    crs=crs, resolution_m=round(float(elev["res_m"]), 3), grid=[mesh["width"], mesh["height"]],
                    mesh_from="display" if display is not None else "surface", bounds_lonlat=list(lonlat),
                    terrain_range_m=[round(float(np.nanmin(terrain)), 1), round(float(np.nanmax(terrain)), 1)],
                    surface_range_m=[round(float(np.nanmin(surface)), 1), round(float(np.nanmax(surface)), 1)])
    if kind == "library" and item:
        from backend.library import tile_manifest
        if item.get("collection") == "dfc2019":
            meta["credit"] = DFC2019_CREDIT
        # curated default for the slider (tile_manifest.json); None = the viewer's automatic value
        meta["default_exaggeration"] = tile_manifest.default_exaggeration(item["id"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
