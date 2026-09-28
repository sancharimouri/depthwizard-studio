"""Real per-job generation: the assets the Workbench boxes and the Depth Wizard Studio load,
made for THIS job's input instead of the Darjeeling demo files (docs/DESKTOP_APP.md).

Writes into <DW2_GENERATED_DIR or data/generated>/<job id>/ the viewer's exact contract
(frontend/src/viewer.js loadRegion):
  satellite.png       the input's own preview image
  relative_depth.png  DAv2-Small relative depth (backend/api/depth_routes.py host), grayscale
  elevation.png       the TERRAIN DEM, colour-ramped (the ramp of scripts/prepare_elevation_texture.py)
  terrain.json        mesh heights from the SURFACE model (DSM) when there is one, else the terrain
  meta.json           which sources were used, and why
Elevation always comes from a real elevation model, never from the image (CLAUDE.md framing):
  library item with a bundled pack (desktop/tiles/build_dem_pack.py): its FABDEM + GLO-30 / VHR DSM
  upload / Search Online scene: its attached DEM (user DEM or FABDEM) + live GLO-30 surface;
                                no attached DEM -> live GLO-30 for both
  DFC2019 library tile: its curated private pack (docs/method-audit/08-dfc2019-terrain-packs/)
  no georeference and no pack (plain PNG/JPG): no elevation; a flat plane, said so in meta/UI
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
from PIL import Image

from backend.terrain.mesh_export import write_terrain_json

ROOT = Path(__file__).resolve().parents[2]
MESH_MAX = 400  # mesh cells per side (Darjeeling's demo mesh is 361 x 325)
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
                                            "terrain.json", "meta.json"}:
        raise GenerationError("unknown generated asset")
    return generated_dir() / job_id / name


# ----------------------------------------------------------------------------- elevation
def _grid_from_bounds(crs, bounds, res_m: float):
    from rasterio.transform import from_bounds
    left, bottom, right, top = bounds
    w = max(8, round((right - left) / res_m))
    h = max(8, round((top - bottom) / res_m))
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
            return {"terrain": r.read(1), "surface": r.read(2), "crs": r.crs, "transform": r.transform,
                    "bounds": tuple(r.bounds), "res_m": abs(r.transform.a),
                    "terrain_source": tags.get("TERRAIN_SOURCE"), "surface_source": tags.get("SURFACE_SOURCE"),
                    "how": tags.get("HOW", how), "crs_label": tags.get("CRS_LABEL"),
                    "ramp": tags.get("RAMP_BAND", "TERRAIN"), "note": tags.get("NOTE")}
    geo = item.get("geo")
    if not geo:
        return None
    # no pack available (e.g. the web backend): live GLO-30 on the item's footprint
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", geo["crs"], always_xy=True).transform(geo["lon"], geo["lat"])
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
        res = max(abs(r.transform.a), 1.0)
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


# ----------------------------------------------------------------------------- textures
def _colour_ramp(z: np.ndarray) -> Image.Image:
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    n = np.clip((z - lo) / (hi - lo) if hi > lo else np.zeros_like(z), 0, 1)
    n = np.where(np.isfinite(n), n, 0)
    rgb = np.stack([np.interp(n, RAMP[:, 0], RAMP[:, c + 1]) for c in range(3)], -1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def _depth_png(resp: dict) -> Image.Image:
    h, w = resp["shape"]
    raw = base64.b64decode(resp["data_b64"])
    if resp.get("encoding") == "u16-zlib":
        d = np.frombuffer(zlib.decompress(raw), "<u2").astype(np.float32).reshape(h, w)
    else:
        d = np.frombuffer(raw, "<f4").reshape(h, w)
    lo, hi = float(d.min()), float(d.max())
    g = ((d - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(d)).astype(np.uint8)
    return Image.fromarray(g, "L")  # same min-max grayscale as backend/depth/depth_engine.save_depth_png


def _mesh_hw(shape) -> tuple[int, int]:
    f = max(1, -(-max(shape) // MESH_MAX))  # ceil
    return shape[0] // f, shape[1] // f


# ----------------------------------------------------------------------------- job
def generate(kind: str, item_id: str, preview: bytes, depth_resp: dict, *, item: dict | None = None,
             input_meta: dict | None = None, input_dir: Path | None = None) -> dict:
    job_id = uuid.uuid4().hex
    out = generated_dir() / job_id
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(preview)).convert("RGB")
    img.save(out / "satellite.png", optimize=True)
    _depth_png(depth_resp).save(out / "relative_depth.png", optimize=True)

    elev = _library_elevation(item) if kind == "library" else _input_elevation(input_meta, input_dir)
    meta = {"job": job_id, "input": {"kind": kind, "id": item_id}, "has_elevation": elev is not None,
            "depth": {k: depth_resp.get(k) for k in ("model", "device", "infer_s", "encoding", "host", "fallback_used")}}
    if elev is None:
        # no georeference: no elevation model exists for this input; a flat plane carries the textures
        w, h = img.size
        aspect = w / h
        flat = np.zeros((64, max(8, round(64 * aspect))), np.float32)
        write_terrain_json(out / "terrain.json", flat, (0.0, 0.0, 0.01 * aspect, 0.01), flat.shape)
        Image.new("RGB", (64, 64), (46, 46, 46)).save(out / "elevation.png")
        meta.update(terrain_source=None, surface_source=None, crs=None, resolution_m=None,
                    note="No georeference, so there is no DEM for this input: relative depth only, on a flat plane.")
    else:
        from rasterio.warp import transform_bounds
        terrain = np.asarray(elev["terrain"], np.float32)
        surface = np.asarray(elev["surface"], np.float32)
        _colour_ramp(surface if elev.get("ramp") == "SURFACE" else terrain).save(out / "elevation.png", optimize=True)
        lonlat = transform_bounds(elev["crs"], "EPSG:4326", *elev["bounds"], densify_pts=21)
        mesh = write_terrain_json(out / "terrain.json", surface, lonlat, _mesh_hw(surface.shape))
        crs = elev.get("crs_label") or str(elev["crs"])
        if elev.get("note"):
            meta["note"] = elev["note"]
        meta.update(terrain_source=elev["terrain_source"], surface_source=elev["surface_source"], how=elev["how"],
                    crs=crs, resolution_m=round(float(elev["res_m"]), 3), grid=[mesh["width"], mesh["height"]],
                    mesh_from="surface", bounds_lonlat=list(lonlat),
                    terrain_range_m=[round(float(np.nanmin(terrain)), 1), round(float(np.nanmax(terrain)), 1)],
                    surface_range_m=[round(float(np.nanmin(surface)), 1), round(float(np.nanmax(surface)), 1)])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
