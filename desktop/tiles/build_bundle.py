"""Build the desktop app's tiered tile library (docs/DESKTOP_APP.md, "Tiered tiles").

  python desktop/tiles/build_bundle.py OUT_DIR

Reads data/library/manifest.json and desktop/tiles/selection.json and writes:
  OUT_DIR/manifest.json     every item, plus "bundled" and "download" (on-demand source)
  OUT_DIR/thumbnails/       all 88 thumbnails
  OUT_DIR/previews/, tiles/ the bundled items only; tiles re-encoded LOSSLESSLY
                            (deflate, horizontal predictor, level 9, 256 px blocks),
                            each verified pixel- and georeference-identical
Bundled: 8 DFC2019 + 11 Sentinel-2 incl. Darjeeling (selection.json) + all 6 Maxar. The rest are
thumbnail-only and downloaded on demand by POST /api/library/{id}/download:
All from the public GitHub Release (DFC2019 too: temporary, publish_dfc_ondemand.py).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.storage import library_store  # noqa: E402

out = Path(sys.argv[1]).resolve()
lib = ROOT / "data" / "library"
man = json.loads((lib / "manifest.json").read_text())
sel = json.loads((Path(__file__).parent / "selection.json").read_text())
bundled = set(sel["dfc2019"]["bundled"]) | set(sel["sentinel2"]["bundled"]) | {
    i["id"] for i in man["items"] if i["collection"] == "vhr"}
assert len(bundled) == 25, len(bundled)  # 8 DFC2019 + 11 Sentinel-2 + 6 Maxar

if out.exists():
    shutil.rmtree(out)
for d in ("thumbnails", "previews", "tiles", "dem"):
    (out / d).mkdir(parents=True)
DEM_PACKS = ROOT / "data" / "library" / "dem"  # desktop/tiles/build_dem_pack.py (all georeferenced items, ~23 MB)


def recompress(src: Path, dst: Path) -> None:
    with rasterio.open(src) as r:
        prof, data, tags = r.profile, r.read(), r.tags()
    prof.update(compress="deflate", predictor=2, zlevel=9, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(dst, "w", **prof) as w:
        w.write(data)
        w.update_tags(**tags)
    with rasterio.open(dst) as r:
        assert np.array_equal(r.read(), data), f"{dst.name}: pixels differ"
        assert r.crs == prof.get("crs") and r.transform == prof.get("transform"), f"{dst.name}: georeference differs"


def published_sizes() -> dict:
    """Asset sizes of the public release: what an on-demand download actually fetches."""
    import subprocess
    try:
        out = subprocess.run(["gh", "release", "view", library_store.GH_TAG, "-R", library_store.GH_REPO,
                              "--json", "assets"], capture_output=True, text=True, check=True, timeout=60).stdout
        return {a["name"]: a["size"] for a in json.loads(out)["assets"]}
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: release sizes unavailable ({type(exc).__name__}); using local file sizes")
        return {}


RELEASE = published_sizes()
items, sizes = [], {"thumbnails": 0, "previews": 0, "tiles": 0, "tiles_original": 0}
for it in man["items"]:
    it = {k: v for k, v in it.items() if k not in ("file", "store", "assets", "r2")}
    iid = it["id"]
    names = {"thumbnail": f"{iid}.jpg", "preview": f"{iid}.jpg", "tile": f"{iid}.tif"}
    shutil.copyfile(lib / "thumbnails" / it["thumbnail"], out / "thumbnails" / names["thumbnail"])
    sizes["thumbnails"] += (out / "thumbnails" / names["thumbnail"]).stat().st_size
    src_tile = Path(os.path.realpath(ROOT / next(i for i in man["items"] if i["id"] == iid)["file"]))
    # Every on-demand item downloads from the public GitHub Release, DFC2019 included:
    # TEMPORARY owner decision (2026-09-26), published by desktop/tiles/publish_dfc_ondemand.py.
    download = {"source": "github-release", "preview": library_store.release_url(f"{iid}__preview.jpg"),
                "tile": library_store.release_url(f"{iid}.tif")}
    pub = (RELEASE.get(f"{iid}__preview.jpg"), RELEASE.get(f"{iid}.tif"))
    download["bytes"] = (sum(pub) if all(pub) else
                         (lib / "previews" / it["preview"]).stat().st_size + src_tile.stat().st_size)
    download["bytes_source"] = "release" if all(pub) else "local"
    if iid in bundled:
        shutil.copyfile(lib / "previews" / it["preview"], out / "previews" / names["preview"])
        recompress(src_tile, out / "tiles" / names["tile"])
        sizes["previews"] += (out / "previews" / names["preview"]).stat().st_size
        sizes["tiles"] += (out / "tiles" / names["tile"]).stat().st_size
        sizes["tiles_original"] += src_tile.stat().st_size
    pack = DEM_PACKS / f"{iid}.tif"
    if pack.is_file():  # elevation for real generation, bundled for every georeferenced item
        shutil.copyfile(pack, out / "dem" / pack.name)
        sizes["dem"] = sizes.get("dem", 0) + pack.stat().st_size
    it.update(thumbnail=names["thumbnail"], preview=names["preview"], tile=names["tile"],
              dem=pack.name if pack.is_file() else None, bundled=iid in bundled, download=download)
    items.append(it)

man_out = {**{k: v for k, v in man.items() if k != "items"}, "items": items,
           "bundle": {"bundled": sorted(bundled), "selection": "desktop/tiles/selection.json"}}
(out / "manifest.json").write_text(json.dumps(man_out, indent=1))
mb = {k: round(v / 1e6, 1) for k, v in sizes.items()}
total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
print(f"items {len(items)} | bundled {len(bundled)} | sizes MB {mb} | bundle total {total:.1f} MB")
