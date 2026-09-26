"""Publish the 40 on-demand DFC2019 tiles to the PUBLIC GitHub Release (2026-09-26).

TEMPORARY, by the project owner's decision: the desktop app must download these on any
machine without a token. The DFC2019 contest terms restrict redistribution; the owner
will replace this source. Run from the repo root:

  .venv/bin/python desktop/tiles/publish_dfc_ondemand.py

1. Stages, for every DFC2019 item NOT bundled in the app (desktop/tiles/selection.json):
   <id>.tif (re-encoded losslessly, verified pixel-identical) and <id>__preview.jpg.
2. Uploads them to sancharimouri/depthwizard2-assets @ library-v1 (gh, --clobber).
3. Replaces the README's "Not included" section with the DFC2019 source and terms.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
REPO, TAG = "sancharimouri/depthwizard2-assets", "library-v1"

man = {i["id"]: i for i in json.loads((ROOT / "data/library/manifest.json").read_text())["items"]}
bundled = set(json.loads((ROOT / "desktop/tiles/selection.json").read_text())["dfc2019"]["bundled"])
ondemand = sorted(i for i, it in man.items() if it["collection"] == "dfc2019" and i not in bundled)
# 42 since 2026-09-27: OMA_364_043 and OMA_315_019 unbundled at the owner's request (selection.json).
assert len(ondemand) == 42, len(ondemand)
# Optional item ids: upload only these (e.g. newly unbundled tiles), not all 42 again.
only = sys.argv[1:]
assert set(only) <= set(ondemand), set(only) - set(ondemand)

stage = Path(tempfile.mkdtemp(prefix="dw2-dfc-"))
for iid in only or ondemand:
    shutil.copyfile(ROOT / "data/library/previews" / man[iid]["preview"], stage / f"{iid}__preview.jpg")
    with rasterio.open(os.path.realpath(ROOT / man[iid]["file"])) as r:
        prof, data, tags = r.profile, r.read(), r.tags()
    prof.update(compress="deflate", predictor=2, zlevel=9, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(stage / f"{iid}.tif", "w", **prof) as w:
        w.write(data)
        w.update_tags(**tags)
    with rasterio.open(stage / f"{iid}.tif") as r:
        assert np.array_equal(r.read(), data), iid
files = sorted(stage.iterdir())
print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB (tiles verified lossless)")

subprocess.run(["gh", "release", "upload", TAG, "-R", REPO, "--clobber", *map(str, files)], check=True)
release = json.loads(subprocess.run(["gh", "release", "view", TAG, "-R", REPO, "--json", "assets"],
                                    capture_output=True, text=True, check=True).stdout)
names = {a["name"] for a in release["assets"]}
missing = [p.name for p in files if p.name not in names]
assert not missing, missing
print(f"uploaded {len(files)} assets to {REPO}@{TAG}")

meta = json.loads(subprocess.run(["gh", "api", f"repos/{REPO}/contents/README.md"],
                                 capture_output=True, text=True, check=True).stdout)
readme = base64.b64decode(meta["content"]).decode()
new_section = f"""## DFC2019 ({len(ondemand)} tiles, `dfc2019-*`): TEMPORARY
Source: 2019 IEEE GRSS Data Fusion Contest, Track 1 (WorldView-3 RGB, Jacksonville and Omaha;
dataset by Johns Hopkins University Applied Physics Laboratory / IARPA CORE3D). Changes: tiles
re-encoded losslessly (deflate), previews resized and contrast-stretched.
**The DFC2019 contest terms restrict redistribution of this data. It is published here temporarily
by the repository owner and will be removed; do not reuse or redistribute it.**
Only the {len(ondemand)} tiles the Depth Wizard desktop app downloads on demand are here (no thumbnails).
"""
# First run replaced "## Not included"; later runs replace the DFC2019 section itself.
marker = "## DFC2019 (" if "## DFC2019 (" in readme else "## Not included"
head, sep, _ = readme.partition(marker)
readme = (head if sep else readme.rstrip() + "\n\n") + new_section
subprocess.run(["gh", "api", "-X", "PUT", f"repos/{REPO}/contents/README.md",
                "-f", "message=README: DFC2019 on-demand tiles (temporary, owner decision)",
                "-f", f"content={base64.b64encode(readme.encode()).decode()}", "-f", f"sha={meta['sha']}"],
               check=True, capture_output=True)
print("README updated")
shutil.rmtree(stage)
