"""Fetch the DAv2-Small checkpoint into a clean offline HF cache (no symlinks) and two
real test inputs (public Sentinel-2 release assets)."""
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

out = Path(sys.argv[1]).resolve()
snap = Path(snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", cache_dir=out / "dl",
                              allow_patterns=["*.json", "*.safetensors"]))
repo = "models--depth-anything--Depth-Anything-V2-Small-hf"
dst = out / "hf" / "hub" / repo
shutil.copytree(snap.parent.parent / "refs", dst / "refs")
shutil.copytree(snap, dst / "snapshots" / snap.name, symlinks=False)
shutil.rmtree(out / "dl")
rel = "https://github.com/sancharimouri/depthwizard2-assets/releases/download/library-v1/"
for name in ("sentinel2-almora__preview.jpg", "sentinel2-almora.tif"):
    urllib.request.urlretrieve(rel + name, out / name)
print("prepared", sorted(p.name for p in out.iterdir()))
