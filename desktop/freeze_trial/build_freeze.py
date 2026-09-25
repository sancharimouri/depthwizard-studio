"""Cross-platform PyInstaller build for the desktop freeze trial (docs/DESKTOP_FREEZE_TRIAL.md).

  python build_freeze.py --root REPO --hf HF_CACHE_DIR --out DIST [--variant full|no-rasterio-submodules]

"full" carries every fix found on macOS; "no-rasterio-submodules" drops
--collect-submodules rasterio to check that fix is still needed on this platform.
ee / googleapiclient are excluded: FABDEM is a live Earth Engine call with per-user
credentials, so it is not part of the offline binary.
"""
import argparse
from pathlib import Path

import PyInstaller.__main__

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="full", choices=["full", "no-rasterio-submodules"])
a = ap.parse_args()
root, out = Path(a.root).resolve(), Path(a.out).resolve()

args = [
    str(root / "desktop/freeze_trial/dw2_entry.py"),
    "--noconfirm", "--onedir", "--name", "dw2-backend",
    "--distpath", str(out / "dist"), "--workpath", str(out / "build"), "--specpath", str(out),
    "--paths", str(root), "--paths", str(root / "bridge"),
    "--add-data", f"{Path(a.hf).resolve()}:hf",
    "--hidden-import", "gradio_client", "--hidden-import", "huggingface_hub",
    "--exclude-module", "ee", "--exclude-module", "googleapiclient",
    "--hidden-import", "transformers.models.depth_anything.modeling_depth_anything",
    "--hidden-import", "transformers.models.dpt.image_processing_dpt",
    "--collect-submodules", "backend", "--collect-submodules", "pyproj",
    "--collect-data", "rasterio", "--collect-data", "pyproj",
]
if a.variant == "full":
    args += ["--collect-submodules", "rasterio"]
from importlib.metadata import PackageNotFoundError, distribution  # noqa: E402

for m in ["transformers", "torch", "tokenizers", "safetensors", "huggingface-hub", "numpy", "tqdm", "regex",
          "requests", "packaging", "filelock", "pyyaml"]:
    try:
        distribution(m)
    except PackageNotFoundError:  # e.g. requests only came in with earthengine-api
        print(f"(no {m} installed: metadata not copied)")
        continue
    args += ["--copy-metadata", m]
PyInstaller.__main__.run(args)
