#!/usr/bin/env python3
"""Publish the curated tile library off this machine (docs/STORAGE.md).

  PUBLIC  GitHub Release  {DW2_ASSETS_REPO} @ {DW2_ASSETS_TAG}
          Sentinel-2 (32) and Maxar VHR (6): tile + thumbnail + preview = 114 assets,
          with attribution in the repo README and the release notes.
  PRIVATE Hugging Face dataset {DW2_LIBRARY_DATASET}
          DFC2019 (50): tiles + thumbnails + previews = 150 files, and manifest.json
          (DFC2019's contest terms forbid dissemination, so never public).

Re-runnable (release assets --clobber; HF commits only changed files). Verifies sizes on
GitHub and sha256 on the Hub. Uses the gh CLI (logged in) and HF_TOKEN (env or .env).

  uv run python scripts/publish_library.py --dry-run
  uv run python scripts/publish_library.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.storage import library_store as ls  # noqa: E402
from backend.storage.hf_checkpoints import token as hf_token  # noqa: E402

LIB = ROOT / "data/library"
EXCLUDED_MARKERS = ("landsat", "cbers", "l1c", "brazil", "token_grid", "resolution_transfer", "gamus")

ATTRIBUTION = f"""# Depth Wizard 2 — public tile assets

Image assets for the Depth Wizard 2 (SIH26175) "Choose from Library" catalog. This repo holds
only these release assets and this README; the application code is not published here.

Release [`{ls.GH_TAG}`](https://github.com/{ls.GH_REPO}/releases/tag/{ls.GH_TAG}) contains, per item,
the GeoTIFF tile (`<id>.tif`), a 256 px thumbnail (`<id>__thumb.jpg`) and a 1024 px preview
(`<id>__preview.jpg`). Thumbnails and previews are resized and contrast-stretched: **modified** from the source.

## Sentinel-2 (32 tiles, `sentinel2-*`)
Contains modified Copernicus Sentinel data (2025). Sentinel-2 L2A true colour (B04/B03/B02), 10 m,
cropped to benchmark footprints. Use is governed by the
[Copernicus Sentinel data legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
(free, full and open access; attribution required).

## Maxar Open Data (6 crops, `vhr-*`)
Imagery © Maxar Technologies (Vantor), Maxar Open Data Program, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). **Non-commercial use only.**
Changes: 2048 × 2048 px crops (Sikkim / Darjeeling Himalaya) cut from source scenes
`10300100CF621C00`, `10300100CE8D0400` (2022-03-07) and `1040010073381800` (2022-03-14),
plus the resized thumbnails/previews above.

## Not included
The 50 DFC2019 tiles of the catalog are **not** published: the 2019 IEEE GRSS Data Fusion
Contest terms forbid dissemination of the data. They are kept in private storage.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, text=True, capture_output=capture)


def plan():
    manifest = json.loads((LIB / "manifest.json").read_text())
    public, private = [], []
    items = []
    for item in manifest["items"]:
        src = (ROOT / item["file"]).resolve()
        assert not any(m in str(src).lower() for m in EXCLUDED_MARKERS), f"excluded artifact: {src}"
        names = ls.asset_names(item)
        files = {"tile": src, "thumbnail": LIB / "thumbnails" / item["thumbnail"],
                 "preview": LIB / "previews" / item["preview"]}
        (public if ls.is_public(item) else private).extend((names[k], files[k]) for k in ("tile", "thumbnail", "preview"))
        clean = {k: v for k, v in item.items() if k not in ("file", "r2", "thumbnail", "preview")}
        clean["store"] = "github" if ls.is_public(item) else "hf-private"
        clean["assets"] = names
        items.append(clean)
    remote_manifest = {**{k: v for k, v in manifest.items() if k != "items"},
                       "published": {"github_release": f"{ls.GH_REPO}@{ls.GH_TAG}", "hf_dataset": ls.HF_DATASET},
                       "items": items}
    missing = [str(p) for _, p in public + private if not p.exists()]
    if missing:
        sys.exit(f"missing local files: {missing[:5]}")
    return public, private, remote_manifest


def publish_github(public) -> None:
    if gh("repo", "view", ls.GH_REPO, check=False).returncode != 0:
        gh("repo", "create", ls.GH_REPO, "--public",
           "--description", "Depth Wizard 2 (SIH26175): public Sentinel-2 + Maxar Open Data tile assets (attribution in README)")
        print(f"created public repo {ls.GH_REPO}")
    # README (also the commit the release tag points at)
    existing = gh("api", f"repos/{ls.GH_REPO}/contents/README.md", check=False)
    body = {"message": "Attribution and licence notes", "content": base64.b64encode(ATTRIBUTION.encode()).decode()}
    if existing.returncode == 0:
        body["sha"] = json.loads(existing.stdout)["sha"]
    subprocess.run(["gh", "api", "-X", "PUT", f"repos/{ls.GH_REPO}/contents/README.md", "--input", "-"],
                   input=json.dumps(body), text=True, check=True, capture_output=True)
    if gh("release", "view", ls.GH_TAG, "--repo", ls.GH_REPO, check=False).returncode != 0:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(ATTRIBUTION)
        gh("release", "create", ls.GH_TAG, "--repo", ls.GH_REPO, "--title", "Library tiles v1",
           "--notes-file", fh.name)
        print(f"created release {ls.GH_TAG}")
    with tempfile.TemporaryDirectory() as tmp:
        staged = []
        for name, src in public:
            dst = Path(tmp) / name
            shutil.copyfile(src, dst)
            staged.append(str(dst))
        for i in range(0, len(staged), 20):
            gh("release", "upload", ls.GH_TAG, *staged[i:i + 20], "--repo", ls.GH_REPO, "--clobber")
            print(f"uploaded release assets {i + 1}-{min(i + 20, len(staged))} / {len(staged)}", flush=True)
    assets = json.loads(gh("release", "view", ls.GH_TAG, "--repo", ls.GH_REPO, "--json", "assets").stdout)["assets"]
    remote = {a["name"]: a["size"] for a in assets}
    for name, src in public:
        assert remote.get(name) == src.stat().st_size, f"size mismatch on GitHub: {name}"
    print(f"verified {len(public)} public release assets ({sum(remote.values()):,d} bytes)")


def publish_hf(private, remote_manifest) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    api.create_repo(ls.HF_DATASET, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(ls.HF_DATASET, repo_type="dataset").private, "dataset is not private; refusing"
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in private:
            dst = Path(tmp) / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        (Path(tmp) / ls.MANIFEST_FILE).write_text(json.dumps(remote_manifest, indent=1))
        (Path(tmp) / "README.md").write_text(
            "# Depth Wizard 2 — private library files\n\nDFC2019 tiles, thumbnails and previews, and the catalog "
            "manifest. **Private, not for redistribution**: the 2019 IEEE GRSS Data Fusion Contest terms forbid "
            "dissemination of the data.\n")
        api.upload_folder(folder_path=tmp, repo_id=ls.HF_DATASET, repo_type="dataset",
                          commit_message="library: DFC2019 files + manifest")
    tree = {f.path: f for f in api.list_repo_tree(ls.HF_DATASET, repo_type="dataset", recursive=True)}
    for name, src in private:
        f = tree[name]
        remote_sha = f.lfs.sha256 if getattr(f, "lfs", None) else None
        if remote_sha is not None:
            assert remote_sha == sha256(src), f"sha256 mismatch on the Hub: {name}"
        else:
            assert f.size == src.stat().st_size, f"size mismatch on the Hub: {name}"
    print(f"verified {len(private)} private files + manifest in {ls.HF_DATASET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    public, private, remote_manifest = plan()
    pub_bytes = sum(p.stat().st_size for _, p in public)
    priv_bytes = sum(p.stat().st_size for _, p in private)
    man_bytes = len(json.dumps(remote_manifest, indent=1).encode())
    print(f"PUBLIC  GitHub release {ls.GH_REPO}@{ls.GH_TAG}: {len(public)} files, {pub_bytes:,d} bytes")
    print(f"PRIVATE HF dataset {ls.HF_DATASET}: {len(private)} files + manifest, {priv_bytes + man_bytes:,d} bytes")
    if a.dry_run:
        return
    publish_github(public)
    publish_hf(private, remote_manifest)


if __name__ == "__main__":
    main()
