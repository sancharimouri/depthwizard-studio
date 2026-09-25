#!/usr/bin/env python3
"""Upload the curated tile library and the Method 6 checkpoints to Cloudflare R2.

One-time migration off this machine's disk (re-runnable: unchanged objects are
skipped by size + sha256). Keys: backend/storage/r2.py.

  What goes up
    library/tiles/{collection}/{tile_id}.tif   the 88 curated tiles (50 DFC2019,
                                               32 Sentinel-2, 6 Maxar VHR): exactly
                                               the items in data/library/manifest.json;
                                               the excluded research artifacts
                                               (Landsat/CBERS/L1C/...) never enter it
    library/thumbnails/{id}.jpg, library/previews/{id}.jpg
    checkpoints/method6/...                    the 5 checkpoints code actually loads
                                               (r2.CHECKPOINTS)
    library/manifest.json                      last, with each item's R2 keys

  Usage (credentials only via the environment, see docs/R2_SETUP.md):
    uv run python scripts/r2_sync.py --dry-run          # list + total size, no network
    uv run --with boto3 python scripts/r2_sync.py        # upload
    uv run --with boto3 python scripts/r2_sync.py --create-bucket
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.storage import r2  # noqa: E402

LIB = ROOT / "data/library"
MANIFEST = LIB / "manifest.json"
INVENTORY = LIB / "r2_inventory.json"
EXCLUDED_MARKERS = ("landsat", "cbers", "l1c", "brazil", "token_grid", "resolution_transfer", "gamus")
FREE_TIER_BYTES = 10 * 1000**3  # R2 free tier: 10 GB-month of storage


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan() -> tuple[list[tuple[str, Path, str]], dict]:
    manifest = json.loads(MANIFEST.read_text())
    uploads: list[tuple[str, Path, str]] = []
    for item in manifest["items"]:
        src = (ROOT / item["file"]).resolve()
        assert not any(m in str(src).lower() for m in EXCLUDED_MARKERS), f"excluded artifact in catalog: {src}"
        keys = r2.library_keys(item)
        item["r2"] = keys
        uploads.append((keys["tile"], src, "image/tiff"))
        uploads.append((keys["thumbnail"], LIB / "thumbnails" / item["thumbnail"], "image/jpeg"))
        uploads.append((keys["preview"], LIB / "previews" / item["preview"], "image/jpeg"))
    for key, rel in r2.CHECKPOINTS.items():
        uploads.append((key, ROOT / rel, "application/octet-stream"))
    missing = [str(p) for _, p, _ in uploads if not p.exists()]
    if missing:
        sys.exit(f"missing local files ({len(missing)}): {missing[:5]}")
    return uploads, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list and size everything; no network")
    ap.add_argument("--create-bucket", action="store_true", help="create the bucket first if it doesn't exist")
    a = ap.parse_args()

    uploads, manifest = plan()
    manifest_bytes = json.dumps(manifest, indent=1).encode()
    groups: dict[str, list[int]] = {}
    for key, path, _ in uploads:
        group = "/".join(key.split("/")[:3]) if key.startswith("library/tiles") else "/".join(key.split("/")[:2])
        groups.setdefault(group, []).append(path.stat().st_size)
    total = sum(sum(v) for v in groups.values()) + len(manifest_bytes)
    print(f"{'group':42s} {'objects':>7s} {'bytes':>14s}")
    for g, sizes in sorted(groups.items()):
        print(f"{g:42s} {len(sizes):7d} {sum(sizes):14,d}")
    print(f"{r2.MANIFEST_KEY:42s} {1:7d} {len(manifest_bytes):14,d}")
    print(f"{'TOTAL':42s} {len(uploads) + 1:7d} {total:14,d}  = {total / 1000**3:.3f} GB "
          f"({100 * total / FREE_TIER_BYTES:.1f}% of the 10 GB free tier)")
    if total > FREE_TIER_BYTES:
        print("WARNING: this exceeds the R2 free tier (10 GB).")
    if a.dry_run:
        return

    if not r2.has_credentials():
        sys.exit("R2 credentials are not set (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY); see docs/R2_SETUP.md.")
    import boto3  # uv run --with boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    import os

    s3 = boto3.client(
        "s3", endpoint_url=r2.endpoint(), region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 5}),
    )
    bucket = r2.bucket()
    if a.create_bucket:
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError:
            s3.create_bucket(Bucket=bucket)
            print(f"created bucket {bucket}")

    inventory = []
    for key, path, ctype in uploads:
        digest = sha256(path)
        size = path.stat().st_size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] == size and head.get("Metadata", {}).get("sha256") == digest:
                inventory.append({"key": key, "bytes": size, "sha256": digest, "status": "unchanged"})
                continue
        except ClientError:
            pass
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": ctype, "Metadata": {"sha256": digest}})
        inventory.append({"key": key, "bytes": size, "sha256": digest, "status": "uploaded"})
        print(f"up  {key}  {size:,d}", flush=True)
    s3.put_object(Bucket=bucket, Key=r2.MANIFEST_KEY, Body=manifest_bytes, ContentType="application/json")
    MANIFEST.write_text(manifest_bytes.decode())  # local copy carries the same r2 keys
    INVENTORY.write_text(json.dumps({"bucket": bucket, "total_bytes": total, "objects": inventory}, indent=1))
    listed = sum(o["Size"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
                 for o in page.get("Contents", []))
    print(f"done: {sum(1 for o in inventory if o['status'] == 'uploaded')} uploaded, "
          f"{sum(1 for o in inventory if o['status'] == 'unchanged')} unchanged; "
          f"bucket now holds {listed:,d} bytes ({listed / 1000**3:.3f} GB)")


if __name__ == "__main__":
    main()
