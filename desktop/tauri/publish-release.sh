#!/bin/bash
# Publish the built, signed desktop release on the PUBLIC GitHub repo that the in-app
# updater reads (plugins.updater.endpoints in src-tauri/tauri.conf.json):
#   https://github.com/sancharimouri/depthwizard2-desktop/releases/latest/download/latest.json
# Run from the repo root after ./desktop/tauri/build-signed.sh:
#   desktop/tauri/publish-release.sh
# Uploads the DMG (installer), DepthWizard.app.tar.gz + .sig (update artifact) and
# latest.json, whose "version" is the built app's own version (they must match).
set -euo pipefail
cd "$(dirname "$0")"
REPO=sancharimouri/depthwizard2-desktop
B=src-tauri/target/release/bundle
APP=$B/macos/DepthWizard.app
VERSION=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")
TAG="v$VERSION"
DMG=$(ls "$B"/dmg/DepthWizard_"$VERSION"_*.dmg)
TGZ=$B/macos/DepthWizard.app.tar.gz
[ -f "$TGZ.sig" ] || { echo "missing $TGZ.sig: build with build-signed.sh"; exit 1; }

OUT=$(mktemp -d)
python3 - "$VERSION" "$TGZ.sig" "$REPO" "$TAG" > "$OUT/latest.json" <<'EOF'
import datetime, json, sys
version, sig, repo, tag = sys.argv[1:]
print(json.dumps({
    "version": version,
    "notes": "Depth Wizard desktop " + version,
    "pub_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "platforms": {"darwin-aarch64": {
        "signature": open(sig).read().strip(),
        "url": f"https://github.com/{repo}/releases/download/{tag}/DepthWizard.app.tar.gz"}},
}, indent=1))
EOF

if ! gh repo view "$REPO" > /dev/null 2>&1; then
  gh repo create "$REPO" --public --description "Depth Wizard (SIH26175) desktop app: releases and in-app updates"
  cat > "$OUT/README.md" <<'EOF'
# Depth Wizard: desktop app (SIH26175)

Releases of the Depth Wizard desktop app (macOS, Apple silicon). The app checks this repo's
latest release on launch (`latest.json`) and offers signed updates in-app.
This repo holds only releases and this README; the application source is not published here.

**Install:** download `DepthWizard_<version>_aarch64.dmg` from the latest release and drag the app
to Applications. The app is not signed with an Apple Developer ID yet, so macOS warns on first
open: right-click the app, choose Open, then confirm.

**Update artifacts** (`DepthWizard.app.tar.gz` + `.sig`) are signed with the project's updater key;
the app refuses any update whose signature does not match.

## What's inside, and its terms
- Relative depth: Depth Anything V2 Small ([Apache-2.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)),
  run with ONNX Runtime (MIT).
- Sentinel-2 tiles: contains modified Copernicus Sentinel data (2025),
  [Sentinel data legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
- Maxar Open Data crops: © Maxar Technologies (Vantor), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial only.
- DFC2019 tiles: 2019 IEEE GRSS Data Fusion Contest (Track 1). **The contest terms restrict
  redistribution; they are included temporarily by the repository owner and will be removed.
  Do not reuse or redistribute them.**
EOF
  gh api -X PUT "repos/$REPO/contents/README.md" -f message="README" \
    -f content="$(base64 < "$OUT/README.md" | tr -d '\n')" > /dev/null
fi

gh release create "$TAG" -R "$REPO" --title "Depth Wizard $VERSION" --latest \
  --notes "Depth Wizard desktop $VERSION (macOS, Apple silicon). Installer: the .dmg. The .app.tar.gz, .sig and latest.json are for in-app updates." \
  "$DMG" "$TGZ" "$TGZ.sig" "$OUT/latest.json"
rm -rf "$OUT"
echo "published $REPO $TAG"
