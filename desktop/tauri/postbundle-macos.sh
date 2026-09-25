#!/bin/bash
# After `npx tauri build`: restore the PyInstaller one-folder's symlinks inside the .app.
# Tauri copies bundle resources by following symlinks, which turned the backend's 48
# library symlinks (libtorch etc., ~457 MB) into duplicate files. Then rebuild the DMG
# from the fixed .app.
set -euo pipefail
cd "$(dirname "$0")"
SRC=src-tauri/sidecar/dw2-backend
BUNDLE=src-tauri/target/release/bundle
APP=$BUNDLE/macos/DepthWizard.app
DST=$APP/Contents/Resources/dw2-backend
n=0
while IFS= read -r -d '' link; do
  rel=${link#"$SRC"/}
  rm -f "$DST/$rel"
  ln -s "$(readlink "$link")" "$DST/$rel"
  n=$((n + 1))
done < <(find "$SRC" -type l -print0)
echo "restored $n symlinks"
DMG=$(ls "$BUNDLE"/dmg/*.dmg)
rm -f "$DMG"
hdiutil create -quiet -volname DepthWizard -srcfolder "$APP" -format UDZO -ov "$DMG"
echo "rebuilt $DMG"
