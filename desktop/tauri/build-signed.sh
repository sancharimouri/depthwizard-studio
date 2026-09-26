#!/bin/bash
# Signed release build (macOS): tauri build + postbundle-macos.sh, updater artifacts signed.
#   ./build-signed.sh [extra tauri build args, e.g. --config updater-test.conf.json]
# The signing key never enters the repo: private key ~/.tauri/depthwizard2-updater.key (mode 600),
# its password in the macOS Keychain (service "depthwizard2-updater-signing"). See docs/DESKTOP_APP.md.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
export TAURI_SIGNING_PRIVATE_KEY="$(cat "$HOME/.tauri/depthwizard2-updater.key")"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$(security find-generic-password -s depthwizard2-updater-signing -w)"
npx tauri build "$@"
./postbundle-macos.sh
