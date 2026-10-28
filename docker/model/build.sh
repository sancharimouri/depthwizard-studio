#!/usr/bin/env bash
# Builds and pushes the model service image (docker/model). The weights are copied from the private bucket at build
# time and checked against their stored SHA-256 before they go into the image.
#
#   WEIGHTS_URI=gs://<bucket> IMAGE=<region>-docker.pkg.dev/<project>/<repo>/depthwizard-model:<tag> docker/model/build.sh
set -euo pipefail
: "${WEIGHTS_URI:?set WEIGHTS_URI to the bucket holding the .onnx files}" "${IMAGE:?set IMAGE to the image to push}"
cd "$(dirname "$0")"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
for m in dav2_small.onnx method6_full_dfc2019_hb_seed42.onnx; do
  gcloud storage cp "$WEIGHTS_URI/$m" "$WEIGHTS_URI/$m.sha256" "$W/"
done
(cd "$W" && sha256sum -c ./*.sha256)
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --build-context weights="$W" -t "$IMAGE" --push .
