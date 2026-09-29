# syntax=docker/dockerfile:1.7
# DepthWizard2 web backend for Cloud Run (docs/container-measurements.md, docs/deploy-cloud-run.md).
# No tile data, no models, no credentials in the image: the library is static on Vercel, depth inference is the
# HF ZeroGPU Space, and secrets come from Secret Manager at run time.
#
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t dw2-backend .

# ------------------------------------------------------------------------------------------------ builder
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends binutils && rm -rf /var/lib/apt/lists/*
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
# The BuildKit cache mount keeps pip's download/wheel cache OUTSIDE the image (so the layer carries no cache, the
# effect of --no-cache-dir) while re-builds with unchanged requirements reuse it.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt
RUN set -eux; \
    SP=/venv/lib/python3.11/site-packages; \
    # hf-xet: huggingface_hub's optional Xet transfer backend; never imported by the web backend (import audit)
    pip uninstall -y hf-xet; \
    # the Google API client's bundled discovery documents (~500 APIs); Earth Engine fetches its own document
    rm -rf $SP/googleapiclient/discovery_cache/documents; \
    # test suites shipped inside packages (numpy/testing is kept: it is a runtime module)
    find $SP -depth -type d -name tests -exec rm -rf {} +; \
    # the bytecode shipped by the wheels (rebuilt once, below)
    find $SP -depth -type d -name __pycache__ -exec rm -rf {} +; \
    # debug symbols of every compiled library
    find $SP -type f \( -name '*.so' -o -name '*.so.*' \) -exec strip --strip-unneeded {} + ; \
    # the installers themselves
    pip uninstall -y pip setuptools wheel || true; \
    # Bytecode is compiled ONCE here. Measured 2026-09-30 (docs/container-measurements.md): without .pyc every import
    # compiles in memory, so start-up was 0.83 vs 0.45 s and peak RSS +10-17 MiB per flow, for 11.8 MB less image.
    # Memory is the priority, so the .pyc files stay. PYTHONDONTWRITEBYTECODE=1 still stops any run-time writes.
    python -m compileall -q -j0 $SP

# ------------------------------------------------------------------------------------------------ runtime
FROM python:3.11-slim
# libexpat1: the rasterio wheel's GDAL needs the system libexpat.so.1, which python:3.11-slim (trixie) lacks
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app
COPY --from=builder /venv /venv
WORKDIR /app
COPY --chown=app:app backend/ backend/
RUN python -m compileall -q backend
ENV PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DW2_NO_DOTENV=1 \
    DW2_UPLOADS_DIR=/tmp/dw2/up \
    DW2_CACHE_DIR=/tmp/dw2/cache \
    DW2_GENERATED_DIR=/tmp/dw2/gen \
    PORT=8080
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/health', timeout=2)"
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --no-server-header"]
