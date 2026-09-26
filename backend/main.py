"""FastAPI entrypoint.

Local dev proxy for the frontend: the CDSE (Copernicus Data Space Ecosystem)
scene search/preview routes (so the client_secret used to query Sentinel Hub
never reaches the browser bundle) and the curated "Choose from Library"
catalog (/api/library), and the Page 1 input routes (/api/input: upload
inspection with GSD-based tier routing, DEM upload, FABDEM fetch).

Run with: uv run uvicorn backend.main:app --reload --port 8000
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.depth_routes import router as depth_router
from backend.api.facts_routes import router as facts_router
from backend.api.generation_routes import router as generation_router
from backend.api.input_routes import router as input_router
from backend.api.library_routes import router as library_router
from backend.api.routes import router

# DW2_NO_DOTENV=1 (the desktop app): python-dotenv searches upward from this file's
# location, so a packaged app would pick up any .env in a folder above its install path.
if os.environ.get("DW2_NO_DOTENV") != "1":
    load_dotenv()

app = FastAPI(title="DepthWizard2 backend")


def _cors_origins() -> list[str]:
    """CORS_ORIGINS: comma-separated exact origins (scheme://host[:port], no trailing slash).

    Nothing is hard-coded: local dev sets http://localhost:5173 in .env; the deployed
    backend sets the Vercel URL in its environment. The Vite dev proxy is same-origin,
    so plain `npm run dev` works even when this is empty.
    """
    return [o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]


CORS_ORIGINS = _cors_origins()
# Optional, e.g. ^https://depthwizard2(-[a-z0-9-]+)?\.vercel\.app$ for Vercel preview deploys
CORS_ORIGIN_REGEX = os.environ.get("CORS_ORIGIN_REGEX", "").strip() or None
if not CORS_ORIGINS and not CORS_ORIGIN_REGEX:
    logging.getLogger(__name__).warning("CORS_ORIGINS is empty: cross-origin browser calls will be refused.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],  # read by the frontend's api() on 429s
)

app.include_router(router)
app.include_router(library_router)
app.include_router(input_router)
app.include_router(facts_router)
app.include_router(depth_router)
app.include_router(generation_router)
