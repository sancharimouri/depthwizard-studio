"""FastAPI entrypoint.

Local dev proxy for the frontend: the CDSE (Copernicus Data Space Ecosystem)
scene search/preview routes (so the client_secret used to query Sentinel Hub
never reaches the browser bundle) and the curated "Choose from Library"
catalog (/api/library), and the Page 1 input routes (/api/input: upload
inspection with GSD-based tier routing, DEM upload, FABDEM fetch).

Run with: uv run uvicorn backend.main:app --reload --port 8000
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.input_routes import router as input_router
from backend.api.library_routes import router as library_router
from backend.api.routes import router

load_dotenv()

app = FastAPI(title="DepthWizard2 backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(library_router)
app.include_router(input_router)
