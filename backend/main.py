"""FastAPI entrypoint.

Local dev proxy for the frontend — currently just the CDSE (Copernicus Data
Space Ecosystem) scene search/preview routes, so the client_secret used to
query Sentinel Hub never reaches the browser bundle.

Run with: uv run uvicorn backend.main:app --reload --port 8000
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
