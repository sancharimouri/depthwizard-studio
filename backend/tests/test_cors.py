"""CORS is an exact allowlist (backend/main.py): a '*' entry is dropped, never honoured."""

from backend import main


def test_wildcard_is_dropped(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://depthwizard-studio.vercel.app, *, http://localhost:5173/")
    assert main._cors_origins() == ["https://depthwizard-studio.vercel.app", "http://localhost:5173"]


def test_empty_means_no_cross_origin(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert main._cors_origins() == []
