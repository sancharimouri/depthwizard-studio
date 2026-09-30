"""The static library bakes ONLY curated facts (scripts/bake_static_library.py, docs/facts-research.md Part E2)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bake_static_library", Path(__file__).resolve().parents[2] / "scripts/bake_static_library.py")
bake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bake)


def _fact(status, text):
    return {"text": text, "source": "S", "source_url": "u", "licence": "L", "retrieved": "2026-09-30",
            "origin": "derived", "confidence": "high", "scope": "tile", "status": status, "value": {"x": 1}, "kind": "relief"}


def test_only_curated_facts_are_baked():
    drafts = {"tiles": {"t1": {"facts": [_fact("draft", "a"), _fact("curated", "b"), _fact("rejected", "c")]}}}
    out = bake.curated_facts(drafts, "t1")
    assert [f["text"] for f in out] == ["b"]
    assert set(out[0]) == set(bake.FACT_FIELDS)  # internal fields (status, value, kind) stay out of the build


def test_no_drafts_or_unknown_tile_bakes_nothing():
    assert bake.curated_facts({}, "t1") == []
    assert bake.curated_facts({"tiles": {"t2": {"facts": [_fact("curated", "x")]}}}, "t1") == []
