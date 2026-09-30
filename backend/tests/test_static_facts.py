"""The static library bakes ONLY curated items, for both groups (scripts/bake_static_library.py, facts v2 Part C)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bake_static_library", Path(__file__).resolve().parents[2] / "scripts/bake_static_library.py")
bake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bake)


def _item(status, kind, text, data=None):
    x = {"kind": kind, "label": kind.title(), "text": text, "status": status, "source": "S", "licence": "L",
         "value": {"x": 1}, "id": "t#1", "retrieved": "2026-09-30"}
    if data is not None:
        x["data"] = data
    return x


def test_only_curated_items_are_baked_in_both_groups():
    drafts = {"tiles": {"t1": {
        "facts": [_item("draft", "wildfire", "a"), _item("curated", "peak", "b")],
        "scenario": {"flood": [_item("curated", "rp100", "c")], "landslide": [_item("draft", "wb_class", "d")],
                     "earthquake": [_item("curated", "epicentres", "", {"events": [[1, 2, 5.0, 2000]]})]}}}}
    out = bake.curated_info(drafts, "t1")
    assert [x["text"] for x in out["facts"]] == ["b"]
    assert out["scenario"]["flood"] == [{"kind": "rp100", "label": "Rp100", "text": "c"}]
    assert out["scenario"]["landslide"] == []
    assert out["scenario"]["earthquake"][0]["data"] == {"events": [[1, 2, 5.0, 2000]]}
    # sources, licences, raw values and status never reach the build
    assert all(set(x) <= set(bake.LINE_FIELDS) for x in out["facts"] + out["scenario"]["flood"])


def test_no_drafts_or_unknown_tile_bakes_nothing():
    empty = {"facts": [], "scenario": {"flood": [], "landslide": [], "earthquake": []}}
    assert bake.curated_info({}, "t1") == empty
    assert bake.curated_info({"tiles": {"t2": {"facts": [_item("curated", "peak", "x")]}}}, "t1") == empty
