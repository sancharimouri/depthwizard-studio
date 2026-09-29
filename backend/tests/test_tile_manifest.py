"""Curated tile manifest (backend/library/tile_manifest.py) and the optional DISPLAY mesh field
(backend/terrain/mesh_export.py). No network, no real library files."""
import json

import numpy as np

from backend.library import tile_manifest
from backend.terrain.mesh_export import write_terrain_json


def _manifest(tmp_path, tiles):
    p = tmp_path / "tile_manifest.json"
    p.write_text(json.dumps({"tiles": tiles}))
    return p


def test_no_manifest_leaves_the_catalog_untouched(monkeypatch):
    monkeypatch.delenv("DW2_TILE_MANIFEST", raising=False)
    items = [{"id": "b"}, {"id": "a"}]
    assert tile_manifest.curate(items) is items
    assert tile_manifest.default_exaggeration("a") is None


def test_curate_filters_orders_and_drops_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("DW2_TILE_MANIFEST", str(_manifest(tmp_path, [
        {"tile_id": "a", "include_in_container": True, "order_index": 2, "default_exaggeration": 1.4},
        {"tile_id": "b", "include_in_container": True, "order_index": 0, "default_exaggeration": None},
        {"tile_id": "c", "include_in_container": False, "order_index": 1, "default_exaggeration": 3.0},
    ])))
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "not-in-manifest"}]
    assert [i["id"] for i in tile_manifest.curate(items)] == ["b", "a"]
    assert tile_manifest.default_exaggeration("a") == 1.4
    assert tile_manifest.default_exaggeration("b") is None


def test_display_field_is_optional_and_separate(tmp_path):
    surface = np.arange(16, dtype=np.float32).reshape(4, 4)
    p = tmp_path / "t.json"
    write_terrain_json(p, surface, (0, 0, 1, 1), (4, 4))
    assert "display" not in json.loads(p.read_text())
    write_terrain_json(p, surface, (0, 0, 1, 1), (4, 4), display=surface * 10)
    d = json.loads(p.read_text())
    assert (d["elevationMin"], d["elevationMax"]) == (0.0, 15.0)           # real surface unchanged
    assert (d["display"]["elevationMin"], d["display"]["elevationMax"]) == (0.0, 150.0)
    assert len(d["display"]["heights"]) == len(d["heights"]) == 16


def test_limit_outliers_flag_only_when_disabled(tmp_path):
    surface = np.arange(16, dtype=np.float32).reshape(4, 4)
    p = tmp_path / "t.json"
    write_terrain_json(p, surface, (0, 0, 1, 1), (4, 4))
    assert "limitOutliers" not in json.loads(p.read_text())     # default: unchanged output, viewer limits as before
    write_terrain_json(p, surface, (0, 0, 1, 1), (4, 4), limit_outliers=False)
    assert json.loads(p.read_text())["limitOutliers"] is False  # DFC2019 lidar: the viewer skips the limiter
