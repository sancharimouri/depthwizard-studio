#!/usr/bin/env python3
"""Part C.0 leaf-on pre-check (2026-09-23, pre-registered in docs/method-audit/07-gamus-generalization/log.md).

Per GAMUS city (DC, NYC, PHL): 40 random tiles drawn from all splits (seed 0). Per tile, the share of
class-6 (tree) pixels with ExG = (2G-R-B)/(R+G+B) > 0.05, the same measure as scripts/tree_greenness_check.py.
Tiles with < 1000 tree pixels are skipped. City value = median over tiles. Leaf-on threshold: >= 0.40.
Tiles are streamed and deleted.
  uv run --no-project --with h5py --with numpy python scripts/gamus_leafon_precheck.py
Output: data/gamus_eval/leafon_precheck.json
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/gamus_eval/leafon_precheck.json"
N_PER_CITY, SEED, THRESH = 40, 0, 0.40


sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamus_io import get, tile_paths, list_files  # noqa: E402


def tile_stat(img_rel):
    rgb = get(img_rel).astype(np.float64)
    cls = get(tile_paths(img_rel)[1])
    m = cls == 6
    if m.sum() < 1000:
        return {"tile": img_rel, "n_tree": int(m.sum()), "frac": None}
    r, g, b = rgb[..., 0][m], rgb[..., 1][m], rgb[..., 2][m]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    return {"tile": img_rel, "n_tree": int(m.sum()), "frac": float((exg > 0.05).mean()),
            "exg_median": float(np.median(exg))}


def main():
    imgs = [f for f in list_files() if f.startswith("images/")]
    rng = random.Random(SEED)
    out = {"rule": f"city leaf-on if median per-tile share of tree px with ExG>0.05 >= {THRESH}",
           "n_per_city": N_PER_CITY, "seed": SEED, "cities": {}}
    for city in ("DC", "NYC", "PHL"):
        pick = rng.sample([f for f in imgs if f.split("/")[-1].startswith(city + "_")], N_PER_CITY)
        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(tile_stat, pick))
        fr = [r["frac"] for r in rows if r["frac"] is not None]
        out["cities"][city] = {"n_tiles_used": len(fr), "median_frac_green": float(np.median(fr)),
                               "p25": float(np.percentile(fr, 25)), "p75": float(np.percentile(fr, 75)),
                               "max_tile": float(np.max(fr)),
                               "n_tiles_ge_thresh": int(sum(f >= THRESH for f in fr)),
                               "leaf_on": bool(np.median(fr) >= THRESH), "tiles": rows}
        print(city, {k: v for k, v in out["cities"][city].items() if k != "tiles"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
