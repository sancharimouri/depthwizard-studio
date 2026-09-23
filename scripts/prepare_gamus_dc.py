#!/usr/bin/env python3
"""
GAMUS Washington DC -> leakage-safe train/test arrays for the Method 6 + GAMUS-DC retrain
(2026-09-23). Own code; only the dataset location (HF `earthflow/GAMUS`) is shared with any
audited repo.

Why a new split: GAMUS's own train/val/test splits are interleaved on the DC tile grid
(e.g. DC_20_30 is `train`, its east neighbour DC_20_31 is `test`). Tiles are 1024x1024 at
~0.33 m and, verified on real pixels (see docs), abut exactly with zero overlap:
col+1 = directly east, row+1 = directly south.

New split (a spatial boundary on the grid):
    test   = rows <= TEST_MAX_ROW  (northern tip of DC)
    buffer = rows TEST_MAX_ROW+1 .. TRAIN_MIN_ROW-1, dropped (2 rows = ~680 m)
    train  = N_TRAIN tiles drawn at random (seed 0) from rows >= TRAIN_MIN_ROW,
             one random 512x512 quadrant each
Checks run here, and the run aborts if any fails: (1) no tile name in both sets;
(2) min grid (Chebyshev) distance between any train and test tile >= 3; (3) no non-flat 32x32
RGB block of any test tile appears byte-identically anywhere in the train set (aligned-block
hash; catches duplicated tiles/crops). Adjacency/overlap of the grid itself is checked
separately (probe in the doc).

Output (not committed; regenerate with this script):
    data/gamus_dc/train_quadrants.npz  rgb uint8 (N,3,512,512), agl float32 (N,512,512), tile, quadrant
    data/gamus_dc/test_tiles.npz       rgb uint8 (M,3,1024,1024), agl float32, cls uint8, tile
    data/gamus_dc/split.json           tile lists + check results
Heights: void is a -5.0 sentinel; downstream valid = finite & agl >= 0.
Requires h5py: `uv run --no-project --with h5py --with numpy python scripts/prepare_gamus_dc.py`
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/gamus_dc"
TMP = OUT / "_tmp"
API = "https://huggingface.co/api/datasets/earthflow/GAMUS"
URL = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"
TEST_MAX_ROW, TRAIN_MIN_ROW = 14, 17
N_TRAIN, SEED = 300, 0


def fetch(rel: str) -> np.ndarray:
    p = TMP / rel.replace("/", "__")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL.format(rel), p)
            with h5py.File(p, "r") as f:
                a = f["image"][...]
            p.unlink()
            return a
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError(rel)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)
    meta = json.loads(urllib.request.urlopen(API).read())
    files = {s["rfilename"] for s in meta["siblings"]}
    paths = {}
    for f in files:
        m = re.match(r"(images|heights|classes)/(\w+)/(DC_(\d+)_(\d+))_(RGB|AGL|CLS)\.h5$", f)
        if m:
            paths.setdefault(m.group(3), {})[m.group(1)] = f
    grid = {t: tuple(int(x) for x in t.split("_")[1:]) for t, d in paths.items() if len(d) == 3}
    test = sorted(t for t, (r, c) in grid.items() if r <= TEST_MAX_ROW)
    pool = sorted(t for t, (r, c) in grid.items() if r >= TRAIN_MIN_ROW)
    rng = random.Random(SEED)
    train = sorted(rng.sample(pool, N_TRAIN))
    quad = {t: rng.randrange(4) for t in train}

    # Check 1 + 2: set and grid separation
    assert not set(train) & set(test), "tile in both sets"
    dmin = min(max(abs(grid[a][0] - grid[b][0]), abs(grid[a][1] - grid[b][1])) for a in train for b in test)
    assert dmin >= 3, f"train/test grid distance {dmin} < 3"
    print(f"grid: {len(grid)} DC tiles; test {len(test)} (rows<={TEST_MAX_ROW}), train pool {len(pool)} "
          f"(rows>={TRAIN_MIN_ROW}), train drawn {len(train)}; min train-test grid distance {dmin}")

    def load_train(t):
        rgb = fetch(paths[t]["images"])
        agl = fetch(paths[t]["heights"]).astype(np.float32)
        q = quad[t]
        r0, c0 = (q // 2) * 512, (q % 2) * 512
        return t, rgb[r0:r0 + 512, c0:c0 + 512].transpose(2, 0, 1).copy(), agl[r0:r0 + 512, c0:c0 + 512].copy()

    def load_test(t):
        rgb = fetch(paths[t]["images"])
        return t, rgb.transpose(2, 0, 1).copy(), fetch(paths[t]["heights"]).astype(np.float32), fetch(paths[t]["classes"]).astype(np.uint8)

    with ThreadPoolExecutor(8) as ex:
        tr = list(ex.map(load_train, train))
        print("train downloaded")
        te = list(ex.map(load_test, test))
        print("test downloaded")

    # Check 3: byte-identical 32x32 block hashes (non-flat blocks only)
    def block_hashes(rgb):
        _, H, W = rgb.shape
        out = set()
        for r in range(0, H - 31, 32):
            for c in range(0, W - 31, 32):
                b = rgb[:, r:r + 32, c:c + 32]
                if b.std() > 2:
                    out.add(hashlib.blake2b(b.tobytes(), digest_size=12).digest())
        return out
    test_h = set().union(*(block_hashes(x[1]) for x in te))
    train_h = set().union(*(block_hashes(x[1]) for x in tr))
    shared = len(test_h & train_h)
    print(f"block-hash check: {len(test_h)} test blocks, {len(train_h)} train blocks, shared {shared}")
    assert shared == 0, "identical image blocks shared between train and test"

    np.savez(OUT / "train_quadrants.npz", rgb=np.stack([x[1] for x in tr]), agl=np.stack([x[2] for x in tr]),
             tile=np.array([x[0] for x in tr]), quadrant=np.array([quad[x[0]] for x in tr]))
    np.savez(OUT / "test_tiles.npz", rgb=np.stack([x[1] for x in te]), agl=np.stack([x[2] for x in te]),
             cls=np.stack([x[3] for x in te]), tile=np.array([x[0] for x in te]))
    (OUT / "split.json").write_text(json.dumps({
        "rule": {"test": f"rows <= {TEST_MAX_ROW}", "buffer_dropped": f"rows {TEST_MAX_ROW + 1}-{TRAIN_MIN_ROW - 1}",
                 "train": f"{N_TRAIN} random (seed {SEED}) of rows >= {TRAIN_MIN_ROW}, one random 512 quadrant each"},
        "checks": {"name_overlap": 0, "min_grid_distance": dmin, "shared_32px_block_hashes": shared,
                   "test_blocks_hashed": len(test_h), "train_blocks_hashed": len(train_h)},
        "test": test, "train": {t: quad[t] for t in train}}, indent=1))
    TMP.rmdir()
    print("saved", OUT)


if __name__ == "__main__":
    main()
