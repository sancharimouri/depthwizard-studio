#!/usr/bin/env python3
"""Leaf-on vs leaf-off check (2026-09-23): excess-green index ExG = (2G - R - B)/(R + G + B) on
tree pixels of GAMUS DC (test npz, class 6), GAMUS NYC (8 random tiles, class 6), DFC2019
(50 benchmark tiles, ASPRS 5), and the VHR a_forest crop (AGL > 5 m). Two stages, merged into one JSON:
  uv run --no-project --with h5py --with numpy python scripts/tree_greenness_check.py nyc   # needs h5py
  .venv/bin/python scripts/tree_greenness_check.py                                         # the rest"""
import json, random, sys, os, urllib.request
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def exg(rgb, m):  # rgb (3,H,W)
    r, g, b = (rgb[i][m].astype(np.float64) for i in range(3))
    s = r + g + b + 1e-6
    x = (2 * g - r - b) / s
    return {"n": int(m.sum()), "exg_median": float(np.median(x)), "frac_exg_gt_0.05": float((x > 0.05).mean())}
OUTF = ROOT / "data/vhr_dsm/_diagnostics/tree_greenness.json"
out = json.loads(OUTF.read_text()) if OUTF.exists() else {}
if sys.argv[1:] == ["nyc"]:
    import h5py
    meta = json.loads(urllib.request.urlopen("https://huggingface.co/api/datasets/earthflow/GAMUS").read())
    imgs = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].startswith("images/") and "/NYC_" in s["rfilename"])
    random.seed(2); U = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"
    def get(rel):
        p = ROOT / "data/gamus_dc/_n.h5"; urllib.request.urlretrieve(U.format(rel), p)
        with h5py.File(p) as f: a = f["image"][...]
        os.remove(p); return a
    ms = []
    for f in random.sample(imgs, 8):
        rgb = get(f).transpose(2, 0, 1); cls = get(f.replace("images/", "classes/").replace("_IMG", "_CLS"))
        ms.append(exg(rgb, cls == 6))
    out["gamus_nyc_trees"] = {k: float(np.median([m[k] for m in ms if m["n"] > 1000])) for k in ("exg_median", "frac_exg_gt_0.05")}
    OUTF.write_text(json.dumps(out, indent=1)); print(out["gamus_nyc_trees"]); sys.exit()
import rasterio
z = np.load(ROOT / "data/gamus_dc/test_tiles.npz")
ms = [exg(z["rgb"][i], z["cls"][i] == 6) for i in range(0, len(z["tile"]), 8)]
out["gamus_dc_test_trees"] = {k: float(np.median([m[k] for m in ms if m["n"] > 1000])) for k in ("exg_median", "frac_exg_gt_0.05")}
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_method6_finetune_twinhead as m6
ms = []
for t in m6.tile_ids():
    rgb, agl, valid = m6.load_tile_rgb_agl(t)
    with rasterio.open(m6.TRUTH_DIR / f"{t}_CLS.tif") as s: cls = s.read(1)
    ms.append(exg(rgb, valid & (cls == 5)))
out["dfc2019_jax50_highveg"] = {k: float(np.median([m[k] for m in ms if m["n"] > 1000])) for k in ("exg_median", "frac_exg_gt_0.05")}
import vhr_dsm_pipeline as vp
from rasterio.windows import Window
scene, cr, cc, _ = vp.CROPS["a_forest"]
r0, c0 = cr * vp.CELL + (vp.CELL - vp.SIZE) // 2, cc * vp.CELL + (vp.CELL - vp.SIZE) // 2
with rasterio.open(ROOT / "data/maxar_sanity" / vp.SCENES[scene]) as s: rgb = s.read([1, 2, 3], window=Window(c0, r0, vp.SIZE, vp.SIZE))
with rasterio.open(ROOT / "data/vhr_dsm/a_forest/agl.tif") as s: agl = s.read(1)
out["vhr_a_forest_agl_gt5"] = exg(rgb, np.isfinite(agl) & (agl > 5))
OUTF.write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(f"{k:26s} {v}")
