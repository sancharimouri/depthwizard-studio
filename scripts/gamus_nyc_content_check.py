"""GAMUS NYC image-content check (2026-09-23): 20 random NYC tiles (seed 1), image + height,
shape/dtype/zero-pixel/valid stats and a 4-tile RGB|AGL preview. Run from a scratch dir holding
gamus_meta.json (HF API listing): uv run --no-project --with h5py --with numpy --with pillow --with matplotlib python ..."""
import json, random, urllib.request, os, numpy as np, h5py
from PIL import Image
from matplotlib import cm
m = json.load(open("gamus_meta.json")); files = {x["rfilename"] for x in m["siblings"]}
imgs = sorted(f for f in files if f.startswith("images/") and "/NYC_" in f)
random.seed(1); pick = random.sample(imgs, 20)
U = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/{}"
def get(rel):
    urllib.request.urlretrieve(U.format(rel), "n.h5")
    with h5py.File("n.h5") as f: a = f["image"][...]
    os.remove("n.h5"); return a
rows, stats = [], []
for i, f in enumerate(pick):
    rgb = get(f); h = get(f.replace("images/", "heights/").replace("_IMG", "_AGL")).astype(np.float32)
    v = h > -4.99
    stats.append(dict(tile=f.split("/")[-1], shape=list(rgb.shape), dtype=str(rgb.dtype), rgb_mean=float(rgb.mean()), rgb_std=float(rgb.std()),
                      frac_zero_px=float((rgb.reshape(-1, 3).max(1) == 0).mean()), agl_valid=float(v.mean()), agl_p95=float(np.percentile(h[v], 95)) if v.any() else None))
    if i < 4:
        rows.append(np.concatenate([rgb[::2, ::2], (cm.viridis(np.clip(h[::2, ::2], 0, 30) / 30)[..., :3] * 255).astype(np.uint8)], 1))
Image.fromarray(np.concatenate(rows, 0)).resize((512, 1024)).save("nyc_check.png")
json.dump(stats, open("/Users/anweshasaha/projects/DepthWizard2/data/vhr_dsm/_diagnostics/gamus_nyc_content_check.json", "w"), indent=1)
for s in stats: print(s)
