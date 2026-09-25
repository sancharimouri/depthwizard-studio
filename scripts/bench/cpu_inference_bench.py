"""CPU inference benchmark (one model x one thread count per process -> clean peak RSS)."""
import json, os, resource, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
which, threads = sys.argv[1], int(sys.argv[2])
os.environ["HF_HOME"] = str(ROOT / "models")
import numpy as np, torch
torch.set_num_threads(threads); dev = torch.device("cpu")
from PIL import Image
img = Image.open(ROOT / "data/library/previews/vhr-a_forest.jpg").convert("RGB")  # 1024x1024
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes -> MB
t0 = time.perf_counter()
if which.startswith("dav2"):
    from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil
    mid = {"dav2-small": "depth-anything/Depth-Anything-V2-Small-hf", "dav2-large": "depth-anything/Depth-Anything-V2-Large-hf"}[which]
    proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()
    load = time.perf_counter() - t0
    x = proc(images=img, return_tensors="pt", do_resize=True, keep_aspect_ratio=False, size={"height": 518, "width": 518})["pixel_values"]
    run = lambda: model(pixel_values=x)
    unit = "one 518x518 image"
else:
    import vhr_dsm_pipeline as vp
    folds, full = vp.load_models(dev)
    load = time.perf_counter() - t0
    rgb = np.asarray(img).transpose(2, 0, 1).copy()
    if which == "m6-1fold":
        run = lambda: vp.tiled_predict(folds[0], rgb, dev); unit = "1024x1024 crop, 1 checkpoint (9 tiles of 512)"
    else:
        run = lambda: [vp.tiled_predict(m, rgb, dev) for m in folds]; unit = "1024x1024 crop, 4-fold ensemble (36 tile passes)"
with torch.no_grad():
    run()  # warm-up
    ts = []
    for _ in range(3 if not which.startswith("m6-4") else 1):
        t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
print(json.dumps({"model": which, "threads": threads, "unit": unit, "load_s": round(load, 2),
                  "infer_s_median": round(float(np.median(ts)), 2), "peak_rss_mb": round(rss())}))
