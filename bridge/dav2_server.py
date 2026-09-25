"""TEMPORARY Colab bridge: DAv2-Small /predict behind a Cloudflare quick tunnel (docs/DEPLOY.md).

Same response as space/app.py: raw 518x518 relative depth in the compact "u16-zlib" encoding. Model loading and
preprocessing are copied verbatim from scripts/bench/cpu_inference_bench.py; the only
addition is moving the model/input to the GPU when Colab gives one.

  POST /predict  multipart field "file" (PNG/JPG/TIFF RGB) -> JSON (518x518 relative depth, u16-zlib)
  GET  /health   -> {"ok": true, "device": ..., "model": ...}
"""
import base64
import io
import time
import zlib

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

mid = "depth-anything/Depth-Anything-V2-Small-hf"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- verbatim from scripts/bench/cpu_inference_bench.py -------------------------
proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()
# ---------------------------------------------------------------------------------


def encode_depth(depth) -> dict:
    """Compact wire format "u16-zlib" (docs/DEPLOY.md): min-max quantised to uint16
    (max error (max-min)/131070, ~3e-5 on real outputs), little-endian, zlib, base64.
    ~0.68 MB instead of 1.43 MB for raw float32 at 518x518. Decode:
    q = frombuffer(zlib.decompress(b64decode(data_b64)), "<u2"); depth = min + q/65535*(max-min)."""
    lo, hi = float(depth.min()), float(depth.max())
    q = np.zeros(depth.shape, "<u2") if hi <= lo else np.round((depth - lo) / (hi - lo) * 65535).astype("<u2")
    return {"encoding": "u16-zlib", "shape": list(depth.shape), "min": lo, "max": hi,
            "data_b64": base64.b64encode(zlib.compress(q.tobytes(), 6)).decode("ascii")}
model = model.to(dev)

app = FastAPI(title="DepthWizard2 DAv2-Small bridge (temporary)")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": dev, "model": mid}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Not a readable image ({type(exc).__name__}).") from exc
    t = time.perf_counter()
    # --- verbatim from scripts/bench/cpu_inference_bench.py ---------------------
    x = proc(images=img, return_tensors="pt", do_resize=True, keep_aspect_ratio=False, size={"height": 518, "width": 518})["pixel_values"]
    # -----------------------------------------------------------------------------
    with torch.no_grad():
        depth = model(pixel_values=x.to(dev)).predicted_depth[0].float().cpu().numpy().astype("<f4")
    return {
        "model": mid,
        "kind": "relative_depth",
        "device": dev,
        "source_size": list(img.size),  # (width, height) of the input image
        "infer_s": round(time.perf_counter() - t, 3),
        **encode_depth(depth),
    }
