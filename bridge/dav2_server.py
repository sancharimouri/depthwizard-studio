"""TEMPORARY Colab bridge: DAv2-Small /predict behind a Cloudflare quick tunnel (docs/DEPLOY.md).

Same response shape as the HF Space attempt (space/app.py). Model loading and
preprocessing are copied verbatim from scripts/bench/cpu_inference_bench.py; the only
addition is moving the model/input to the GPU when Colab gives one.

  POST /predict  multipart field "file" (PNG/JPG/TIFF RGB) -> JSON (raw 518x518 relative depth)
  GET  /health   -> {"ok": true, "device": ..., "model": ...}
"""
import base64
import io
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

mid = "depth-anything/Depth-Anything-V2-Small-hf"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- verbatim from scripts/bench/cpu_inference_bench.py -------------------------
proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()
# ---------------------------------------------------------------------------------
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
        "shape": list(depth.shape),
        "dtype": "float32",
        "source_size": list(img.size),  # (width, height) of the input image
        "min": float(depth.min()),
        "max": float(depth.max()),
        "infer_s": round(time.perf_counter() - t, 3),
        "data_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
    }
