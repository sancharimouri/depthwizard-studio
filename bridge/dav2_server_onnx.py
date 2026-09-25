"""DAv2-Small /predict on ONNX Runtime: the torch-free runtime for the desktop app.

Same API and JSON as bridge/dav2_server.py (the torch version, still used by the
Colab bridge and the HF Space). The model is DAv2-Small exported to ONNX by
desktop/freeze_trial/export_onnx.py; preprocessing is the processor's own recipe
(PIL bicubic to 518x518, /255, ImageNet mean/std) in numpy.

Parity vs the torch path (desktop/freeze_trial/parity_onnx.py, 2026-09-26, 3 reference
images): preprocessing bit-identical; depth max|diff| 4.5e-6 / 2.8e-5 / 8.3e-6,
Pearson 1.0000000000. See docs/DESKTOP_APP.md.

  POST /predict  multipart field "file" -> JSON (518x518 relative depth, u16-zlib)
  GET  /health   -> {"ok": true, "device": "cpu", "model": ..., "runtime": "onnxruntime"}
"""
import base64
import io
import os
import time
import zlib

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

mid = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_PATH = os.environ.get("DW2_DAV2_ONNX", "dav2_small.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI(title="DepthWizard2 DAv2-Small (ONNX Runtime)")


def preprocess(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((518, 518), Image.BICUBIC), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def encode_depth(depth) -> dict:
    """Identical to bridge/dav2_server.py encode_depth (compact "u16-zlib" wire format)."""
    lo, hi = float(depth.min()), float(depth.max())
    q = np.zeros(depth.shape, "<u2") if hi <= lo else np.round((depth - lo) / (hi - lo) * 65535).astype("<u2")
    return {"encoding": "u16-zlib", "shape": list(depth.shape), "min": lo, "max": hi,
            "data_b64": base64.b64encode(zlib.compress(q.tobytes(), 6)).decode("ascii")}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": "cpu", "model": mid, "runtime": "onnxruntime"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Not a readable image ({type(exc).__name__}).") from exc
    t = time.perf_counter()
    depth = session.run(None, {"pixel_values": preprocess(img)})[0][0].astype("<f4")
    return {
        "model": mid,
        "kind": "relative_depth",
        "device": "cpu",
        "source_size": list(img.size),  # (width, height) of the input image
        "infer_s": round(time.perf_counter() - t, 3),
        **encode_depth(depth),
    }
