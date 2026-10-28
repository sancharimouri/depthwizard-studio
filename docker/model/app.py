"""DepthWizard2 model service on ONNX Runtime (CPU): DAv2-Small relative depth and Method 6 height above ground.

Adapted from bridge/dav2_server_onnx.py (same /predict API and JSON as the HF Space). The weights are baked into the
image at build time (docker/model/build.sh), so the service needs no storage access at run time.

  GET  /health                                -> {"ok": true, ...}
  POST /predict  multipart "file"             -> 518x518 relative depth, u16-zlib (as the Space)
  POST /agl      multipart "file", "gsd_m"    -> Method 6 height above ground in metres, u16-zlib, at the input size

Method 6 is trained on DFC2019 at 0.3 m. Its accuracy collapses on coarser imagery fed at its own pixel count
(Pearson 0.745 at 0.3 m, 0.187 at 1.2 m; resolution-transfer.md, protocol P), so /agl accepts only images of
AGL_MAX_GSD_M or finer and resamples them to 0.3 m spacing first, then runs 512 px tiles reflect-padded to 518
(the training input).
"""
import base64
import io
import os
import time
import zlib

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

Image.MAX_IMAGE_PIXELS = 64_000_000

MODELS = os.environ.get("DW2_MODELS_DIR", "/models")
DAV2_ID = "depth-anything/Depth-Anything-V2-Small-hf"
M6_ID = "method6_full_dfc2019_hb_seed42"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

TRAIN_GSD_M = 0.3      # Method 6's training resolution (DFC2019)
AGL_MAX_GSD_M = 0.6    # coarsest input accepted: at most 4x the pixels after resampling to 0.3 m
AGL_MAX_SIDE = 2048    # largest side after resampling (16 tiles, about 30 s on 2 vCPU)
TILE, PAD = 512, 6     # 512 px tiles reflect-padded to 518, as in training
AGL_NOTE = ("Model-predicted height above ground (Method 6), validated on DFC2019 (US cities, satellite imagery); "
            "did not generalize to GAMUS by RMSE.")


def _session(name: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("DW2_THREADS", "2"))
    return ort.InferenceSession(os.path.join(MODELS, name), so, providers=["CPUExecutionProvider"])


dav2 = _session("dav2_small.onnx")
m6 = _session(f"{M6_ID}.onnx")

app = FastAPI(title="DepthWizard2 models (ONNX Runtime)")


def normalise(a: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 1x3xHxW float32, /255 and ImageNet mean/std."""
    return ((a.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def encode(values: np.ndarray) -> dict:
    """The Space's compact "u16-zlib" wire format: min/max plus a 16-bit quantised grid."""
    lo, hi = float(values.min()), float(values.max())
    q = np.zeros(values.shape, "<u2") if hi <= lo else np.round((values - lo) / (hi - lo) * 65535).astype("<u2")
    return {"encoding": "u16-zlib", "shape": list(values.shape), "min": lo, "max": hi,
            "data_b64": base64.b64encode(zlib.compress(q.tobytes(), 6)).decode("ascii")}


async def read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Not a readable image ({type(exc).__name__}).") from exc


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": "cpu", "runtime": "onnxruntime", "models": [DAV2_ID, M6_ID],
            "agl_max_gsd_m": AGL_MAX_GSD_M}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    img = await read_image(file)
    t = time.perf_counter()
    x = normalise(np.asarray(img.resize((518, 518), Image.BICUBIC)))
    depth = dav2.run(None, {"pixel_values": x})[0][0].astype("<f4")
    return {"model": DAV2_ID, "kind": "relative_depth", "device": "cpu", "source_size": list(img.size),
            "infer_s": round(time.perf_counter() - t, 3), **encode(depth)}


@app.post("/agl")
async def agl(file: UploadFile = File(...), gsd_m: float = Form(...)) -> dict:
    if not (0 < gsd_m <= AGL_MAX_GSD_M):
        raise HTTPException(status_code=422, detail=f"Method 6 runs only on imagery of {AGL_MAX_GSD_M} m/pixel or "
                                                    f"finer (got {gsd_m} m); it is trained at {TRAIN_GSD_M} m.")
    img = await read_image(file)
    w, h = img.size
    scale = gsd_m / TRAIN_GSD_M
    gw, gh = max(1, round(w * scale)), max(1, round(h * scale))
    if max(gw, gh) > AGL_MAX_SIDE:
        raise HTTPException(status_code=413, detail=f"Image too large: {gw}x{gh} px at {TRAIN_GSD_M} m "
                                                    f"(limit {AGL_MAX_SIDE} px a side).")
    t = time.perf_counter()
    a = np.asarray(img.resize((gw, gh), Image.BICUBIC) if (gw, gh) != (w, h) else img)
    th, tw = -(-gh // TILE) * TILE, -(-gw // TILE) * TILE
    a = np.pad(a, ((0, th - gh), (0, tw - gw), (0, 0)), mode="reflect" if min(gh, gw) > 1 else "edge")
    mu = np.zeros((th, tw), np.float32)
    sigma = np.zeros((th, tw), np.float32)
    # ponytail: tiles don't overlap, so seams can show at tile edges; overlap them (e.g. a 192 px margin) if they do
    for r in range(0, th, TILE):
        for c in range(0, tw, TILE):
            x = np.pad(normalise(a[r:r + TILE, c:c + TILE]), ((0, 0), (0, 0), (0, PAD), (0, PAD)), mode="reflect")
            m, lv = (np.asarray(o).reshape(TILE + PAD, TILE + PAD) for o in m6.run(None, {"pixel_values": x}))
            mu[r:r + TILE, c:c + TILE] = m[:TILE, :TILE]
            sigma[r:r + TILE, c:c + TILE] = np.exp(0.5 * lv[:TILE, :TILE])
    infer_s = round(time.perf_counter() - t, 3)
    back = lambda g: np.asarray(Image.fromarray(g[:gh, :gw]).resize((w, h), Image.BILINEAR), "<f4")  # noqa: E731
    return {"model": M6_ID, "kind": "agl_m", "note": AGL_NOTE, "device": "cpu", "source_size": [w, h],
            "gsd_m": gsd_m, "grid_size": [gw, gh], "tiles": (th // TILE) * (tw // TILE), "infer_s": infer_s,
            "agl": encode(back(mu)), "sigma": encode(back(sigma))}
