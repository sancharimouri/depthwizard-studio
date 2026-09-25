"""DepthWizard2 inference Space: DAv2-Small relative depth on plain CPU (CPU Basic).

Called by the Render backend through gradio_client (api_name="/predict"); not a
user-facing UI. The model loading and preprocessing lines are copied verbatim from
scripts/bench/cpu_inference_bench.py (the benchmarked path), because a Space cannot
import from the main repo.

Output (JSON): the raw DAv2 `predicted_depth` for the 518x518 model input in the
compact "u16-zlib" encoding (see encode_depth), plus its shape and range. Relative depth
(larger = nearer), NOT elevation; resize/normalise on the caller side.
"""
import base64
import time
import zlib

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

mid = "depth-anything/Depth-Anything-V2-Small-hf"
torch.set_num_threads(2)  # CPU Basic = 2 vCPU

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


def predict(img):
    if img is None:
        raise gr.Error("No image.")
    img = img.convert("RGB")
    t = time.perf_counter()
    # --- verbatim from scripts/bench/cpu_inference_bench.py ---------------------
    x = proc(images=img, return_tensors="pt", do_resize=True, keep_aspect_ratio=False, size={"height": 518, "width": 518})["pixel_values"]
    # -----------------------------------------------------------------------------
    with torch.no_grad():
        depth = model(pixel_values=x).predicted_depth[0].numpy().astype("<f4")
    return {
        "model": mid,
        "kind": "relative_depth",
        "source_size": list(img.size),  # (width, height) of the input image
        "infer_s": round(time.perf_counter() - t, 3),
        **encode_depth(depth),
    }


demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.JSON(),
                    title="DepthWizard2 DAv2-Small (CPU)", api_name="predict", flagging_mode="never")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
