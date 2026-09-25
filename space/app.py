"""DepthWizard2 inference Space: DAv2-Small relative depth on plain CPU (CPU Basic).

Called by the Render backend through gradio_client (api_name="/predict"); not a
user-facing UI. The model loading and preprocessing lines are copied verbatim from
scripts/bench/cpu_inference_bench.py (the benchmarked path), because a Space cannot
import from the main repo.

Output (JSON): the raw DAv2 `predicted_depth` for the 518x518 model input, as
little-endian float32 bytes in base64, plus its shape and range. Relative depth
(larger = nearer), NOT elevation; resize/normalise on the caller side.
"""
import base64
import time

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil

mid = "depth-anything/Depth-Anything-V2-Small-hf"
torch.set_num_threads(2)  # CPU Basic = 2 vCPU

# --- verbatim from scripts/bench/cpu_inference_bench.py -------------------------
proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()
# ---------------------------------------------------------------------------------


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
        "shape": list(depth.shape),
        "dtype": "float32",
        "source_size": list(img.size),  # (width, height) of the input image
        "min": float(depth.min()),
        "max": float(depth.max()),
        "infer_s": round(time.perf_counter() - t, 3),
        "data_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
    }


demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.JSON(),
                    title="DepthWizard2 DAv2-Small (CPU)", api_name="predict", flagging_mode="never")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
