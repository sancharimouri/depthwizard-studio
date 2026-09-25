"""ONNX vs torch parity for DAv2-Small (docs/DESKTOP_APP.md). Usage: parity_onnx.py MODEL.onnx HF_HOME REPO_ROOT"""
import os, sys, numpy as np, torch
os.environ["HF_HOME"] = sys.argv[2]; os.environ["HF_HUB_OFFLINE"] = "1"
import onnxruntime as ort
from PIL import Image
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil
mid = "depth-anything/Depth-Anything-V2-Small-hf"
proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()
sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def np_preprocess(img):  # torch-free: PIL bicubic 518x518, /255, ImageNet normalise, NCHW
    a = np.asarray(img.convert("RGB").resize((518, 518), Image.BICUBIC), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

def u16(d):  # the wire format the UI receives (encode_depth)
    lo, hi = float(d.min()), float(d.max()); q = np.round((d - lo) / (hi - lo) * 65535).astype("<u2")
    return lo + q.astype(np.float64) / 65535 * (hi - lo)

stats = lambda a, b: (float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()), float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
R = sys.argv[3]
for name in ("sentinel2-almora", "dfc2019-JAX_004_006", "vhr-a_forest"):
    img = Image.open(f"{R}/data/library/previews/{name}.jpg")
    x_tf = proc(images=img, return_tensors="pt", do_resize=True, keep_aspect_ratio=False, size={"height": 518, "width": 518})["pixel_values"]
    with torch.no_grad(): ref = model(pixel_values=x_tf).predicted_depth[0].numpy()
    ort_same = sess.run(None, {"pixel_values": x_tf.numpy()})[0][0]
    x_np = np_preprocess(img)
    ort_full = sess.run(None, {"pixel_values": x_np})[0][0]
    pre = float(np.abs(x_np - x_tf.numpy()).max())
    a, b, c = stats(ort_same, ref), stats(ort_full, ref), stats(u16(ort_full), u16(ref))
    print(f"{name:20s} range {ref.min():.3f}..{ref.max():.3f}")
    print(f"   A model only (same input)      max|diff| {a[0]:.2e}  pearson {a[1]:.10f}")
    print(f"   B preprocessing np vs transformers  max|diff| {pre:.2e}")
    print(f"   C full pipeline (np + ORT)     max|diff| {b[0]:.2e}  pearson {b[1]:.10f}")
    print(f"   D as the UI receives (u16)     max|diff| {c[0]:.2e}  pearson {c[1]:.10f}")
