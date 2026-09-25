"""Export DAv2-Small to ONNX for the desktop app (bridge/dav2_server_onnx.py).

Loading code verbatim from scripts/bench/cpu_inference_bench.py. Needs torch + transformers
(not bundled in the app). Usage: python export_onnx.py OUT.onnx HF_HOME_WITH_THE_MODEL
"""
import os, sys, torch
os.environ["HF_HOME"] = sys.argv[2]; os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil
mid = "depth-anything/Depth-Anything-V2-Small-hf"
proc = DPTImageProcessorPil.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid, use_safetensors=True).eval()

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values): return self.m(pixel_values=pixel_values).predicted_depth

x = torch.randn(1, 3, 518, 518)
torch.onnx.export(Wrap(model), (x,), sys.argv[1], input_names=["pixel_values"], output_names=["predicted_depth"],
                  opset_version=17, dynamo=False)
import onnx; onnx.checker.check_model(sys.argv[1]); print("exported", os.path.getsize(sys.argv[1]) / 1e6, "MB")
