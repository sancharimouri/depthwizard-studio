"""Method 6 full model -> ONNX, SCRATCH ONLY (docs/method6-checkpoint-audit.md §7.4): exported to build/, never
committed or shipped.

  .venv/bin/python scripts/method6_onnx_export.py data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019_hb_seed42.pt

- Exports TwinHeadDav2GSD(enable_gsd_film=False) with the checkpoint's height_scale baked in: input
  pixel_values 1x3x518x518 (ImageNet-normalised, reflect-padded quadrant), outputs mu (AGL, m) and log_var.
  Opset 17, TorchScript exporter, as desktop/freeze_trial/export_onnx.py.
- Parity vs PyTorch (CPU, float32) on 3 DFC2019 tiles x 4 quadrants: max |mu diff| (m) and Pearson.
Writes build/method6_onnx/<stem>.onnx and parity.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/method6_onnx"
os.environ.setdefault("HF_HOME", str(ROOT / "build/hf"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import method6_checkpoint_audit as A  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        mu, log_var = self.m(pixel_values)
        return mu, log_var


def prep(rgb_c: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_c / 255.0).float()
    x = ((x - A.IMAGENET_MEAN[0]) / A.IMAGENET_STD[0]).float()
    return A.pad_to(x, A.PAD_TO)[None]


def main():
    ckpt = Path(sys.argv[1]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / f"{ckpt.stem}.onnx"
    cpu = torch.device("cpu")
    model, meta, _ = A.load(ckpt, cpu)
    model = model.float().eval()
    x = torch.randn(1, 3, A.PAD_TO, A.PAD_TO)
    t0 = time.time()
    torch.onnx.export(Wrap(model), (x,), str(onnx_path), input_names=["pixel_values"], output_names=["mu", "log_var"],
                      opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(str(onnx_path))
    export_s = time.time() - t0

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tids = A.tile_ids()
    tiles = [tids[0], tids[len(tids) // 2], tids[-1]]
    diffs, pt_all, ox_all, per = [], [], [], []
    with torch.no_grad():
        for t in tiles:
            arrs = A.load_tile_rgb_agl(t)
            for q in range(4):
                rgb_c, _agl, _v = A.quad(arrs, q)
                xin = prep(rgb_c)
                mu_pt = model(xin)[0][0, 0].numpy()
                mu_ox = sess.run(["mu"], {"pixel_values": xin.numpy()})[0][0, 0]
                d = float(np.max(np.abs(mu_pt - mu_ox)))
                diffs.append(d)
                pt_all.append(mu_pt.ravel()); ox_all.append(mu_ox.ravel())
                per.append({"tile": t, "quadrant": q, "max_abs_diff_m": d})
    P, O = np.concatenate(pt_all).astype(np.float64), np.concatenate(ox_all).astype(np.float64)
    res = {"checkpoint": str(ckpt.relative_to(ROOT)), "onnx": str(onnx_path.relative_to(ROOT)),
           "onnx_bytes": onnx_path.stat().st_size, "opset": 17, "input": [1, 3, A.PAD_TO, A.PAD_TO],
           "outputs": ["mu (AGL m)", "log_var"], "height_scale_baked": float(meta["height_scale"]),
           "export_seconds": round(export_s, 1), "parity_tiles": tiles, "n_quadrants": len(per),
           "max_abs_diff_m": float(max(diffs)), "mean_abs_diff_m": float(np.mean(np.abs(P - O))),
           "pearson": float(np.corrcoef(P, O)[0, 1]), "per_quadrant": per}
    (OUT / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_quadrant"}, indent=1))


if __name__ == "__main__":
    main()
