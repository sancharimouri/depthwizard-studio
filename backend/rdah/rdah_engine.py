"""RDAH-Net inference for DepthWizard2."""

from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
import rasterio

from backend.rdah.model import HeightPredTransformer


class RDAHEngine:
    def __init__(self, checkpoint: Path):
        self.device = self._select_device()

        self.model = HeightPredTransformer()

        ckpt = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. "
                f"Missing={missing}, Unexpected={unexpected}"
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _pad_to_1024(
        rgb: np.ndarray,
        depth: np.ndarray,
    ):
        """Pad both inputs to a size compatible with RDAH attention."""

        h, w = rgb.shape[:2]

        target_h = max(1024, h)
        target_w = max(1024, w)

        # RDAH uses 8x8 attention blocks.
        target_h = ((target_h + 7) // 8) * 8
        target_w = ((target_w + 7) // 8) * 8

        rgb_pad = np.zeros(
            (target_h, target_w, 3),
            dtype=rgb.dtype,
        )

        depth_pad = np.zeros(
            (target_h, target_w),
            dtype=depth.dtype,
        )

        rgb_pad[:h, :w] = rgb
        depth_pad[:h, :w] = depth

        return rgb_pad, depth_pad

    @staticmethod
    def _prepare_rgb(rgb: np.ndarray) -> torch.Tensor:
        """Match RDAH RGB preprocessing."""

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32)

            lo = rgb.min()
            hi = rgb.max()

            if hi > lo:
                rgb = (rgb - lo) / (hi - lo) * 255.0
            else:
                rgb = np.zeros_like(rgb)

            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # HWC -> CHW and [0,255] -> [0,1]
        tensor = torch.from_numpy(
            rgb.astype(np.float32)
        ).permute(2, 0, 1) / 255.0

        # Official ImageNet normalization.
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
        ).view(3, 1, 1)

        return (tensor - mean) / std

    @staticmethod
    def _prepare_depth(
        relative_depth: np.ndarray,
    ) -> torch.Tensor:
        """Prepare 1-channel DAv2 relative depth."""

        return torch.from_numpy(
            relative_depth.astype(np.float32)
        ).unsqueeze(0)

    def predict(
        self,
        rgb: np.ndarray,
        relative_depth: np.ndarray,
    ) -> np.ndarray:

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"RGB must be HxWx3, got {rgb.shape}"
            )

        if relative_depth.ndim != 2:
            raise ValueError(
                f"Relative depth must be HxW, got {relative_depth.shape}"
            )

        original_h, original_w = rgb.shape[:2]

        # Make sure depth matches RGB before padding.
        if relative_depth.shape != (original_h, original_w):
            depth_img = Image.fromarray(
                relative_depth.astype(np.float32),
                mode="F",
            )

            depth_img = depth_img.resize(
                (original_w, original_h),
                Image.Resampling.BILINEAR,
            )

            relative_depth = np.asarray(
                depth_img,
                dtype=np.float32,
            )

        rgb_resized = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize(
                (1024, 1024),
                Image.Resampling.BILINEAR,
            )
        )

        depth_resized = np.asarray(
            Image.fromarray(
                relative_depth.astype(np.float32),
                mode="F",
            ).resize(
                (1024, 1024),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

        rgb_tensor = self._prepare_rgb(rgb_resized)
        depth_tensor = self._prepare_depth(depth_resized)

        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(
                depth_tensor,
                rgb_tensor,
            )

        prediction = (
            prediction
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        prediction = np.asarray(
            Image.fromarray(
                prediction,
                mode="F",
            ).resize(
                (original_w, original_h),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

        return prediction


def run_darjeeling(
    rgb_path: Path,
    depth_path: Path,
    output_dir: Path,
    checkpoint: Path,
) -> dict:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.moveaxis(rgb, 0, -1)

    relative_depth = np.load(depth_path)

    engine = RDAHEngine(checkpoint)

    start = time.perf_counter()

    prediction = engine.predict(
        rgb,
        relative_depth,
    )

    seconds = time.perf_counter() - start

    npy_path = (
        output_dir /
        "Darjeeling_RDAH_nDSM.npy"
    )

    png_path = (
        output_dir /
        "Darjeeling_RDAH_nDSM.png"
    )

    np.save(npy_path, prediction)

    finite = np.isfinite(prediction)

    if finite.any():
        lo = float(np.nanpercentile(prediction, 2))
        hi = float(np.nanpercentile(prediction, 98))

        scaled = np.clip(
            (prediction - lo) /
            (hi - lo + 1e-8),
            0,
            1,
        )

        Image.fromarray(
            (scaled * 255).astype(np.uint8)
        ).save(png_path)

    return {
        "device": engine.device,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(relative_depth.shape),
        "prediction_shape": list(prediction.shape),
        "prediction_min": float(np.nanmin(prediction)),
        "prediction_max": float(np.nanmax(prediction)),
        "prediction_mean": float(np.nanmean(prediction)),
        "nan_count": int(np.isnan(prediction).sum()),
        "inf_count": int(np.isinf(prediction).sum()),
        "seconds": seconds,
        "npy": str(npy_path),
        "png": str(png_path),
    }