#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """
    Convert RGB uint8 image to model input.
    """

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    return image[None, ...].astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path(
            "data/dfc2019/raw/RGB/Track1-RGB"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/"
            "semantic/"
            "hotosm_dinov3s_buildings/"
            "model.onnx"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/dfc2019/experiments/"
            "semantic/building"
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path(
            "data/dfc2019/experiments/"
            "dav2_baseline/depth"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Building model not found:\n{args.model}"
        )

    if not args.rgb_dir.exists():
        raise FileNotFoundError(
            f"RGB directory not found:\n{args.rgb_dir}"
        )

    print("Loading ONNX model...")

    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected one model input, "
            f"found {len(inputs)}"
        )

    input_name = inputs[0].name

    print(
        f"Model input: {input_name}"
    )

    # Use the exact DAv2 benchmark cache as the tile list.
    # The benchmark contains the 50 tiles we have already evaluated,
    # so we do NOT run building segmentation over all 2,783 DFC tiles.

    if not args.depth_dir.exists():
        raise FileNotFoundError(
            f"DAv2 depth directory not found:\n{args.depth_dir}"
        )

    depth_files = sorted(
        args.depth_dir.glob("*_depth.npy")
    )

    if not depth_files:
        raise RuntimeError(
            f"No DAv2 benchmark depth files found in:\n"
            f"{args.depth_dir}"
        )

    tile_ids = [
        depth_path.stem.removesuffix("_depth")
        for depth_path in depth_files
    ]

    print(
        f"Found {len(tile_ids)} benchmark tiles "
        f"from DAv2 cache."
    )

    print(
        f"Model input size: {args.size}x{args.size}"
    )

    for index, tile_id in enumerate(tile_ids, start=1):

        rgb_candidates = [
            args.rgb_dir / f"{tile_id}_RGB.tif",
            args.rgb_dir / f"{tile_id}_RGB.TIF",
        ]

        rgb_path = next(
            (p for p in rgb_candidates if p.exists()),
            None,
        )

        if rgb_path is None:
            print(
                f"[{index}/{len(tile_ids)}] "
                f"[SKIP] Missing RGB: {tile_id}"
            )
            continue

        # IMPORTANT:
        # save as JAX_004_006.npy, NOT JAX_004_006_RGB.npy
        output_path = (
            args.output_dir / f"{tile_id}.npy"
        )

        if output_path.exists():
            print(
                f"[{index}/{len(tile_ids)}] "
                f"[SKIP] Already exists: "
                f"{output_path.name}"
            )
            continue

        print(
            f"\n[{index}/{len(tile_ids)}] "
            f"[RUN] {tile_id}"
        )

        # ----------------------------------------------
        # Read RGB
        # ----------------------------------------------

        image = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"[SKIP] Could not read {rgb_path}"
            )
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_h, original_w = image.shape[:2]

        # ----------------------------------------------
        # Preprocess
        # ----------------------------------------------

        model_input = preprocess(
            image,
            args.size,
        )

        # ----------------------------------------------
        # Inference
        # ----------------------------------------------

        outputs = session.run(
            None,
            {
                input_name: model_input,
            },
        )

        if not outputs:
            raise RuntimeError(
                "Model returned no outputs."
            )

        logits = np.asarray(outputs[0])

        # Expected segmentation output:
        #
        # [1, 1, H, W]
        # or
        # [1, H, W]

        if logits.ndim == 4:
            logits = logits[0]

        if logits.ndim == 3:
            logits = logits[0]

        if logits.ndim != 2:
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{logits.shape}"
            )

        probability = sigmoid(
            logits.astype(np.float32)
        )

        # ----------------------------------------------
        # Restore original tile resolution
        # ----------------------------------------------

        probability = cv2.resize(
            probability,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        probability = np.clip(
            probability,
            0.0,
            1.0,
        ).astype(np.float32)

        # ----------------------------------------------
        # Save
        # ----------------------------------------------

        np.save(
            output_path,
            probability,
        )

        print(
            f"Saved: {output_path}"
        )

        print(
            f"  shape = {probability.shape}"
        )

        print(
            f"  min   = {probability.min():.4f}"
        )

        print(
            f"  mean  = {probability.mean():.4f}"
        )

        print(
            f"  max   = {probability.max():.4f}"
        )

    print(
        "\nDone. Only the DAv2 benchmark tiles were processed."
    )



if __name__ == "__main__":
    main()
