#!/usr/bin/env python3
"""
Method 4 — Learned Scale Modulation

DFC2019, 4-fold spatial holdout.

For each fold:
  * Hold out one quadrant from every tile.
  * Train a small CNN on the other 3 quadrants across all 50 tiles.
  * Input = RGB + normalized DAv2.
  * Output = multiplicative depth scale + additive metric residual.
  * Evaluate only on the held-out quadrants.

No DFC CLS labels are used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset


SEED = 42
PATCH = 64
TRAIN_PATCHES_PER_QUADRANT = 12
MIN_VALID_FRACTION = 0.85
EPOCHS = 12
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4


@dataclass
class Tile:
    tile_id: str
    depth_path: Path
    rgb_path: Path
    truth_path: Path


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tile(tile: Tile):
    depth = np.load(tile.depth_path).astype(np.float32)
    with rasterio.open(tile.rgb_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    with rasterio.open(tile.truth_path) as src:
        truth = src.read(1).astype(np.float32)

    rgb = np.moveaxis(rgb, 0, -1)

    if not (depth.shape == truth.shape == rgb.shape[:2]):
        raise ValueError(
            f"{tile.tile_id}: shape mismatch: "
            f"depth={depth.shape}, rgb={rgb.shape}, truth={truth.shape}"
        )

    valid = (
        np.isfinite(depth)
        & np.isfinite(truth)
        & (truth >= 0)
    )

    return rgb, depth, truth, valid


def quadrant_bounds(h: int, w: int, q: int):
    hm, wm = h // 2, w // 2
    if q == 0:
        return 0, hm, 0, wm
    if q == 1:
        return 0, hm, wm, w
    if q == 2:
        return hm, h, 0, wm
    return hm, h, wm, w


def sample_patch_origins(valid: np.ndarray, bounds, count: int, patch: int, rng):
    r0, r1, c0, c1 = bounds
    candidates = []

    max_r = r1 - patch
    max_c = c1 - patch
    if max_r < r0 or max_c < c0:
        return []

    # Random candidate search; small and deterministic.
    attempts = max(count * 100, 500)
    for _ in range(attempts):
        rr = int(rng.integers(r0, max_r + 1))
        cc = int(rng.integers(c0, max_c + 1))
        vm = valid[rr:rr + patch, cc:cc + patch]
        if vm.mean() >= MIN_VALID_FRACTION:
            candidates.append((rr, cc))
            if len(candidates) >= count:
                break
    return candidates


class PatchDataset(Dataset):
    def __init__(self, samples, rgb_stats, depth_stats):
        self.samples = samples
        self.rgb_mean, self.rgb_std = rgb_stats
        self.depth_mean, self.depth_std = depth_stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, depth, truth = self.samples[idx]

        rgb = rgb / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        depth_n = (depth - self.depth_mean) / self.depth_std

        x = np.concatenate(
            [rgb, depth_n[..., None]],
            axis=-1,
        )

        x = torch.from_numpy(
            np.transpose(x, (2, 0, 1)).astype(np.float32)
        )
        y = torch.from_numpy(truth[None].astype(np.float32))

        mask = torch.isfinite(y) & (y >= 0)
        y = torch.nan_to_num(y, nan=0.0)
        return x, y, mask


class ScaleModulationNet(nn.Module):
    """Small RGB+DAv2 correction head."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.scale_head = nn.Conv2d(64, 1, 1)
        self.residual_head = nn.Conv2d(64, 1, 1)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x, raw_depth):
        z = self.features(x)

        # Scale starts at 1.0 and is constrained to [0.25, 4.0].
        scale = torch.exp(
            math.log(4.0) * torch.tanh(self.scale_head(z))
        )

        # Metric residual constrained to ±50 m.
        residual = 50.0 * torch.tanh(
            self.residual_head(z)
        )

        pred = raw_depth * scale + residual
        return pred, scale, residual


def huber_masked(pred, target, mask):
    loss = F.smooth_l1_loss(
        pred[mask],
        target[mask],
        beta=1.0,
    )
    return loss


def smoothness_loss(scale):
    dx = torch.abs(scale[:, :, :, 1:] - scale[:, :, :, :-1]).mean()
    dy = torch.abs(scale[:, :, 1:, :] - scale[:, :, :-1, :]).mean()
    return dx + dy


def fit_stats(train_tiles):
    # Train-only normalization statistics; test quadrants are never used.
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sq = np.zeros(3, dtype=np.float64)
    rgb_n = 0
    depth_sum = 0.0
    depth_sq = 0.0
    depth_n = 0

    for rgb, depth, _, valid in train_tiles:
        v = valid
        rgb_v = rgb[v]
        depth_v = depth[v]

        rgb_sum += rgb_v.sum(axis=0)
        rgb_sq += (rgb_v.astype(np.float64) ** 2).sum(axis=0)
        rgb_n += len(rgb_v)

        depth_sum += float(depth_v.sum())
        depth_sq += float((depth_v.astype(np.float64) ** 2).sum())
        depth_n += len(depth_v)

    rgb_mean = rgb_sum / rgb_n
    rgb_var = np.maximum(rgb_sq / rgb_n - rgb_mean ** 2, 1e-6)
    rgb_std = np.sqrt(rgb_var)

    depth_mean = depth_sum / depth_n
    depth_var = max(depth_sq / depth_n - depth_mean ** 2, 1e-6)
    depth_std = math.sqrt(depth_var)

    # RGB normalization operates after /255, hence convert stats.
    rgb_mean = rgb_mean / 255.0
    rgb_std = rgb_std / 255.0

    return (rgb_mean.astype(np.float32), rgb_std.astype(np.float32)), (
        float(depth_mean),
        float(depth_std),
    )


def make_training_samples(cached, held_out_q, rgb_stats, depth_stats, rng):
    samples = []
    for rgb, depth, truth, valid in cached:
        h, w = truth.shape
        for q in range(4):
            if q == held_out_q:
                continue
            bounds = quadrant_bounds(h, w, q)
            origins = sample_patch_origins(
                valid,
                bounds,
                TRAIN_PATCHES_PER_QUADRANT,
                PATCH,
                rng,
            )
            for rr, cc in origins:
                r = rgb[rr:rr + PATCH, cc:cc + PATCH]
                d = depth[rr:rr + PATCH, cc:cc + PATCH]
                y = truth[rr:rr + PATCH, cc:cc + PATCH]
                v = valid[rr:rr + PATCH, cc:cc + PATCH]
                d = d.copy()
                y = y.copy()
                d[~v] = 0.0
                y[~v] = np.nan
                samples.append((r, d, y))
    return samples


def collect_baseline_samples(cached, held_out_q, rng, per_tile=5000):
    xs = []
    ys = []
    for rgb, depth, truth, valid in cached:
        h, w = truth.shape
        train_mask = np.ones_like(valid, dtype=bool)
        r0s = []
        for q in range(4):
            if q != held_out_q:
                r0s.append(quadrant_bounds(h, w, q))
        train_mask[:] = False
        for r0, r1, c0, c1 in r0s:
            train_mask[r0:r1, c0:c1] = True
        train_mask &= valid
        yy, xx = np.where(train_mask)
        if len(yy) == 0:
            continue
        n = min(per_tile, len(yy))
        idx = rng.choice(len(yy), size=n, replace=False)
        xs.append(depth[yy[idx], xx[idx]])
        ys.append(truth[yy[idx], xx[idx]])
    return np.concatenate(xs), np.concatenate(ys)


def predict_quadrant(model, rgb, depth, bounds, rgb_stats, depth_stats, device):
    r0, r1, c0, c1 = bounds
    rgb_mean, rgb_std = rgb_stats
    depth_mean, depth_std = depth_stats

    out = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)

    for rr in range(r0, r1, PATCH):
        for cc in range(c0, c1, PATCH):
            re = min(rr + PATCH, r1)
            ce = min(cc + PATCH, c1)

            rgb_p = rgb[rr:re, cc:ce]
            d_p = depth[rr:re, cc:ce]
            ph = re - rr
            pw = ce - cc

            # Pad edge patches to PATCH x PATCH.
            rgb_pad = np.zeros((PATCH, PATCH, 3), dtype=np.float32)
            d_pad = np.zeros((PATCH, PATCH), dtype=np.float32)
            rgb_pad[:ph, :pw] = rgb_p
            d_pad[:ph, :pw] = d_p

            rgb_n = rgb_pad / 255.0
            rgb_n = (rgb_n - rgb_mean) / rgb_std
            depth_n = (d_pad - depth_mean) / depth_std

            x = np.concatenate(
                [rgb_n, depth_n[..., None]],
                axis=-1,
            )
            x = torch.from_numpy(
                np.transpose(x, (2, 0, 1))[None].astype(np.float32)
            ).to(device)
            raw_depth = torch.from_numpy(
                d_pad[None, None].astype(np.float32)
            ).to(device)

            with torch.no_grad():
                pred, _, _ = model(x, raw_depth)

            pred = pred[0, 0].detach().cpu().numpy()
            out[rr - r0:re - r0, cc - c0:ce - c0] = pred[:ph, :pw]

    return out


def compute_metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(np.float64)
    p = p[mask].astype(np.float64)

    return {
        "mae_m": float(np.mean(np.abs(y - p))),
        "rmse_m": float(np.sqrt(np.mean((y - p) ** 2))),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "n_pixels": int(len(y)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("data/dfc2019/experiments/dav2_baseline/depth"),
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path("data/dfc2019/raw/RGB/Track1-RGB"),
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path("data/dfc2019/raw/Truth/Track1-Truth"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/dfc2019/experiments/method4"),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patches-per-quadrant", type=int, default=TRAIN_PATCHES_PER_QUADRANT)
    args = parser.parse_args()

    seed_everything()
    device = get_device()
    print(f"Device: {device}")

    depth_files = sorted(args.depth_dir.glob("*_depth.npy"))
    if len(depth_files) != 50:
        print(f"WARNING: expected 50 benchmark depth files, found {len(depth_files)}")

    tiles = []
    for dp in depth_files:
        tile_id = dp.stem.removesuffix("_depth")
        rp = args.rgb_dir / f"{tile_id}_RGB.tif"
        tp = args.truth_dir / f"{tile_id}_AGL.tif"
        if rp.exists() and tp.exists():
            tiles.append(Tile(tile_id, dp, rp, tp))

    if not tiles:
        raise RuntimeError("No complete DFC benchmark tiles found.")

    print(f"Tiles: {len(tiles)}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Cache all benchmark arrays once.
    cached = []
    for tile in tiles:
        cached.append(load_tile(tile))

    fold_results = []
    rng = np.random.default_rng(SEED)

    for held_out_q in range(4):
        print("\n" + "=" * 72)
        print(f"FOLD {held_out_q + 1}/4 — held-out quadrant {held_out_q}")
        print("=" * 72)

        # Train-only stats.
        train_stats_subset = []
        for data in cached:
            rgb, depth, truth, valid = data
            h, w = truth.shape
            r0, r1, c0, c1 = quadrant_bounds(h, w, held_out_q)
            train_valid = valid.copy()
            train_valid[r0:r1, c0:c1] = False
            train_stats_subset.append((rgb, depth, truth, train_valid))

        rgb_stats, depth_stats = fit_stats(train_stats_subset)
        print("Train RGB mean:", rgb_stats[0])
        print("Train RGB std :", rgb_stats[1])
        print("Train depth mean/std:", depth_stats)

        samples = []
        for rgb, depth, truth, valid in cached:
            h, w = truth.shape
            for q in range(4):
                if q == held_out_q:
                    continue
                bounds = quadrant_bounds(h, w, q)
                origins = sample_patch_origins(
                    valid,
                    bounds,
                    args.patches_per_quadrant,
                    PATCH,
                    rng,
                )
                for rr, cc in origins:
                    r = rgb[rr:rr + PATCH, cc:cc + PATCH].copy()
                    d = depth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    y = truth[rr:rr + PATCH, cc:cc + PATCH].copy()
                    v = valid[rr:rr + PATCH, cc:cc + PATCH]
                    d[~v] = 0.0
                    y[~v] = np.nan
                    samples.append((r, d, y))

        print(f"Training patches: {len(samples)}")

        dataset = PatchDataset(samples, rgb_stats, depth_stats)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model = ScaleModulationNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        model.train()
        for epoch in range(1, args.epochs + 1):
            running = []
            for x, y, mask in loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                raw_depth = x[:, 3:4] * depth_stats[1] + depth_stats[0]

                optimizer.zero_grad(set_to_none=True)
                pred, scale, _ = model(x, raw_depth)

                loss_data = huber_masked(pred, y, mask)
                loss_smooth = smoothness_loss(scale)
                loss = loss_data + 0.01 * loss_smooth

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running.append(float(loss.detach().cpu()))

            print(
                f"epoch {epoch:02d}/{args.epochs}: "
                f"loss={np.mean(running):.5f}"
            )

        # Save fold model.
        model_path = args.outdir / f"method4_fold{held_out_q}.pt"
        torch.save(model.state_dict(), model_path)

        # Global affine baseline trained only on training quadrants.
        base_x, base_y = collect_baseline_samples(
            cached,
            held_out_q,
            rng,
        )
        base_model = LinearRegression()
        base_model.fit(base_x.reshape(-1, 1), base_y)

        fold_tile_results = []

        model.eval()
        for tile, data in zip(tiles, cached):
            rgb, depth, truth, valid = data
            h, w = truth.shape
            bounds = quadrant_bounds(h, w, held_out_q)
            r0, r1, c0, c1 = bounds

            pred4 = predict_quadrant(
                model,
                rgb,
                depth,
                bounds,
                rgb_stats,
                depth_stats,
                device,
            )

            y_test = truth[r0:r1, c0:c1]
            v_test = valid[r0:r1, c0:c1]

            baseline_pred = base_model.predict(
                depth[r0:r1, c0:c1].reshape(-1, 1)
            ).reshape(r1 - r0, c1 - c0)

            yv = y_test[v_test]
            pv4 = pred4[v_test]
            pvb = baseline_pred[v_test]

            fold_tile_results.append(
                {
                    "tile": tile.tile_id,
                    "baseline": compute_metrics(yv, pvb),
                    "method4": compute_metrics(yv, pv4),
                }
            )

        def agg(key):
            vals = [r[key] for r in fold_tile_results]
            return {
                "mae_m": float(np.mean([x["mae_m"] for x in vals])),
                "rmse_m": float(np.mean([x["rmse_m"] for x in vals])),
                "pearson": float(np.mean([x["pearson"] for x in vals])),
                "spearman": float(np.mean([x["spearman"] for x in vals])),
                "n_tiles": len(vals),
            }

        fold_result = {
            "fold": held_out_q,
            "held_out_quadrant": held_out_q,
            "baseline": agg("baseline"),
            "method4": agg("method4"),
            "model_path": str(model_path),
            "tiles": fold_tile_results,
        }
        fold_results.append(fold_result)

        print("\nFold summary:")
        print("Baseline:", fold_result["baseline"])
        print("Method4 :", fold_result["method4"])

    def final_aggregate(key):
        blocks = [f[key] for f in fold_results]
        return {
            "mae_m": float(np.mean([b["mae_m"] for b in blocks])),
            "rmse_m": float(np.mean([b["rmse_m"] for b in blocks])),
            "pearson": float(np.mean([b["pearson"] for b in blocks])),
            "spearman": float(np.mean([b["spearman"] for b in blocks])),
            "n_folds": 4,
        }

    output = {
        "experiment": "DFC2019 Method 4 learned scale modulation",
        "protocol": (
            "4-fold spatial holdout. One quadrant from every tile is held out; "
            "the model trains on the remaining three quadrants across all benchmark tiles."
        ),
        "input": "RGB + DAv2 relative depth",
        "output": "DAv2 * learned scale + learned residual",
        "n_tiles": len(tiles),
        "baseline": final_aggregate("baseline"),
        "method4": final_aggregate("method4"),
        "folds": fold_results,
    }

    output_path = args.outdir / "method4_spatial_cv_results.json"
    output_path.write_text(json.dumps(output, indent=2))

    print("\n" + "=" * 72)
    print("FINAL METHOD 4 RESULT")
    print("=" * 72)
    print("Baseline:", output["baseline"])
    print("Method 4 :", output["method4"])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

