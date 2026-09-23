#!/usr/bin/env python3
"""
DepthWizard2 - Method 3 Semantic QC PDF

Creates a 10-page PDF:
    1 page per city
    5 panels per page:
        [1] RGB
        [2] Building density (50 m)
        [3] Building edge
        [4] Building fraction
        [5] Distance to nearest building

The script is designed for large GeoTIFFs and avoids loading full-resolution
rasters when rendering the PDF. It downsamples with rasterio before plotting.

It also handles:
    - NaN / nodata / -9999 values
    - RGB uint8/uint16 imagery
    - percentile contrast stretching for RGB
    - robust percentile scaling for derived layers
    - aspect-preserving display
    - long city names
    - missing files (reported clearly and skipped)
    - exact city ordering

Typical command:
    python scripts/make_method3_qc_pdf.py

Output:
    data/sentinel2/method3_semantic_QC.pdf

Optional:
    --cities kolkata bardhaman ...
    --max-size 1400
    --dpi 150
    --output /path/to/file.pdf

Dependencies:
    rasterio
    numpy
    matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

# Headless backend - important for terminal/SSH environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(
    "/Users/anweshasaha/projects/DepthWizard2"
)

DEFAULT_SENTINEL_ROOT = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "sentinel2"
)

DEFAULT_OUTPUT = (
    DEFAULT_SENTINEL_ROOT
    / "method3_semantic_QC.pdf"
)

CITY_ORDER = [
    "kolkata",
    "bardhaman",
    "delhi",
    "mumbai",
    "bengaluru",
    "hyderabad",
    "jaipur",
    "kochi",
    "darjeeling",
    "sundarbans",
]

CITY_LABELS = {
    "kolkata": "Kolkata",
    "bardhaman": "Bardhaman",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "bengaluru": "Bengaluru",
    "hyderabad": "Hyderabad",
    "jaipur": "Jaipur",
    "kochi": "Kochi",
    "darjeeling": "Darjeeling",
    "sundarbans": "Sundarbans",
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def city_files(
    sentinel_root: Path,
    city: str,
) -> dict[str, Path]:

    city_dir = sentinel_root / city
    semantic_dir = city_dir / "semantic"

    return {
        "rgb": city_dir / f"{CITY_LABELS[city]}_RGB.tif",
        "density": semantic_dir / "building_density_50m.tif",
        "edge": semantic_dir / "building_edge.tif",
        "fraction": semantic_dir / "building_fraction.tif",
        "distance": semantic_dir / "distance_to_building_m.tif",
    }


# ---------------------------------------------------------------------------
# Raster reading
# ---------------------------------------------------------------------------

def _safe_resample_factor(
    height: int,
    width: int,
    max_size: int,
) -> float:

    largest = max(height, width)

    if largest <= max_size:
        return 1.0

    return max_size / largest


def read_single_band_preview(
    path: Path,
    max_size: int,
) -> tuple[np.ndarray, dict]:
    """
    Read one raster band at a reduced resolution.

    Returns:
        array
        metadata dict
    """

    with rasterio.open(path) as src:
        factor = _safe_resample_factor(
            src.height,
            src.width,
            max_size,
        )

        out_height = max(
            1,
            int(round(src.height * factor)),
        )
        out_width = max(
            1,
            int(round(src.width * factor)),
        )

        arr = src.read(
            1,
            out_shape=(
                out_height,
                out_width,
            ),
            resampling=rasterio.enums.Resampling.average,
            masked=True,
        )

        # Rasterio may return an integer-typed masked array (e.g. uint8).
        # Convert to float BEFORE filling masked pixels with NaN.
        arr = arr.astype(np.float32).filled(np.nan)
        arr = np.asarray(
            arr,
            dtype=np.float32,
        )

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "nodata": src.nodata,
            "min": None,
            "max": None,
        }

    return arr, metadata


def read_rgb_preview(
    path: Path,
    max_size: int,
) -> tuple[np.ndarray, dict]:

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(
                f"RGB raster requires >=3 bands: {path}"
            )

        factor = _safe_resample_factor(
            src.height,
            src.width,
            max_size,
        )

        out_height = max(
            1,
            int(round(src.height * factor)),
        )
        out_width = max(
            1,
            int(round(src.width * factor)),
        )

        arr = src.read(
            [1, 2, 3],
            out_shape=(
                3,
                out_height,
                out_width,
            ),
            resampling=rasterio.enums.Resampling.average,
            masked=True,
        )

        # Convert integer RGB data to float before filling masked pixels
        # with NaN. Filling uint8/uint16 masks directly with NaN fails.
        arr = arr.astype(np.float32).filled(np.nan)
        arr = np.asarray(
            arr,
            dtype=np.float32,
        )

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }

    # Convert CHW -> HWC.
    rgb = np.moveaxis(
        arr,
        0,
        -1,
    )

    return rgb, metadata


# ---------------------------------------------------------------------------
# Display scaling
# ---------------------------------------------------------------------------

def robust_limits(
    arr: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    values = arr[
        np.isfinite(arr)
    ]

    if values.size == 0:
        return 0.0, 1.0

    vmin = float(
        np.percentile(
            values,
            low,
        )
    )

    vmax = float(
        np.percentile(
            values,
            high,
        )
    )

    if not np.isfinite(vmin):
        vmin = float(np.nanmin(values))

    if not np.isfinite(vmax):
        vmax = float(np.nanmax(values))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return vmin, vmax


def normalize_rgb(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Percentile contrast stretch per channel.
    This makes Sentinel RGB TIFFs readable even if they are not already
    display-stretched.
    """

    output = np.zeros_like(
        rgb,
        dtype=np.float32,
    )

    for channel in range(3):
        band = rgb[..., channel]

        lo, hi = robust_limits(
            band,
            2,
            98,
        )

        scaled = (
            band - lo
        ) / max(
            hi - lo,
            1e-6,
        )

        output[..., channel] = np.clip(
            scaled,
            0.0,
            1.0,
        )

    # Replace NaN with black.
    output[
        ~np.isfinite(output)
    ] = 0.0

    return output


def sanitize_display_array(
    arr: np.ndarray,
) -> np.ma.MaskedArray:

    invalid = ~np.isfinite(arr)

    return np.ma.masked_where(
        invalid,
        arr,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_raster_panel(
    ax,
    arr: np.ndarray,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None,
) -> None:

    data = sanitize_display_array(
        arr
    )

    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(
            arr,
            2,
            98,
        )

    image = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_axis_off()

    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.02,
    )

    if colorbar_label:
        cbar.set_label(
            colorbar_label,
            fontsize=8,
        )

    cbar.ax.tick_params(
        labelsize=7
    )


def add_stats_text(
    ax,
    arr: np.ndarray,
) -> None:

    values = arr[
        np.isfinite(arr)
    ]

    if values.size == 0:
        return

    text = (
        f"mean={np.mean(values):.3f}\n"
        f"p95={np.percentile(values,95):.3f}\n"
        f"max={np.max(values):.3f}"
    )

    ax.text(
        0.015,
        0.015,
        text,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.78,
            "edgecolor": "none",
        },
    )


# ---------------------------------------------------------------------------
# One city page
# ---------------------------------------------------------------------------

def make_city_page(
    pdf: PdfPages,
    sentinel_root: Path,
    city: str,
    max_size: int,
) -> bool:

    files = city_files(
        sentinel_root,
        city,
    )

    missing = [
        name
        for name, path in files.items()
        if not path.exists()
    ]

    if missing:
        print(
            f"[SKIP] {city}: missing {', '.join(missing)}"
        )
        return False

    print(
        f"[PAGE] {CITY_LABELS[city]}"
    )

    rgb, rgb_meta = read_rgb_preview(
        files["rgb"],
        max_size,
    )

    density, _ = read_single_band_preview(
        files["density"],
        max_size,
    )

    edge, _ = read_single_band_preview(
        files["edge"],
        max_size,
    )

    fraction, _ = read_single_band_preview(
        files["fraction"],
        max_size,
    )

    distance, _ = read_single_band_preview(
        files["distance"],
        max_size,
    )

    rgb_display = normalize_rgb(
        rgb
    )

    # Landscape page gives each of the 5 panels enough width.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
    )

    # Flatten and use first five; sixth becomes an information panel.
    ax_rgb = axes[0, 0]
    ax_density = axes[0, 1]
    ax_edge = axes[0, 2]
    ax_fraction = axes[1, 0]
    ax_distance = axes[1, 1]
    ax_info = axes[1, 2]

    ax_rgb.imshow(
        rgb_display,
        interpolation="nearest",
    )
    ax_rgb.set_title(
        "1. Sentinel-2 RGB",
        fontsize=11,
        fontweight="bold",
    )
    ax_rgb.set_axis_off()

    # For building density/fraction, [0,1] is physically meaningful.
    plot_raster_panel(
        ax_density,
        density,
        "2. Building density (50 m)",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="fraction",
    )

    add_stats_text(
        ax_density,
        density,
    )

    plot_raster_panel(
        ax_edge,
        edge,
        "3. Building edge",
        cmap="magma",
        vmin=0.0,
        vmax=max(
            1.0,
            float(
                np.nanpercentile(
                    edge,
                    99,
                )
            ),
        ),
        colorbar_label="edge strength",
    )

    add_stats_text(
        ax_edge,
        edge,
    )

    plot_raster_panel(
        ax_fraction,
        fraction,
        "4. Building fraction",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="building fraction",
    )

    add_stats_text(
        ax_fraction,
        fraction,
    )

    distance_values = distance[
        np.isfinite(distance)
    ]

    if distance_values.size:
        distance_vmax = float(
            np.percentile(
                distance_values,
                98,
            )
        )

        distance_vmax = max(
            distance_vmax,
            1.0,
        )
    else:
        distance_vmax = 1.0

    plot_raster_panel(
        ax_distance,
        distance,
        "5. Distance to building",
        cmap="plasma",
        vmin=0.0,
        vmax=distance_vmax,
        colorbar_label="metres",
    )

    add_stats_text(
        ax_distance,
        distance,
    )

    # Information panel.
    ax_info.set_axis_off()

    fraction_values = fraction[
        np.isfinite(fraction)
    ]

    density_values = density[
        np.isfinite(density)
    ]

    lines = [
        CITY_LABELS[city],
        "",
        "Input:",
        str(files["rgb"]),
        "",
        "Raster:",
        f"{rgb_meta['width']} x {rgb_meta['height']} px",
        f"CRS: {rgb_meta['crs']}",
        "",
        "Building fraction:",
    ]

    if fraction_values.size:
        lines.extend(
            [
                f"mean = {fraction_values.mean():.4f}",
                f"p95  = {np.percentile(fraction_values,95):.4f}",
                f">0   = {np.mean(fraction_values > 0):.2%}",
                f">0.5 = {np.mean(fraction_values > 0.5):.2%}",
            ]
        )

    lines.extend(
        [
            "",
            "Building density:",
        ]
    )

    if density_values.size:
        lines.extend(
            [
                f"mean = {density_values.mean():.4f}",
                f"p95  = {np.percentile(density_values,95):.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Source:",
            "Microsoft GlobalMLBuildingFootprints",
            "Release: 2026-08-13",
            "",
            "Note:",
            "External semantic prior.",
            "Not scene-specific ground truth.",
        ]
    )

    ax_info.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        wrap=True,
    )

    fig.suptitle(
        f"DepthWizard2 - Method 3 Semantic Prior QC: "
        f"{CITY_LABELS[city]}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.012,
        (
            "Semantic QC | RGB + building density + edge + "
            "building fraction + distance-to-building"
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.025,
            1,
            0.97,
        ]
    )

    pdf.savefig(
        fig,
        dpi=150,
    )

    plt.close(
        fig
    )

    return True


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 10-page Method-3 semantic QC PDF, "
            "one city per page."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--sentinel-root",
        type=Path,
        default=DEFAULT_SENTINEL_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1400,
        help=(
            "Maximum rendered raster dimension. "
            "Lower uses less RAM."
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=CITY_ORDER,
        default=CITY_ORDER,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF raster rendering DPI.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.max_size < 200:
        raise ValueError(
            "--max-size should be at least 200."
        )

    sentinel_root = (
        args.sentinel_root.resolve()
    )

    output = (
        args.output.resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Creating:\n{output}"
    )

    pages_created = 0
    missing_cities = []

    with PdfPages(
        output
    ) as pdf:

        for city in args.cities:
            try:
                ok = make_city_page(
                    pdf=pdf,
                    sentinel_root=sentinel_root,
                    city=city,
                    max_size=args.max_size,
                )

                if ok:
                    pages_created += 1
                else:
                    missing_cities.append(
                        city
                    )

            except Exception as exc:
                print(
                    f"[ERROR] {city}: {exc}"
                )
                raise

    print()
    print(
        "=" * 78
    )
    print(
        "QC PDF COMPLETE"
    )
    print(
        "=" * 78
    )
    print(
        f"Output: {output}"
    )
    print(
        f"Pages:  {pages_created}"
    )

    if missing_cities:
        print(
            "Skipped:"
        )
        for city in missing_cities:
            print(
                f"  - {city}"
            )

    if pages_created != len(args.cities):
        print(
            "\nWARNING: expected "
            f"{len(args.cities)} pages but created "
            f"{pages_created}."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)
