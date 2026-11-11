#!/usr/bin/env python3
"""Create cluster-grouped 2017/2024 imagery galleries with gain overlays.

Expected layout:
    tiles/
      tile_.../
        composites/s1s2_2017.tif
        composites/s1s2_2024.tif
        labels/gain_confidence.tif
        metadata.json

The cluster CSV must contain `tile_id` and `cluster` columns. Metadata is read
from each tile's metadata.json; no separate metadata file is needed.

RGB uses raster bands [3, 2, 1], matching the notebook's B4/B3/B2 display.
Gain pixels are finite positive gain-confidence values; zero and NaN are
background. For the confidence overlay, values are displayed on a 50–100
scale by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, Normalize

RGB_BANDS = (3, 2, 1)  # 1-based raster bands: B4, B3, B2


def load_clusters(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cluster CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"tile_id", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    df = df.dropna(subset=["tile_id", "cluster"]).copy()
    df["tile_id"] = df["tile_id"].astype(str)
    if df.empty:
        raise ValueError(f"No clustered tiles found in {path}")
    return df


def read_tile_metadata(tile_dir: Path) -> dict:
    path = tile_dir / "metadata.json"
    if not path.is_file():
        return {}

    try:
        with path.open() as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path}: {exc}")
        return {}

    return metadata


def read_rgb(path: Path) -> np.ndarray:
    """Read B4/B3/B2 as HWC float imagery, matching the notebook's band order."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing composite: {path}")

    with rasterio.open(path) as src:
        if src.count < max(RGB_BANDS):
            raise ValueError(
                f"{path} has {src.count} bands; RGB display requires bands "
                f"{RGB_BANDS}."
            )
        rgb = src.read(RGB_BANDS).astype(np.float32)

    return np.ascontiguousarray(np.moveaxis(rgb, 0, -1))


def read_gain_confidence(tile_dir: Path) -> np.ndarray:
    """Read gain confidence; NaN and zero are background."""
    path = tile_dir / "labels" / "gain_confidence.tif"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gain confidence raster: {path}")

    with rasterio.open(path) as src:
        confidence = src.read(1).astype(np.float32)

    # Keep NaNs so they can be masked transparently in the overlay.
    confidence[~np.isfinite(confidence)] = np.nan
    return confidence


def percentile_rgb_limits(
    records: list[dict],
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get one shared per-channel stretch for all selected tiles and years."""
    channel_samples = [[], [], []]

    for record in records:
        for year in record["years"]:
            rgb = record["rgb"][year]
            # Subsample pixels to limit memory while estimating percentiles.
            flat = rgb.reshape(-1, 3)
            stride = max(1, len(flat) // 20000)
            sample = flat[::stride]

            for channel in range(3):
                values = sample[:, channel]
                values = values[np.isfinite(values)]
                if values.size:
                    channel_samples[channel].append(values)

    lows = np.zeros(3, dtype=np.float32)
    highs = np.ones(3, dtype=np.float32)

    for channel, samples in enumerate(channel_samples):
        if not samples:
            continue
        values = np.concatenate(samples)
        lows[channel], highs[channel] = np.percentile(values, [lower, upper])
        if not np.isfinite(highs[channel]) or highs[channel] <= lows[channel]:
            lows[channel] = float(np.nanmin(values))
            highs[channel] = float(np.nanmax(values))
            if highs[channel] <= lows[channel]:
                highs[channel] = lows[channel] + 1.0

    return lows, highs


def fixed_stretch(
    rgb: np.ndarray,
    limits: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Percentile stretch RGB with common per-channel limits across panels."""
    low, high = limits
    stretched = (rgb - low[None, None, :]) / (high - low)[None, None, :]
    return np.clip(np.nan_to_num(stretched, nan=0.0), 0.0, 1.0)


def load_records(
    clusters: pd.DataFrame,
    tiles_dir: Path,
    years: tuple[int, int],
    per_cluster: int,
    cluster_filter: set[str] | None,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Load selected images and confidence masks, grouped by cluster."""
    grouped: dict[str, list[dict]] = {}
    missing: list[str] = []

    for cluster_value in sorted(clusters["cluster"].unique(), key=lambda x: str(x)):
        cluster_id = str(cluster_value)
        if cluster_filter is not None and cluster_id not in cluster_filter:
            continue

        rows = clusters[clusters["cluster"] == cluster_value]
        if per_cluster > 0:
            rows = rows.head(per_cluster)

        records = []
        for _, row in rows.iterrows():
            tile_id = str(row["tile_id"])
            tile_dir = tiles_dir / tile_id

            if not tile_dir.is_dir():
                missing.append(tile_id)
                continue

            try:
                images = {
                    year: read_rgb(tile_dir / "composites" / f"s1s2_{year}.tif")
                    for year in years
                }
                confidence = read_gain_confidence(tile_dir)

                expected_shape = images[years[0]].shape[:2]
                for year in years:
                    if images[year].shape[:2] != expected_shape:
                        raise ValueError(
                            f"{year} RGB shape {images[year].shape[:2]} does not "
                            f"match {expected_shape}"
                        )
                if confidence.shape != expected_shape:
                    raise ValueError(
                        f"Gain confidence shape {confidence.shape} does not "
                        f"match RGB shape {expected_shape}"
                    )

            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                print(f"Skipping {tile_id}: {exc}")
                continue

            records.append(
                {
                    "tile_id": tile_id,
                    "cluster": cluster_id,
                    "years": years,
                    "rgb": images,
                    "confidence": confidence,
                    "metadata": read_tile_metadata(tile_dir),
                }
            )

        grouped[cluster_id] = records

    return grouped, missing


def make_overlay(
    ax: plt.Axes,
    confidence: np.ndarray,
    layer: str,
    opacity: float,
    confidence_min: float,
    confidence_max: float,
) -> tuple[str, Normalize, object]:
    """Draw confidence or binary gain overlay; return colorbar details."""
    gain = np.isfinite(confidence) & (confidence > 0)

    if layer == "confidence":
        values = np.ma.masked_where(~gain, confidence)
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad(alpha=0)
        norm = Normalize(vmin=confidence_min, vmax=confidence_max)
        ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            alpha=opacity,
            interpolation="none",
        )
        return (
            f"Gain confidence ({confidence_min:g}–{confidence_max:g})",
            norm,
            cmap,
        )

    values = np.ma.masked_where(~gain, gain.astype(np.float32))
    cmap = ListedColormap(["#ff2020"])
    cmap.set_bad(alpha=0)
    norm = Normalize(vmin=0, vmax=1)
    ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        alpha=opacity,
        interpolation="none",
    )
    return "Gain (binary)", norm, cmap


def create_galleries(
    grouped: dict[str, list[dict]],
    output_dir: Path,
    limits: tuple[np.ndarray, np.ndarray],
    layer: str,
    opacity: float,
    columns: int,
    confidence_min: float,
    confidence_max: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, records in grouped.items():
        if not records:
            print(f"Cluster {cluster_id}: no tiles to display.")
            continue

        rows = int(np.ceil(len(records) / columns))
        fig, axes = plt.subplots(
            rows,
            columns * 2,
            figsize=(columns * 6.4, rows * 3.8),
            squeeze=False,
        )

        colorbar_info = None

        for index, record in enumerate(records):
            row_index = index // columns
            first_column = (index % columns) * 2
            tile_id = record["tile_id"]
            metadata = record["metadata"]
            place = " | ".join(
                str(metadata[key])
                for key in ("country", "region", "biome")
                if metadata.get(key)
            )
            tile_title = tile_id if not place else f"{tile_id} | {place}"

            for year_offset, year in enumerate(record["years"]):
                ax = axes[row_index, first_column + year_offset]
                rgb = fixed_stretch(record["rgb"][year], limits)
                ax.imshow(rgb, interpolation="none")

                colorbar_info = make_overlay(
                    ax,
                    record["confidence"],
                    layer,
                    opacity,
                    confidence_min,
                    confidence_max,
                )

                ax.set_title(f"{tile_title}\n{year}", fontsize=8)
                ax.axis("off")

        for ax in axes.flat:
            if not ax.has_data():
                ax.axis("off")

        fig.suptitle(
            f"Cluster {cluster_id}: imagery and {layer} overlay",
            fontsize=14,
        )

        if colorbar_info is not None:
            colorbar_label, norm, cmap = colorbar_info
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                ax=axes.ravel().tolist(),
                fraction=0.015,
                pad=0.02,
            )
            cbar.set_label(colorbar_label)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output = output_dir / f"cluster_{cluster_id}_tiles.png"
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display clustered tiles with 2017/2024 imagery and gain overlays."
    )
    parser.add_argument(
        "--clusters",
        "--cluster_csv",
        dest="clusters",
        type=Path,
        required=True,
        help="Clustered CSV containing tile_id and cluster.",
    )
    parser.add_argument(
        "--tiles-dir",
        "--tiles_dir",
        dest="tiles_dir",
        type=Path,
        required=True,
        help="Parent directory containing tile_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--year",
        dest="years",
        type=int,
        nargs=2,
        default=(2017, 2024),
        metavar=("FIRST", "LAST"),
        help="First and last imagery years to display (default: 2017 2024).",
    )
    parser.add_argument(
        "--layer",
        choices=("confidence", "binary"),
        default="confidence",
        help="Overlay continuous gain confidence or binary gain pixels.",
    )
    parser.add_argument(
        "--cluster",
        action="append",
        help="Cluster ID to display; repeat to select multiple clusters.",
    )
    parser.add_argument(
        "--per-cluster",
        "--per_cluster",
        dest="per_cluster",
        type=int,
        default=12,
        help="Maximum tiles per cluster; 0 displays all tiles.",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--confidence-min", type=float, default=50.0)
    parser.add_argument("--confidence-max", type=float, default=100.0)
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    args = parser.parse_args()

    if args.per_cluster < 0:
        parser.error("--per-cluster must be non-negative")
    if args.columns < 1:
        parser.error("--columns must be at least 1")
    if not 0 <= args.opacity <= 1:
        parser.error("--opacity must be between 0 and 1")
    if args.confidence_max <= args.confidence_min:
        parser.error("--confidence-max must exceed --confidence-min")
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Percentiles must satisfy 0 <= lower < upper <= 100")

    years = tuple(args.years)
    clusters = load_clusters(args.clusters)
    cluster_filter = set(args.cluster) if args.cluster else None

    grouped, missing = load_records(
        clusters=clusters,
        tiles_dir=args.tiles_dir,
        years=years,
        per_cluster=args.per_cluster,
        cluster_filter=cluster_filter,
    )

    records = [
        record for cluster_records in grouped.values() for record in cluster_records
    ]
    if not records:
        raise SystemExit(
            "No usable tiles found. Check the cluster CSV, tile directory, "
            "and requested cluster IDs."
        )

    limits = percentile_rgb_limits(
        records,
        lower=args.lower_percentile,
        upper=args.upper_percentile,
    )
    print(
        "Shared RGB stretch limits by channel "
        f"(R,G,B): low={limits[0].tolist()}, high={limits[1].tolist()}"
    )

    create_galleries(
        grouped=grouped,
        output_dir=args.out_dir,
        limits=limits,
        layer=args.layer,
        opacity=args.opacity,
        columns=args.columns,
        confidence_min=args.confidence_min,
        confidence_max=args.confidence_max,
    )

    if missing:
        print(
            f"Warning: {len(missing)} tile directories were not found under "
            f"{args.tiles_dir}. First missing IDs: {missing[:10]}"
        )


if __name__ == "__main__":
    main()
