#!/usr/bin/env python3

"""
Pixel-level temporal feature extraction for forest-gain typology analysis.

Input:
    DataCollection/data/test_tiles/

Output:
    DataCollection/feature_analysis/pixel_typology_data/

Samples per tile:
    100 labelled gain pixels
    50 unlabelled candidate-background pixels

Gain definition:
    finite gain_confidence > 0

Unlabelled candidate-background definition:
    NaN gain_confidence

IMPORTANT:
    NaN confidence is not assumed to mean confirmed non-gain.
    It is retained as a separate sampling class.

All sampled pixels must have valid observations across all eight
years and all four data sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path.home() / "Documents" / "ForestGainFoundations"

TILE_ROOT = PROJECT_DIR / "DataCollection" / "data" / "test_tiles"

OUTPUT_DIR = PROJECT_DIR / "DataCollection" / "feature_analysis" / "pixel_typology_data"


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

YEARS = list(range(2017, 2025))

S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]

S1_BANDS = [
    "S1_VV",
    "S1_VH",
    "S1_VVVH",
]

AE_BANDS = [f"AE_{i:02d}" for i in range(64)]

TESSERA_BANDS = [f"TS_{i:03d}" for i in range(128)]

SOURCES = {
    "s1": {
        "path": "composites/s1s2_{year}.tif",
        "bands": S1_BANDS,
        "indices": [11, 12, 13],
    },
    "s2": {
        "path": "composites/s1s2_{year}.tif",
        "bands": S2_BANDS,
        "indices": list(range(1, 11)),
    },
    "alphaearth": {
        "path": "embeddings/aee_{year}.tif",
        "bands": AE_BANDS,
        "indices": list(range(1, 65)),
    },
    "tessera": {
        "path": "embeddings/tessera_{year}.tif",
        "bands": TESSERA_BANDS,
        "indices": list(range(1, 129)),
    },
}

VALID_MASK_BAND_INDEX = 14
S2_SCALE = 10000.0

GAIN_PER_TILE = 100
UNLABELLED_PER_TILE = 50

RANDOM_SEED = 42


# ---------------------------------------------------------
# RASTER HELPERS
# ---------------------------------------------------------


def read_bands(path, band_indices):
    """
    Read selected raster bands.

    Returns:
        Float32 array with shape (bands, height, width).
        Nodata and masked values are converted to NaN.
    """

    with rasterio.open(path) as src:
        arr = src.read(
            band_indices,
            masked=True,
        )

    return arr.filled(np.nan).astype(np.float32)


def read_band(path, band_index):
    """Read one raster band as a float32 array."""

    return read_bands(path, [band_index])[0]


# ---------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------


def make_feature_names():
    names = []

    for year in YEARS:
        for source_name, source in SOURCES.items():
            for band in source["bands"]:
                names.append(f"{source_name}_{year}_{band}")

    return names


# ---------------------------------------------------------
# TILE DISCOVERY
# ---------------------------------------------------------


def discover_tiles():
    if not TILE_ROOT.exists():
        raise FileNotFoundError(f"Tile directory not found: {TILE_ROOT}")

    tiles = sorted(
        p
        for p in TILE_ROOT.iterdir()
        if p.is_dir() and (p / "labels" / "gain_confidence.tif").exists()
    )

    print(f"Found {len(tiles)} tiles.")

    if not tiles:
        raise RuntimeError("No tiles containing gain_confidence.tif found.")

    return tiles


# ---------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------


def sample_indices(mask, n, rng):
    """Sample up to n pixels from a boolean mask."""

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)

    n = min(n, len(indices))

    return rng.choice(
        indices,
        size=n,
        replace=False,
    )


# ---------------------------------------------------------
# PIXEL EXTRACTION
# ---------------------------------------------------------


def extract_tile(tile_dir, rng):

    confidence_path = tile_dir / "labels" / "gain_confidence.tif"

    confidence = read_band(
        confidence_path,
        1,
    )

    height, width = confidence.shape

    # -----------------------------------------------------
    # Determine observation-valid pixels independently
    # of the gain-confidence raster.
    # -----------------------------------------------------

    observation_valid = np.ones(
        (height, width),
        dtype=bool,
    )

    for year in YEARS:

        composite_path = tile_dir / f"composites/s1s2_{year}.tif"

        if not composite_path.exists():
            raise FileNotFoundError(composite_path)

        annual_valid = read_band(
            composite_path,
            VALID_MASK_BAND_INDEX,
        )

        if annual_valid.shape != confidence.shape:
            raise ValueError(f"Validity mask shape mismatch: {composite_path}")

        observation_valid &= np.isfinite(annual_valid) & (annual_valid > 0)

    # -----------------------------------------------------
    # Separate gain from unlabelled pixels.
    # -----------------------------------------------------

    gain_mask = observation_valid & np.isfinite(confidence) & (confidence > 0)

    unlabelled_mask = observation_valid & ~np.isfinite(confidence)

    gain_idx = sample_indices(
        gain_mask,
        GAIN_PER_TILE,
        rng,
    )

    unlabelled_idx = sample_indices(
        unlabelled_mask,
        UNLABELLED_PER_TILE,
        rng,
    )

    selected_idx = np.concatenate(
        [
            gain_idx,
            unlabelled_idx,
        ]
    )

    if len(selected_idx) == 0:
        return None

    labels = np.concatenate(
        [
            np.ones(
                len(gain_idx),
                dtype=np.uint8,
            ),
            np.zeros(
                len(unlabelled_idx),
                dtype=np.uint8,
            ),
        ]
    )

    label_types = np.concatenate(
        [
            np.full(
                len(gain_idx),
                "gain",
                dtype=object,
            ),
            np.full(
                len(unlabelled_idx),
                "unlabelled",
                dtype=object,
            ),
        ]
    )

    rows = selected_idx // width
    cols = selected_idx % width

    # -----------------------------------------------------
    # Extract annual features.
    # -----------------------------------------------------

    feature_blocks = []

    for year in YEARS:

        # Sentinel-1 and Sentinel-2 share a composite.
        composite_path = tile_dir / f"composites/s1s2_{year}.tif"

        composite = read_bands(
            composite_path,
            list(range(1, 14)),
        )

        # S1: bands 11-13 in the raster.
        s1 = composite[
            10:13,
            rows,
            cols,
        ].T

        # S2: bands 1-10, converted from DN to reflectance.
        s2 = (
            composite[
                0:10,
                rows,
                cols,
            ].T
            / S2_SCALE
        )

        feature_blocks.extend(
            [
                s1,
                s2,
            ]
        )

        # AlphaEarth and Tessera.
        for source_name in [
            "alphaearth",
            "tessera",
        ]:

            source = SOURCES[source_name]

            embedding_path = tile_dir / source["path"].format(year=year)

            if not embedding_path.exists():
                raise FileNotFoundError(embedding_path)

            embedding = read_bands(
                embedding_path,
                source["indices"],
            )

            if embedding.shape[1:] != (
                height,
                width,
            ):
                raise ValueError(f"Embedding shape mismatch: {embedding_path}")

            sampled = embedding[
                :,
                rows,
                cols,
            ].T

            feature_blocks.append(sampled)

    features = np.concatenate(
        feature_blocks,
        axis=1,
    )

    # -----------------------------------------------------
    # Retain only pixels with complete trajectories.
    # -----------------------------------------------------

    complete = np.isfinite(features).all(axis=1)

    features = features[complete]
    rows = rows[complete]
    cols = cols[complete]
    selected_idx = selected_idx[complete]
    labels = labels[complete]
    label_types = label_types[complete]

    if len(features) == 0:
        return None

    # -----------------------------------------------------
    # Coordinates.
    # -----------------------------------------------------

    with rasterio.open(tile_dir / f"composites/s1s2_{YEARS[0]}.tif") as src:

        transform = src.transform
        crs = str(src.crs)

    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )

    confidence_values = confidence[
        rows,
        cols,
    ]

    metadata = pd.DataFrame(
        {
            "tile_id": tile_dir.name,
            "pixel_index": selected_idx,
            "row": rows,
            "col": cols,
            "x": xs,
            "y": ys,
            "crs": crs,
            "sample_type": label_types,
            "gain_label": labels,
            "gain_confidence": confidence_values,
        }
    )

    return features, metadata


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    tiles = discover_tiles()

    all_features = []
    all_metadata = []

    total_gain = 0
    total_unlabelled = 0
    failed_tiles = []

    for i, tile_dir in enumerate(
        tiles,
        start=1,
    ):

        print(f"\n[{i}/{len(tiles)}] " f"Processing {tile_dir.name}")

        try:

            result = extract_tile(
                tile_dir,
                rng,
            )

            if result is None:
                print("  No complete sampled pixels.")
                continue

            features, metadata = result

            n_gain = int((metadata["sample_type"] == "gain").sum())

            n_unlabelled = int((metadata["sample_type"] == "unlabelled").sum())

            total_gain += n_gain
            total_unlabelled += n_unlabelled

            all_features.append(features)
            all_metadata.append(metadata)

            print(
                f"  Gain: {n_gain} | "
                f"Unlabelled: {n_unlabelled} | "
                f"Total: {len(metadata)}"
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            failed_tiles.append(
                {
                    "tile": tile_dir.name,
                    "error": str(exc),
                }
            )

    if not all_features:
        raise RuntimeError("No pixel features were successfully extracted.")

    # -----------------------------------------------------
    # Combine all tiles.
    # -----------------------------------------------------

    X = np.concatenate(
        all_features,
        axis=0,
    )

    metadata = pd.concat(
        all_metadata,
        ignore_index=True,
    )

    feature_names = make_feature_names()

    if X.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature mismatch: {X.shape[1]} "
            f"columns but {len(feature_names)} names."
        )

    # -----------------------------------------------------
    # Save outputs.
    # -----------------------------------------------------

    np.savez_compressed(
        OUTPUT_DIR / "pixel_features.npz",
        X=X,
    )

    metadata.to_csv(
        OUTPUT_DIR / "pixel_metadata.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "feature_names.json",
        "w",
    ) as f:

        json.dump(
            feature_names,
            f,
            indent=2,
        )

    summary = {
        "tile_root": str(TILE_ROOT),
        "output_directory": str(OUTPUT_DIR),
        "n_tiles_found": len(tiles),
        "n_tiles_processed": int(metadata["tile_id"].nunique()),
        "n_tiles_failed": len(failed_tiles),
        "n_pixels": int(len(metadata)),
        "n_gain_pixels": int(total_gain),
        "n_unlabelled_pixels": int(total_unlabelled),
        "n_features": int(X.shape[1]),
        "years": YEARS,
        "sources": list(SOURCES.keys()),
        "gain_per_tile_requested": GAIN_PER_TILE,
        "unlabelled_per_tile_requested": UNLABELLED_PER_TILE,
        "random_seed": RANDOM_SEED,
        "failed_tiles": failed_tiles,
        "warning": (
            "NaN confidence pixels are unlabelled candidates, "
            "not confirmed non-gain."
        ),
    }

    with open(
        OUTPUT_DIR / "extraction_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Final report.
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("EXTRACTION COMPLETE")
    print("=" * 55)

    print(json.dumps(summary, indent=2))

    print("\nSaved files:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path.name}")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
