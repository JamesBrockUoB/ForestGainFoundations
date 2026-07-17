from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import geoai
import numpy as np
import rasterio
from config import settings
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform


def tile_bbox(tile: dict) -> tuple[float, float, float, float]:
    return (
        tile["min_lon"],
        tile["min_lat"],
        tile["max_lon"],
        tile["max_lat"],
    )


def _align_to_tile_grid(src_path: str, dest_path: Path, tile: dict) -> None:
    ct = tile_crs_transform(tile)
    dst_transform = Affine(ct[0], ct[1], ct[2], ct[3], ct[4], ct[5])
    size = settings.tile_pixels

    with rasterio.open(src_path) as src:
        count = src.count
        piece = np.full((count, size, size), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, list(range(1, count + 1))),
            destination=piece,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=settings.crs_wkt,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        meta = src.meta.copy()

    meta.update(
        crs=settings.crs_wkt,
        transform=dst_transform,
        width=size,
        height=size,
        dtype="float32",
        count=count,
        nodata=np.nan,
    )
    with rasterio.open(dest_path, "w", **meta) as dst:
        dst.write(piece)


def download_aee(tile: dict, embeddings_dir: Path, logger: logging.Logger) -> None:
    """
    "geoai" source: direct HTTPS windowed COG reads from Source
    Cooperative, no Earth Engine involved. See export/aee.py's
    submit_aee_exports for the "gee" (Drive export task) alternative.
    """
    bbox = tile_bbox(tile)

    for year in settings.period_years:
        dest = embeddings_dir / f"aee_{year}.tif"
        if dest.exists():
            continue

        with tempfile.TemporaryDirectory() as tmp:
            logger.info(f"{tile['tile_id']} | AEE {year} (geoai)")
            files = geoai.download_google_satellite_embedding(
                bbox=bbox,
                output_dir=tmp,
                years=[year],
                crs=None,
                dequantize=True,
            )
            if len(files) != 1:
                raise RuntimeError(
                    f"AEE {year}: expected 1 file, got {len(files)}: {files}"
                )
            _align_to_tile_grid(files[0], dest, tile)


def download_embeddings(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    download_aee(tile, embeddings_dir, logger)
