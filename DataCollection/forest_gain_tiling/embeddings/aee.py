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
    bbox = tile_bbox(tile)
    years = settings.period_years

    if all((embeddings_dir / f"aee_{y}.tif").exists() for y in years):
        return

    with tempfile.TemporaryDirectory() as tmp:
        logger.info(
            f"{tile['tile_id']} | AEE {years[0]}-{years[-1]} (geoai, single batched call)"
        )
        geoai.download_google_satellite_embedding(
            bbox=bbox,
            output_dir=tmp,
            years=years,
            crs=None,
            dequantize=True,
        )
        for year in years:
            dest = embeddings_dir / f"aee_{year}.tif"
            if dest.exists():
                continue
            src = Path(tmp) / f"aef_{year}.tif"
            if not src.exists():
                raise RuntimeError(
                    f"AEE {year}: expected output {src} not produced "
                    f"(no intersecting tiles for this bbox/year?)"
                )
            _align_to_tile_grid(src, dest, tile)


def download_embeddings(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    download_aee(tile, embeddings_dir, logger)
