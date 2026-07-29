from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from config import settings
from geotessera import GeoTessera
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform

_gt_client: GeoTessera | None = None
_gt_embeddings_dir: str | None = None


def _get_client() -> GeoTessera:
    """One client per process (avoids reloading the registry manifest
    every call). embeddings_dir is wiped after each year to keep raw
    tile data from accumulating on disk."""
    global _gt_client, _gt_embeddings_dir
    if _gt_client is None:
        _gt_embeddings_dir = tempfile.mkdtemp(prefix="tessera_raw_")
        _gt_client = GeoTessera(embeddings_dir=_gt_embeddings_dir)
    return _gt_client


def tile_bbox(tile: dict) -> tuple[float, float, float, float]:
    return (
        tile["min_lon"],
        tile["min_lat"],
        tile["max_lon"],
        tile["max_lat"],
    )


def _align_to_tile_grid(src_paths: list[str], dest_path: Path, tile: dict) -> None:
    """Reproject one or more source GeoTIFFs (possibly different native
    UTM CRSs) onto this tile's exact pixel grid, combining any overlap."""
    ct = tile_crs_transform(tile)
    dst_transform = Affine(ct[0], ct[1], ct[2], ct[3], ct[4], ct[5])
    size = settings.tile_pixels

    with rasterio.open(src_paths[0]) as ref:
        count = ref.count
        base_meta = ref.meta.copy()

    combined = np.full((count, size, size), np.nan, dtype="float32")

    for src_path in src_paths:
        with rasterio.open(src_path) as src:
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
        fill = np.isnan(combined) & ~np.isnan(piece)
        combined[fill] = piece[fill]

    meta = base_meta.copy()
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
        dst.write(combined)


def download_tessera(tile: dict, embeddings_dir: Path, logger: logging.Logger) -> None:
    bbox = tile_bbox(tile)
    gt = _get_client()

    for year in settings.period_years:
        dest = embeddings_dir / f"tessera_{year}.tif"
        if dest.exists():
            continue

        with tempfile.TemporaryDirectory() as tmp:
            tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
            files = gt.export_embedding_geotiffs(
                tiles_to_fetch=tiles_to_fetch,
                output_dir=tmp,
                bands=None,
                compress="lzw",
            )
            if not files:
                raise RuntimeError(f"TESSERA {year}: no files returned for bbox={bbox}")

            _align_to_tile_grid(files, dest, tile)

        # reclaim raw npy/landmask data immediately, no accumulation
        shutil.rmtree(_gt_embeddings_dir, ignore_errors=True)
        Path(_gt_embeddings_dir).mkdir(parents=True, exist_ok=True)


def download_embeddings(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    download_tessera(tile, embeddings_dir, logger)
