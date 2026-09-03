from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from config import settings
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform

_YEAR_TIMEOUT_S = 180  # generous but bounded


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


def _fetch_and_align_year(
    embeddings_dir_str: str,
    bbox: tuple,
    year: int,
    dest_path_str: str,
    tile: dict,
) -> None:
    """
    Runs in a separate PROCESS (not thread), specifically so a stuck
    fetch can be forcibly terminated at the OS level -- Python cannot
    kill a thread under any circumstances, only a process. Also builds
    its own fresh GeoTessera client with no state shared across attempts
    or with the parent, so there's no cross-attempt poisoning to guard
    against the way a shared singleton would need.
    """
    from geotessera import GeoTessera

    gt = GeoTessera(embeddings_dir=embeddings_dir_str)
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
        _align_to_tile_grid(files, Path(dest_path_str), tile)


def download_tessera(tile: dict, embeddings_dir: Path, logger: logging.Logger) -> None:
    """
    Fetch tessera embeddings for all years required for this period.
    Launch one subprocess per missing year and wait for them concurrently.
    Each subprocess is bounded by the unchanged _YEAR_TIMEOUT_S and will
    be forcibly terminated if it exceeds that timeout.
    """
    bbox = tile_bbox(tile)

    # Determine missing years
    years_to_fetch: list[tuple[int, Path]] = []
    for year in settings.period_years:
        dest = embeddings_dir / f"tessera_{year}.tif"
        if not dest.exists():
            years_to_fetch.append((year, dest))

    if not years_to_fetch:
        return

    procs: list[tuple[mp.Process, str, int, Path]] = []
    try:
        for year, dest in years_to_fetch:
            raw_dir = tempfile.mkdtemp(prefix=f"tessera_raw_{year}_")
            proc = mp.Process(
                target=_fetch_and_align_year,
                args=(raw_dir, bbox, year, str(dest), tile),
            )
            proc.start()
            procs.append((proc, raw_dir, year, dest))

        # Wait for each process with the same per-process timeout semantics
        for proc, raw_dir, year, dest in procs:
            proc.join(timeout=_YEAR_TIMEOUT_S)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                # Clean up raw dirs and raise an error to indicate the timeout
                for _p, rd, _y, _d in procs:
                    shutil.rmtree(rd, ignore_errors=True)
                raise RuntimeError(
                    f"TESSERA {year}: fetch exceeded {_YEAR_TIMEOUT_S}s "
                    f"timeout for bbox={bbox} -- subprocess terminated"
                )

            if proc.exitcode != 0:
                for _p, rd, _y, _d in procs:
                    shutil.rmtree(rd, ignore_errors=True)
                raise RuntimeError(
                    f"TESSERA {year}: subprocess failed (exitcode={proc.exitcode})"
                )

            if not dest.exists():
                for _p, rd, _y, _d in procs:
                    shutil.rmtree(rd, ignore_errors=True)
                raise RuntimeError(
                    f"TESSERA {year}: subprocess exited cleanly but {dest} "
                    f"was not written"
                )

    finally:
        for _p, rd, _y, _d in procs:
            shutil.rmtree(rd, ignore_errors=True)


def download_embeddings(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    download_tessera(tile, embeddings_dir, logger)
