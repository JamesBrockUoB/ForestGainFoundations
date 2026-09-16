from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from config import settings
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform

_YEAR_TIMEOUT_S = 60

# Always spawn, never fork. This is routinely started from a background
# thread (see gee/pipeline.py process_tile) while other threads are alive
# in the same process (GEE polling, etc.). Forking a multi-threaded
# process can inherit a lock (requests/GDAL/ee) held mid-acquisition by
# another thread and hang forever -- spawn avoids that entirely, and as a
# bonus doesn't copy the parent's whole heap into every child via COW.
_MP_CTX = mp.get_context("spawn")

# Cap GDAL's per-process block cache. Left at the default, it can grow to
# a large fraction of system RAM on its own, on top of the numpy arrays
# below -- with several year-subprocesses running concurrently that adds
# up fast.
_GDAL_CACHEMAX_MB = 256


def tile_bbox(tile: dict) -> tuple[float, float, float, float]:
    return (
        tile["min_lon"],
        tile["min_lat"],
        tile["max_lon"],
        tile["max_lat"],
    )


def _align_to_tile_grid(
    src_paths: list[str],
    dest_path: Path,
    tile: dict,
) -> None:
    """Reproject source GeoTIFFs onto this tile's exact pixel grid."""
    with rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHEMAX_MB):
        ct = tile_crs_transform(tile)
        dst_transform = Affine(
            ct[0],
            ct[1],
            ct[2],
            ct[3],
            ct[4],
            ct[5],
        )
        size = settings.tile_pixels

        # Read metadata from the first source.
        with rasterio.open(src_paths[0]) as ref:
            count = ref.count
            base_meta = ref.meta.copy()

        combined = np.full(
            (count, size, size),
            np.nan,
            dtype="float32",
        )

        # Open all source TIFFs once and reuse them.
        with ExitStack() as stack:
            sources = [
                stack.enter_context(rasterio.open(src_path)) for src_path in src_paths
            ]

            for src in sources:
                piece = np.full(
                    (count, size, size),
                    np.nan,
                    dtype="float32",
                )

                reproject(
                    source=rasterio.band(
                        src,
                        list(range(1, count + 1)),
                    ),
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

                # Don't keep the previous source's temporary array around.
                del piece

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

        del combined


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


def _reap(proc: mp.Process) -> bool:
    """
    Make sure a process is really gone. Returns True if it had to be
    force-terminated/killed (i.e. it was still running past its budget).
    """
    if not proc.is_alive():
        return False
    proc.terminate()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return True


def download_tessera(tile: dict, embeddings_dir: Path, logger: logging.Logger) -> None:
    """
    Fetch tessera embeddings for all years required for this period.
    Launch one subprocess per missing year and wait for them concurrently.
    Each subprocess is bounded by _YEAR_TIMEOUT_S and forcibly terminated
    if it exceeds that timeout.

    Every subprocess started here is *always* joined/terminated/killed
    before this function returns, no matter which year fails first or
    whether an exception is raised while walking the list -- previously,
    the first year's failure would raise immediately and leave any other
    already-started subprocesses (and the RAM/handles they held) running
    in the background, uncounted and uncleaned. Left running, those
    orphans stacked up across retries, which is what was actually eating
    the RAM.
    """
    bbox = tile_bbox(tile)

    years_to_fetch: list[tuple[int, Path]] = []
    for year in settings.period_years:
        dest = embeddings_dir / f"tessera_{year}.tif"
        if not dest.exists():
            years_to_fetch.append((year, dest))

    if not years_to_fetch:
        return

    procs: list[tuple[mp.Process, str, int, Path]] = []
    errors: list[str] = []

    try:
        for year, dest in years_to_fetch:
            raw_dir = tempfile.mkdtemp(prefix=f"tessera_raw_{year}_")
            proc = _MP_CTX.Process(
                target=_fetch_and_align_year,
                args=(raw_dir, bbox, year, str(dest), tile),
            )
            proc.start()
            procs.append((proc, raw_dir, year, dest))

        # Wait for every process, collecting errors as we go rather than
        # bailing on the first one -- so nothing is left running when we
        # eventually raise.
        for proc, raw_dir, year, dest in procs:
            proc.join(timeout=_YEAR_TIMEOUT_S)

            if _reap(proc):
                errors.append(
                    f"TESSERA {year}: fetch exceeded {_YEAR_TIMEOUT_S}s "
                    f"timeout for bbox={bbox} -- subprocess terminated"
                )
                continue

            if proc.exitcode != 0:
                errors.append(
                    f"TESSERA {year}: subprocess failed (exitcode={proc.exitcode})"
                )
                continue

            if not dest.exists():
                errors.append(
                    f"TESSERA {year}: subprocess exited cleanly but {dest} "
                    f"was not written"
                )

        if errors:
            raise RuntimeError("; ".join(errors))

    finally:
        # Belt-and-braces: no matter what happened above -- including an
        # exception raised before every process was even reached in the
        # loop -- nothing started by this call is allowed to still be
        # alive when we return.
        for proc, raw_dir, year, dest in procs:
            _reap(proc)
            shutil.rmtree(raw_dir, ignore_errors=True)


def download_embeddings(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    download_tessera(tile, embeddings_dir, logger)
