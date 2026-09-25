from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import shutil
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from config import settings
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform


class TesseraNoDataError(RuntimeError):
    pass


_MP_CTX = mp.get_context("spawn")
_GDAL_CACHEMAX_MB = 256

# Per-year fetch budget. If a subprocess hasn't reported back within this
# window it's reaped and treated as a failure, freeing its slot for the
# next pending year -- without this, one stuck fetch blocks a slot (and,
# at max concurrency, potentially the whole tile) forever.
_YEAR_TIMEOUT_S = 60

# How many years fetch concurrently. This is the actual RAM/CPU knob:
# each running subprocess holds its own reprojected arrays plus a
# _GDAL_CACHEMAX_MB cache, so peak memory scales with this number, not
# with the total year count. Override by adding tessera_max_concurrent_years
# to settings; otherwise defaults to 2.
_DEFAULT_MAX_CONCURRENT_YEARS = 2
_MAX_CONCURRENT_YEARS = getattr(
    settings, "tessera_max_concurrent_years", _DEFAULT_MAX_CONCURRENT_YEARS
)


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

        with rasterio.open(src_paths[0]) as ref:
            count = ref.count
            base_meta = ref.meta.copy()

        combined = np.full(
            (count, size, size),
            np.nan,
            dtype="float32",
        )

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
    result_queue,
) -> None:
    try:
        from geotessera import GeoTessera

        gt = GeoTessera(embeddings_dir=embeddings_dir_str)

        with tempfile.TemporaryDirectory() as tmp:
            tiles_to_fetch = gt.registry.load_blocks_for_region(
                bounds=bbox,
                year=year,
            )

            files = gt.export_embedding_geotiffs(
                tiles_to_fetch=tiles_to_fetch,
                output_dir=tmp,
                bands=None,
                compress="lzw",
            )

            if not files:
                raise TesseraNoDataError(
                    f"TESSERA {year}: no files returned for bbox={bbox}"
                )

            _align_to_tile_grid(
                files,
                Path(dest_path_str),
                tile,
            )

        result_queue.put(
            {
                "year": year,
                "success": True,
                "error_type": None,
                "error": None,
            }
        )

    except TesseraNoDataError as exc:
        result_queue.put(
            {
                "year": year,
                "success": False,
                "error_type": "no_data",
                "error": str(exc),
            }
        )

    except Exception as exc:
        result_queue.put(
            {
                "year": year,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def _reap(proc: mp.Process) -> bool:
    if not proc.is_alive():
        return False

    proc.terminate()
    proc.join(timeout=10)

    if proc.is_alive():
        proc.kill()
        proc.join()

    return True


def download_tessera(
    tile: dict,
    embeddings_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Fetch every missing year for this tile, running at most
    _MAX_CONCURRENT_YEARS subprocesses at a time. As soon as a running
    subprocess finishes -- success, failure, or timeout -- its slot is
    immediately handed to the next pending year, so peak RAM/CPU is
    bounded by the concurrency cap rather than by the total year count.
    """
    bbox = tile_bbox(tile)

    years_to_fetch: list[tuple[int, Path]] = []
    for year in settings.years:
        dest = embeddings_dir / f"tessera_{year}.tif"
        if not dest.exists():
            years_to_fetch.append((year, dest))

    if not years_to_fetch:
        return

    max_concurrent = max(1, min(len(years_to_fetch), _MAX_CONCURRENT_YEARS))
    logger.info(
        f"TESSERA: fetching {len(years_to_fetch)} year(s), "
        f"{max_concurrent} at a time"
    )

    result_queue = _MP_CTX.Queue()
    pending: list[tuple[int, Path]] = list(years_to_fetch)
    # year -> (proc, raw_dir, dest, launched_at)
    running: dict[int, tuple[mp.Process, str, Path, float]] = {}
    errors: list[str] = []
    no_data_errors: list[str] = []
    aborted = False

    def _launch(year: int, dest: Path) -> None:
        raw_dir = tempfile.mkdtemp(prefix=f"tessera_raw_{year}_")
        proc = _MP_CTX.Process(
            target=_fetch_and_align_year,
            args=(raw_dir, bbox, year, str(dest), tile, result_queue),
        )
        proc.start()
        running[year] = (proc, raw_dir, dest, time.monotonic())

    def _finish(year: int) -> None:
        """Reap a running process, clean up its scratch dir, and pull the
        next pending year into its freed slot (unless we're aborting)."""
        proc, raw_dir, _dest, _launched_at = running.pop(year)
        proc.join(timeout=5)
        if proc.is_alive():
            _reap(proc)
        shutil.rmtree(raw_dir, ignore_errors=True)
        if not aborted and pending:
            nyear, ndest = pending.pop(0)
            _launch(nyear, ndest)

    def _reap_timed_out() -> None:
        now = time.monotonic()
        for year, (proc, _raw_dir, dest, launched_at) in list(running.items()):
            if now - launched_at > _YEAR_TIMEOUT_S:
                errors.append(
                    f"TESSERA {year}: fetch exceeded {_YEAR_TIMEOUT_S}s "
                    f"timeout for bbox={bbox} -- subprocess terminated"
                )
                _finish(year)

    try:
        while pending and len(running) < max_concurrent:
            year, dest = pending.pop(0)
            _launch(year, dest)

        while running:
            _reap_timed_out()
            if not running:
                break

            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                # a process may have died without ever pushing a result
                for year, (proc, _raw_dir, dest, _launched_at) in list(running.items()):
                    if not proc.is_alive():
                        if proc.exitcode != 0:
                            errors.append(
                                f"TESSERA {year}: subprocess failed "
                                f"(exitcode={proc.exitcode})"
                            )
                        elif not dest.exists():
                            errors.append(
                                f"TESSERA {year}: subprocess exited "
                                f"cleanly but {dest} was not written"
                            )
                        _finish(year)
                continue

            year = result["year"]
            if year not in running:
                continue  # stray/duplicate result -- ignore

            if not result["success"]:
                if result["error_type"] == "no_data":
                    no_data_errors.append(result["error"])
                else:
                    errors.append(
                        f"TESSERA {year}: {result['error_type']}: " f"{result['error']}"
                    )

            _finish(year)

            if no_data_errors and not aborted:
                aborted = True
                pending.clear()
                for y, (proc, raw_dir, _dest, _launched_at) in list(running.items()):
                    _reap(proc)
                    shutil.rmtree(raw_dir, ignore_errors=True)
                running.clear()
                raise TesseraNoDataError("; ".join(no_data_errors))

        if errors:
            raise RuntimeError("; ".join(errors))

    finally:
        for year, (proc, raw_dir, _dest, _launched_at) in list(running.items()):
            _reap(proc)
            shutil.rmtree(raw_dir, ignore_errors=True)

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass


def download_embeddings(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_tessera(
        tile,
        embeddings_dir,
        logger,
    )
