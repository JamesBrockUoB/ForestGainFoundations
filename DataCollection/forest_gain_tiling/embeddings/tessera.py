from __future__ import annotations

import logging
import multiprocessing as mp
import queue
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


class TesseraNoDataError(RuntimeError):
    pass


_MP_CTX = mp.get_context("spawn")
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
    bbox = tile_bbox(tile)

    years_to_fetch: list[tuple[int, Path]] = []

    for year in settings.period_years:
        dest = embeddings_dir / f"tessera_{year}.tif"

        if not dest.exists():
            years_to_fetch.append((year, dest))

    if not years_to_fetch:
        return

    result_queue = _MP_CTX.Queue()
    procs: list[tuple[mp.Process, str, int, Path]] = []
    errors: list[str] = []
    no_data_errors: list[str] = []
    completed: set[int] = set()

    try:
        for year, dest in years_to_fetch:
            raw_dir = tempfile.mkdtemp(prefix=f"tessera_raw_{year}_")

            proc = _MP_CTX.Process(
                target=_fetch_and_align_year,
                args=(
                    raw_dir,
                    bbox,
                    year,
                    str(dest),
                    tile,
                    result_queue,
                ),
            )

            proc.start()

            procs.append(
                (
                    proc,
                    raw_dir,
                    year,
                    dest,
                )
            )

        remaining = {year for _, _, year, _ in procs}

        while remaining:
            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                dead_without_result = []

                for proc, raw_dir, year, dest in procs:
                    if year in remaining and not proc.is_alive():
                        dead_without_result.append((proc, year, dest))

                for proc, year, dest in dead_without_result:
                    remaining.discard(year)

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

                continue

            year = result["year"]
            remaining.discard(year)
            completed.add(year)

            if result["success"]:
                continue

            if result["error_type"] == "no_data":
                no_data_errors.append(result["error"])
            else:
                errors.append(
                    f"TESSERA {year}: {result['error_type']}: " f"{result['error']}"
                )

            if no_data_errors:
                for proc, raw_dir, proc_year, dest in procs:
                    if proc_year not in completed:
                        _reap(proc)

                raise TesseraNoDataError("; ".join(no_data_errors))

        for proc, raw_dir, year, dest in procs:
            proc.join(timeout=1)

            if proc.is_alive():
                _reap(proc)

            if proc.exitcode != 0:
                errors.append(
                    f"TESSERA {year}: subprocess failed " f"(exitcode={proc.exitcode})"
                )
            elif not dest.exists():
                errors.append(
                    f"TESSERA {year}: subprocess exited cleanly "
                    f"but {dest} was not written"
                )

        if errors:
            raise RuntimeError("; ".join(errors))

    finally:
        for proc, raw_dir, year, dest in procs:
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
