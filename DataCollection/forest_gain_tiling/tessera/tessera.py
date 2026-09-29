from __future__ import annotations

import logging
import multiprocessing as mp
import os
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

# Maximum time allowed for an individual year subprocess.
_YEAR_TIMEOUT_S = 60

# Maximum number of years fetched concurrently.
_DEFAULT_MAX_CONCURRENT_YEARS = 2

_MAX_CONCURRENT_YEARS = getattr(
    settings,
    "tessera_max_concurrent_years",
    _DEFAULT_MAX_CONCURRENT_YEARS,
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

        # Never write directly to the final path. If the subprocess is
        # terminated during the write, the incomplete file must not look
        # like a successfully downloaded year.
        tmp_path = dest_path.with_name(dest_path.name + ".part")

        try:
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(combined)

            os.replace(tmp_path, dest_path)

        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        del combined


def _fetch_and_align_year(
    embeddings_dir: str,
    bbox: tuple[float, float, float, float],
    year: int,
    dest_path: str,
    tile: dict,
    result_queue,
) -> None:
    try:
        from geotessera import GeoTessera

        tessera = GeoTessera(
            embeddings_dir=embeddings_dir,
            bbox=bbox,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tiles_to_fetch = tessera.registry.load_blocks_for_region(
                bounds=bbox,
                year=year,
            )

            files = tessera.export_embedding_geotiffs(
                tiles_to_fetch=tiles_to_fetch,
                output_dir=tmp,
                bands=None,
                compress="lzw",
            )

            if not files:
                raise TesseraNoDataError(f"TESSERA {year}: no data available")

            _align_to_tile_grid(
                files,
                Path(dest_path),
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


def _reap(proc: mp.Process) -> None:
    if not proc.is_alive():
        return

    proc.terminate()
    proc.join(timeout=10)

    if proc.is_alive():
        proc.kill()
        proc.join()


def download_tessera(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    years: list[int] | None = None,
) -> None:
    """
    Fetch every missing TESSERA year for this tile.

    At most _MAX_CONCURRENT_YEARS subprocesses run simultaneously.
    A subprocess that exceeds _YEAR_TIMEOUT_S is terminated and its
    year is treated as failed.

    Existing .tif files are skipped.
    """

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bbox = tile_bbox(tile)
    target_years = list(settings.years if years is None else years)

    years_to_fetch: list[tuple[int, Path]] = []

    for year in target_years:
        dest = embeddings_dir / f"tessera_{year}.tif"

        if not dest.exists():
            years_to_fetch.append((year, dest))

    if not years_to_fetch:
        return

    max_concurrent = max(
        1,
        min(
            len(years_to_fetch),
            _MAX_CONCURRENT_YEARS,
        ),
    )

    logger.info(
        f"TESSERA: fetching {len(years_to_fetch)} year(s), "
        f"{max_concurrent} at a time"
    )

    result_queue = _MP_CTX.Queue()

    pending: list[tuple[int, Path]] = list(years_to_fetch)

    # year -> (process, scratch_dir, destination, launch_time)
    running: dict[
        int,
        tuple[mp.Process, str, Path, float],
    ] = {}

    errors: list[str] = []
    no_data_errors: list[str] = []
    aborted = False

    def launch(year: int, dest: Path) -> None:
        scratch_dir = tempfile.mkdtemp(prefix=f"tessera_raw_{year}_")

        proc = _MP_CTX.Process(
            target=_fetch_and_align_year,
            args=(
                scratch_dir,
                bbox,
                year,
                str(dest),
                tile,
                result_queue,
            ),
        )

        proc.start()

        running[year] = (
            proc,
            scratch_dir,
            dest,
            time.monotonic(),
        )

    def finish(year: int) -> None:
        proc, scratch_dir, _dest, _launched_at = running.pop(year)

        proc.join(timeout=5)

        if proc.is_alive():
            _reap(proc)

        shutil.rmtree(
            scratch_dir,
            ignore_errors=True,
        )

        if not aborted and pending:
            next_year, next_dest = pending.pop(0)
            launch(next_year, next_dest)

    def reap_timed_out() -> None:
        now = time.monotonic()

        for year, (
            proc,
            _scratch_dir,
            _dest,
            launched_at,
        ) in list(running.items()):
            if now - launched_at > _YEAR_TIMEOUT_S:
                errors.append(f"{year}: timeout after {_YEAR_TIMEOUT_S}s")
                _reap(proc)
                finish(year)

    try:
        while pending and len(running) < max_concurrent:
            year, dest = pending.pop(0)
            launch(year, dest)

        while running:
            reap_timed_out()

            if not running:
                break

            try:
                result = result_queue.get(timeout=0.5)

            except queue.Empty:
                for year, (
                    proc,
                    _scratch_dir,
                    dest,
                    _launched_at,
                ) in list(running.items()):
                    if not proc.is_alive():
                        if proc.exitcode != 0:
                            errors.append(
                                f"{year}: subprocess died "
                                f"(exitcode={proc.exitcode})"
                            )
                        elif not dest.exists():
                            errors.append(
                                f"{year}: exited cleanly, " "no output written"
                            )

                        finish(year)

                continue

            year = result["year"]

            if year not in running:
                # Stray or duplicate result.
                continue

            if not result["success"]:
                if result["error_type"] == "no_data":
                    no_data_errors.append(result["error"])
                else:
                    errors.append(
                        f"TESSERA {year}: "
                        f"{result['error_type']}: "
                        f"{result['error']}"
                    )

            finish(year)

            if no_data_errors and not aborted:
                aborted = True
                pending.clear()

                for (
                    _year,
                    (proc, scratch_dir, _dest, _launched_at),
                ) in list(running.items()):
                    _reap(proc)
                    shutil.rmtree(
                        scratch_dir,
                        ignore_errors=True,
                    )

                running.clear()

                raise TesseraNoDataError("; ".join(no_data_errors))

        if errors:
            grouped: dict[str, list[str]] = {}

            for error in errors:
                year, reason = error.split(
                    ": ",
                    1,
                )
                grouped.setdefault(
                    reason,
                    [],
                ).append(year)

            summary = "; ".join(
                f"{reason} ({', '.join(years)})" for reason, years in grouped.items()
            )

            raise RuntimeError(summary)

    finally:
        for (
            _year,
            (proc, scratch_dir, _dest, _launched_at),
        ) in list(running.items()):
            _reap(proc)
            shutil.rmtree(
                scratch_dir,
                ignore_errors=True,
            )

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
