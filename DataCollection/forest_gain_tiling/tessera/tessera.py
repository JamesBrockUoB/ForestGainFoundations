from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from threading import Event

import numpy as np
import rasterio
from config import settings
from export.year_chunks import chunk_years
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tiling.grid import crs_transform as tile_crs_transform
from tiling.grid import footprint_bounds_lonlat


class TesseraNoDataError(RuntimeError):
    pass


class TesseraPermanentError(RuntimeError):
    pass


_MP_CTX = mp.get_context("spawn")

_GDAL_CACHEMAX_MB = 256

# Maximum number of years fetched concurrently.
_DEFAULT_MAX_CONCURRENT_YEARS = 2

_MAX_CONCURRENT_YEARS = getattr(
    settings,
    "tessera_max_concurrent_years",
    _DEFAULT_MAX_CONCURRENT_YEARS,
)

# Reject a tile when more than this fraction of it has no TESSERA embedding
# (sea, or blocks that do not cover the tile). Smaller gaps are written as NaN
# and masked out downstream.
_MAX_NODATA_FRAC = getattr(
    settings, "tessera_max_nodata_frac", 1.0
)  # 1.0 = no limit, accept any fraction of NaN

# --- GeoTessera >= 0.11 dataset selection -------------------------------------------------------
# GeoTesseraZarr streams the global v1.1 "dclimate" run by default (2017-2025). Embeddings from
# different (version, variant) pairs are NOT interchangeable, so the pair is pinned explicitly and
# stamped into every aligned GeoTIFF; cached files from any other dataset are discarded and refetched.
# All four settings are optional.
_norm_version = (
    lambda v: str(v).strip().lower().lstrip("v")
)  # noqa: E731  ("v1.1" -> "1.1")
_DATASET_VERSION = _norm_version(getattr(settings, "tessera_dataset_version", "1.1"))
_DATASET_VARIANT = (
    str(getattr(settings, "tessera_dataset_variant", "dclimate")).strip().lower()
)
_CACHE_DIR = getattr(
    settings, "tessera_cache_dir", None
)  # persists Zarr metadata between runs
_VIA_GATEWAY = getattr(
    settings, "tessera_via_gateway", None
)  # True/False; None = library default

_TAG_VERSION = "TESSERA_DATASET_VERSION"
_TAG_VARIANT = "TESSERA_DATASET_VARIANT"

# Errors that will not fix themselves on retry (e.g. a zero-byte / truncated
# upstream file, or the wrong dataset being served). Matched on message text as a
# fallback; the EOFError cause check in _is_permanent is the sturdier signal.
_PERMANENT_MARKERS = ("No data left in file", "dataset mismatch")


def _is_permanent(exc: BaseException) -> bool:
    """True if the failure is deterministic (bad/empty upstream data)."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, EOFError):
            return True
        if any(m in str(cur) for m in _PERMANENT_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def open_tessera():
    """GeoTesseraZarr client for the pinned (version, variant), e.g. v1.1 / dclimate."""
    from geotessera import GeoTesseraZarr
    from geotessera.registry import zarr_store_url

    kwargs = {}
    if _CACHE_DIR:
        kwargs["cache_dir"] = str(_CACHE_DIR)
    if _VIA_GATEWAY is not None:
        kwargs["via_gateway"] = bool(_VIA_GATEWAY)

    gt = GeoTesseraZarr(
        zarr_store_url(f"v{_DATASET_VERSION}", _DATASET_VARIANT), **kwargs
    )

    ds = getattr(
        gt, "dataset", None
    )  # None for stores that are not a published dataset
    expected = f"{_DATASET_VERSION}-{_DATASET_VARIANT}"
    if ds is not None and str(ds.name).lower() != expected:
        raise RuntimeError(
            f"TESSERA dataset mismatch: opened {ds.name}, expected {expected}"
        )
    return gt


def tile_is_current(path: Path) -> bool:
    """True if an aligned tessera_<year>.tif was made from the pinned dataset."""
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
    except Exception:
        return False
    return (
        _norm_version(tags.get(_TAG_VERSION, "")) == _DATASET_VERSION
        and str(tags.get(_TAG_VARIANT, "")).strip().lower() == _DATASET_VARIANT
    )


def _years_to_fetch(
    embeddings_dir: Path,
    target_years: list[int],
    logger: logging.Logger,
    tile_id: str,
) -> list[tuple[int, Path]]:
    """Years with no usable file. A file from another dataset (e.g. pre-0.11 v1.0, or untagged) is deleted."""
    todo: list[tuple[int, Path]] = []
    for year in target_years:
        dest = embeddings_dir / f"tessera_{year}.tif"
        if dest.exists():
            if tile_is_current(dest):
                continue
            logger.warning(
                f"{tile_id} | TESSERA {year}: cached file is not from "
                f"v{_DATASET_VERSION}/{_DATASET_VARIANT} (or has no dataset tag); refetching"
            )
            dest.unlink(missing_ok=True)
        todo.append((year, dest))
    return todo


def tile_bbox(tile: dict) -> tuple[float, float, float, float]:
    return footprint_bounds_lonlat(
        tile["x_min_m"],
        tile["y_min_m"],
        tile["x_max_m"],
        tile["y_max_m"],
        pad_deg=0.002,
    )


def _align_to_tile_grid(
    src_paths: list[str],
    dest_path: Path,
    tile: dict,
    year: int,
) -> None:
    """
    Reproject the per-UTM-zone GeoTIFFs written by export_geotiffs
    (tessera_<year>_utm<NN>.tif) onto the tile's EPSG grid. A tile that spans
    a zone boundary gets several files; the first one with data wins per pixel.
    """
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
            src_tags = ref.tags()

        # Refuse anything that is not the pinned dataset before spending time on it.
        src_ver = src_tags.get(_TAG_VERSION)
        src_var = src_tags.get(_TAG_VARIANT)
        if (src_ver is not None and _norm_version(src_ver) != _DATASET_VERSION) or (
            src_var is not None and str(src_var).strip().lower() != _DATASET_VARIANT
        ):
            raise RuntimeError(
                f"TESSERA {year}: dataset mismatch: got {src_ver}/{src_var}, "
                f"expected {_DATASET_VERSION}/{_DATASET_VARIANT}"
            )

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
                    resampling=Resampling.nearest,
                    dst_nodata=np.nan,
                )

                fill = np.isnan(combined) & ~np.isnan(piece)
                combined[fill] = piece[fill]

                del piece

        # Pixels without an embedding are NaN in v1.1 dclimate (it does not mark water separately);
        # exact zeros in every band are also treated as no-data (older stores used them for sea).
        # Reject the tile if too much of it is affected, otherwise write the gaps as NaN.
        # Raised before anything is written to disk.
        nodata = np.isnan(combined).all(axis=0) | (combined == 0).all(axis=0)
        nodata_frac = float(nodata.mean())
        if nodata_frac > _MAX_NODATA_FRAC:
            raise TesseraNoDataError(
                f"TESSERA {year}: {nodata_frac:.0%} of tile has no embedding "
                f"(limit {_MAX_NODATA_FRAC:.0%})"
            )
        combined[:, nodata] = np.nan

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

        tags = {
            _TAG_VERSION: src_ver or _DATASET_VERSION,
            _TAG_VARIANT: src_var or _DATASET_VARIANT,
        }
        for k in ("TESSERA_DATASET_VERSION_PATH", "TESSERA_SOURCE"):
            if k in src_tags:
                tags[k] = src_tags[k]

        # Never write directly to the final path. If the subprocess is
        # terminated during the write, the incomplete file must not look
        # like a successfully downloaded year.
        tmp_path = dest_path.with_name(dest_path.name + ".part")

        try:
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(combined)
                dst.update_tags(**tags)

            os.replace(tmp_path, dest_path)

        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        del combined


def _fetch_and_align_year(
    bbox: tuple[float, float, float, float],
    year: int,
    dest_path: str,
    tile: dict,
    scratch_dir: str,
    result_queue,
) -> None:
    """Runs in a subprocess. scratch_dir is owned (and removed) by the parent, so a killed
    subprocess cannot leak its temporary export."""
    try:
        gt = open_tessera()

        # Writes tessera_<year>_utm<NN>.tif, one file per intersecting UTM zone.
        files = gt.export_geotiffs(
            bbox,
            year,
            scratch_dir,
            bands=None,
            compress="lzw",
        )

        if not files:
            raise TesseraNoDataError(f"TESSERA {year}: no data available")

        _align_to_tile_grid(
            [str(f) for f in files],
            Path(dest_path),
            tile,
            year,
        )

        result_queue.put(
            {
                "year": year,
                "success": True,
                "error_type": None,
                "error": None,
                "permanent": False,
                "tb": None,
            }
        )

    except TesseraNoDataError as exc:
        result_queue.put(
            {
                "year": year,
                "success": False,
                "error_type": "no_data",
                "error": str(exc),
                "permanent": True,
                "tb": traceback.format_exc(),
            }
        )

    except Exception as exc:
        result_queue.put(
            {
                "year": year,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "permanent": _is_permanent(exc),
                "tb": traceback.format_exc(),
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


def _new_scratch(embeddings_dir: Path, year: int) -> str:
    return tempfile.mkdtemp(prefix=f".scratch_{year}_", dir=embeddings_dir)


def download_tessera(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    years: list[int] | None = None,
) -> None:
    """
    Fetch every missing TESSERA year for this tile.

    At most _MAX_CONCURRENT_YEARS subprocesses run simultaneously.
    A subprocess that exceeds the per-year timeout is terminated and its
    year is treated as failed.

    Existing .tif files are skipped if they were made from the pinned
    dataset (v1.1 / dclimate by default); otherwise they are refetched.
    """
    tile_id = tile["tile_id"]

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bbox = tile_bbox(tile)
    target_years = list(settings.years if years is None else years)

    years_to_fetch = _years_to_fetch(embeddings_dir, target_years, logger, tile_id)

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
        f"{tile_id} | TESSERA v{_DATASET_VERSION}/{_DATASET_VARIANT}: "
        f"fetching {len(years_to_fetch)} year(s), {max_concurrent} at a time"
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
        scratch_dir = _new_scratch(embeddings_dir, year)

        proc = _MP_CTX.Process(
            target=_fetch_and_align_year,
            args=(
                bbox,
                year,
                str(dest),
                tile,
                scratch_dir,
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
            if now - launched_at > settings.tessera_year_timeout_s:
                errors.append(
                    f"{year}: timeout after {settings.tessera_year_timeout_s}s"
                )
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
                            errors.append(f"{year}: exited cleanly, no output written")

                        finish(year)

                continue

            year = result["year"]

            if year not in running:
                # Stray or duplicate result.
                continue

            if not result["success"]:
                if result.get("permanent"):
                    # One-line error is raised below and logged once by the
                    # caller; the full traceback is DEBUG-only.
                    if result.get("tb"):
                        logger.debug(
                            f"{tile_id} | TESSERA {year} traceback "
                            f"(bbox={bbox}):\n{result['tb']}"
                        )

                    aborted = True
                    pending.clear()

                    finish(year)

                    # Stop any other years still running; the tile is dead.
                    for (
                        _y,
                        (proc, scratch_dir, _dest, _launched_at),
                    ) in list(running.items()):
                        _reap(proc)
                        shutil.rmtree(scratch_dir, ignore_errors=True)
                    running.clear()

                    raise TesseraPermanentError(
                        f"TESSERA {year}: "
                        f"{result['error_type']}: "
                        f"{result['error']}"
                    )

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


def download_tessera_with_retry(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
    retries: int = 2,
    years: list[int] | None = None,
    raise_no_data: bool = False,
) -> bool:
    tile_id = tile["tile_id"]

    # 1 initial attempt + `retries` retry attempts
    total_attempts = 1 + retries

    for attempt in range(total_attempts):
        if cancel_event.is_set():
            logger.warning(f"{tile_id} | TESSERA cancelled")
            return False

        try:
            if attempt == 0:
                logger.info(f"{tile_id} | starting TESSERA")
            else:
                logger.info(f"{tile_id} | starting TESSERA retry {attempt}/{retries}")

            download_tessera(tile, output_dir, logger, years)
            logger.info(f"{tile_id} | TESSERA complete")
            return True

        except TesseraPermanentError as exc:
            logger.error(f"{tile_id} | TESSERA permanent failure, not retrying: {exc}")
            raise

        except TesseraNoDataError as exc:
            logger.error(
                f"{tile_id} | TESSERA has no data for this tile; "
                f"not retrying: {exc}"
            )
            if raise_no_data:
                raise
            return False

        except Exception as exc:
            logger.error(f"{tile_id} | TESSERA failed: {exc}")

        # Only wait and log if we actually have retries left
        if attempt < total_attempts - 1:
            wait = 2**attempt + 1

            logger.warning(
                f"{tile_id} | TESSERA scheduled retry "
                f"{attempt + 1}/{retries} in {wait}s"
            )

            if cancel_event.wait(wait):
                logger.warning(f"{tile_id} | TESSERA cancelled during retry wait")
                return False

    logger.error(f"{tile_id} | TESSERA exhausted retries")
    return False


def fetch_tessera_in_chunks(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
) -> bool:
    """Fetch TESSERA in chunks, splitting failed chunks recursively."""

    def fetch(years: list[int]) -> bool:
        if cancel_event.is_set():
            return False

        retries = 2 if len(years) > 1 else 4

        try:
            ok = download_tessera_with_retry(
                tile,
                output_dir,
                logger,
                cancel_event,
                retries=retries,
                years=years,
                raise_no_data=True,
            )
        except TesseraNoDataError:
            # Deterministic (no / too little data): splitting into smaller
            # requests would only re-download the same blocks.
            return False

        if ok:
            return True

        if len(years) == 1:
            return False

        logger.warning(
            f"{tile['tile_id']} | TESSERA {years} failed; "
            "retrying as smaller requests"
        )

        mid = len(years) // 2
        return fetch(years[:mid]) and fetch(years[mid:])

    for chunk in chunk_years(
        list(settings.years),
        settings.tessera_years_per_request,
    ):
        if not fetch(chunk):
            cancel_event.set()
            return False

    return True


def download_tessera_until_acquired(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    years: list[int] | None = None,
) -> None:
    """
    Fetch every missing TESSERA year once and wait for completion.

    This is used for TESSERA_MISSING recovery and is intentionally
    not time-sensitive. Each year is allowed to run to completion
    without retrying failed downloads.

    Files from another dataset (or without a dataset tag) count as missing.

    Raises:
        TesseraNoDataError: the year has no data, or is permanently
            unreadable (bad upstream file). Callers should REJECT
            the tile.
        RuntimeError: a TESSERA subprocess fails or exits without
            producing output.
    """
    tile_id = tile["tile_id"]

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    bbox = tile_bbox(tile)
    target_years = list(settings.years if years is None else years)

    pending: list[int] = [
        year
        for year, _dest in _years_to_fetch(
            embeddings_dir, target_years, logger, tile_id
        )
    ]

    if not pending:
        return

    max_concurrent = max(1, min(len(pending), _MAX_CONCURRENT_YEARS))

    logger.info(
        f"{tile_id} | TESSERA v{_DATASET_VERSION}/{_DATASET_VARIANT}: "
        f"fetching {len(pending)} year(s), {max_concurrent} at a time"
    )

    result_queue = _MP_CTX.Queue()

    # year -> (process, scratch_dir, dest)
    running: dict[int, tuple[mp.Process, str, Path]] = {}

    def launch(year: int) -> None:
        dest = embeddings_dir / f"tessera_{year}.tif"
        scratch_dir = _new_scratch(embeddings_dir, year)

        proc = _MP_CTX.Process(
            target=_fetch_and_align_year,
            args=(bbox, year, str(dest), tile, scratch_dir, result_queue),
        )
        proc.start()

        running[year] = (proc, scratch_dir, dest)

    try:
        while pending and len(running) < max_concurrent:
            launch(pending.pop(0))

        while running:
            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                for year, (proc, scratch_dir, dest) in list(running.items()):
                    if not proc.is_alive():
                        proc.join()
                        shutil.rmtree(scratch_dir, ignore_errors=True)
                        del running[year]

                        if proc.exitcode != 0:
                            reason = f"subprocess died (exitcode={proc.exitcode})"
                            raise RuntimeError(f"{year}: {reason}")

                        if not dest.exists():
                            raise RuntimeError(
                                f"{year}: exited cleanly, no output written"
                            )

                        # Finished and wrote output; the success result
                        # is still in flight or was already consumed.
                        continue

                continue

            year = result["year"]

            if year not in running:
                continue

            proc, scratch_dir, dest = running.pop(year)
            proc.join(timeout=5)
            if proc.is_alive():
                _reap(proc)
            shutil.rmtree(scratch_dir, ignore_errors=True)

            if result["success"]:
                logger.info(f"TESSERA {year}: acquired")
                if pending:
                    launch(pending.pop(0))
                continue

            msg = f"{result['error_type']}: {result['error']}"

            if result["error_type"] == "no_data":
                for y, (p, sd, _d) in list(running.items()):
                    _reap(p)
                    shutil.rmtree(sd, ignore_errors=True)
                running.clear()
                raise TesseraNoDataError(result["error"])

            if result.get("permanent"):
                # One-line error via the raised exception; traceback DEBUG-only.
                logger.debug(
                    f"{tile_id} | TESSERA {year} traceback "
                    f"(bbox={bbox}):\n{result.get('tb') or msg}"
                )
                raise TesseraNoDataError(f"{year}: {msg} (bbox={bbox})")

            raise RuntimeError(f"{year}: failed: {msg}")

    finally:
        for _year, (proc, scratch_dir, _dest) in list(running.items()):
            _reap(proc)
            shutil.rmtree(scratch_dir, ignore_errors=True)

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
