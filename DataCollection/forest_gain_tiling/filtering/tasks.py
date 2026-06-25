from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time

import ee
from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from filtering.aoi_batches import iter_aoi_pending_tiles
from filtering.aoi_filter import filter_aoi


def run_filter_local(logger: logging.Logger, limit_aois: int | None = None) -> None:
    """
    Sequential AOI-batched filter run.
    Iterates AOIs with pending tiles, fetches one aggregated raster per
    AOI, and updates all of that AOI's pending tiles in one pass.
    """
    ds = Datasets()

    total_valid = 0
    total_rejected = 0
    total_failed = 0
    n_aois = 0
    t0 = time.time()

    for aoi_id, tiles in iter_aoi_pending_tiles(status=str(TileStatus.PENDING)):
        if limit_aois is not None and n_aois >= limit_aois:
            break

        logger.info(f"AOI {aoi_id}: starting ({len(tiles)} tiles)")
        t_aoi = time.time()

        counts = filter_aoi(aoi_id, tiles, ds, logger)

        elapsed_aoi = time.time() - t_aoi
        total_valid += counts.get("valid", 0)
        total_rejected += counts.get("rejected", 0)
        total_failed += counts.get("failed", 0)
        n_aois += 1

        logger.info(
            f"AOI {aoi_id}: done in {elapsed_aoi:.1f}s | "
            f"valid={counts.get('valid', 0)} rejected={counts.get('rejected', 0)} "
            f"failed={counts.get('failed', 0)}"
        )

        elapsed = (time.time() - t0) / 60
        rate = n_aois / elapsed if elapsed > 0 else 0
        logger.info(
            f"AOIs processed: {n_aois:,} | "
            f"valid={total_valid:,} rejected={total_rejected:,} "
            f"failed={total_failed:,} | "
            f"{rate:.1f} aois/min | {elapsed:.1f}min elapsed"
        )

    elapsed = (time.time() - t0) / 60
    logger.info(
        f"Filter complete: {n_aois:,} AOIs | "
        f"valid={total_valid:,} rejected={total_rejected:,} failed={total_failed:,} | "
        f"{elapsed:.1f}min elapsed"
    )


def _mp_worker(aoi_queue: mp.Queue, result_queue: mp.Queue, worker_id: int) -> None:
    credentials = ee.ServiceAccountCredentials(None, settings.gee_credentials)
    time.sleep(worker_id * 5)
    ee.Initialize(credentials, project=settings.gee_project)

    ds = Datasets()
    logger = logging.getLogger(f"gee.filter.worker.{worker_id}")

    while True:
        item = aoi_queue.get()
        if item is None:
            break

        aoi_id, tiles = item
        logger.info(f"AOI {aoi_id}: starting ({len(tiles)} tiles)")
        t_aoi = time.time()

        counts = filter_aoi(aoi_id, tiles, ds, logger)

        logger.info(
            f"AOI {aoi_id}: done in {time.time()-t_aoi:.1f}s | "
            f"valid={counts.get('valid', 0)} rejected={counts.get('rejected', 0)} "
            f"failed={counts.get('failed', 0)}"
        )

        result_queue.put((aoi_id, counts))


def _mp_writer(result_queue: mp.Queue, total: int, logger: logging.Logger) -> None:
    done = 0
    total_valid = 0
    total_rejected = 0
    total_failed = 0
    t0 = time.time()

    while done < total:
        try:
            _aoi_id, counts = result_queue.get(timeout=600)
        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total})")
            continue

        done += 1
        total_valid += counts.get("valid", 0)
        total_rejected += counts.get("rejected", 0)
        total_failed += counts.get("failed", 0)

        if done % 50 == 0:
            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                f"AOIs {done}/{total} | "
                f"valid={total_valid:,} rejected={total_rejected:,} "
                f"failed={total_failed:,} | "
                f"{rate:.1f} aois/min | {elapsed:.1f}min elapsed"
            )

    logger.info(
        f"Filter writer complete: {done:,}/{total:,} AOIs | "
        f"valid={total_valid:,} rejected={total_rejected:,} failed={total_failed:,}"
    )


def run_filter_hpc(logger: logging.Logger, limit_aois: int | None = None) -> None:
    """
    HPC AOI-batched filter run.
    Each worker pulls an AOI (aoi_id, tiles) off the queue, fetches its
    aggregated raster, and updates the DB directly.
    """
    aoi_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    aois = []
    for aoi_id, tiles in iter_aoi_pending_tiles(status=str(TileStatus.PENDING)):
        aois.append((aoi_id, tiles))
        if limit_aois is not None and len(aois) >= limit_aois:
            break

    if not aois:
        logger.info("No AOIs with pending tiles.")
        return

    workers = [
        mp.Process(target=_mp_worker, args=(aoi_queue, result_queue, i))
        for i in range(settings.num_workers)
    ]
    writer_thread = threading.Thread(
        target=_mp_writer,
        args=(result_queue, len(aois), logger),
        daemon=False,
    )

    for w in workers:
        w.start()
    writer_thread.start()

    logger.info(
        f"Started {settings.num_workers} filter workers + 1 writer thread "
        f"for {len(aois):,} AOIs"
    )

    for item in aois:
        aoi_queue.put(item)
    for _ in range(settings.num_workers):
        aoi_queue.put(None)

    for w in workers:
        w.join()
    writer_thread.join()
