from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from typing import Any

import ee
from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from filtering.checks import filter_tile
from registry.store import update_tile


def filter_single_tile(
    tile: dict[str, Any], ds: Datasets, logger: logging.Logger
) -> str:
    tile_id = tile["tile_id"]
    try:
        new_status, reason = filter_tile(tile, ds)
        if new_status == str(TileStatus.REJECTED):
            logger.info(f"reject: {tile_id} — {reason}")
            update_tile(tile_id, status=TileStatus.REJECTED, rejection_reason=reason)
        else:
            logger.info(f"valid: {tile_id}")
            update_tile(tile_id, status=TileStatus.VALID)
        return new_status
    except Exception as exc:
        logger.error(f"error filtering {tile_id}: {exc}")
        update_tile(tile_id, status=TileStatus.FAILED, error=str(exc))
        return str(TileStatus.FAILED)


def run_filter_local(
    candidates: list[dict], ds: Datasets, logger: logging.Logger
) -> None:
    total = len(candidates)
    for i, tile in enumerate(candidates, 1):
        logger.info(f"Filtering {i}/{total}: {tile['tile_id']}")
        filter_single_tile(tile, ds, logger)
        time.sleep(0.2)


def _mp_worker(tile_queue: mp.Queue, result_queue: mp.Queue, worker_id: int) -> None:
    credentials = ee.ServiceAccountCredentials(None, settings.gee_credentials)
    time.sleep(worker_id * 5)
    ee.Initialize(credentials, project=settings.gee_project)

    ds = Datasets()
    logger = logging.getLogger(f"gee.filter.worker.{worker_id}")

    while True:
        tile = tile_queue.get()
        if tile is None:
            break

        status = filter_single_tile(tile, ds, logger)
        result_queue.put((tile["tile_id"], status))


def _mp_writer(result_queue: mp.Queue, total: int, logger: logging.Logger) -> None:
    from registry.store import _get_db

    db = _get_db()
    done = 0
    t0 = time.time()

    while done < total:
        try:
            _tile_id, _status = result_queue.get(timeout=600)
        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total})")
            continue

        done += 1

        if done % 50 == 0:
            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0
            counts = db.status_counts()
            logger.info(
                f"Progress {done}/{total} | "
                f"valid={counts.get(str(TileStatus.VALID), 0)} "
                f"rejected={counts.get(str(TileStatus.REJECTED), 0)} "
                f"failed={counts.get(str(TileStatus.FAILED), 0)} | "
                f"{rate:.1f} tiles/min | {elapsed:.1f}min elapsed"
            )

    logger.info("Filter writer complete")


def run_filter_hpc(
    candidates: list[dict], ds: Datasets, logger: logging.Logger
) -> None:
    tile_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(target=_mp_worker, args=(tile_queue, result_queue, i))
        for i in range(settings.num_workers)
    ]
    writer_thread = threading.Thread(
        target=_mp_writer,
        args=(result_queue, len(candidates), logger),
        daemon=False,
    )

    for w in workers:
        w.start()
    writer_thread.start()

    logger.info(f"Started {settings.num_workers} filter workers + 1 writer thread")

    for tile in candidates:
        tile_queue.put(tile)
    for _ in range(settings.num_workers):
        tile_queue.put(None)

    logger.info(f"Queued {len(candidates)} tiles for filtering")

    for w in workers:
        w.join()
    writer_thread.join()
