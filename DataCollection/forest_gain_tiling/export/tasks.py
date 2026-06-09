from __future__ import annotations

import logging
import multiprocessing as mp
import random
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import ee

from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from export.drive import rclone_to_hpc
from stack.stacks import build_full_stack, build_full_valid
from labels.gain import build_gain_layer
from labels.viability import score_viability
from registry.store import Registry, save_registry, update_tile
from tiling.grid import crs_transform, tile_geom


def process_tile(
    tile: dict[str, Any], registry: Registry, ds: Datasets, logger: logging.Logger
) -> str:
    tile_id = tile["tile_id"]
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    try:
        gain_validated, gain_binary = build_gain_layer(geom, ds)

        gain_stats = gain_binary.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=settings.scale,
            crs=settings.crs,
            crsTransform=ct,
            maxPixels=1_000_000_000,
        )
        gain_pct = (
            ee.Number(
                ee.Algorithms.If(gain_stats.get("gain"), gain_stats.get("gain"), 0)
            )
            .multiply(100)
            .getInfo()
        )

        if gain_pct < settings.gain_pct_min:
            reason = f"gain_pct={gain_pct:.3f} < {settings.gain_pct_min}"
            logger.info(f"reject (low gain {gain_pct:.2f}%): {tile_id}")
            update_tile(
                registry, tile_id, status=TileStatus.REJECTED, rejection_reason=reason
            )
            return str(TileStatus.REJECTED)

        viability = score_viability(geom, gain_validated, ds)
        if (
            viability["ndvi_delta"] <= settings.ndvi_delta_min
            or viability["gain_canopy_mean"] < settings.gain_canopy_min
        ):
            reason = f"viability={viability}"
            logger.info(f"reject (viability): {tile_id} {viability}")
            update_tile(
                registry, tile_id, status=TileStatus.REJECTED, rejection_reason=reason
            )
            return str(TileStatus.REJECTED)

        full_valid = build_full_valid(geom)
        stack = build_full_stack(tile, geom, gain_validated, full_valid, ds)

        task = ee.batch.Export.image.toDrive(
            image=stack,
            description=tile_id,
            folder=settings.drive_folder,
            fileNamePrefix=tile_id,
            region=geom,
            scale=settings.scale,
            crs=settings.crs,
            crsTransform=ct,
            maxPixels=10_000_000_000_000,
            fileFormat="GeoTIFF",
        )
        task.start()
        update_tile(
            registry,
            tile_id,
            status=TileStatus.SUBMITTED,
            gee_task_id=task.id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"submitted: {tile_id}  task={task.id}")

        return _poll_task(task, tile_id, registry, logger)

    except Exception as exc:
        logger.error(f"error processing {tile_id}: {exc}")
        update_tile(registry, tile_id, status=TileStatus.FAILED, error=str(exc))
        return str(TileStatus.FAILED)


def _poll_task(
    task: ee.batch.Task, tile_id: str, registry: Registry, logger: logging.Logger
) -> str:
    while True:
        state = task.status()["state"]

        if state == "COMPLETED":
            if settings.hpc_path:
                rclone_to_hpc(tile_id, settings.hpc_path, logger)
            update_tile(
                registry,
                tile_id,
                status=TileStatus.COMPLETE,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return str(TileStatus.COMPLETE)

        if state == "FAILED":
            err = task.status().get("error_message", "unknown")
            logger.error(f"GEE task failed: {tile_id} — {err}")
            update_tile(registry, tile_id, status=TileStatus.FAILED, error=err)
            return str(TileStatus.FAILED)

        if state in ("CANCELLED", "CANCEL_REQUESTED"):
            update_tile(
                registry, tile_id, status=TileStatus.FAILED, error="task cancelled"
            )
            return str(TileStatus.FAILED)

        time.sleep(settings.poll_interval)


def run_local(
    candidates: list[dict], registry: Registry, ds: Datasets, logger: logging.Logger
) -> None:
    total = len(candidates)
    for i, tile in enumerate(candidates, 1):
        logger.info(f"Tile {i}/{total}: {tile['tile_id']}")
        process_tile(tile, registry, ds, logger)
        time.sleep(0.2)


def _mp_worker(tile_queue: mp.Queue, result_queue: mp.Queue, worker_id: int) -> None:
    credentials = ee.ServiceAccountCredentials(None, settings.gee_credentials)
    time.sleep(worker_id * 5)
    ee.Initialize(credentials, project=settings.gee_project)

    ds = Datasets()
    local_registry: Registry = {}
    logger = logging.getLogger(f"gee.worker.{worker_id}")

    while True:
        tile = tile_queue.get()
        if tile is None:
            break

        tile_id = tile["tile_id"]
        local_registry[tile_id] = dict(tile)

        for attempt in range(8):
            status = process_tile(tile, local_registry, ds, logger)
            if status != str(TileStatus.FAILED):
                break
            err = local_registry[tile_id].get("error", "")
            if any(k in err.lower() for k in ("429", "concurrent", "quota", "memory")):
                wait = (2**attempt) + random.uniform(0, 2)
                logger.warning(
                    f"Worker {worker_id} | {tile_id} | retry {attempt+1}/8 in {wait:.1f}s"
                )
                time.sleep(wait)
            else:
                break
        else:
            logger.error(f"Worker {worker_id} | {tile_id}: exhausted retries")

        result_queue.put((tile_id, local_registry[tile_id]))


def _mp_writer(
    result_queue: mp.Queue, total: int, registry: Registry, logger: logging.Logger
) -> None:
    done = 0
    t0 = time.time()

    while done < total:
        try:
            tile_id, entry = result_queue.get(timeout=600)
        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total})")
            continue

        registry[tile_id] = entry
        done += 1

        if done % 20 == 0:
            save_registry(registry)
            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0
            counts = Counter(e["status"] for e in registry.values())
            logger.info(
                f"Progress {done}/{total} | "
                f"complete={counts[str(TileStatus.COMPLETE)]} "
                f"rejected={counts[str(TileStatus.REJECTED)]} "
                f"failed={counts[str(TileStatus.FAILED)]} | "
                f"{rate:.1f} tiles/min | {elapsed:.1f}min elapsed"
            )

    save_registry(registry)
    logger.info("Writer complete — registry flushed")


def run_hpc(
    candidates: list[dict], registry: Registry, ds: Datasets, logger: logging.Logger
) -> None:
    tile_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(target=_mp_worker, args=(tile_queue, result_queue, i))
        for i in range(settings.num_workers)
    ]
    writer_thread = threading.Thread(
        target=_mp_writer,
        args=(result_queue, len(candidates), registry, logger),
        daemon=False,
    )

    for w in workers:
        w.start()
    writer_thread.start()

    logger.info(f"Started {settings.num_workers} HPC workers + 1 writer thread")

    for tile in candidates:
        tile_queue.put(tile)
    for _ in range(settings.num_workers):
        tile_queue.put(None)

    logger.info(f"Queued {len(candidates)} tiles")

    for w in workers:
        w.join()
    writer_thread.join()
