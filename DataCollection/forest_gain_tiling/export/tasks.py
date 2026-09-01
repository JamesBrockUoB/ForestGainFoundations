from __future__ import annotations

import logging
import multiprocessing as mp
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
from config import settings
from embeddings.tasks import process_all_embeddings_with_retry
from enums import TileStatus
from export.aee import submit_aee_exports
from export.composites import submit_composite_exports
from export.drive import (
    check_hpc_available,
    rclone_all_products,
)
from export.labels import submit_label_exports
from export.metadata import write_tile_metadata
from export.static import submit_static_exports
from gee.auth import get_ee_credentials
from gee.cleanup import _cleanup_failed_tile
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer
from registry.store import update_tile
from stack.stacks import build_full_valid
from tiling.grid import crs_transform, tile_geom


def get_local_output_dir(tile_id: str) -> Path:
    return settings.data_dir / "test_tiles" / tile_id


def _wait_for_all(
    tasks: dict[str, ee.batch.Task],
    logger: logging.Logger,
    tile_id: str,
) -> bool:
    """
    Poll every submitted GEE export task until each is COMPLETED, or
    return False (cancelling any still-running tasks) the moment one
    FAILS/CANCELLED. This is what makes "mark complete" wait for every
    product rather than just the first to finish.
    """
    pending = dict(tasks)

    while pending:
        for key, task in list(pending.items()):
            state = task.status()["state"]

            if state == "COMPLETED":
                del pending[key]

            elif state in ("FAILED", "CANCELLED", "CANCEL_REQUESTED"):
                err = task.status().get("error_message", "unknown")
                logger.error(f"{tile_id} | export failed: {key} — {err}")

                for other_key, other_task in pending.items():
                    if other_key != key:
                        try:
                            other_task.cancel()
                        except Exception:
                            pass

                return False

        if pending:
            time.sleep(settings.poll_interval)

    return True


def _verify_tile_outputs(output_dir: Path, gee_product_keys: list[str]) -> list[str]:
    """
    Final gate before COMPLETE: confirm every GEE-derived product and
    every expected embedding file for the active period actually landed
    on disk. Embedding years are scoped to settings.period_years (p1:
    2017-2020, p2: 2020-2024) — NOT a fixed 2017-2024 range, since each
    period only ever needs its own years. Returns a list of missing
    relative product keys (empty = fully verified).
    """
    missing: list[str] = []

    for key in gee_product_keys:
        category, name = key.split("/", 1)
        if not (output_dir / category / f"{name}.tif").exists():
            missing.append(key)

    for year in settings.period_years:
        if not (output_dir / "embeddings" / f"tessera_{year}.tif").exists():
            missing.append(f"embeddings/tessera_{year}")
        if not (output_dir / "embeddings" / f"aee_{year}.tif").exists():
            missing.append(f"embeddings/aee_{year}")

    return missing


def process_tile(
    tile: dict[str, Any],
    ds: Datasets,
    logger: logging.Logger,
    local_output: bool = False,
) -> str:
    """
    Process one tile: submit every GEE export (composites/static/labels/
    [aee if aee_source=="gee"]), start embeddings downloading IMMEDIATELY
    in parallel (TESSERA, and AEE if aee_source=="geoai" -- neither
    depends on GEE task state), wait for all GEE tasks, rclone, then join
    the embeddings thread and verify everything landed. A tile is only
    ever marked COMPLETE after every product has been confirmed on disk.
    """
    tile_id = tile["tile_id"]
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    tasks: dict[str, ee.batch.Task] = {}
    output_dir: Path | None = None
    cancel_event = threading.Event()
    t_embeddings: threading.Thread | None = None

    try:
        _, _, gain_confidence = build_gain_layer(geom, ds)
        full_valid = build_full_valid(geom)

        tasks.update(submit_composite_exports(geom, ct, full_valid, tile_id))
        tasks.update(submit_static_exports(geom, ct, full_valid, tile_id))
        tasks.update(
            submit_label_exports(geom, ct, full_valid, ds, gain_confidence, tile_id)
        )
        if settings.aee_source == "gee":
            tasks.update(submit_aee_exports(geom, ct, tile_id))

        update_tile(
            tile_id,
            status=TileStatus.SUBMITTED,
            gee_task_id=",".join(t.id for t in tasks.values()),
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"{tile_id} | submitted {len(tasks)} export tasks")

        if local_output:
            output_dir = get_local_output_dir(tile_id)
            dest_root = str(output_dir.parent)
        else:
            if not settings.hpc_path:
                raise RuntimeError("HPC_PATH is not configured")
            output_dir = Path(settings.hpc_path) / tile_id
            dest_root = settings.hpc_path

        # Embeddings (TESSERA always, + AEE when aee_source=="geoai") has
        # no dependency on GEE task state -- start it now so it overlaps
        # the entire _wait_for_all polling window below, not just the
        # rclone step after. process_all_embeddings_with_retry already
        # skips the geoai-AEE download when aee_source=="gee", so this is always safe to
        # start unconditionally regardless of aee_source.
        embeddings_result: dict[str, bool] = {}

        def _run_embeddings() -> None:
            embeddings_result["ok"] = process_all_embeddings_with_retry(
                tile,
                output_dir,
                logger,
                cancel_event,
            )

        t_embeddings = threading.Thread(target=_run_embeddings)
        t_embeddings.start()

        if not _wait_for_all(tasks, logger, tile_id):
            raise RuntimeError("one or more GEE export tasks failed")

        logger.info(f"{tile_id} | all exports complete")

        logger.info(f"{tile_id} | rcloning data")
        products = [tuple(key.split("/", 1)) for key in tasks.keys()]
        rclone_ok = rclone_all_products(tile_id, products, dest_root, logger)

        # Embeddings has been running since before _wait_for_all started --
        # by this point it's very likely already finished (or close to
        # it), so this join is typically near-instant
        t_embeddings.join()

        if not rclone_ok:
            raise RuntimeError("rclone transfer failed")

        if not embeddings_result.get("ok"):
            raise RuntimeError("embedding acquisition failed")

        missing = _verify_tile_outputs(output_dir, list(tasks.keys()))
        if missing:
            raise RuntimeError(f"missing outputs after processing: {missing}")

        logger.info(f"{tile_id} | rcloning complete")

        write_tile_metadata(tile, output_dir, logger)

        update_tile(
            tile_id,
            status=TileStatus.COMPLETE,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"{tile_id} | complete")
        return str(TileStatus.COMPLETE)

    except Exception as exc:
        logger.exception(f"{tile_id} | processing failed")

        return _cleanup_failed_tile(
            tile_id=tile_id,
            reason=str(exc),
            logger=logger,
            tasks=tasks,
            output_dir=output_dir,
            embeddings_thread=t_embeddings,
            cancel_event=cancel_event,
        )


def run_local(
    candidates: list[dict],
    ds: Datasets,
    logger: logging.Logger,
    local_output: bool = False,
) -> None:
    """Process tiles sequentially."""
    if not local_output:
        if not settings.hpc_path:
            raise RuntimeError("HPC_PATH is not configured")
        if not check_hpc_available(settings.hpc_path, logger):
            logger.error(f"HPC destination unreachable: {settings.hpc_path}")
            return

    total = len(candidates)
    for i, tile in enumerate(candidates, 1):
        logger.info(f"Tile {i}/{total}: {tile['tile_id']}")
        process_tile(tile, ds, logger, local_output)
        time.sleep(0.2)


def _mp_worker(
    tile_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
    local_output: bool,
) -> None:
    """HPC worker."""
    time.sleep(worker_id * 5)
    ee.Initialize(get_ee_credentials(), project=settings.gee_project)

    ds = Datasets()
    logger = logging.getLogger(f"gee.worker.{worker_id}")

    while True:
        tile = tile_queue.get()
        if tile is None:
            break

        tile_id = tile["tile_id"]

        for attempt in range(8):
            status = process_tile(tile, ds, logger, local_output)
            if status != str(TileStatus.FAILED):
                break

            from registry.store import load_registry_entry

            entry = load_registry_entry(tile_id)
            error = entry.get("error", "") if entry else ""

            if any(
                k in error.lower() for k in ("429", "quota", "concurrent", "memory")
            ):
                wait = 2**attempt + random.uniform(0, 2)
                logger.warning(f"{tile_id} retry {attempt + 1}/8 in {wait:.1f}s")
                time.sleep(wait)
            else:
                break

        result_queue.put(tile_id)


def _mp_writer(result_queue: mp.Queue, total: int, logger: logging.Logger) -> None:
    from registry.store import _get_db

    db = _get_db()
    done = 0
    start = time.time()

    while done < total:
        _ = result_queue.get()
        done += 1

        if done % 20 == 0:
            elapsed = (time.time() - start) / 60
            rate = done / elapsed if elapsed else 0
            counts = db.status_counts()
            logger.info(
                f"{done}/{total} "
                f"complete={counts.get(str(TileStatus.COMPLETE),0)} "
                f"failed={counts.get(str(TileStatus.FAILED),0)} "
                f"{rate:.1f} tiles/min"
            )


def run_hpc(
    candidates: list[dict],
    logger: logging.Logger,
) -> None:
    """
    Process tiles with HPC workers. `ds` is accepted only to match
    run_local's call signature from main.py's cmd_run — each worker
    process builds its own Datasets() in _mp_worker, since ee-bound
    objects can't cross a process boundary, so the ds passed in here is
    unused.
    """
    if not settings.hpc_path:
        raise RuntimeError("HPC_PATH is not configured")
    if not check_hpc_available(settings.hpc_path, logger):
        logger.error(f"HPC destination unreachable: {settings.hpc_path}")
        return
    tile_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(
            target=_mp_worker,
            args=(tile_queue, result_queue, i, False),
        )
        for i in range(settings.num_workers)
    ]
    writer = threading.Thread(
        target=_mp_writer, args=(result_queue, len(candidates), logger)
    )

    for worker in workers:
        worker.start()
    writer.start()

    for tile in candidates:
        tile_queue.put(tile)
    for _ in workers:
        tile_queue.put(None)

    for worker in workers:
        worker.join()
    writer.join()
