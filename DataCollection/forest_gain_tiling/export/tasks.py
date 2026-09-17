from __future__ import annotations

import logging
import multiprocessing as mp
import random
import shutil
import subprocess
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
    rclone_push,
    rclone_read_bytes,
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


def get_embeddings_scratch_dir() -> Path:
    return settings.data_dir / "tessera_scratch"


def _wait_for_all(
    tasks: dict[str, ee.batch.Task],
    logger: logging.Logger,
    tile_id: str,
    submitted_times: dict[str, float] | None = None,
    poll_interval: float | None = None,
    cancel_event: threading.Event | None = None,
) -> bool:
    if poll_interval is None:
        poll_interval = settings.poll_interval

    pending = dict(tasks)
    first_running: dict[str, float] = {}
    completed_at: dict[str, float] = {}

    if submitted_times is None:
        submitted_times = {k: time.time() for k in tasks.keys()}

    while pending:
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(
                f"{tile_id} | cancellation requested; cancelling GEE exports"
            )
            for task in pending.values():
                try:
                    task.cancel()
                except Exception:
                    pass
            return False

        for key, task in list(pending.items()):
            try:
                status = task.status()
            except Exception:
                status = {}
            state = status.get("state")

            if state and state not in ("READY", "PENDING") and key not in first_running:
                first_running[key] = time.time()
                qtime = first_running[key] - submitted_times.get(
                    key, first_running[key]
                )
                logger.info(f"{tile_id} | {key} started (queue {qtime:.1f}s)")

            if state == "COMPLETED":
                end = time.time()
                start = first_running.get(key, submitted_times.get(key, end))
                runt = end - start
                qtime = start - submitted_times.get(key, start)
                total = end - submitted_times.get(key, end)
                logger.info(
                    f"{tile_id} | {key} completed (queue {qtime:.1f}s, run {runt:.1f}s, total {total:.1f}s)"
                )
                completed_at[key] = end
                del pending[key]

            elif state in ("FAILED", "CANCELLED", "CANCEL_REQUESTED"):
                err = status.get("error_message", "unknown")
                logger.error(f"{tile_id} | export failed: {key} — {err}")
                for other_key, other_task in pending.items():
                    if other_key != key:
                        try:
                            other_task.cancel()
                        except Exception:
                            pass
                return False

        if pending:
            time.sleep(poll_interval)

    if submitted_times and completed_at:
        first_submit = min(submitted_times.values())
        last_complete = max(completed_at.values())
        logger.info(
            f"{tile_id} | all exports finished (wall {(last_complete-first_submit):.1f}s)"
        )

    return True


def _dest_file_exists(dest_root: str, rel_path: str) -> bool:
    full_path = f"{dest_root}/{rel_path}"

    if ":" not in dest_root:
        return Path(full_path).exists()

    parent, filename = full_path.rsplit("/", 1)
    result = subprocess.run(["rclone", "lsf", parent], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return filename in result.stdout.splitlines()


def _verify_tile_outputs(
    tile_id: str,
    dest_root: str,
    gee_product_keys: list[str],
) -> list[str]:
    missing: list[str] = []

    for key in gee_product_keys:
        category, name = key.split("/", 1)
        rel_path = f"{tile_id}/{category}/{name}.tif"
        if not _dest_file_exists(dest_root, rel_path):
            missing.append(key)

    for year in settings.period_years:
        rel_path = f"{tile_id}/embeddings/tessera_{year}.tif"
        if not _dest_file_exists(dest_root, rel_path):
            missing.append(f"embeddings/tessera_{year}")

    return missing


def process_tile(
    tile: dict[str, Any],
    ds: Datasets,
    logger: logging.Logger,
    local_output: bool = False,
) -> str:
    tile_id = tile["tile_id"]
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    tasks: dict[str, ee.batch.Task] = {}
    output_dir: Path | None = None
    cancel_event = threading.Event()
    t_embeddings: threading.Thread | None = None
    rclone_completed = False

    try:
        _, _, gain_confidence = build_gain_layer(geom, ds)
        full_valid = build_full_valid(geom)

        tasks.update(submit_composite_exports(geom, ct, full_valid, tile_id))
        tasks.update(submit_static_exports(geom, ct, full_valid, tile_id))
        tasks.update(
            submit_label_exports(geom, ct, full_valid, ds, gain_confidence, tile_id)
        )
        tasks.update(submit_aee_exports(geom, ct, tile_id))

        submitted_times = {k: time.time() for k in tasks.keys()}

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
            output_dir = get_local_output_dir(tile_id)
            dest_root = settings.hpc_path

        embeddings_result: dict[str, bool] = {}

        embeddings_scratch = None if local_output else get_embeddings_scratch_dir()
        embeddings_target_dir = output_dir if local_output else embeddings_scratch

        def _run_embeddings() -> None:
            try:
                if embeddings_scratch is not None:
                    if embeddings_scratch.exists():
                        shutil.rmtree(embeddings_scratch)
                    embeddings_scratch.mkdir(parents=True, exist_ok=True)

                ok = process_all_embeddings_with_retry(
                    tile, embeddings_target_dir, logger, cancel_event
                )
                embeddings_result["ok"] = ok

                if not ok:
                    cancel_event.set()
                    return

                if embeddings_scratch is not None:
                    scratch_embeddings_dir = embeddings_scratch / "embeddings"

                    if not scratch_embeddings_dir.exists():
                        raise RuntimeError(
                            f"{tile_id} | TESSERA embeddings directory missing: "
                            f"{scratch_embeddings_dir}"
                        )

                    final_embeddings_dir = output_dir / "embeddings"
                    final_embeddings_dir.mkdir(parents=True, exist_ok=True)

                    for tif in scratch_embeddings_dir.glob("*.tif"):
                        destination = final_embeddings_dir / tif.name
                        if destination.exists():
                            destination.unlink()
                        shutil.move(str(tif), str(destination))

            finally:
                if embeddings_scratch is not None:
                    try:
                        if embeddings_scratch.exists():
                            shutil.rmtree(embeddings_scratch, ignore_errors=False)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logger.exception(
                            "%s | failed to remove TESSERA scratch: %s",
                            tile_id,
                            embeddings_scratch,
                        )

        t_embeddings = threading.Thread(target=_run_embeddings)
        t_embeddings.start()

        if not _wait_for_all(
            tasks,
            logger,
            tile_id,
            submitted_times=submitted_times,
            cancel_event=cancel_event,
        ):
            raise RuntimeError(
                "one or more GEE export tasks failed or processing was cancelled"
            )

        logger.info(f"{tile_id} | all exports complete")

        logger.info(f"{tile_id} | rcloning data")
        products = [tuple(key.split("/", 1)) for key in tasks.keys()]
        rclone_ok = rclone_all_products(tile_id, products, dest_root, logger)
        rclone_completed = rclone_ok

        t_embeddings.join()

        if not rclone_ok:
            logger.error(
                f"{tile_id} | rclone transfer failed; preserving local outputs for retry"
            )
            update_tile(
                tile_id,
                status=TileStatus.FAILED,
                error="rclone transfer failed; local outputs preserved",
            )
            return str(TileStatus.FAILED)

        if not embeddings_result.get("ok"):
            raise RuntimeError("embedding acquisition failed")

        if not local_output:
            if not rclone_push(
                str(output_dir / "embeddings"),
                f"{dest_root}/{tile_id}/embeddings",
                logger,
            ):
                raise RuntimeError("failed to push TESSERA embeddings to destination")

        missing = _verify_tile_outputs(tile_id, dest_root, list(tasks.keys()))
        if missing:
            raise RuntimeError(f"missing outputs after processing: {missing}")

        logger.info(f"{tile_id} | rcloning complete")

        if local_output:
            write_tile_metadata(tile, output_dir, logger)
        else:
            remote_labels_dir = f"{dest_root}/{tile_id}/labels"
            gain_bytes = rclone_read_bytes(
                f"{remote_labels_dir}/gain_confidence.tif", logger
            )
            if gain_bytes is None:
                raise RuntimeError(
                    f"{tile_id} | could not read gain_confidence.tif from "
                    f"{dest_root} for metadata computation"
                )
            pseudo_bytes = rclone_read_bytes(
                f"{remote_labels_dir}/pseudo_labels.tif", logger
            )
            write_tile_metadata(tile, output_dir, logger, gain_bytes, pseudo_bytes)
            if not rclone_push(
                str(output_dir / "metadata.json"),
                f"{dest_root}/{tile_id}/metadata.json",
                logger,
            ):
                raise RuntimeError("failed to push metadata.json to destination")

            try:
                shutil.rmtree(output_dir)
            except OSError as exc:
                logger.warning(
                    f"{tile_id} | failed to remove local output dir {output_dir}: {exc}"
                )

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
            drive_already_cleared=rclone_completed,
        )


def run_local(
    candidates: list[dict],
    ds: Datasets,
    logger: logging.Logger,
    local_output: bool = False,
) -> None:
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
