from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
from config import settings
from embeddings.tasks import process_all_embeddings_with_retry
from enums import TileStatus
from export.aee import build_year_aee
from export.atlas import build_atlas_layout, split_atlas_file, submit_atlas_export
from export.composites import build_year_composite
from export.drive import (
    check_hpc_available,
    place_split_file,
    rclone_download_atlas,
)
from export.labels import build_label_layers
from export.metadata import write_tile_metadata
from export.static import build_static_layers
from export.tasks import _wait_for_all, get_local_output_dir
from gee.auth import get_ee_credentials
from gee.cleanup import _cleanup_failed_tile
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer
from registry.store import update_tile
from stack.stacks import build_full_valid
from tiling.grid import tile_geom


def _output_dir_for(tile_id: str, local_output: bool) -> Path:
    if local_output:
        return get_local_output_dir(tile_id)
    if not settings.hpc_path:
        raise RuntimeError("HPC_PATH is not configured")
    return Path(settings.hpc_path) / tile_id


def _dest_root(local_output: bool) -> str:
    if local_output:
        return str(settings.data_dir / "test_tiles")
    if not settings.hpc_path:
        raise RuntimeError("HPC_PATH is not configured")
    return settings.hpc_path


def process_tile_batch(
    tiles: list[dict[str, Any]],
    ds: Datasets,
    logger: logging.Logger,
    local_output: bool = False,
    batch_id: str | None = None,
    local_execution: bool = False,
) -> dict[str, str]:
    """
    Process a batch of tiles' GEE-derived products (composites, static,
    labels, [aee if gee]) as one packed "atlas" export per product,
    instead of one export per tile per product. Tiles need not be
    spatially adjacent — packing is purely pixel-space bookkeeping via
    export.atlas. Embeddings (TESSERA / geoai-AEE) stay per-tile, same
    as process_tile, since they don't touch GEE's export/task quota.

    Returns {tile_id: final_status_str}.
    """
    batch_id = batch_id or uuid.uuid4().hex[:8]
    batch_label = f"batch_{batch_id}"

    tiles_by_id = {t["tile_id"]: t for t in tiles}
    layout = build_atlas_layout(tiles)

    results: dict[str, str] = {}
    output_dirs = {tid: _output_dir_for(tid, local_output) for tid in tiles_by_id}
    dest_root = _dest_root(local_output)

    # --- 1) per-tile precompute (gain layer, full_valid mask, static &
    # label layers) — once per tile, reused across every product's
    # per-tile builder closure below, same as process_tile does for a
    # single tile. ---
    full_valid_by_tile: dict[str, ee.Image] = {}
    static_layers_by_tile: dict[str, dict[str, ee.Image]] = {}
    label_layers_by_tile: dict[str, dict[str, ee.Image]] = {}

    for tile in tiles:
        tid = tile["tile_id"]
        geom = tile_geom(tile)
        try:
            _, _, gain_confidence = build_gain_layer(geom, ds)
            full_valid_by_tile[tid] = build_full_valid(geom)
            static_layers_by_tile[tid] = build_static_layers(geom)
            label_layers_by_tile[tid] = build_label_layers(geom, ds, gain_confidence)
        except Exception as exc:
            logger.error(f"{batch_label} | {tid} | precompute failed: {exc}")
            results[tid] = _cleanup_failed_tile(
                tile_id=tid,
                reason=f"batch precompute failed: {exc}",
                logger=logger,
                tasks={},
                output_dir=output_dirs[tid],
                embeddings_thread=None,
                cancel_event=threading.Event(),
            )

    active_tiles = [t for t in tiles if t["tile_id"] not in results]
    if not active_tiles:
        return results

    # --- 2) submit one atlas export per product across active_tiles ---
    tasks: dict[str, ee.batch.Task] = {}

    for year in settings.period_years:

        def _composite_builder(tile, year=year):
            tid = tile["tile_id"]
            geom = tile_geom(tile)
            return build_year_composite(geom, year).updateMask(full_valid_by_tile[tid])

        tasks[f"composites/s1s2_{year}"] = submit_atlas_export(
            active_tiles,
            layout,
            _composite_builder,
            batch_id=batch_id,
            category="composites",
            name=f"s1s2_{year}",
        )

    static_names = list(next(iter(static_layers_by_tile.values())).keys())
    for name in static_names:

        def _static_builder(tile, name=name):
            tid = tile["tile_id"]
            return static_layers_by_tile[tid][name].updateMask(full_valid_by_tile[tid])

        tasks[f"static/{name}"] = submit_atlas_export(
            active_tiles,
            layout,
            _static_builder,
            batch_id=batch_id,
            category="static",
            name=name,
        )

    label_names = list(next(iter(label_layers_by_tile.values())).keys())
    for name in label_names:

        def _label_builder(tile, name=name):
            tid = tile["tile_id"]
            return label_layers_by_tile[tid][name].updateMask(full_valid_by_tile[tid])

        tasks[f"labels/{name}"] = submit_atlas_export(
            active_tiles,
            layout,
            _label_builder,
            batch_id=batch_id,
            category="labels",
            name=name,
        )

    if settings.aee_source == "gee":
        for year in settings.period_years:

            def _aee_builder(tile, year=year):
                return build_year_aee(tile_geom(tile), year)

            tasks[f"embeddings/aee_{year}"] = submit_atlas_export(
                active_tiles,
                layout,
                _aee_builder,
                batch_id=batch_id,
                category="embeddings",
                name=f"aee_{year}",
            )

    submitted_times = {k: time.time() for k in tasks}
    task_id_str = ",".join(t.id for t in tasks.values())

    for tile in active_tiles:
        update_tile(
            tile["tile_id"],
            status=TileStatus.SUBMITTED,
            gee_task_id=task_id_str,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

    logger.info(
        f"{batch_label} | submitted {len(tasks)} atlas export tasks for {len(active_tiles)} tiles"
    )

    # --- 3) start per-tile embeddings immediately, in parallel with the
    # atlas export wait. Local execution runs TESSERA serially to avoid RAM spikes
    cancel_event = threading.Event()
    embeddings_threads: dict[str, threading.Thread] = {}
    embeddings_results: dict[str, bool] = {}

    if local_execution and len(active_tiles) > 1:
        for tile in active_tiles:
            tid = tile["tile_id"]

            if cancel_event.is_set():
                embeddings_results[tid] = False
                continue

            try:
                embeddings_results[tid] = process_all_embeddings_with_retry(
                    tile,
                    output_dirs[tid],
                    logger,
                    cancel_event,
                )
            except Exception as exc:
                logger.error(
                    f"{batch_label} | {tid} | embedding acquisition failed: {exc}"
                )
                embeddings_results[tid] = False
    else:
        for tile in active_tiles:
            tid = tile["tile_id"]

            def _run_embeddings(tile=tile, tid=tid):
                embeddings_results[tid] = process_all_embeddings_with_retry(
                    tile,
                    output_dirs[tid],
                    logger,
                    cancel_event,
                )

            th = threading.Thread(target=_run_embeddings)
            th.start()
            embeddings_threads[tid] = th

    # --- 4) wait for atlas exports ---
    if not _wait_for_all(tasks, logger, batch_label, submitted_times=submitted_times):
        cancel_event.set()
        for th in embeddings_threads.values():
            th.join()
        for tile in active_tiles:
            tid = tile["tile_id"]
            results[tid] = _cleanup_failed_tile(
                tile_id=tid,
                reason="one or more batch export tasks failed",
                logger=logger,
                tasks=tasks,
                output_dir=output_dirs[tid],
                embeddings_thread=None,
                cancel_event=cancel_event,
            )
        return results

    logger.info(f"{batch_label} | all atlas exports complete")

    # --- 5) download each atlas file, split into per-tile files, place
    # each split file at its final destination ---
    staging_dir = settings.data_dir / "atlas_staging" / batch_id
    split_ok = True

    for key in tasks:
        category, name = key.split("/", 1)
        drive_name = f"atlas__{batch_id}__{category}__{name}.tif"
        local_atlas_path = staging_dir / f"{category}__{name}.tif"

        if not rclone_download_atlas(drive_name, local_atlas_path, logger):
            logger.error(f"{batch_label} | failed to download atlas file {drive_name}")
            split_ok = False
            break

        tile_split_dir = staging_dir / "split" / category
        try:
            split_atlas_file(
                local_atlas_path,
                layout,
                tiles_by_id,
                output_paths=lambda tid, d=tile_split_dir, name=name: d
                / f"{tid}__{name}.tif",
            )
        except Exception as exc:
            logger.error(f"{batch_label} | split failed for {key}: {exc}")
            split_ok = False
            break

        for tile in active_tiles:
            tid = tile["tile_id"]
            local_split_path = tile_split_dir / f"{tid}__{name}.tif"
            placed = place_split_file(
                local_split_path,
                tid,
                category,
                name,
                dest_root,
                local_output,
                logger,
            )
            if not placed:
                logger.warning(
                    f"{batch_label} | {tid} | failed to place {category}/{name}"
                )

    for th in embeddings_threads.values():
        th.join()

    if not split_ok:
        cancel_event.set()
        for tile in active_tiles:
            tid = tile["tile_id"]
            results[tid] = _cleanup_failed_tile(
                tile_id=tid,
                reason="atlas download/split failed",
                logger=logger,
                tasks=tasks,
                output_dir=output_dirs[tid],
                embeddings_thread=None,
                cancel_event=cancel_event,
            )
        return results

    # --- 6) per-tile verification + metadata + status, same gate as
    # process_tile ---
    for tile in active_tiles:
        tid = tile["tile_id"]

        if not embeddings_results.get(tid):
            results[tid] = _cleanup_failed_tile(
                tile_id=tid,
                reason="embedding acquisition failed",
                logger=logger,
                tasks=tasks,
                output_dir=output_dirs[tid],
                embeddings_thread=None,
                cancel_event=cancel_event,
            )
            continue

        missing = [
            key
            for key in tasks
            if not (
                output_dirs[tid]
                / Path(key.split("/", 1)[0])
                / f"{key.split('/', 1)[1]}.tif"
            ).exists()
        ]
        for year in settings.period_years:
            if not (output_dirs[tid] / "embeddings" / f"tessera_{year}.tif").exists():
                missing.append(f"embeddings/tessera_{year}")
            if not (output_dirs[tid] / "embeddings" / f"aee_{year}.tif").exists():
                missing.append(f"embeddings/aee_{year}")

        if missing:
            results[tid] = _cleanup_failed_tile(
                tile_id=tid,
                reason=f"missing outputs after batch processing: {missing}",
                logger=logger,
                tasks=tasks,
                output_dir=output_dirs[tid],
                embeddings_thread=None,
                cancel_event=cancel_event,
            )
            continue

        try:
            write_tile_metadata(tile, output_dirs[tid], logger)
        except Exception as exc:
            logger.warning(f"{batch_label} | {tid} | metadata write failed: {exc}")

        update_tile(
            tid,
            status=TileStatus.COMPLETE,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"{batch_label} | {tid} | complete")
        results[tid] = str(TileStatus.COMPLETE)

    return results


def run_batched_local(
    candidates: list[dict],
    ds: Datasets,
    logger: logging.Logger,
    tile_batch_size: int,
    local_output: bool = False,
) -> None:
    """Process tiles in fixed-size atlas batches, sequentially."""
    if not local_output:
        dest = _dest_root(local_output)
        if not check_hpc_available(dest, logger):
            logger.error(f"HPC destination unreachable: {dest}")
            return

    total_batches = -(-len(candidates) // tile_batch_size)
    for i in range(0, len(candidates), tile_batch_size):
        batch = candidates[i : i + tile_batch_size]
        batch_num = i // tile_batch_size + 1
        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} tiles")
        process_tile_batch(batch, ds, logger, local_output=local_output, local_execution=True)
        time.sleep(0.2)


def _mp_batch_worker(
    batch_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
    local_output: bool,
) -> None:
    time.sleep(worker_id * 5)
    ee.Initialize(get_ee_credentials(), project=settings.gee_project)

    ds = Datasets()
    logger = logging.getLogger(f"gee.batchworker.{worker_id}")

    while True:
        item = batch_queue.get()
        if item is None:
            break
        batch_idx, batch = item
        results = process_tile_batch(
            batch,
            ds,
            logger,
            local_output=local_output,
            batch_id=f"w{worker_id}_{batch_idx}",
            local_execution=False,
        )
        result_queue.put((batch_idx, results))


def run_batched_hpc(
    candidates: list[dict],
    logger: logging.Logger,
    tile_batch_size: int,
) -> None:
    """Process tiles in fixed-size atlas batches, distributed across
    HPC worker processes — each worker handles whole batches
    sequentially rather than individual tiles."""
    if not settings.hpc_path:
        raise RuntimeError("HPC_PATH is not configured")
    if not check_hpc_available(settings.hpc_path, logger):
        logger.error(f"HPC destination unreachable: {settings.hpc_path}")
        return

    batches = [
        candidates[i : i + tile_batch_size]
        for i in range(0, len(candidates), tile_batch_size)
    ]

    batch_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(target=_mp_batch_worker, args=(batch_queue, result_queue, i, False))
        for i in range(settings.num_workers)
    ]
    for w in workers:
        w.start()

    for idx, batch in enumerate(batches, start=1):
        batch_queue.put((idx, batch))
    for _ in workers:
        batch_queue.put(None)

    done = 0
    total_tiles = 0
    while done < len(batches):
        _idx, results = result_queue.get()
        done += 1
        total_tiles += len(results)
        logger.info(
            f"batch {done}/{len(batches)} done | {total_tiles} tiles processed so far"
        )

    for w in workers:
        w.join()
