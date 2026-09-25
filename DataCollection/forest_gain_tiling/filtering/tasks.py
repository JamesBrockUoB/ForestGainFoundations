from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from itertools import islice

import ee
from config import settings
from enums import TileStatus
from filtering.tile_batches import (
    count_pending,
    iter_pending_tile_batches,
    iter_stratified_pending_tile_batches,
)
from filtering.tile_filter import filter_batch_cheap, filter_batch_imagery
from gee.auth import get_ee_credentials
from gee_datasets.registry import Datasets
from tqdm import tqdm

# Stage keys must match main.py's `--stage` choices (["cheap", "imagery"]).
STAGES = {
    "cheap": {
        "input_status": str(TileStatus.PENDING),
        "fn": filter_batch_cheap,
    },
    "imagery": {
        "input_status": str(TileStatus.CHEAP_VALID),
        "fn": filter_batch_imagery,
    },
}


def _resolve_batch_iter(
    cfg: dict,
    batch_size: int,
    limit_batches: int | None,
    stratify_field: str | None,
    stratify_mode: str,
    tile_limit: int | None,
    logger: logging.Logger,
):
    """
    Shared batch-source resolution for both local and HPC runners.
    Returns (batch_iter, total_batches_estimate).

    Stratified mode ignores limit_batches — tile_limit is the knob there,
    since "N batches" doesn't map onto "allocated draw across strata" the
    same way it does for the plain xi/yi-ordered scan.
    """
    if stratify_field:
        if tile_limit is None:
            raise ValueError("tile_limit is required when stratify_field is set")

        total_batches = -(-tile_limit // batch_size)
        batch_iter = iter_stratified_pending_tile_batches(
            cfg["input_status"],
            batch_size,
            stratify_field,
            tile_limit,
            mode=stratify_mode,
            logger=logger,
        )
        logger.info(
            f"starting | batch_size={batch_size} | "
            f"stratify={stratify_field} mode={stratify_mode} | "
            f"tile_limit={tile_limit:,} | total_batches(est)={total_batches:,}"
        )
        return batch_iter, total_batches

    total_pending = count_pending(cfg["input_status"])
    total_batches = -(-total_pending // batch_size)
    if limit_batches is not None:
        total_batches = min(total_batches, limit_batches)

    batch_iter = iter_pending_tile_batches(cfg["input_status"], batch_size)
    if limit_batches is not None:
        batch_iter = islice(batch_iter, limit_batches)

    logger.info(
        f"starting | batch_size={batch_size} | "
        f"pending={total_pending:,} | total_batches={total_batches:,}"
    )
    return batch_iter, total_batches


def run_filter_local(
    logger: logging.Logger,
    stage: str,
    batch_size: int,
    limit_batches: int | None = None,
    stratify_field: str | None = None,
    stratify_mode: str = "prop",
    tile_limit: int | None = None,
) -> None:
    cfg = STAGES[stage]
    ds = Datasets() if stage == "cheap" else None

    batch_iter, total_batches = _resolve_batch_iter(
        cfg,
        batch_size,
        limit_batches,
        stratify_field,
        stratify_mode,
        tile_limit,
        logger,
    )

    totals: dict[str, int] = {}
    n_tiles = 0
    n_batches = 0
    t0 = time.time()

    with tqdm(
        batch_iter, total=total_batches, desc=f"filter[{stage}]", unit="batch"
    ) as pbar:
        for tiles in pbar:
            n_batches += 1
            n_tiles += len(tiles)

            if stage == "cheap":
                counts = cfg["fn"](tiles, ds, logger, batch_label=f"batch {n_batches}")
            else:
                counts = cfg["fn"](tiles, logger, batch_label=f"batch {n_batches}")

            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v

            elapsed_min = (time.time() - t0) / 60
            rate = n_tiles / elapsed_min if elapsed_min > 0 else 0
            pbar.set_postfix(
                {**totals, "tiles": n_tiles, "t/min": f"{rate:.0f}"}, refresh=False
            )

    elapsed = (time.time() - t0) / 60
    logger.info(
        f"complete: {n_batches:,} batches, {n_tiles:,} tiles | "
        + " ".join(f"{k}={v:,}" for k, v in totals.items())
        + f" | {elapsed:.1f}min elapsed"
    )


def _mp_worker(
    stage: str, logfile, batch_queue: mp.Queue, result_queue: mp.Queue, worker_id: int
) -> None:
    time.sleep(worker_id * 5)
    ee.Initialize(get_ee_credentials(), project=settings.gee_project)
    ds = Datasets() if stage == "cheap" else None  # <-- build once per worker

    fn = STAGES[stage]["fn"]
    logger = logging.getLogger(f"gee.filter.{stage}.worker.{worker_id}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)

    while True:
        item = batch_queue.get()
        if item is None:
            break
        batch_idx, tiles = item
        t0 = time.time()
        if stage == "cheap":
            counts = fn(tiles, ds, logger, batch_label=f"batch {batch_idx}")
        else:
            counts = fn(tiles, logger, batch_label=f"batch {batch_idx}")
        logger.debug(f"batch {batch_idx}: done in {time.time()-t0:.1f}s")
        result_queue.put((batch_idx, len(tiles), counts))


def _mp_writer(result_queue: mp.Queue, total: int, logger: logging.Logger) -> None:
    done = 0
    n_tiles = 0
    totals: dict[str, int] = {}
    t0 = time.time()

    with tqdm(total=total, desc="filter[hpc]", unit="batch") as pbar:
        while done < total:
            try:
                _idx, ntiles, counts = result_queue.get(timeout=600)
            except Exception:
                logger.warning(f"Result queue timeout ({done}/{total})")
                continue

            done += 1
            n_tiles += ntiles
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v

            elapsed_min = (time.time() - t0) / 60
            rate = n_tiles / elapsed_min if elapsed_min > 0 else 0
            pbar.set_postfix(
                {**totals, "tiles": n_tiles, "t/min": f"{rate:.0f}"}, refresh=False
            )
            pbar.update(1)

    logger.info(
        f"writer complete: {done:,}/{total:,} batches, {n_tiles:,} tiles | "
        + " ".join(f"{k}={v:,}" for k, v in totals.items())
    )


def run_filter_hpc(
    logger: logging.Logger,
    stage: str,
    batch_size: int,
    limit_batches: int | None = None,
    stratify_field: str | None = None,
    stratify_mode: str = "prop",
    tile_limit: int | None = None,
) -> None:
    cfg = STAGES[stage]
    batch_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    t0 = time.time()  # covers batch materialisation + worker spin-up/staggering

    batch_iter, _ = _resolve_batch_iter(
        cfg,
        batch_size,
        limit_batches,
        stratify_field,
        stratify_mode,
        tile_limit,
        logger,
    )

    # HPC path materialises batches up front (same as before this change) —
    # the stratified iterator already does its own DB work per stratum
    # before yielding, so this is just draining it into a list.
    batches = []
    for tiles in batch_iter:
        batches.append(tiles)
        if (
            stratify_field is None
            and limit_batches is not None
            and len(batches) >= limit_batches
        ):
            break

    if not batches:
        logger.info(f"no tiles pending for stage '{stage}'.")
        return

    # find the active logfile so workers can attach their own handlers to it
    logfile = next(
        h.baseFilename
        for h in logging.getLogger("gee").handlers
        if isinstance(h, logging.FileHandler)
    )

    workers = [
        mp.Process(
            target=_mp_worker, args=(stage, logfile, batch_queue, result_queue, i)
        )
        for i in range(settings.num_workers)
    ]
    writer_thread = threading.Thread(
        target=_mp_writer, args=(result_queue, len(batches), logger), daemon=False
    )

    for w in workers:
        w.start()
    writer_thread.start()

    logger.info(
        f"started {settings.num_workers} workers | "
        f"{len(batches):,} batches ({sum(len(b) for b in batches):,} tiles)"
        + (
            f" | stratify={stratify_field} mode={stratify_mode}"
            if stratify_field
            else ""
        )
    )

    for idx, tiles in enumerate(batches, start=1):
        batch_queue.put((idx, tiles))
    for _ in range(settings.num_workers):
        batch_queue.put(None)

    for w in workers:
        w.join()
    writer_thread.join()

    elapsed = (time.time() - t0) / 60
    n_tiles = sum(len(b) for b in batches)
    rate = n_tiles / elapsed if elapsed > 0 else 0
    logger.info(
        f"hpc filter complete: {len(batches):,} batches, {n_tiles:,} tiles | "
        f"{elapsed:.1f}min elapsed | {rate:.0f} tiles/min"
    )
