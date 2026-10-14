"""End-to-end tile pipeline: cheap filter -> imagery filter -> export, repeated
until the requested number of tiles has been EXPORTED (status COMPLETE).

The limit is an export quota, not a filtering budget. The pipeline filters in
waves, sizing each wave from the yield observed so far (tiles that survive both
filter stages per tile drawn), and exports as soon as enough VALID tiles exist.
Tiles that end up TESSERA_MISSING count towards the quota; once everything
else has finished, those tiles (only the ones created by this run) are retried
via retry_tessera_missing. Tiles that fail or are rejected do not count; the
loop tops up until the quota is met or no PENDING tiles are left.

Uses only the existing building blocks (run_filter_*, run_local/run_hpc,
filter_candidates, stratified_sample), so --stratify behaves as in `filter` and
`run`: waves are drawn by stratum, and each export round is a stratified sample
of the VALID pool.
"""

from __future__ import annotations

import logging
import math
import random

from config import settings
from enums import TileStatus
from export.tasks import retry_tessera_missing, run_hpc, run_local
from filtering.tasks import run_filter_hpc, run_filter_local
from gee_datasets.registry import Datasets
from registry.store import _get_db
from tiling.selection import (
    filter_candidates,
    load_or_compute_strata_ratios,
    log_strata_counts,
    stratified_sample,
)

# note: ratios come back as {stratum: fraction}; see load_or_compute_strata_ratios

_PENDING = str(TileStatus.PENDING)
_CHEAP = str(TileStatus.CHEAP_VALID)
_VALID = str(TileStatus.VALID)
_COMPLETE = str(TileStatus.COMPLETE)
_TESSERA_MISSING = str(TileStatus.TESSERA_MISSING)


def _counts() -> dict[str, int]:
    return _get_db().status_counts()


def _filter_stage(
    logger: logging.Logger,
    stage: str,
    batch_size: int,
    *,
    limit_batches: int | None = None,
    stratify_field: str | None = None,
    stratify_mode: str = "prop",
    tile_limit: int | None = None,
) -> None:
    fn = run_filter_hpc if settings.use_hpc else run_filter_local
    fn(
        logger,
        stage=stage,
        batch_size=batch_size,
        limit_batches=limit_batches,
        stratify_field=stratify_field,
        stratify_mode=stratify_mode,
        tile_limit=tile_limit,
    )


def _stratified_pick(
    pool: list[dict],
    field: str,
    n: int,
    mode: str,
    ratios: dict[str, float] | None,
) -> list[dict]:
    """stratified_sample rounds each stratum's quota (round(n * ratio)), so for
    small n it can return fewer than n tiles, or none, even when the pool is
    large. Ask for progressively more until n tiles come back, then shuffle
    before trimming so the trim does not favour whichever strata come first."""
    target = min(n, len(pool))
    ask = n
    chosen: list[dict] = []
    for _ in range(12):
        chosen = stratified_sample(pool, field, ask, mode, ratios=ratios)
        if len(chosen) >= target:
            break
        ask = math.ceil(ask * 1.6) + 1
    random.shuffle(chosen)
    return chosen[:target]


def run_pipeline(
    logger: logging.Logger,
    *,
    limit: int,
    stratify_field: str | None = None,
    stratify_mode: str = "prop",
    local_output: bool = False,
    batch_size_cheap: int,
    batch_size_imagery: int,
    initial_yield: float = 0.25,
    max_wave: int = 20000,
    max_waves: int = 50,
) -> int:
    """Returns the number of tiles exported (COMPLETE) during this run."""
    if limit <= 0:
        raise ValueError("limit must be a positive number of tiles")

    ratios = None
    if stratify_field and stratify_mode == "prop":
        # Fail fast on a missing/stale ratio cache before spending GEE calls.
        ratios = load_or_compute_strata_ratios(logger)[stratify_field]

    ds = None if settings.use_hpc else Datasets()

    c0 = _counts()
    start_done = c0.get(_COMPLETE, 0) + c0.get(_TESSERA_MISSING, 0)
    prior_missing = {
        t["tile_id"] for t in _get_db().list_tiles(status=_TESSERA_MISSING)
    }

    def exported() -> int:
        # COMPLETE plus TESSERA_MISSING (the latter are retried at the end).
        c = _counts()
        return c.get(_COMPLETE, 0) + c.get(_TESSERA_MISSING, 0) - start_done

    drawn_total = 0
    valid_gained_total = 0
    yield_est = initial_yield
    waves = 0
    stalled = 0
    exhausted = False

    logger.info(
        f"Pipeline: export {limit:,} tiles"
        + (
            f" | stratify={stratify_field} mode={stratify_mode}"
            if stratify_field
            else ""
        )
        + (" | local output" if local_output and not settings.use_hpc else "")
    )

    while True:
        remaining = limit - exported()
        if remaining <= 0:
            break

        # 1) Finish tiles that already passed the cheap stage (earlier runs).
        c = _counts()
        if c.get(_CHEAP, 0) > 0:
            logger.info(f"[pipeline] imagery filter on {c[_CHEAP]:,} CHEAP_VALID tiles")
            _filter_stage(logger, "imagery", batch_size_imagery)
            c = _counts()

        pool_n = c.get(_VALID, 0)

        # 2) Export once the VALID pool can cover what is still missing (or
        #    when no more tiles can be drawn).
        if pool_n >= remaining or exhausted or waves >= max_waves:
            if pool_n == 0:
                if waves >= max_waves and not exhausted:
                    logger.warning(
                        f"[pipeline] stopped: reached --max-waves={max_waves}"
                    )
                break

            pool = filter_candidates(
                status=_VALID,
                tile_id=None,
                aoi_id=None,
                biome=None,
                region=None,
                logger=logger,
            )
            if stratify_field:
                chosen = _stratified_pick(
                    pool, stratify_field, remaining, stratify_mode, ratios
                )
                log_strata_counts(chosen, stratify_field, logger, stratify_mode)
            else:
                chosen = pool[:remaining]

            if not chosen:
                break

            logger.info(
                f"[pipeline] exporting {len(chosen):,} tiles "
                f"({exported():,}/{limit:,} done)"
            )
            if settings.use_hpc:
                run_hpc(chosen, logger)
            else:
                run_local(chosen, ds, logger, local_output=local_output)

            # Guard against rounds that process nothing (e.g. destination
            # unreachable): processed tiles always leave the VALID status.
            stalled = stalled + 1 if _counts().get(_VALID, 0) >= pool_n else 0
            if stalled >= 2:
                logger.error(
                    "[pipeline] export is not processing tiles; stopping. "
                    "Check the log above (destination unreachable?)."
                )
                break
            continue

        # 3) Draw and filter another wave of PENDING tiles.
        pending_before = c.get(_PENDING, 0)
        if pending_before == 0:
            exhausted = True
            logger.warning("[pipeline] no PENDING tiles left to filter")
            continue

        shortage = remaining - pool_n
        y = max(0.005, yield_est)
        wave = math.ceil(shortage / y * 1.25)
        wave = min(max_wave, max(batch_size_cheap, wave), pending_before)

        logger.info(
            f"[pipeline] wave {waves + 1}: drawing {wave:,} tiles "
            f"(need {shortage:,} more VALID, est. yield {y:.1%})"
        )
        if stratify_field:
            _filter_stage(
                logger,
                "cheap",
                batch_size_cheap,
                stratify_field=stratify_field,
                stratify_mode=stratify_mode,
                tile_limit=wave,
            )
        else:
            _filter_stage(
                logger,
                "cheap",
                batch_size_cheap,
                limit_batches=math.ceil(wave / batch_size_cheap),
            )

        drawn = pending_before - _counts().get(_PENDING, 0)
        if drawn <= 0:
            exhausted = True
            logger.warning("[pipeline] the cheap stage drew no tiles")
            continue

        _filter_stage(logger, "imagery", batch_size_imagery)

        gained = max(0, _counts().get(_VALID, 0) - pool_n)
        drawn_total += drawn
        valid_gained_total += gained
        yield_est = valid_gained_total / drawn_total
        waves += 1
        logger.info(
            f"[pipeline] wave {waves}: {drawn:,} drawn -> {gained:,} VALID "
            f"(yield {gained / drawn:.1%}, cumulative {yield_est:.1%})"
        )

    # Everything else is finished: fetch TESSERA for tiles that came back
    # TESSERA_MISSING during this run (earlier ones are left to retry-tessera).
    new_missing = [
        t["tile_id"]
        for t in _get_db().list_tiles(status=_TESSERA_MISSING)
        if t["tile_id"] not in prior_missing
    ]
    if new_missing:
        logger.info(f"[pipeline] retrying TESSERA for {len(new_missing):,} tiles")
        for tid in new_missing:
            retry_tessera_missing(
                logger,
                tile_id=tid,
                local_output=local_output and not settings.use_hpc,
            )

    done = exported()
    c = _counts()
    still_missing = sum(
        1
        for t in _get_db().list_tiles(status=_TESSERA_MISSING)
        if t["tile_id"] not in prior_missing
    )
    if still_missing:
        logger.warning(
            f"[pipeline] {still_missing:,} tiles still TESSERA_MISSING after retry "
            "(run `retry-tessera` later)"
        )
    logger.info(
        f"[pipeline] finished: {done:,}/{limit:,} tiles exported"
        f" | waves={waves} drawn={drawn_total:,}"
        f" | VALID left={c.get(_VALID, 0):,}"
        f" | tessera_missing={c.get(_TESSERA_MISSING, 0):,}"
    )
    if done < limit:
        logger.warning(
            f"[pipeline] quota not met ({limit - done:,} short): "
            + ("no PENDING tiles left." if exhausted else "see the reasons above.")
        )
    return done
