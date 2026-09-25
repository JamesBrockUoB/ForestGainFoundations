from __future__ import annotations

import logging
import time

from config import settings
from enums import TileStatus
from filtering.raster_stats import (
    check_tessera_coverage,
    fetch_cheap_stats,
    fetch_imagery_stats,
)
from registry.store import update_tile


def evaluate_cheap_stats(
    stats: dict[str, float],
    logger: logging.Logger | None = None,
) -> tuple[str, str | None]:

    gain_pct = stats["gain_frac"] * 100.0
    ndvi_trend = stats["ndvi_trend"]

    if gain_pct == 0.0:
        return str(TileStatus.REJECTED), "no_gain"

    if gain_pct < settings.gain_pct_min:
        return str(TileStatus.REJECTED), "low_gain_pct"

    if ndvi_trend <= settings.ndvi_trend_min:
        return str(TileStatus.REJECTED), "low_viability"

    return str(TileStatus.CHEAP_VALID), None


def evaluate_imagery_stats(
    stats: dict[str, float], logger: logging.Logger | None = None
) -> tuple[str, str | None]:
    """
    Requires every per-year S1 and S2 band to clear
    settings.imagery_min_valid_frac.
    """
    low = {b: v for b, v in stats.items() if v < settings.imagery_min_valid_frac}

    if low:
        if logger:
            logger.debug(f"low_imagery_coverage: {low}")
        return str(TileStatus.REJECTED), "low_imagery_coverage"

    return str(TileStatus.VALID), None


def filter_batch_cheap(
    tiles: list[dict],
    ds,
    logger: logging.Logger,
    batch_label: str = "",
) -> dict[str, int]:
    logger.debug(f"{batch_label} cheap: {len(tiles)} tiles")

    t0 = time.time()

    try:
        covered_tiles, missing_tessera = check_tessera_coverage(
            tiles,
            logger=logger,
        )

        logger.debug(
            f"  {batch_label} TESSERA coverage check: "
            f"{time.time()-t0:.1f}s | "
            f"covered={len(covered_tiles)} | "
            f"missing={len(missing_tessera)}"
        )

    except Exception as exc:
        logger.error(
            f"  {batch_label} TESSERA coverage check failed after "
            f"{time.time()-t0:.1f}s — {exc}"
        )

        for t in tiles:
            update_tile(
                t["tile_id"],
                status=TileStatus.FAILED,
                error=f"TESSERA coverage check failed: {exc}",
            )

        return {"failed": len(tiles)}

    counts = {
        "cheap_valid": 0,
        "rejected": 0,
        "tessera_no_coverage": 0,
    }

    for tile_id, missing_years in missing_tessera.items():
        update_tile(
            tile_id,
            status=TileStatus.REJECTED,
            rejection_reason="missing_tessera_coverage",
        )
        counts["rejected"] += 1
        counts["tessera_no_coverage"] += 1

        logger.debug(
            f"  {batch_label} rejected {tile_id}: "
            f"TESSERA missing years={missing_years}"
        )

    if not covered_tiles:
        logger.debug(f"  {batch_label}: all tiles rejected by TESSERA coverage")
        return counts

    t0 = time.time()

    try:
        stats_by_tile = fetch_cheap_stats(covered_tiles, ds)
        logger.debug(f"  {batch_label} cheap fetch: {time.time()-t0:.1f}s")

    except Exception as exc:
        logger.error(
            f"  {batch_label} cheap fetch failed after "
            f"{time.time()-t0:.1f}s — {exc}"
        )

        for t in covered_tiles:
            update_tile(
                t["tile_id"],
                status=TileStatus.FAILED,
                error=str(exc),
            )

        return {"failed": len(covered_tiles)}

    for t in covered_tiles:
        tile_id = t["tile_id"]

        stats = stats_by_tile.get(tile_id)

        if stats is None:
            update_tile(
                tile_id,
                status=TileStatus.FAILED,
                error="missing stats result",
            )
            counts["rejected"] += 1
            continue

        status, reason = evaluate_cheap_stats(
            stats,
            logger=logger,
        )

        if status == str(TileStatus.REJECTED):
            update_tile(
                tile_id,
                status=TileStatus.REJECTED,
                rejection_reason=reason,
            )
            counts["rejected"] += 1

        else:
            update_tile(
                tile_id,
                status=TileStatus.CHEAP_VALID,
            )
            counts["cheap_valid"] += 1

    logger.debug(
        f"  {batch_label} cheap eval: "
        f"cheap_valid={counts['cheap_valid']} "
        f"rejected={counts['rejected']} "
        f"tessera_no_coverage={counts['tessera_no_coverage']}"
    )

    return counts


def filter_batch_imagery(
    tiles: list[dict],
    logger: logging.Logger,
    batch_label: str = "",
) -> dict[str, int]:
    logger.debug(f"{batch_label} imagery: {len(tiles)} tiles")

    t0 = time.time()

    try:
        stats_by_tile = fetch_imagery_stats(tiles)

        logger.debug(f"  {batch_label} imagery fetch: {time.time()-t0:.1f}s")

    except Exception as exc:
        logger.error(
            f"  {batch_label} imagery fetch failed after "
            f"{time.time()-t0:.1f}s — {exc}"
        )

        for t in tiles:
            update_tile(
                t["tile_id"],
                status=TileStatus.FAILED,
                error=str(exc),
            )

        return {"failed": len(tiles)}

    counts = {
        "valid": 0,
        "rejected": 0,
    }

    for t in tiles:
        tile_id = t["tile_id"]

        stats = stats_by_tile.get(tile_id)

        if stats is None:
            update_tile(
                tile_id,
                status=TileStatus.FAILED,
                error="missing imagery stats result",
            )
            counts["rejected"] += 1
            continue

        status, reason = evaluate_imagery_stats(
            stats,
            logger=logger,
        )

        if status == str(TileStatus.REJECTED):
            update_tile(
                tile_id,
                status=TileStatus.REJECTED,
                rejection_reason=reason,
            )
            counts["rejected"] += 1

        else:
            update_tile(
                tile_id,
                status=TileStatus.VALID,
            )
            counts["valid"] += 1

    logger.debug(
        f"  {batch_label} imagery eval: "
        f"valid={counts['valid']} "
        f"rejected={counts['rejected']}"
    )

    return counts
