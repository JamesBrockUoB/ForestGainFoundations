from __future__ import annotations

import logging
import random
from typing import Any, Iterator

from config import settings
from registry.store import _get_db
from tiling.selection import STRATA_FIELDS, load_or_compute_strata_ratios


def iter_pending_tile_batches(
    status: str, batch_size: int, period: str | None = None
) -> Iterator[list[dict[str, Any]]]:
    period = period or settings.period
    db = _get_db()

    last_xi: int | None = None
    last_yi: int | None = None

    with db._conn() as conn:
        while True:
            if last_xi is None:
                rows = conn.execute(
                    """
                    SELECT
                        tile_id,
                        xi,
                        yi,
                        x_min_m,
                        y_min_m,
                        x_max_m,
                        y_max_m,
                        min_lat,
                        max_lat
                    FROM tiles
                    WHERE status = ? AND period = ?
                    ORDER BY xi, yi
                    LIMIT ?
                    """,
                    (status, period, batch_size),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        tile_id,
                        xi,
                        yi,
                        x_min_m,
                        y_min_m,
                        x_max_m,
                        y_max_m,
                        min_lat,
                        max_lat
                    FROM tiles
                    WHERE status = ? AND period = ?
                      AND (xi > ? OR (xi = ? AND yi > ?))
                    ORDER BY xi, yi
                    LIMIT ?
                    """,
                    (
                        status,
                        period,
                        last_xi,
                        last_xi,
                        last_yi,
                        batch_size,
                    ),
                ).fetchall()

            if not rows:
                return

            tiles = [
                {
                    "tile_id": r["tile_id"],
                    "xi": r["xi"],
                    "yi": r["yi"],
                    "x_min_m": r["x_min_m"],
                    "y_min_m": r["y_min_m"],
                    "x_max_m": r["x_max_m"],
                    "y_max_m": r["y_max_m"],
                    "min_lat": r["min_lat"],
                    "max_lat": r["max_lat"],
                }
                for r in rows
            ]

            last_xi = tiles[-1]["xi"]
            last_yi = tiles[-1]["yi"]

            yield tiles


def count_pending(status: str, period: str | None = None) -> int:
    period = period or settings.period
    return _get_db().count_tiles(status=status, period=period)


def count_pending_by_stratum(
    status: str, stratify_field: str, period: str | None = None
) -> dict[str, int]:
    """
    Actual available-tile counts per stratum for `status` — delegates to
    RegistryDB's existing biome_counts/region_counts/country_counts
    (status_filter=..., period=...), each a single GROUP BY query.
    """
    if stratify_field not in STRATA_FIELDS:
        raise ValueError(f"stratify_field must be one of {STRATA_FIELDS}")

    period = period or settings.period
    db = _get_db()

    if stratify_field == "biome":
        return db.biome_counts(status_filter=status, period=period)
    elif stratify_field == "region":
        return db.region_counts(status_filter=status, period=period)
    else:  # "country"
        return db.country_counts(status_filter=status, period=period)


def iter_stratified_pending_tile_batches(
    status: str,
    batch_size: int,
    period: str,
    stratify_field: str,
    tile_limit: int,
    mode: str = "prop",
    logger: logging.Logger | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """
    Like iter_pending_tile_batches, but draws up to `tile_limit` tiles
    total, allocated across strata (biome/region/country) rather than
    however xi/yi ordering happens to hand them out.

    mode='prop': quota per stratum comes from the cached population
      ratios (load_or_compute_strata_ratios) — matches the true
      biome/region/country distribution. Correct for building toward
      a representative full dataset.
    mode='equal': quota is tile_limit // n_strata for every stratum
      present at this `status`, ignoring population ratios entirely.
      Use this for small proto-datasets, where you want every stratum
      exercised rather than proportionally represented — at n=100-200,
      proportional allocation rounds thin strata down to 0-1 tiles and
      they effectively vanish from the run.

    Cost note: one query per stratum value, each with ORDER BY RANDOM()
    — fine at proto-dataset scale (thousands of tiles). ORDER BY RANDOM()
    in SQLite is O(n log n) over each stratum's matching rows; swap for
    reservoir sampling over a cursor if any stratum's pool reaches the
    millions.
    """
    if stratify_field not in STRATA_FIELDS:
        raise ValueError(f"stratify_field must be one of {STRATA_FIELDS}")
    if mode not in ("prop", "equal"):
        raise ValueError(f"mode must be 'prop' or 'equal', got {mode!r}")

    available = count_pending_by_stratum(status, stratify_field, period)

    if mode == "equal":
        strata = list(available.keys())
        if not strata:
            if logger:
                logger.warning(
                    f"No tiles at status={status} for period={period} — "
                    f"nothing to stratify."
                )
            return
        quotas = {s: tile_limit // len(strata) for s in strata}
    else:
        ratios = load_or_compute_strata_ratios(period, logger)[stratify_field]
        quotas = {s: round(tile_limit * r) for s, r in ratios.items()}

    db = _get_db()
    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, tuple[int, int]] = {}

    with db._conn() as conn:
        for stratum, quota in quotas.items():
            if quota <= 0:
                continue

            have = available.get(stratum, 0)
            if have < quota:
                shortfalls[stratum] = (quota, have)

            rows = conn.execute(
                f"""
                SELECT
                    tile_id, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m,
                    min_lat, max_lat
                FROM tiles
                WHERE status = ? AND period = ? AND {stratify_field} = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (status, period, stratum, quota),
            ).fetchall()

            selected.extend(
                {
                    "tile_id": r["tile_id"],
                    "xi": r["xi"],
                    "yi": r["yi"],
                    "x_min_m": r["x_min_m"],
                    "y_min_m": r["y_min_m"],
                    "x_max_m": r["x_max_m"],
                    "y_max_m": r["y_max_m"],
                    "min_lat": r["min_lat"],
                    "max_lat": r["max_lat"],
                }
                for r in rows
            )

    if logger:
        if shortfalls:
            logger.warning(
                f"Stratified filter ({stratify_field}, mode={mode}): "
                f"{len(shortfalls)} strata short of quota — "
                + ", ".join(
                    f"{s}: wanted {q}, have {h}" for s, (q, h) in shortfalls.items()
                )
            )
        logger.info(
            f"Stratified filter ({stratify_field}, mode={mode}): "
            f"selected {len(selected):,} of {tile_limit:,} requested tiles"
        )

    random.shuffle(selected)  # interleave strata across batches, not block by stratum

    for i in range(0, len(selected), batch_size):
        yield selected[i : i + batch_size]
