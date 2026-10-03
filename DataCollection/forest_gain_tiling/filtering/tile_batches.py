from __future__ import annotations

import logging
import math
import random
from typing import Any, Iterator

from registry.store import _get_db
from tiling.selection import STRATA_FIELDS, load_or_compute_strata_ratios

_TILE_COLS = (
    "tile_id, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m, "
    "min_lon, min_lat, max_lon, max_lat"
)


def _row_to_tile(r) -> dict[str, Any]:
    return {
        k: r[k]
        for k in (
            "tile_id",
            "xi",
            "yi",
            "x_min_m",
            "y_min_m",
            "x_max_m",
            "y_max_m",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
        )
    }


def iter_spatial_pending_tile_batches(
    status: str,
    batch_size: int,
    block_size: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """
    Spatially compact batches for full (unstratified) runs.

    Tiles are grouped into block_size x block_size grid-cell blocks
    (default ~ sqrt(batch_size), so one block ~ one batch when dense),
    blocks are visited in serpentine order, and batches are filled
    across consecutive blocks. Each batch is therefore one compact
    region instead of a long xi-column.
    """
    db = _get_db()
    B = block_size or max(1, math.isqrt(batch_size - 1) + 1)

    with db._conn() as conn:
        origin = conn.execute(
            "SELECT MIN(xi) AS x0, MIN(yi) AS y0 FROM tiles WHERE status = ?",
            (status,),
        ).fetchone()
        if origin["x0"] is None:
            return
        x0, y0 = origin["x0"], origin["y0"]

        # offset by the minimum so integer division is a true floor
        blocks = conn.execute(
            """
            SELECT DISTINCT (xi - ?) / ? AS bx, (yi - ?) / ? AS by
            FROM tiles WHERE status = ?
            """,
            (x0, B, y0, B, status),
        ).fetchall()
        order = sorted(
            ((r["bx"], r["by"]) for r in blocks),
            key=lambda b: (b[0], b[1] if b[0] % 2 == 0 else -b[1]),
        )

        buffer: list[dict[str, Any]] = []
        for bx, by in order:
            rows = conn.execute(
                f"""
                SELECT {_TILE_COLS} FROM tiles
                WHERE status = ?
                  AND xi >= ? AND xi < ?
                  AND yi >= ? AND yi < ?
                ORDER BY yi, xi
                """,
                (
                    status,
                    x0 + bx * B,
                    x0 + (bx + 1) * B,
                    y0 + by * B,
                    y0 + (by + 1) * B,
                ),
            ).fetchall()
            buffer.extend(_row_to_tile(r) for r in rows)

            while len(buffer) >= batch_size:
                yield buffer[:batch_size]
                buffer = buffer[batch_size:]

        if buffer:
            yield buffer


def iter_pending_tile_batches(
    status: str,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
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
                        min_lon,
                        min_lat,
                        max_lon,
                        max_lat
                    FROM tiles
                    WHERE status = ?
                    ORDER BY xi, yi
                    LIMIT ?
                    """,
                    (status, batch_size),
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
                        min_lon,
                        min_lat,
                        max_lon,
                        max_lat
                    FROM tiles
                    WHERE status = ?
                      AND (xi > ? OR (xi = ? AND yi > ?))
                    ORDER BY xi, yi
                    LIMIT ?
                    """,
                    (
                        status,
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
                    "min_lon": r["min_lon"],
                    "min_lat": r["min_lat"],
                    "max_lon": r["max_lon"],
                    "max_lat": r["max_lat"],
                }
                for r in rows
            ]

            last_xi = tiles[-1]["xi"]
            last_yi = tiles[-1]["yi"]

            yield tiles


def count_pending(status: str) -> int:
    return _get_db().count_tiles(status=status)


def count_pending_by_stratum(
    status: str,
    stratify_field: str,
) -> dict[str, int]:
    """
    Actual available-tile counts per stratum for `status` — delegates to
    RegistryDB's existing biome_counts/region_counts/country_counts
    (status_filter=...), each a single GROUP BY query.
    """
    if stratify_field not in STRATA_FIELDS:
        raise ValueError(f"stratify_field must be one of {STRATA_FIELDS}")

    db = _get_db()

    if stratify_field == "biome":
        return db.biome_counts(status_filter=status)
    elif stratify_field == "region":
        return db.region_counts(status_filter=status)
    else:  # "country"
        return db.country_counts(status_filter=status)


def iter_stratified_pending_tile_batches(
    status: str,
    batch_size: int,
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

    available = count_pending_by_stratum(status, stratify_field)

    if mode == "equal":
        strata = list(available.keys())
        if not strata:
            if logger:
                logger.warning(f"No tiles at status={status} — nothing to stratify.")
            return
        quotas = {s: tile_limit // len(strata) for s in strata}
    else:
        ratios = load_or_compute_strata_ratios(logger)[stratify_field]
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
                    min_lon, min_lat, max_lon, max_lat
                FROM tiles
                WHERE status = ? AND {stratify_field} = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (status, stratum, quota),
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
                    "min_lon": r["min_lon"],
                    "min_lat": r["min_lat"],
                    "max_lon": r["max_lon"],
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
