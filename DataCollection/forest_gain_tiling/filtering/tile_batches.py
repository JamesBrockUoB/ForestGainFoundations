from __future__ import annotations

from typing import Any, Iterator

from config import settings
from registry.store import _get_db


def iter_pending_tile_batches(
    status: str, batch_size: int, period: str | None = None
) -> Iterator[list[dict[str, Any]]]:
    """
    Stream tiles matching `status` (and `period`) in fixed-size batches,
    ordered by grid position (xi, yi) so each batch's bounding-box extent
    stays spatially tight rather than sprawling.

    `period` defaults to settings.period. This filter is mandatory, not
    just a convenience: filter_batch_cheap/filter_batch_imagery compute one
    shared gain layer / imagery-availability image for the whole batch,
    keyed to settings.period's year range. Two tiles can legitimately share
    the same (xi, yi) grid cell across different periods (tile_id now
    encodes period — see tiling/grid.py), so without this filter a single
    batch could silently mix p1 and p2 tiles at the same row/col and apply
    the wrong period's gain/imagery calculation to one of them.

    Uses keyset pagination on (xi, yi) instead of OFFSET, so it's correct
    even as statuses change mid-run: each call asks "give me the next
    `batch_size` tiles still in `status`/`period`, with grid position past
    the last one I saw." Tiles already moved to another status are simply
    excluded by the WHERE clause. Resuming after an interruption just
    restarts the same query from the beginning (last_xi/last_yi reset) —
    since already-processed tiles are no longer in `status`, it naturally
    picks up only what's left.
    """
    period = period or settings.period
    db = _get_db()
    last_xi: int | None = None
    last_yi: int | None = None

    with db._conn() as conn:
        while True:
            if last_xi is None:
                rows = conn.execute(
                    """
                    SELECT tile_id, xi, yi, x_min_m, y_min_m
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
                    SELECT tile_id, xi, yi, x_min_m, y_min_m
                    FROM tiles
                    WHERE status = ? AND period = ?
                      AND (xi > ? OR (xi = ? AND yi > ?))
                    ORDER BY xi, yi
                    LIMIT ?
                    """,
                    (status, period, last_xi, last_xi, last_yi, batch_size),
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
                }
                for r in rows
            ]
            last_xi, last_yi = tiles[-1]["xi"], tiles[-1]["yi"]
            yield tiles


def count_pending(status: str, period: str | None = None) -> int:
    period = period or settings.period
    db = _get_db()
    with db._conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM tiles WHERE status = ? AND period = ?",
            (status, period),
        ).fetchone()
        return row["cnt"]
