from __future__ import annotations

from typing import Any, Iterator

from config import settings
from registry.store import _get_db


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
                        y_max_m
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
                        y_max_m
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
                }
                for r in rows
            ]

            last_xi = tiles[-1]["xi"]
            last_yi = tiles[-1]["yi"]

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
