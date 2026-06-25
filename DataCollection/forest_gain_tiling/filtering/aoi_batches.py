from __future__ import annotations

from typing import Any, Iterator

from enums import TileStatus
from registry.store import _get_db


def iter_aoi_pending_tiles(
    status: str = str(TileStatus.PENDING),
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """
    Stream (aoi_id, tiles) groups for tiles matching `status`.

    Each tile dict includes tile_id, xi, yi, x_min_m, y_min_m.
    A tile belonging to multiple AOIs is yielded once per AOI it belongs to —
    callers should be idempotent (e.g. skip tiles already updated to a
    non-`status` state by an earlier AOI in the same run).
    """
    db = _get_db()

    with db._conn() as conn:
        aoi_ids = [
            r["aoi_id"]
            for r in conn.execute("SELECT DISTINCT aoi_id FROM tile_aois").fetchall()
        ]

        for aoi_id in aoi_ids:
            rows = conn.execute(
                """
                SELECT t.tile_id, t.xi, t.yi, t.x_min_m, t.y_min_m, t.status
                FROM tiles t
                INNER JOIN tile_aois ta ON t.tile_id = ta.tile_id
                WHERE ta.aoi_id = ? AND t.status = ?
                """,
                (aoi_id, status),
            ).fetchall()

            if not rows:
                continue

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
            yield aoi_id, tiles


def count_aois_with_pending(status: str = str(TileStatus.PENDING)) -> int:
    db = _get_db()
    with db._conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT ta.aoi_id) as cnt
            FROM tile_aois ta
            INNER JOIN tiles t ON t.tile_id = ta.tile_id
            WHERE t.status = ?
            """,
            (status,),
        ).fetchone()
        return row["cnt"]
