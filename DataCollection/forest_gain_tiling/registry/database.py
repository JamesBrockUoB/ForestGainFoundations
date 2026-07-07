"""SQLite database layer for tile registry storage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import settings
from enums import TileStatus


class RegistryDB:
    """SQLite database wrapper for tile registry with streaming/pagination support.

    A single database holds tiles for every period; the `period` column
    (e.g. "p1", "p2") distinguishes them. Most read/write methods accept an
    optional `period` filter — pass it explicitly (or rely on the caller
    already having filtered) whenever an operation should not mix periods,
    since gain/imagery validity is period-specific even for a tile at the
    same grid location.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.registry_db_path
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialise database schema if not exists."""
        with self._conn() as conn:
            # Main tiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_id TEXT PRIMARY KEY,
                    period TEXT NOT NULL,
                    xi INTEGER NOT NULL,
                    yi INTEGER NOT NULL,
                    x_min_m REAL NOT NULL,
                    y_min_m REAL NOT NULL,
                    x_max_m REAL NOT NULL,
                    y_max_m REAL NOT NULL,
                    min_lon REAL NOT NULL,
                    min_lat REAL NOT NULL,
                    max_lon REAL NOT NULL,
                    max_lat REAL NOT NULL,
                    biome TEXT NOT NULL,
                    region TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    gee_task_id TEXT,
                    submitted_at TEXT,
                    completed_at TEXT,
                    rejection_reason TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            # Junction table for AOI-tile relationships
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tile_aois (
                    tile_id TEXT NOT NULL,
                    aoi_id TEXT NOT NULL,
                    PRIMARY KEY (tile_id, aoi_id),
                    FOREIGN KEY (tile_id) REFERENCES tiles(tile_id)
                )
                """)

            # Indexes on tiles table
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON tiles(status)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_period ON tiles(period)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_period_status ON tiles(period, status)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_biome ON tiles(biome)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_region ON tiles(region)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated ON tiles(updated_at DESC)
                """)

            # Indexes on tile_aois table
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tile_aois_aoi ON tile_aois(aoi_id)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tile_aois_tile ON tile_aois(tile_id)
                """)

            conn.commit()

    def insert_or_ignore(self, tile: dict[str, Any]) -> bool:
        """
        Insert a tile, returning True if inserted, False if already exists.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO tiles (
                    tile_id, period, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m,
                    min_lon, min_lat, max_lon, max_lat, biome, region,
                    status, gee_task_id, submitted_at, completed_at,
                    rejection_reason, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile["tile_id"],
                    tile.get("period", settings.period),
                    tile["xi"],
                    tile["yi"],
                    tile["x_min_m"],
                    tile["y_min_m"],
                    tile["x_max_m"],
                    tile["y_max_m"],
                    tile["min_lon"],
                    tile["min_lat"],
                    tile["max_lon"],
                    tile["max_lat"],
                    tile["biome"],
                    tile["region"],
                    str(tile.get("status", TileStatus.PENDING)),
                    tile.get("gee_task_id"),
                    tile.get("submitted_at"),
                    tile.get("completed_at"),
                    tile.get("rejection_reason"),
                    tile.get("error"),
                    now,
                    now,
                ),
            )
            inserted = cursor.rowcount > 0

            # Insert AOI relationships if newly inserted
            if inserted:
                for aoi_id in tile.get("aoi_ids", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO tile_aois (tile_id, aoi_id) VALUES (?, ?)",
                        (tile["tile_id"], aoi_id),
                    )

            conn.commit()
            return inserted

    def insert_batch(
        self, tiles: list[dict[str, Any]], batch_size: int = 1000000
    ) -> int:
        """Insert multiple tiles and their AOI relationships in batches."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0

        with self._conn() as conn:
            for i in range(0, len(tiles), batch_size):
                batch = tiles[i : i + batch_size]

                params_list = [
                    (
                        tile["tile_id"],
                        tile.get("period", settings.period),
                        tile["xi"],
                        tile["yi"],
                        tile["x_min_m"],
                        tile["y_min_m"],
                        tile["x_max_m"],
                        tile["y_max_m"],
                        tile["min_lon"],
                        tile["min_lat"],
                        tile["max_lon"],
                        tile["max_lat"],
                        tile["biome"],
                        tile["region"],
                        str(tile.get("status", TileStatus.PENDING)),
                        tile.get("gee_task_id"),
                        tile.get("submitted_at"),
                        tile.get("completed_at"),
                        tile.get("rejection_reason"),
                        tile.get("error"),
                        now,
                        now,
                    )
                    for tile in batch
                ]

                cursor = conn.executemany(
                    """
                    INSERT OR IGNORE INTO tiles (
                        tile_id, period, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m,
                        min_lon, min_lat, max_lon, max_lat, biome, region,
                        status, gee_task_id, submitted_at, completed_at,
                        rejection_reason, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params_list,
                )
                inserted += cursor.rowcount

                # Batch all AOI relationships
                aoi_params = []
                for tile in batch:
                    for aoi_id in tile.get("aoi_ids", []):
                        aoi_params.append((tile["tile_id"], aoi_id))

                if aoi_params:
                    conn.executemany(
                        "INSERT OR IGNORE INTO tile_aois (tile_id, aoi_id) VALUES (?, ?)",
                        aoi_params,
                    )

                conn.commit()

        return inserted

    def get_tile(self, tile_id: str) -> dict[str, Any] | None:
        """Fetch a single tile by ID (with AOI IDs)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tiles WHERE tile_id = ?", (tile_id,)
            ).fetchone()
            if not row:
                return None

            # Get AOI IDs from junction table
            aoi_rows = conn.execute(
                "SELECT aoi_id FROM tile_aois WHERE tile_id = ?", (tile_id,)
            ).fetchall()
            aoi_ids = [r["aoi_id"] for r in aoi_rows]

            return self._row_to_dict(row, aoi_ids)

    def update_tile(self, tile_id: str, **kwargs: Any) -> None:
        """Update specific fields on a tile."""
        now = datetime.now(timezone.utc).isoformat()
        kwargs["updated_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [tile_id]

        with self._conn() as conn:
            conn.execute(f"UPDATE tiles SET {set_clause} WHERE tile_id = ?", values)
            conn.commit()

    def list_tiles(
        self,
        status: str | None = None,
        period: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Stream tiles with optional status/period filters, with pagination.
        Use limit/offset for memory-efficient iteration over large datasets.
        """
        query = "SELECT * FROM tiles"
        clauses = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY updated_at DESC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                # Get AOI IDs from junction table for each tile
                aoi_rows = conn.execute(
                    "SELECT aoi_id FROM tile_aois WHERE tile_id = ?", (row["tile_id"],)
                ).fetchall()
                aoi_ids = [r["aoi_id"] for r in aoi_rows]
                result.append(self._row_to_dict(row, aoi_ids))
            return result

    def count_tiles(self, status: str | None = None, period: str | None = None) -> int:
        """Count tiles, optionally filtered by status and/or period."""
        query = "SELECT COUNT(*) as cnt FROM tiles"
        clauses = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        with self._conn() as conn:
            result = conn.execute(query, params).fetchone()
            return result["cnt"]

    def status_counts(self, period: str | None = None) -> dict[str, int]:
        """Get counts by status, optionally scoped to one period."""
        query = "SELECT status, COUNT(*) as cnt FROM tiles"
        params: list[Any] = []
        if period is not None:
            query += " WHERE period = ?"
            params.append(period)
        query += " GROUP BY status"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["status"]: row["cnt"] for row in rows}

    def biome_counts(
        self, status_filter: str | None = None, period: str | None = None
    ) -> dict[str, int]:
        """Get counts by biome, optionally filtered by status and/or period."""
        query = "SELECT biome, COUNT(*) as cnt FROM tiles"
        clauses = []
        params: list[Any] = []

        if status_filter is not None:
            clauses.append("status = ?")
            params.append(status_filter)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " GROUP BY biome ORDER BY cnt DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["biome"]: row["cnt"] for row in rows}

    def region_counts(
        self, status_filter: str | None = None, period: str | None = None
    ) -> dict[str, int]:
        """Get counts by region, optionally filtered by status and/or period."""
        query = "SELECT region, COUNT(*) as cnt FROM tiles"
        clauses = []
        params: list[Any] = []

        if status_filter is not None:
            clauses.append("status = ?")
            params.append(status_filter)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " GROUP BY region ORDER BY cnt DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["region"]: row["cnt"] for row in rows}

    def rejection_counts(self, period: str | None = None) -> dict[str, int]:
        """Get rejection reason counts, optionally scoped to one period."""
        query = (
            "SELECT rejection_reason, COUNT(*) as cnt FROM tiles "
            "WHERE status = ? AND rejection_reason IS NOT NULL"
        )
        params: list[Any] = [str(TileStatus.REJECTED)]
        if period is not None:
            query += " AND period = ?"
            params.append(period)
        query += " GROUP BY rejection_reason ORDER BY cnt DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["rejection_reason"]: row["cnt"] for row in rows}

    def reset_tiles(
        self,
        status: str | None = None,
        period: str | None = None,
        clear_history: bool = False,
        to_status: str = str(TileStatus.PENDING),
    ) -> int:
        """
        Reset tile statuses back to `to_status` in a single bulk UPDATE.

        status: if given, only reset tiles currently in this status.
            If None, reset every tile not already in `to_status` (within
            `period`, if given).
        period: if given, only reset tiles in this period. Strongly
            recommended — resetting across periods indiscriminately will
            queue tiles from a period you didn't mean to touch back
            through the filter pipeline.
        to_status: the status to reset tiles into (default 'pending').
            Use this to resume a specific pipeline stage — e.g. resetting
            FAILED tiles that died during the imagery stage back to
            'cheap_valid' instead of all the way to 'pending', so stage 1
            isn't redone.
        clear_history: if True, also null out gee_task_id, submitted_at,
            completed_at, rejection_reason, and error — a full wipe back
            to a blank row. If False (default), only the status flag is
            flipped and the old error/rejection_reason remain visible for
            debugging.

        Returns the number of rows affected.
        """
        now = datetime.now(timezone.utc).isoformat()

        set_clause = "status = ?, updated_at = ?"
        params: list[Any] = [to_status, now]

        if clear_history:
            set_clause += (
                ", gee_task_id = NULL, submitted_at = NULL, "
                "completed_at = NULL, rejection_reason = NULL, error = NULL"
            )

        query = f"UPDATE tiles SET {set_clause}"
        clauses = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        else:
            clauses.append("status != ?")
            params.append(to_status)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        query += " WHERE " + " AND ".join(clauses)

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def clear_all(self) -> None:
        """Clear all tiles and AOI relationships (use with caution)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM tile_aois")
            conn.execute("DELETE FROM tiles")
            conn.commit()

    def _row_to_dict(
        self, row: sqlite3.Row, aoi_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Convert a database row to a dictionary with proper types."""
        if aoi_ids is None:
            aoi_ids = []

        return {
            "tile_id": row["tile_id"],
            "period": row["period"],
            "xi": row["xi"],
            "yi": row["yi"],
            "x_min_m": row["x_min_m"],
            "y_min_m": row["y_min_m"],
            "x_max_m": row["x_max_m"],
            "y_max_m": row["y_max_m"],
            "min_lon": row["min_lon"],
            "min_lat": row["min_lat"],
            "max_lon": row["max_lon"],
            "max_lat": row["max_lat"],
            "biome": row["biome"],
            "region": row["region"],
            "aoi_ids": aoi_ids,
            "status": row["status"],
            "gee_task_id": row["gee_task_id"],
            "submitted_at": row["submitted_at"],
            "completed_at": row["completed_at"],
            "rejection_reason": row["rejection_reason"],
            "error": row["error"],
        }
