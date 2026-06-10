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
    """SQLite database wrapper for tile registry with streaming/pagination support."""

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
        """Initialize database schema if not exists."""
        with self._conn() as conn:
            # Main tiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_id TEXT PRIMARY KEY,
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

            # Junction table for AOI-tile relationships (indexed for fast lookups)
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
                CREATE INDEX IF NOT EXISTS idx_biome ON tiles(biome)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_region ON tiles(region)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated ON tiles(updated_at DESC)
                """)

            # Indexes on tile_aois table (fast audit lookups)
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
                    tile_id, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m,
                    min_lon, min_lat, max_lon, max_lat, biome, region,
                    status, gee_task_id, submitted_at, completed_at,
                    rejection_reason, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile["tile_id"],
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
        self, tiles: list[dict[str, Any]], batch_size: int = 100000
    ) -> int:
        """
        Insert multiple tiles in a single transaction.
        Also populates the tile_aois junction table.
        Returns count of newly inserted tiles.
        """
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0

        with self._conn() as conn:
            for i in range(0, len(tiles), batch_size):
                batch = tiles[i : i + batch_size]

                # Insert tiles
                params_list = [
                    (
                        tile["tile_id"],
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
                        tile_id, xi, yi, x_min_m, y_min_m, x_max_m, y_max_m,
                        min_lon, min_lat, max_lon, max_lat, biome, region,
                        status, gee_task_id, submitted_at, completed_at,
                        rejection_reason, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params_list,
                )
                inserted += cursor.rowcount

                # Insert AOI relationships (junction table)
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

    def get_aoi_tile_counts(self, aoi_id: str) -> dict[str, int]:
        """
        Get status counts for a single AOI (used for audit).
        Fast indexed query - uses junction table instead of string search.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT t.status, COUNT(*) as cnt
                FROM tiles t
                INNER JOIN tile_aois ta ON t.tile_id = ta.tile_id
                WHERE ta.aoi_id = ?
                GROUP BY t.status
                """,
                (aoi_id,),
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}

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
        self, status: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Stream tiles with optional status filter, with pagination.
        Use limit/offset for memory-efficient iteration over large datasets.
        """
        query = "SELECT * FROM tiles"
        params = []

        if status is not None:
            query += " WHERE status = ?"
            params.append(status)

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

    def count_tiles(self, status: str | None = None) -> int:
        """Count tiles, optionally filtered by status."""
        query = "SELECT COUNT(*) as cnt FROM tiles"
        params = []

        if status is not None:
            query += " WHERE status = ?"
            params.append(status)

        with self._conn() as conn:
            result = conn.execute(query, params).fetchone()
            return result["cnt"]

    def status_counts(self) -> dict[str, int]:
        """Get counts by status."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM tiles GROUP BY status"
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}

    def biome_counts(self, status_filter: str | None = None) -> dict[str, int]:
        """Get counts by biome, optionally filtered by status."""
        query = "SELECT biome, COUNT(*) as cnt FROM tiles"
        params = []

        if status_filter is not None:
            query += " WHERE status = ?"
            params.append(status_filter)

        query += " GROUP BY biome ORDER BY cnt DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["biome"]: row["cnt"] for row in rows}

    def region_counts(self, status_filter: str | None = None) -> dict[str, int]:
        """Get counts by region, optionally filtered by status."""
        query = "SELECT region, COUNT(*) as cnt FROM tiles"
        params = []

        if status_filter is not None:
            query += " WHERE status = ?"
            params.append(status_filter)

        query += " GROUP BY region ORDER BY cnt DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row["region"]: row["cnt"] for row in rows}

    def rejection_counts(self) -> dict[str, int]:
        """Get rejection reason counts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT rejection_reason, COUNT(*) as cnt FROM tiles "
                "WHERE status = ? AND rejection_reason IS NOT NULL "
                "GROUP BY rejection_reason ORDER BY cnt DESC",
                (str(TileStatus.REJECTED),),
            ).fetchall()
            return {row["rejection_reason"]: row["cnt"] for row in rows}

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
