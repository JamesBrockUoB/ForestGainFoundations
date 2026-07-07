"""High-level registry operations with SQLite backend."""

from __future__ import annotations

from typing import Any

from config import settings
from enums import TileStatus
from registry.database import RegistryDB

# Global database instance
_db: RegistryDB | None = None


def _get_db() -> RegistryDB:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = RegistryDB()
    return _db


def load_registry_entry(tile_id: str) -> dict[str, Any] | None:
    """Load a single registry entry."""
    return _get_db().get_tile(tile_id)


def save_tile_entry(tile: dict[str, Any]) -> bool:
    """
    Insert a tile into registry if not exists.
    Returns True if newly inserted, False if already existed.
    """
    return _get_db().insert_or_ignore(tile)


def save_tiles_batch(tiles: list[dict[str, Any]], batch_size: int = 1000000) -> int:
    """
    Insert multiple tiles efficiently in batches.
    Returns count of newly inserted tiles.
    """
    return _get_db().insert_batch(tiles, batch_size=batch_size)


def reset_tiles(
    status: str | None = None,
    period: str | None = None,
    clear_history: bool = False,
    to_status: str = str(TileStatus.PENDING),
) -> int:
    """Bulk-reset tile statuses to `to_status`. Returns rows affected.

    `period` defaults to None (all periods) at this layer — callers that
    want to scope resets to the active period (main.py's `reset` command
    does) should pass settings.period explicitly.
    """
    return _get_db().reset_tiles(
        status=status, period=period, clear_history=clear_history, to_status=to_status
    )


def update_tile(tile_id: str, **kwargs: Any) -> None:
    """Update specific fields on a tile and persist immediately."""
    if "status" in kwargs and isinstance(kwargs["status"], TileStatus):
        kwargs["status"] = str(kwargs["status"])
    _get_db().update_tile(tile_id, **kwargs)


def iter_tiles(
    status: str | None = None,
    period: str | None = None,
    batch_size: int = 1000,
) -> list[dict[str, Any]]:
    """
    Stream tiles in batches.
    Use for large-scale iteration without memory buildup.

    `period` defaults to settings.period — pass period=None explicitly to
    iterate across every period in the registry.
    """
    if period is None:
        period = settings.period

    db = _get_db()
    offset = 0
    while True:
        batch = db.list_tiles(
            status=status, period=period, limit=batch_size, offset=offset
        )
        if not batch:
            break
        for tile in batch:
            yield tile
        offset += batch_size


def get_registry_stats(period: str | None = None) -> dict[str, Any]:
    """Get aggregate statistics about the registry, optionally scoped to a period."""
    db = _get_db()
    return {
        "total": db.count_tiles(period=period),
        "by_status": db.status_counts(period=period),
        "by_biome": db.biome_counts(period=period),
        "by_region": db.region_counts(period=period),
        "rejections": db.rejection_counts(period=period),
    }


def registry_summary(period: str | None = None) -> str:
    """Generate summary statistics of registry state, optionally scoped to a period."""
    db = _get_db()

    status_counts = db.status_counts(period=period)
    biome_counts = db.biome_counts(
        status_filter=str(TileStatus.COMPLETE), period=period
    )
    region_counts = db.region_counts(
        status_filter=str(TileStatus.COMPLETE), period=period
    )
    rejection_counts = db.rejection_counts(period=period)

    scope_label = period if period is not None else "ALL PERIODS"

    lines = [
        "",
        "═" * 60,
        f"  TILE REGISTRY SUMMARY  ({scope_label})",
        "═" * 60,
        f"  Total tiles    : {db.count_tiles(period=period):>10,}",
    ]
    for s in TileStatus:
        lines.append(f"  {s.value:<14} : {status_counts.get(s.value, 0):>10,}")

    if rejection_counts:
        lines += ["", "  Rejected by reason:"]
        for r, n in sorted(rejection_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {r:<35} {n:>8,}")

    if biome_counts:
        lines += ["", "  Complete by biome:"]
        for b, n in sorted(biome_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {b:<45} {n:>7,}")

    if region_counts:
        lines += ["", "  Complete by region:"]
        for r, n in sorted(region_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {r:<30} {n:>7,}")

    lines += ["═" * 60, ""]
    return "\n".join(lines)
