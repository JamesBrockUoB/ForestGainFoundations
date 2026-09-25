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
    clear_history: bool = False,
    to_status: str = str(TileStatus.PENDING),
) -> int:
    """Bulk-reset tile statuses to `to_status`. Returns rows affected."""
    return _get_db().reset_tiles(
        status=status, clear_history=clear_history, to_status=to_status
    )


def update_tile(tile_id: str, **kwargs: Any) -> None:
    """Update specific fields on a tile and persist immediately."""
    if "status" in kwargs and isinstance(kwargs["status"], TileStatus):
        kwargs["status"] = str(kwargs["status"])
    _get_db().update_tile(tile_id, **kwargs)


def iter_tiles(
    status: str | None = None,
    batch_size: int = 1000,
) -> list[dict[str, Any]]:
    """
    Stream tiles in batches.
    Use for large-scale iteration without memory buildup.
    """

    db = _get_db()
    offset = 0
    while True:
        batch = db.list_tiles(status=status, limit=batch_size, offset=offset)
        if not batch:
            break
        for tile in batch:
            yield tile
        offset += batch_size


def get_registry_stats() -> dict[str, Any]:
    """Get aggregate statistics about the registry"""
    db = _get_db()
    return {
        "total": db.count_tiles(),
        "by_status": db.status_counts(),
        "by_biome": db.biome_counts(),
        "by_region": db.region_counts(),
        "by_country": db.country_counts(),
        "rejections": db.rejection_counts(),
    }


def registry_summary(
    verbose: int = 0,
) -> str:
    """Generate registry summary, optionally including biome/region/country breakdowns."""
    db = _get_db()

    status_counts = db.status_counts()
    total = db.count_tiles()

    lines = [
        "",
        "═" * 60,
        f"  REGISTRY SUMMARY",
        "═" * 60,
        f"  Total tiles : {total:>10,}",
        "",
        "  By status:",
    ]

    for status, cnt in sorted(status_counts.items(), key=lambda x: x[0]):
        lines.append(f"    {status:<20} {cnt:>8,}")

    if verbose:
        biome_counts = db.biome_counts()
        region_counts = db.region_counts()
        country_counts = db.country_counts()

        lines += ["", "  By biome:"]
        for biome, count in sorted(biome_counts.items(), key=lambda x: -x[1]):
            lines.append(
                f"    {biome:<45} {count:>8,}  "
                f"({100 * count / max(total, 1):5.1f}%)"
            )

        lines += ["", "  By region:"]
        for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
            lines.append(
                f"    {region:<30} {count:>8,}  "
                f"({100 * count / max(total, 1):5.1f}%)"
            )

        lines += ["", "  By country:"]
        for country, count in sorted(country_counts.items(), key=lambda x: -x[1]):
            lines.append(
                f"    {country:<30} {count:>8,}  "
                f"({100 * count / max(total, 1):5.1f}%)"
            )

    lines += ["═" * 60, ""]

    return "\n".join(lines)
