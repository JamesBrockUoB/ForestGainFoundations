"""High-level registry operations with SQLite backend."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from config import settings
from enums import TileStatus
from models import AoiAuditEntry
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


def update_tile(tile_id: str, **kwargs: Any) -> None:
    """Update specific fields on a tile and persist immediately."""
    if "status" in kwargs and isinstance(kwargs["status"], TileStatus):
        kwargs["status"] = str(kwargs["status"])
    _get_db().update_tile(tile_id, **kwargs)


def iter_tiles(
    status: str | None = None, batch_size: int = 1000
) -> list[dict[str, Any]]:
    """
    Stream tiles in batches (never materializes entire grid).
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
    """Get aggregate statistics about the registry."""
    db = _get_db()
    return {
        "total": db.count_tiles(),
        "by_status": db.status_counts(),
        "by_biome": db.biome_counts(),
        "by_region": db.region_counts(),
        "rejections": db.rejection_counts(),
    }


def build_aoi_audit(
    valid_aois: list[dict],
) -> dict[str, dict[str, Any]]:
    """Build AOI coverage audit by querying database incrementally."""
    db = _get_db()
    aoi_tile_counts: dict[str, Counter] = defaultdict(Counter)

    # Stream through all tiles without materializing entire dataset
    for entry in iter_tiles():
        for aoi_id in entry.get("aoi_ids", []):
            aoi_tile_counts[aoi_id][entry["status"]] += 1

    result: dict[str, dict] = {}
    for aoi in valid_aois:
        aoi_id = aoi["id"]
        counts = aoi_tile_counts.get(aoi_id, Counter())
        complete = counts.get(str(TileStatus.COMPLETE), 0)
        result[aoi_id] = AoiAuditEntry(
            biome=aoi.get("biome_name", "Unknown"),
            region=aoi.get("region", "Unknown"),
            tile_counts=dict(counts),
            total_tiles=sum(counts.values()),
            complete_tiles=complete,
            has_coverage=complete > 0,
        ).__dict__

    return result


def save_aoi_audit(audit: dict) -> None:
    """Save AOI audit to JSON file (for external analysis)."""
    import json

    tmp = settings.aoi_audit_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2)
    tmp.replace(settings.aoi_audit_path)


def registry_summary() -> str:
    """Generate summary statistics of registry state."""
    db = _get_db()

    status_counts = db.status_counts()
    biome_counts = db.biome_counts(status_filter=str(TileStatus.COMPLETE))
    region_counts = db.region_counts(status_filter=str(TileStatus.COMPLETE))
    rejection_counts = db.rejection_counts()

    lines = [
        "",
        "═" * 60,
        "  TILE REGISTRY SUMMARY",
        "═" * 60,
        f"  Total tiles    : {db.count_tiles():>10,}",
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


def audit_summary(audit: dict) -> str:
    """Generate summary of AOI coverage audit."""
    total = len(audit)
    covered = sum(1 for v in audit.values() if v["has_coverage"])
    uncovered = total - covered

    by_biome: dict[str, dict] = defaultdict(lambda: {"total": 0, "covered": 0})
    for v in audit.values():
        b = v["biome"]
        by_biome[b]["total"] += 1
        by_biome[b]["covered"] += int(v["has_coverage"])

    lines = [
        "",
        "═" * 60,
        "  AOI COVERAGE AUDIT",
        "═" * 60,
        f"  Total AOIs     : {total:>10,}",
        f"  With coverage  : {covered:>10,}  ({100*covered/max(total,1):.1f}%)",
        f"  No coverage    : {uncovered:>10,}  ({100*uncovered/max(total,1):.1f}%)",
        "",
        "  By biome (total | covered | gap%):",
    ]
    for b, c in sorted(by_biome.items(), key=lambda x: -x[1]["total"]):
        gap = 100 * (c["total"] - c["covered"]) / max(c["total"], 1)
        lines.append(
            f"    {b:<45} {c['total']:>6,}  {c['covered']:>6,}  {gap:5.1f}% gap"
        )

    lines += ["═" * 60, ""]
    return "\n".join(lines)
