from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from config import settings
from enums import TileStatus
from models import AoiAuditEntry

Registry = dict[str, dict[str, Any]]


def load_registry() -> Registry:
    if not settings.registry_path.exists():
        return {}
    with open(settings.registry_path) as f:
        entries = json.load(f)
    return {e["tile_id"]: e for e in entries}


def save_registry(registry: Registry) -> None:
    tmp = settings.registry_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(registry.values()), f, indent=2)
    tmp.replace(settings.registry_path)


def update_tile(registry: Registry, tile_id: str, **kwargs: Any) -> None:
    registry[tile_id].update(kwargs)
    if "status" in kwargs and isinstance(kwargs["status"], TileStatus):
        registry[tile_id]["status"] = str(kwargs["status"])
    save_registry(registry)


def build_aoi_audit(
    registry: Registry, valid_aois: list[dict]
) -> dict[str, dict[str, Any]]:
    aoi_tile_counts: dict[str, Counter] = defaultdict(Counter)
    for entry in registry.values():
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
    tmp = settings.aoi_audit_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2)
    tmp.replace(settings.aoi_audit_path)


def registry_summary(registry: Registry) -> str:
    status_counts = Counter(e["status"] for e in registry.values())
    biome_counts = Counter(
        e.get("biome", "Unknown")
        for e in registry.values()
        if e["status"] == str(TileStatus.COMPLETE)
    )
    region_counts = Counter(
        e.get("region", "Unknown")
        for e in registry.values()
        if e["status"] == str(TileStatus.COMPLETE)
    )
    rejection_counts = Counter(
        e.get("rejection_reason", "unknown")
        for e in registry.values()
        if e["status"] == str(TileStatus.REJECTED)
    )

    lines = [
        "",
        "═" * 60,
        "  TILE REGISTRY SUMMARY",
        "═" * 60,
        f"  Total tiles    : {len(registry):>10,}",
    ]
    for s in TileStatus:
        lines.append(f"  {s.value:<14} : {status_counts.get(s.value, 0):>10,}")

    if rejection_counts:
        lines += ["", "  Rejected by reason:"]
        for r, n in rejection_counts.most_common():
            lines.append(f"    {r:<35} {n:>8,}")

    if biome_counts:
        lines += ["", "  Complete by biome:"]
        for b, n in biome_counts.most_common():
            lines.append(f"    {b:<45} {n:>7,}")

    if region_counts:
        lines += ["", "  Complete by region:"]
        for r, n in region_counts.most_common():
            lines.append(f"    {r:<30} {n:>7,}")

    lines += ["═" * 60, ""]
    return "\n".join(lines)


def audit_summary(audit: dict) -> str:
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
