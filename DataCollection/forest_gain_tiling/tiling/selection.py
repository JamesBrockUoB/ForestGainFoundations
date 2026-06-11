"""Tile selection and filtering with streaming support."""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from registry.store import iter_tiles


def filter_candidates(
    status: str,
    aoi_id: str | None = None,
    biome: str | None = None,
    region: str | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    Stream and filter candidate tiles from database.
    Returns list after applying all filters (for compatibility with stratified_sample).
    Never materializes the entire grid - only the filtered results.
    """
    candidates = []

    for tile in iter_tiles(status=status):
        # Filter by aoi_id
        if aoi_id and aoi_id not in tile.get("aoi_ids", []):
            continue

        # Filter by biome (case-insensitive substring)
        if biome and biome.lower() not in tile.get("biome", "").lower():
            continue

        # Filter by region (case-insensitive substring)
        if region and region.lower() not in tile.get("region", "").lower():
            continue

        candidates.append(tile)

    if logger:
        logger.info(f"Found {len(candidates):,} candidate tiles after filtering")

    return candidates


def stratified_sample(
    candidates: list[dict],
    stratify_key: str,
    limit: int,
    mode: str = "prop",
) -> list[dict]:
    """
    Stratified sampling from candidates.
    mode='prop': sample proportionally by stratum
    mode='equal': equal count from each stratum
    """
    # Group by stratum
    strata: dict[str, list[dict]] = {}
    for tile in candidates:
        key = tile.get(stratify_key, "Unknown")
        if key not in strata:
            strata[key] = []
        strata[key].append(tile)

    sampled = []

    if mode == "equal":
        # Equal count per stratum
        per_stratum = max(1, limit // len(strata))
        for stratum_tiles in strata.values():
            sampled.extend(
                random.sample(stratum_tiles, min(per_stratum, len(stratum_tiles)))
            )
    else:  # mode == "prop"
        # Proportional sampling
        for stratum_tiles in strata.values():
            stratum_count = int(limit * len(stratum_tiles) / len(candidates))
            stratum_count = max(1, stratum_count)
            sampled.extend(
                random.sample(stratum_tiles, min(stratum_count, len(stratum_tiles)))
            )

    return sampled[:limit]


def log_strata_counts(
    candidates: list[dict], key: str, logger: logging.Logger, mode: str
) -> None:
    """Log the stratified counts."""
    counts = Counter(t.get(key, "Unknown") for t in candidates)
    logger.info(f"Stratified sample ({mode} mode):")
    for stratum, count in counts.most_common():
        logger.info(f"  {stratum}: {count}")
