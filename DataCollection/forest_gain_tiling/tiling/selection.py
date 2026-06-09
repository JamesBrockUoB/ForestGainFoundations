from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any


def filter_candidates(
    registry: dict[str, Any],
    target_status: str,
    *,
    aoi_id: str | None = None,
    biome: str | None = None,
    region: str | None = None,
    logger: logging.Logger | None = None,
) -> list[dict]:
    candidates = [e for e in registry.values() if e["status"] == target_status]

    if aoi_id:
        candidates = [e for e in candidates if aoi_id in e.get("aoi_ids", [])]
        if logger:
            logger.info(f"Filtered to AOI {aoi_id}: {len(candidates):,} tiles")

    if biome:
        candidates = [
            e for e in candidates if biome.lower() in e.get("biome", "").lower()
        ]
        if logger:
            logger.info(f"Filtered to biome '{biome}': {len(candidates):,} tiles")

    if region:
        candidates = [
            e for e in candidates if region.lower() in e.get("region", "").lower()
        ]
        if logger:
            logger.info(f"Filtered to region '{region}': {len(candidates):,} tiles")

    return candidates


def stratified_sample(
    candidates: list[dict],
    key: str,
    limit: int,
    mode: str = "prop",
) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for tile in candidates:
        buckets[tile.get(key, "Unknown")].append(tile)

    n_strata = len(buckets)
    total = len(candidates)

    if mode == "equal":
        base_alloc: dict[str, int] = {k: limit // n_strata for k in buckets}
    else:
        base_alloc = {k: int(limit * len(v) / total) for k, v in buckets.items()}

    allocated = sum(base_alloc.values())
    remainder = limit - allocated
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        if remainder <= 0:
            break
        base_alloc[k] += 1
        remainder -= 1

    sampled: list[dict] = []
    for k, alloc in base_alloc.items():
        sampled.extend(buckets[k][:alloc])

    return sampled


def log_strata_counts(
    candidates: list[dict], key: str, logger: logging.Logger, mode: str
) -> None:
    strata_counts = Counter(t.get(key, "Unknown") for t in candidates)
    logger.info(
        f"Stratified sample ({mode}) by {key}: "
        f"{len(candidates):,} tiles across {len(strata_counts)} strata"
    )
    for stratum, n in strata_counts.most_common():
        logger.info(f"  {stratum:<45} {n:>6,}")
