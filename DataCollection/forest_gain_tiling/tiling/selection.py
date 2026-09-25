"""Tile selection, filtering, and stratification-ratio management."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from registry.store import _get_db, iter_tiles

STRATA_FIELDS = ("biome", "region", "country")


def _strata_ratios_path() -> Path:
    return settings.valid_aois_path.parent / f"strata_ratios.json"


def save_strata_ratios(
    counts_by_field: dict[str, Counter],
    total: int,
) -> None:
    """
    Persist population-level stratification ratios. Call this from `plan`
    right after the tile grid is built, using the counters accumulated
    while streaming tiles to the registry — that's the full, unfiltered
    population and the only correct denominator for ratios.

    Do NOT compute these ratios from VALID (or any post-filter) tiles —
    filtering rejects tiles non-uniformly across biome/region/country
    (cloud cover, gain thresholds, etc. all vary geographically), so
    ratios derived from survivors bake that bias in as if it were the
    true distribution.
    """
    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "total_tiles": total,
        "fields": {
            field: {k: v / total for k, v in counts.items()}
            for field, counts in counts_by_field.items()
        },
    }

    path = _strata_ratios_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _compute_strata_ratios_from_registry() -> dict:
    """
    Fallback: rebuilds the ratio cache directly from the registry's
    per-field aggregate queries (biome_counts/region_counts/country_counts.
    Used when `plan` hasn't been (re)run since the cache went
    stale or missing.
    """
    db = _get_db()
    total = db.count_tiles()

    if total == 0:
        raise RuntimeError("No tiles found for — run `plan` first.")

    counts_by_field = {
        "biome": Counter(db.biome_counts()),
        "region": Counter(db.region_counts()),
        "country": Counter(db.country_counts()),
    }

    save_strata_ratios(counts_by_field, total)

    with open(_strata_ratios_path()) as f:
        return json.load(f)["fields"]


def load_or_compute_strata_ratios(
    logger: logging.Logger | None = None,
    force_recompute: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Load cached population strata ratios. Computes (and
    caches) them from the registry only if the cache is missing, stale,
    or force_recompute=True.

    Returns {"biome": {name: ratio, ...}, "region": {...}, "country": {...}}.
    """
    path = _strata_ratios_path()

    if path.exists() and not force_recompute:
        with open(path) as f:
            cached = json.load(f)

        db_total = _get_db().count_tiles()

        if cached["total_tiles"] != db_total:
            msg = (
                "Strata ratio cache was computed over "
                f"{cached['total_tiles']:,} tiles; registry now has "
                f"{db_total:,}. Ratios may be stale — re-run `plan`, or "
                "call load_or_compute_strata_ratios(force_recompute=True) "
                "if the AOI universe genuinely changed."
            )
            (logger.warning if logger else print)(msg)

        return cached["fields"]

    if logger:
        logger.info("No strata ratio cache — computing from registry…")

    return _compute_strata_ratios_from_registry()


def filter_candidates(
    status: str,
    tile_id: str | None = None,
    aoi_id: str | None = None,
    biome: str | None = None,
    region: str | None = None,
    country: str | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    Stream and filter candidate tiles from database.
    Returns list after applying all filters (for compatibility with stratified_sample).
    Never materializes the entire grid - only the filtered results.
    """
    candidates = []

    for tile in iter_tiles(status=status):
        if tile_id and tile.get("tile_id") != tile_id:
            continue

        if aoi_id and aoi_id not in tile.get("aoi_ids", []):
            continue

        if biome and biome.lower() not in tile.get("biome", "").lower():
            continue

        if region and region.lower() not in tile.get("region", "").lower():
            continue

        if country and country.lower() not in tile.get("country", "").lower():
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
    ratios: dict[str, float] | None = None,
) -> list[dict]:
    """
    Stratified sampling from candidates.

    mode='prop':
      - if `ratios` is given, quotas are drawn from that FIXED population
        distribution (see load_or_compute_strata_ratios) — this is what
        you want whenever `candidates` has already been through any
        filtering, since deriving ratios from `candidates` itself just
        re-encodes whatever bias the filtering introduced.
      - if `ratios` is None, falls back to proportions derived from
        `candidates` directly — only correct when `candidates` IS the
        full unfiltered population already (e.g. sampling PENDING tiles
        pre-cheap-filter with no prior stage having run).
    mode='equal': even split across strata present in `candidates`,
      ignores `ratios` entirely.
    """
    strata: dict[str, list[dict]] = {}
    for tile in candidates:
        key = tile.get(stratify_key, "Unknown")
        strata.setdefault(key, []).append(tile)

    sampled = []

    if mode == "equal":
        per_stratum = max(1, limit // len(strata))
        for stratum_tiles in strata.values():
            sampled.extend(
                random.sample(stratum_tiles, min(per_stratum, len(stratum_tiles)))
            )

    else:  # mode == "prop"
        if ratios is None:
            total = len(candidates)
            for stratum_tiles in strata.values():
                stratum_count = max(1, int(limit * len(stratum_tiles) / total))
                sampled.extend(
                    random.sample(stratum_tiles, min(stratum_count, len(stratum_tiles)))
                )
        else:
            for stratum, stratum_tiles in strata.items():
                ratio = ratios.get(stratum, 0.0)
                stratum_count = round(limit * ratio)
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
