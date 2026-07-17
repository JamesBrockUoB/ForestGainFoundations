from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from config import settings
from embeddings.aee import download_embeddings as download_aee
from embeddings.tessera import download_embeddings as download_tessera


def _process_embedding_source(
    name: str,
    download_fn: Callable[[dict, Path, logging.Logger], None],
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    tile_id = tile["tile_id"]
    try:
        logger.info(f"{tile_id} | starting {name}")
        download_fn(tile, output_dir, logger)
        logger.info(f"{tile_id} | {name} complete")
        return True
    except Exception as exc:
        logger.error(f"{tile_id} | {name} failed: {exc}")
        return False


def _process_embedding_source_with_retry(
    name: str,
    download_fn: Callable[[dict, Path, logging.Logger], None],
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    retries: int = 5,
) -> bool:
    tile_id = tile["tile_id"]
    for attempt in range(retries):
        if _process_embedding_source(name, download_fn, tile, output_dir, logger):
            return True
        wait = (2**attempt) + 1
        logger.warning(f"{tile_id} | {name} retry {attempt + 1}/{retries} in {wait}s")
        time.sleep(wait)
    logger.error(f"{tile_id} | {name} exhausted retries")
    return False


def process_tessera_with_retry(
    tile: dict[str, Any], output_dir: Path, logger: logging.Logger, retries: int = 5
) -> bool:
    return _process_embedding_source_with_retry(
        "TESSERA", download_tessera, tile, output_dir, logger, retries
    )


def process_aee_with_retry(
    tile: dict[str, Any], output_dir: Path, logger: logging.Logger, retries: int = 5
) -> bool:
    return _process_embedding_source_with_retry(
        "AEE", download_aee, tile, output_dir, logger, retries
    )


def process_all_embeddings_with_retry(
    tile: dict[str, Any], output_dir: Path, logger: logging.Logger, retries: int = 5
) -> bool:
    """
    TESSERA always runs here. AEE only runs here when
    settings.aee_source == "geoai" — when it's "gee", AEE was already
    submitted as a Drive export task alongside composites/static/labels
    (export/aee.py's submit_aee_exports, wired into process_tile) and
    has already landed on disk via rclone by the time this function
    runs; calling download_aee again here would be redundant.
    """
    tessera_ok = process_tessera_with_retry(tile, output_dir, logger, retries)
    aee_ok = (
        process_aee_with_retry(tile, output_dir, logger, retries)
        if settings.aee_source == "geoai"
        else True
    )
    return tessera_ok and aee_ok
