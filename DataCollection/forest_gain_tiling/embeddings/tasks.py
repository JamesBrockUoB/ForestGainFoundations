from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from embeddings.tessera import download_embeddings


def process_tessera(
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    tile_id = tile["tile_id"]
    try:
        logger.info(f"{tile_id} | starting embeddings")
        download_embeddings(tile, output_dir, logger)
        logger.info(f"{tile_id} | embeddings complete")
        return True
    except Exception as exc:
        logger.error(f"{tile_id} | embeddings failed: {exc}")
        return False


def process_tessera_with_retry(
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    retries: int = 5,
) -> bool:
    tile_id = tile["tile_id"]
    for attempt in range(retries):
        if process_tessera(tile, output_dir, logger):
            return True
        wait = (2**attempt) + 1
        logger.warning(
            f"{tile_id} | embedding retry {attempt + 1}/{retries} in {wait}s"
        )
        time.sleep(wait)
    logger.error(f"{tile_id} | embeddings exhausted retries")
    return False
