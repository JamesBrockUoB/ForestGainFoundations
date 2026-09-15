from __future__ import annotations

import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable

from config import settings
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
    cancel_event: Event,
    retries: int = 5,
) -> bool:
    tile_id = tile["tile_id"]

    for attempt in range(retries):
        if cancel_event.is_set():
            logger.warning(f"{tile_id} | {name} cancelled")
            return False

        if _process_embedding_source(
            name,
            download_fn,
            tile,
            output_dir,
            logger,
        ):
            return True

        if attempt < retries - 1:
            wait = (2**attempt) + 1

            logger.warning(
                f"{tile_id} | {name} retry {attempt + 1}/{retries} in {wait}s"
            )

            # Interruptible sleep
            if cancel_event.wait(wait):
                logger.warning(f"{tile_id} | {name} cancelled during retry wait")
                return False

    logger.error(f"{tile_id} | {name} exhausted retries")
    return False


def process_tessera_with_retry(
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
    retries: int = 5,
) -> bool:
    ok = _process_embedding_source_with_retry(
        "TESSERA",
        download_tessera,
        tile,
        output_dir,
        logger,
        cancel_event,
        retries,
    )
    if not ok:
        cancel_event.set()
    return ok


def process_all_embeddings_with_retry(
    tile: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
    retries: int = 5,
) -> bool:
    """
    Downloads TESSERA embeddings for the tile.
    If TESSERA fails and exhausts retries, cancel_event is set.
    """
    tile_id = tile["tile_id"]

    tessera_ok = process_tessera_with_retry(
        tile,
        output_dir,
        logger,
        cancel_event,
        retries,
    )

    if not tessera_ok:
        logger.error(f"{tile_id} | embedding download failed, aborting tile")

    return tessera_ok
