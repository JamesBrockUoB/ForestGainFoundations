from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from config import settings
from export.year_chunks import chunk_years
from tessera.tessera import TesseraNoDataError, download_tessera


def download_tessera_with_retry(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
    retries: int = 4,
    years: list[int] | None = None,
) -> bool:
    tile_id = tile["tile_id"]

    for attempt in range(retries):
        if cancel_event.is_set():
            logger.warning(f"{tile_id} | TESSERA cancelled")
            return False

        try:
            logger.info(f"{tile_id} | starting TESSERA")
            download_tessera(tile, output_dir, logger, years)
            logger.info(f"{tile_id} | TESSERA complete")
            return True

        except TesseraNoDataError as exc:
            logger.error(
                f"{tile_id} | TESSERA has no data for this tile; "
                f"not retrying: {exc}"
            )
            return False

        except Exception as exc:
            logger.error(f"{tile_id} | TESSERA failed: {exc}")

        if attempt < retries - 1:
            wait = 2**attempt + 1

            logger.warning(
                f"{tile_id} | TESSERA retry " f"{attempt + 1}/{retries} in {wait}s"
            )

            if cancel_event.wait(wait):
                logger.warning(f"{tile_id} | TESSERA cancelled during retry wait")
                return False

    logger.error(f"{tile_id} | TESSERA exhausted retries")
    return False


def fetch_tessera_in_chunks(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    cancel_event: Event,
) -> bool:
    """Fetch TESSERA in chunks, splitting failed chunks recursively."""

    def fetch(years: list[int]) -> bool:
        if cancel_event.is_set():
            return False

        retries = 2 if len(years) > 1 else 4

        if download_tessera_with_retry(
            tile,
            output_dir,
            logger,
            cancel_event,
            retries=retries,
            years=years,
        ):
            return True

        if len(years) == 1:
            return False

        logger.warning(
            f"{tile['tile_id']} | TESSERA {years} failed; "
            "retrying as smaller requests"
        )

        mid = len(years) // 2
        return fetch(years[:mid]) and fetch(years[mid:])

    for chunk in chunk_years(
        list(settings.years),
        settings.tessera_years_per_request,
    ):
        if not fetch(chunk):
            cancel_event.set()
            return False

    return True
