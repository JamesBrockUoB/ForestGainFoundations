from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

import ee
from enums import TileStatus
from registry.store import update_tile


def _cleanup_failed_tile(
    *,
    tile_id: str,
    reason: str,
    logger: logging.Logger,
    tasks: dict[str, ee.batch.Task],
    output_dir: Path | None,
    embeddings_thread: threading.Thread | None,
    cancel_event: threading.Event,
) -> str:
    """
    Cancel all outstanding work and remove partial outputs.

    This is called on ANY failure before returning FAILED.
    """

    logger.error(f"{tile_id} | {reason}")

    #
    # Stop GEE exports
    #
    for key, task in tasks.items():
        try:
            state = task.status()["state"]

            if state not in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "CANCEL_REQUESTED",
            ):
                task.cancel()
                logger.info(f"{tile_id} | cancelled {key}")

        except Exception as exc:
            logger.warning(f"{tile_id} | failed cancelling {key}: {exc}")

    #
    # Stop embedding downloads
    #
    cancel_event.set()

    if embeddings_thread is not None:
        embeddings_thread.join(timeout=60)

    #
    # Remove partial tile directory
    #
    if output_dir and output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            logger.info(f"{tile_id} | removed {output_dir}")
        except Exception as exc:
            logger.warning(f"{tile_id} | failed removing output: {exc}")

    #
    # Registry
    #
    update_tile(
        tile_id,
        status=TileStatus.FAILED,
        error=reason,
    )

    return str(TileStatus.FAILED)
