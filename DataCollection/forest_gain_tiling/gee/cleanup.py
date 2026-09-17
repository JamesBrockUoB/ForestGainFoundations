from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

import ee
from config import settings
from enums import TileStatus
from registry.store import update_tile


def _cancel_gee_tasks(
    tile_id: str,
    tasks: dict[str, ee.batch.Task],
    logger: logging.Logger,
) -> None:
    for key, task in tasks.items():
        try:
            status = task.status()
            state = status.get("state")

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


def _delete_drive_exports(
    tile_id: str,
    tasks: dict[str, ee.batch.Task],
    logger: logging.Logger,
) -> None:
    if not settings.drive_remote:
        logger.warning(
            f"{tile_id} | DRIVE_REMOTE is not configured; "
            "cannot remove failed tile exports from Drive"
        )
        return

    if not settings.drive_folder:
        logger.warning(
            f"{tile_id} | DRIVE_FOLDER is not configured; "
            "cannot remove failed tile exports from Drive"
        )
        return

    import subprocess

    for key in tasks:
        try:
            category, name = key.split("/", 1)
        except ValueError:
            logger.warning(f"{tile_id} | invalid GEE product key: {key}")
            continue

        drive_name = f"{tile_id}__{category}__{name}.tif"

        source = f"{settings.drive_remote}:" f"{settings.drive_folder}/" f"{drive_name}"

        result = subprocess.run(
            [
                "rclone",
                "deletefile",
                source,
                "--drive-use-trash=false",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            pass
        else:
            stderr = (result.stderr or "").strip()

            if (
                "not found" in stderr.lower()
                or "object not found" in stderr.lower()
                or "file not found" in stderr.lower()
            ):
                pass
            else:
                logger.warning(
                    f"{tile_id} | failed removing Drive export "
                    f"{drive_name}: {stderr or '<empty>'}"
                )


def _cleanup_failed_tile(
    *,
    tile_id: str,
    reason: str,
    logger: logging.Logger,
    tasks: dict[str, ee.batch.Task],
    output_dir: Path | None,
    embeddings_thread: threading.Thread | None,
    cancel_event: threading.Event,
    drive_already_cleared: bool = False,
) -> str:
    logger.error(f"{tile_id} | {reason}")

    cancel_event.set()

    _cancel_gee_tasks(tile_id, tasks, logger)

    if drive_already_cleared:
        logger.info(f"{tile_id} | Drive exports already moved; skipping Drive cleanup")
    else:
        _delete_drive_exports(tile_id, tasks, logger)

    if embeddings_thread is not None:
        embeddings_thread.join()

    scratch_dir = settings.data_dir / "tessera_scratch" / tile_id

    if scratch_dir.exists():
        try:
            shutil.rmtree(
                scratch_dir,
                ignore_errors=False,
            )

        except FileNotFoundError:
            pass

        except Exception:
            logger.exception(
                f"{tile_id} | failed removing TESSERA scratch: " f"{scratch_dir}"
            )

    if output_dir is not None and output_dir.exists():
        try:
            shutil.rmtree(
                output_dir,
                ignore_errors=False,
            )

            logger.info(f"{tile_id} | removed output: {output_dir}")

        except FileNotFoundError:
            pass

        except Exception:
            logger.exception(f"{tile_id} | failed removing output: " f"{output_dir}")

    update_tile(
        tile_id,
        status=TileStatus.FAILED,
        error=reason,
    )

    return str(TileStatus.FAILED)
