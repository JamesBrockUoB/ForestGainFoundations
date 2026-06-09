from __future__ import annotations

import logging
import subprocess

from config import settings


def rclone_to_hpc(tile_id: str, hpc_path: str, logger: logging.Logger) -> bool:
    result = subprocess.run(
        [
            "rclone",
            "moveto",
            "--drive-use-trash=false",
            f"{settings.drive_remote}:{settings.drive_folder}/{tile_id}.tif",
            f"{hpc_path}/{tile_id}.tif",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"rclone failed for {tile_id}: {result.stderr[:120]}")
        return False

    logger.info(f"rclone complete: {tile_id}")
    return True
