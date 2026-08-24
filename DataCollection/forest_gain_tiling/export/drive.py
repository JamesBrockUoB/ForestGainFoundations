from __future__ import annotations

import logging
import subprocess

from config import settings


def rclone_product(
    tile_id: str,
    category: str,
    name: str,
    dest_root: str,
    logger: logging.Logger,
) -> bool:
    """
    Move one exported product from the flat Drive export folder into
    dest_root/tile_id/category/name.tif. Drive itself has no per-tile
    subfolders, so the Drive-side filename is
    "{tile_id}__{category}__{name}.tif" — that prefix is what keeps
    concurrent tiles from colliding, stripped back off on the way to the
    destination's proper subfolder.
    """
    drive_name = f"{tile_id}__{category}__{name}.tif"
    dest_path = f"{dest_root}/{tile_id}/{category}/{name}.tif"

    result = subprocess.run(
        [
            "rclone",
            "moveto",
            "--drive-use-trash=false",
            f"{settings.drive_remote}:{settings.drive_folder}/{drive_name}",
            dest_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(
            f"rclone failed for {tile_id} {category}/{name}: {result.stderr[:200]}"
        )
        return False

    logger.debug(f"rclone complete: {tile_id} {category}/{name}")
    return True


def rclone_all_products(
    tile_id: str,
    products: list[tuple[str, str]],
    dest_root: str,
    logger: logging.Logger,
) -> bool:
    """Stops at the first failure — one missing file already means the
    tile can't be marked complete."""
    for category, name in products:
        if not rclone_product(tile_id, category, name, dest_root, logger):
            return False
    return True


def check_hpc_available(
    dest_root: str,
    logger: logging.Logger,
    timeout: float = 15.0,
) -> bool:
    """
    Check whether the HPC rclone remote is reachable.

    Does not require dest_root itself to exist — the export code can create
    the destination directory when moving files.
    """
    if ":" not in dest_root:
        logger.error(f"Invalid HPC rclone destination: {dest_root}")
        return False

    remote, path = dest_root.split(":", 1)

    try:
        result = subprocess.run(
            [
                "rclone",
                "lsd",
                f"{remote}:{path}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            logger.info(f"HPC rclone destination available: {dest_root}")
            return True

        # The directory may not exist yet. Check the remote itself.
        result = subprocess.run(
            [
                "rclone",
                "lsd",
                f"{remote}:",
                "--max-depth",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            logger.info(
                f"HPC rclone remote reachable, destination does not yet exist: "
                f"{dest_root}"
            )
            return True

        logger.warning(
            f"HPC rclone remote unreachable: {remote} | "
            f"{result.stderr.strip()[:300]}"
        )
        return False

    except subprocess.TimeoutExpired:
        logger.warning(f"HPC rclone check timed out: {remote}")
        return False

    except OSError as e:
        logger.error(f"Failed to run rclone for HPC check: {e}")
        return False
