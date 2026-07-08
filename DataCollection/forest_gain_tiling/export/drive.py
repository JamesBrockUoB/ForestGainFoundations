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
