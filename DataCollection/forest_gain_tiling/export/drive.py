import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings


def _build_rclone_base_args() -> list[str]:
    """
    Build a tuned base rclone argument list, reading defaults from settings
    if present. Individual rclone products will append source and dest paths.
    """
    transfers = getattr(settings, "rclone_transfers", 8)
    checkers = getattr(settings, "rclone_checkers", 8)
    fast_list = getattr(settings, "rclone_fast_list", True)

    args = ["rclone", "moveto", "--drive-use-trash=false"]
    if fast_list:
        args.append("--fast-list")
    args += [f"--transfers={transfers}", f"--checkers={checkers}"]
    drive_chunk = getattr(settings, "rclone_drive_chunk_size", None)
    if drive_chunk:
        args.append(f"--drive-chunk-size={drive_chunk}")
    return args


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

    cmd = _build_rclone_base_args() + [
        f"{settings.drive_remote}:{settings.drive_folder}/{drive_name}",
        dest_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

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
    max_workers: int | None = None,
) -> bool:
    """Run rclone moves in parallel across products for a single tile.
    Stops at the first failure — one missing file already means the
    tile can't be marked complete.
    """
    if not products:
        return True

    cpu_count = os.cpu_count() or 4
    default_workers = min(len(products), max(2, cpu_count * 2))
    max_workers = max_workers or default_workers

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(rclone_product, tile_id, category, name, dest_root, logger): (category, name)
            for category, name in products
        }

        try:
            for fut in as_completed(futures):
                ok = fut.result()
                if not ok:
                    # cancel outstanding tasks and return False
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    failed = futures[fut]
                    logger.warning(f"{tile_id} | rclone failed for {failed}; aborting tile rclone")
                    return False
        except Exception as exc:
            for pending in futures:
                if not pending.done():
                    pending.cancel()
            logger.warning(f"{tile_id} | rclone encountered an exception: {exc}")
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
