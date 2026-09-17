import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings

_drive_folder_lock = threading.Lock()


def _build_rclone_base_args() -> list[str]:
    args = ["rclone", "moveto", "--drive-use-trash=false"]
    if settings.rclone_fast_list:
        args.append("--fast-list")
    if not settings.rclone_verify_checksum:
        args.append("--size-only")
    args += [
        f"--transfers={settings.rclone_transfers}",
        f"--checkers={settings.rclone_checkers}",
        f"--contimeout={settings.rclone_contimeout}",
        f"--timeout={settings.rclone_timeout}",
        f"--low-level-retries={settings.rclone_low_level_retries}",
    ]
    if settings.rclone_drive_chunk_size:
        args.append(f"--drive-chunk-size={settings.rclone_drive_chunk_size}")
    return args


def _run_rclone_moveto(src: str, dest: str, logger: logging.Logger) -> bool:
    cmd = _build_rclone_base_args() + [src, dest]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"rclone failed: {src} -> {dest}: {result.stderr}")
        return False
    logger.debug(f"rclone complete: {src} -> {dest}")
    return True


_drive_folder_id_cache: str | None = None


def ensure_drive_source_folder(logger: logging.Logger) -> str:
    global _drive_folder_id_cache
    with _drive_folder_lock:
        if _drive_folder_id_cache is not None:
            return _drive_folder_id_cache

        remote = settings.drive_remote
        folder_name = settings.drive_folder

        def _list_matching_folders() -> list[dict]:
            result = subprocess.run(
                ["rclone", "lsjson", f"{remote}:", "--dirs-only"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not list Drive root while resolving folder "
                    f"'{folder_name}': {result.stderr}"
                )
            entries = json.loads(result.stdout)
            return [e for e in entries if e.get("Name") == folder_name]

        matches = _list_matching_folders()

        if not matches:
            mk = subprocess.run(
                ["rclone", "mkdir", f"{remote}:{folder_name}"],
                capture_output=True,
                text=True,
            )
            if mk.returncode != 0:
                raise RuntimeError(
                    f"Failed to create Drive folder '{folder_name}': {mk.stderr}"
                )
            matches = _list_matching_folders()
            if not matches:
                raise RuntimeError(
                    f"Created Drive folder '{folder_name}' but could not find it "
                    f"immediately afterwards"
                )

        if len(matches) > 1:
            logger.warning(
                f"Found {len(matches)} Drive folders named '{folder_name}'; "
                f"merging into one canonical folder"
            )
            matches.sort(key=lambda e: e.get("ID", ""))
            canonical = matches[0]
            for dup in matches[1:]:
                dup_spec = f"{remote},root_folder_id={dup['ID']}:"
                canon_spec = f"{remote},root_folder_id={canonical['ID']}:"
                mv = subprocess.run(
                    ["rclone", "move", dup_spec, canon_spec],
                    capture_output=True,
                    text=True,
                )
                if mv.returncode != 0:
                    raise RuntimeError(
                        f"Failed to merge duplicate Drive folder {dup['ID']} into "
                        f"{canonical['ID']}: {mv.stderr}"
                    )
                rmdir = subprocess.run(
                    ["rclone", "rmdir", dup_spec],
                    capture_output=True,
                    text=True,
                )
                if rmdir.returncode != 0:
                    logger.warning(
                        f"Merged duplicate Drive folder {dup['ID']} but could not "
                        f"remove the now-empty folder: {rmdir.stderr}"
                    )
        else:
            canonical = matches[0]

        _drive_folder_id_cache = canonical["ID"]
        logger.info(
            f"Pinned Drive source folder '{folder_name}' to ID {_drive_folder_id_cache}"
        )
        return _drive_folder_id_cache


def _ensure_dest_dirs(dest_dirs: set[str], logger: logging.Logger) -> bool:
    for dest_dir in sorted(dest_dirs):
        result = subprocess.run(
            ["rclone", "mkdir", dest_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"rclone mkdir failed for {dest_dir}: {result.stderr}")
            return False
        logger.debug(f"rclone mkdir ok (or already existed): {dest_dir}")
    return True


def rclone_read_bytes(src: str, logger: logging.Logger) -> bytes | None:
    result = subprocess.run(["rclone", "cat", src], capture_output=True)
    if result.returncode != 0:
        logger.warning(
            f"rclone cat failed: {src}: {result.stderr.decode(errors='replace')}"
        )
        return None
    return result.stdout


def rclone_push(src: str, dest: str, logger: logging.Logger) -> bool:
    cmd = _build_rclone_base_args() + [src, dest]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"rclone push failed: {src} -> {dest}: {result.stderr}")
        return False
    return True


def rclone_product(
    tile_id: str,
    category: str,
    name: str,
    dest_root: str,
    logger: logging.Logger,
    folder_id: str | None = None,
) -> bool:
    drive_name = f"{tile_id}__{category}__{name}.tif"
    dest_path = f"{dest_root}/{tile_id}/{category}/{name}.tif"

    if folder_id is None:
        folder_id = ensure_drive_source_folder(logger)
    src = f"{settings.drive_remote},root_folder_id={folder_id}:{drive_name}"
    return _run_rclone_moveto(src, dest_path, logger)


def rclone_all_products(
    tile_id: str,
    products: list[tuple[str, str]],
    dest_root: str,
    logger: logging.Logger,
    max_workers: int | None = None,
) -> bool:
    if not products:
        return True

    folder_id = ensure_drive_source_folder(logger)

    dest_dirs = {f"{dest_root}/{tile_id}/{category}" for category, _ in products}
    if not _ensure_dest_dirs(dest_dirs, logger):
        logger.warning(f"{tile_id} | failed to prepare destination directories")
        return False

    default_workers = min(len(products), settings.rclone_max_workers)
    max_workers = max_workers or default_workers

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                rclone_product, tile_id, category, name, dest_root, logger, folder_id
            ): (category, name)
            for category, name in products
        }

        try:
            for fut in as_completed(futures):
                ok = fut.result()
                if not ok:
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    failed = futures[fut]
                    logger.warning(
                        f"{tile_id} | rclone failed for {failed}; aborting tile rclone"
                    )
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
    if ":" not in dest_root:
        logger.error(f"Invalid HPC rclone destination: {dest_root}")
        return False

    remote, path = dest_root.split(":", 1)

    try:
        result = subprocess.run(
            ["rclone", "lsd", f"{remote}:{path}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            logger.info(f"HPC rclone destination available: {dest_root}")
            return True

        result = subprocess.run(
            ["rclone", "lsd", f"{remote}:", "--max-depth", "1"],
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
