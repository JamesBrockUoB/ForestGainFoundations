"""Dedicated export path for inspector tiles; never touches the registry."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import ee

from config import settings
from embeddings.tasks import process_all_embeddings_with_retry
from export.aee import submit_aee_exports
from export.composites import submit_composite_exports
from export.drive import rclone_all_products
from export.labels import submit_label_exports
from export.metadata import write_tile_metadata
from export.static import submit_static_exports
from export.tasks import _verify_tile_outputs, _wait_for_all
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer
from stack.stacks import build_full_valid
from tiling.grid import crs_transform, tile_geom


def export_inspector_tile(
    tile: dict[str, Any],
    ds: Datasets,
    logger: logging.Logger,
    output_root: Path,
) -> Path:
    """Export a point-centred tile to ``output_root`` without registry writes.

    It uses the same product builders as the production exporter, but owns its
    task lifecycle and local directory so no ad-hoc tile can affect a planned
    tile's state or destination.
    """
    tile_id = tile["tile_id"]
    output_dir = output_root / tile_id
    geom = tile_geom(tile)
    tasks: dict[str, ee.batch.Task] = {}
    cancel_event = threading.Event()
    embeddings_thread: threading.Thread | None = None
    embeddings_result: dict[str, bool] = {}

    try:
        _, _, gain_confidence = build_gain_layer(geom, ds)
        full_valid = build_full_valid(geom)
        transform = crs_transform(tile)

        tasks.update(submit_composite_exports(geom, transform, full_valid, tile_id))
        tasks.update(submit_static_exports(geom, transform, full_valid, tile_id))
        tasks.update(
            submit_label_exports(geom, transform, full_valid, ds, gain_confidence, tile_id)
        )
        if settings.aee_source == "gee":
            tasks.update(submit_aee_exports(geom, transform, tile_id))

        def run_embeddings() -> None:
            embeddings_result["ok"] = process_all_embeddings_with_retry(
                tile, output_dir, logger, cancel_event
            )

        embeddings_thread = threading.Thread(target=run_embeddings)
        embeddings_thread.start()

        if not _wait_for_all(tasks, logger, tile_id):
            raise RuntimeError("one or more Earth Engine export tasks failed")

        products = [tuple(key.split("/", 1)) for key in tasks]
        if not rclone_all_products(tile_id, products, str(output_root), logger):
            raise RuntimeError("rclone transfer failed")

        embeddings_thread.join()
        if not embeddings_result.get("ok"):
            raise RuntimeError("embedding acquisition failed")

        missing = _verify_tile_outputs(output_dir, list(tasks))
        if missing:
            raise RuntimeError(f"missing outputs after export: {missing}")

        write_tile_metadata(tile, output_dir, logger)
        return output_dir

    except Exception:
        cancel_event.set()
        for task in tasks.values():
            try:
                if task.status()["state"] not in {
                    "COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"
                }:
                    task.cancel()
            except Exception:
                pass
        if embeddings_thread is not None:
            embeddings_thread.join(timeout=60)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
