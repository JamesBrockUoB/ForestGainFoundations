from __future__ import annotations

import ee
from config import settings
from gee_datasets.registry import Datasets


def build_label_layers(
    gain_confidence: ee.Image,
) -> dict[str, ee.Image]:
    layers: dict[str, ee.Image] = {
        "gain_confidence": gain_confidence.updateMask(gain_confidence).rename(
            "gain_confidence"
        ),
    }

    return layers


def submit_label_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    gain_confidence: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    tasks: dict[str, ee.batch.Task] = {}
    layers = build_label_layers(gain_confidence)

    for name, image in layers.items():
        key = f"labels/{name}"
        prefix = f"{tile_id}__labels__{name}"

        task = ee.batch.Export.image.toDrive(
            image=image.updateMask(full_valid).toFloat(),
            description=prefix,
            folder=settings.drive_folder,
            fileNamePrefix=prefix,
            region=geom,
            scale=settings.scale,
            crs=settings.crs_wkt,
            crsTransform=crs_transform,
            maxPixels=10_000_000_000_000,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
        )
        task.start()
        tasks[key] = task

    return tasks
