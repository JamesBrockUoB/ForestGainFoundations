from __future__ import annotations

import ee
from config import settings
from gee_datasets.registry import Datasets
from labels.pseudo import build_pseudo_labels


def build_label_layers(
    geom: ee.Geometry,
    ds: Datasets,
    gain_confidence: ee.Image,
) -> dict[str, ee.Image]:
    """
    - gain_confidence: always present.
    - pseudo_labels: only when settings.pseudo_labels_available (ForTy
      is a fixed 2020 snapshot). p2 tiles therefore only export gain_mask under labels/
    """
    layers: dict[str, ee.Image] = {
        "gain_confidence": gain_confidence.updateMask(gain_confidence).rename(
            "gain_confidence"
        ),
    }

    if settings.pseudo_labels_available:
        layers["pseudo_labels"] = build_pseudo_labels(geom, gain_confidence, ds)

    return layers


def submit_label_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    ds: Datasets,
    gain_confidence: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    tasks: dict[str, ee.batch.Task] = {}
    layers = build_label_layers(geom, ds, gain_confidence)

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
