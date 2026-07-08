from __future__ import annotations

import ee
from config import settings

AEE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def aee_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """
    Google DeepMind AlphaEarth Foundations annual embedding for `year`,
    clipped to geom.
    """
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(AEE_COLLECTION)
        .filterDate(start, end)
        .filterBounds(geom)
        .mosaic()
        .clip(geom)
    )


def submit_aee_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    """
    One export task per year in settings.period_years — same pattern as
    submit_composite_exports / submit_static_exports / submit_label_exports.
    Lands at embeddings/aee_<year>.tif via the same Drive->rclone path as
    every other product, so _verify_tile_outputs' existing expectations
    (embeddings/aee_<year>.tif) are unchanged.
    """
    tasks: dict[str, ee.batch.Task] = {}

    for year in settings.period_years:
        name = f"aee_{year}"
        key = f"embeddings/{name}"
        prefix = f"{tile_id}__embeddings__{name}"

        image = aee_composite(geom, year).updateMask(full_valid).toFloat()

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=prefix,
            folder=settings.drive_folder,
            fileNamePrefix=prefix,
            region=geom,
            scale=settings.scale,
            crs=settings.crs_wkt,
            crsTransform=crs_transform,
            maxPixels=10_000_000_000_000,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks[key] = task

    return tasks
