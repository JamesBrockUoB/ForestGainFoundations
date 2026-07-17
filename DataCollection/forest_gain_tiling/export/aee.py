from __future__ import annotations

import ee
from config import settings


def build_year_aee(geom: ee.Geometry, year: int) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(geom)
        .mosaic()
        .clip(geom)
    )


def submit_aee_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    """
    One Drive export task per year, landing at embeddings/aee_<year>.tif
    via the same Export->Drive->rclone path as composites/static/labels.
    Only submitted when settings.aee_source == "gee" — the default
    ("geoai") fetches AEE via direct HTTPS reads in the synchronous
    embeddings step instead (see embeddings/aee.py), costing no GEE
    task-queue quota. Not masked by full_valid — AEE is an independent
    embedding source, not derived from S1/S2 coverage.
    """
    tasks: dict[str, ee.batch.Task] = {}

    for year in settings.period_years:
        image = build_year_aee(geom, year).toFloat()
        name = f"aee_{year}"
        key = f"embeddings/{name}"
        prefix = f"{tile_id}__embeddings__{name}"

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
