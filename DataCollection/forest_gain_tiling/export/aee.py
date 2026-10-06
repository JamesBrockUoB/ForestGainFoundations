from __future__ import annotations

import ee
from config import settings
from export.year_chunks import chunk_name, chunk_years, prefix_bands
from gee_datasets.registry import Datasets


def build_year_aee(geom: ee.Geometry, year: int) -> ee.Image:
    DATASETS = Datasets()
    return (
        ee.ImageCollection(DATASETS.satellite_embedding)
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
    tasks: dict[str, ee.batch.Task] = {}

    for years in chunk_years(list(settings.years), settings.export_years_per_task):
        year_images = []
        for year in years:
            image = build_year_aee(geom, year).toFloat()
            year_images.append(prefix_bands(image, year) if len(years) > 1 else image)

        combined = ee.Image.cat(year_images)

        name = chunk_name("aee", years)
        key = f"embeddings/{name}"
        prefix = f"{tile_id}__embeddings__{name}"

        task = ee.batch.Export.image.toDrive(
            image=combined,
            description=prefix,
            folder=settings.drive_folder,
            fileNamePrefix=prefix,
            region=geom,
            crs=settings.crs_wkt,
            crsTransform=crs_transform,
            maxPixels=10_000_000_000_000,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks[key] = task

    return tasks
