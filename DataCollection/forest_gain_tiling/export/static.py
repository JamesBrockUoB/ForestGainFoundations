from __future__ import annotations

import ee
from config import settings

# Snapshot matched to each period's start
WDPA_SNAPSHOT_BY_PERIOD = {
    "p1": "201707",  # nearest available snapshot at/before p1's 2017 start
    "p2": "202001",  # nearest available snapshot at/before p2's 2020 start
}


def _protected_area_mask(geom: ee.Geometry) -> ee.Image:
    snapshot = WDPA_SNAPSHOT_BY_PERIOD[settings.period]
    fc = ee.FeatureCollection(f"WCMC/WDPA/{snapshot}/polygons")
    return ee.Image().byte().paint(fc, 1).unmask(0).rename("protected_area")


def build_static_layers(geom: ee.Geometry) -> dict[str, ee.Image]:
    fabdem = (
        ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
        .filterBounds(geom)
        .mosaic()
        .clip(geom)
    )

    fabdem_native = fabdem.setDefaultProjection(crs="EPSG:4326", scale=30)
    slope_native = ee.Terrain.slope(fabdem_native)
    slope = slope_native.resample("bilinear").reproject(
        crs=settings.crs_wkt, scale=settings.scale
    )

    return {
        "fabdem": fabdem.rename("DEM"),
        "slope": slope.rename("slope"),
        "protected_area": _protected_area_mask(geom).clip(geom),
    }


def submit_static_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    tasks: dict[str, ee.batch.Task] = {}
    layers = build_static_layers(geom)

    for name, image in layers.items():
        key = f"static/{name}"
        prefix = f"{tile_id}__static__{name}"

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
        )
        task.start()
        tasks[key] = task

    return tasks
