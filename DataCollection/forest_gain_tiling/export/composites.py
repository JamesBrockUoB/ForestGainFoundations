from __future__ import annotations

import ee
from config import settings


def _mask_s2_scl(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    return img.updateMask(
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(0))
    )


def _add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5*((NIR-RED)/(NIR+6.0*RED-7.5*BLUE+1.0))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    return img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"]).addBands([ndvi, evi])


def _date_range(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year+1}-01-01"


def s2_availability(geom: ee.Geometry, year: int) -> ee.Image:
    """
    Coverage check only — uses settings.s2_check_bands (a small proxy
    subset), NOT the full band list used for spectral analysis. See
    s2_composite for the full-band version used in actual exports.
    """
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .map(_mask_s2_scl)
        .select(list(settings.s2_check_bands))
    )
    return ic.map(lambda i: i.mask().reduce(ee.Reducer.min())).reduce(ee.Reducer.max())


def s1_availability(geom: ee.Geometry, year: int) -> ee.Image:
    """
    S1 counterpart to s2_availability — fraction-style valid-pixel presence
    check using settings.s1_check_bands (VV, VH by default), IW mode only.
    """
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(list(settings.s1_check_bands))
    )
    return ic.map(lambda i: i.mask().reduce(ee.Reducer.min())).reduce(ee.Reducer.max())


def s2_coverage_frac(geom: ee.Geometry, year: int) -> ee.Number:
    """Fraction of pixels in geom with at least one valid S2 observation in `year`."""
    valid = s2_availability(geom, year)
    stats = valid.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=settings.scale,
        crs=settings.crs_wkt,
        maxPixels=1_000_000_000,
    )
    return ee.Number(ee.Algorithms.If(stats.get("valid"), stats.get("valid"), 0))


def s2_composite(geom: ee.Geometry, year: int) -> ee.Image:
    start, end = _date_range(year)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(_mask_s2_scl)
        .select(
            [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
            ]
        )
        .map(_add_indices)
    )

    return s2.median()


def s2_peak_ndvi(geom: ee.Geometry, year: int) -> ee.Image:
    centroid = ee.Geometry(geom).centroid(maxError=1)
    north = ee.Number(centroid.coordinates().get(1)).gt(0)

    start = ee.String(
        ee.Algorithms.If(
            north,
            f"{year}-05-01",
            f"{year}-11-01",
        )
    )

    end = ee.String(
        ee.Algorithms.If(
            north,
            f"{year}-09-30",
            f"{year + 1}-03-31",
        )
    )

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(_mask_s2_scl)
        .map(_add_indices)
        .select(["NDVI"])
        .median()
    )


def s1_composite(geom: ee.Geometry, year: int) -> ee.Image:  # noqa: ARG001
    def _mask_edge(img):
        edge = img.lt(-30.0)
        return img.updateMask(img.mask().And(edge.Not()))

    med = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .map(_mask_edge)
        .median()
    )
    return med.addBands(med.select("VV").divide(med.select("VH")).rename("VVVH"))


_BAND_SUFFIXES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "NDVI",
    "EVI",
    "VV",
    "VH",
    "VVVH",
]


def build_year_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """Combined S1+S2 composite for a single year — exported as its own
    per-year file (composites/s1s2_<year>.tif)."""
    return (
        s2_composite(geom, year)
        .addBands(s1_composite(geom, year))
        .rename(list(_BAND_SUFFIXES))
    )


def submit_composite_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    """
    Submit one export task per year in settings.period_years. Returns a
    dict keyed "composites/s1s2_<year>" -> started ee.batch.Task.
    """
    tasks: dict[str, ee.batch.Task] = {}

    for year in settings.period_years:
        image = build_year_composite(geom, year).updateMask(full_valid).toFloat()
        name = f"s1s2_{year}"
        key = f"composites/{name}"
        prefix = f"{tile_id}__composites__{name}"

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
            skipEmptyTiles=True,
        )
        task.start()
        tasks[key] = task

    return tasks
