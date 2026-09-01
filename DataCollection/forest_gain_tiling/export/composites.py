from __future__ import annotations

import ee
from config import settings

NATIVE_10M_BANDS = ["B2", "B3", "B4", "B8"]
NATIVE_20M_BANDS = ["B5", "B6", "B7", "B8A", "B11", "B12"]

CLOUD_SCORE_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_PLUS_BAND = "cs_cdf"


def hemisphere_from_tile(min_lat: float, max_lat: float) -> bool:
    """True if the tile's centroid is in the northern hemisphere.

    Only used for the leaf-on NDVI trend signal
    """
    return (min_lat + max_lat) / 2.0 >= 0


def _join_cloud_score_plus(ic: ee.ImageCollection, geom, start: str, end: str) -> ee.ImageCollection:
    """Link Cloud Score+ QA band onto each S2 image via
    ImageCollection.linkCollection — Google's recommended pattern for
    this dataset. The linked band is attached directly as a band on
    each image, matched by system:index, rather than nested behind a
    property lookup (the older Join.saveFirst pattern this replaces)."""
    cs_col = (
        ee.ImageCollection(CLOUD_SCORE_PLUS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(geom)
    )
    return ic.linkCollection(cs_col, [CLOUD_SCORE_PLUS_BAND])


def _mask_cloud_score_plus(img: ee.Image, threshold: float = settings.cloud_score_thresh) -> ee.Image:
    """Mask using the linked Cloud Score+ cs_cdf band — a plain band on
    img after linkCollection, no unwrapping needed."""
    cs = img.select(CLOUD_SCORE_PLUS_BAND)
    return img.updateMask(cs.gte(threshold))


def _add_ndvi(img: ee.Image) -> ee.Image:
    return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))


def _date_range(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year+1}-01-01"


def leaf_on_window(year: int, *, north: bool) -> tuple[str, str]:
    """Leaf-on / peak-growing-season window — used only by
    s2_peak_ndvi / s2_ndvi_trend."""
    if north:
        return f"{year}-05-01", f"{year}-09-30"
    return f"{year}-11-01", f"{year + 1}-03-31"


def s2_availability(geom, year: int) -> ee.Image:
    """
    Full-year, Cloud Score+ masked coverage check
    """
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    )
    ic = _join_cloud_score_plus(ic, geom, start, end)
    ic = ic.map(_mask_cloud_score_plus).select(settings.s2_check_band)
    return ic.count().gt(0).unmask(0)


def s1_observation_count(tile: ee.Feature, year: int) -> ee.Number:
    """
    Acquisition-level S1 observation count for one tile.
    """
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(settings.s1_check_band)
    )
    return ic.filterBounds(tile.geometry()).size()


def s1_availability(tile: ee.Feature, year: int) -> ee.Number:
    """
    Acquisition-level S1 availability for one tile. 1 if the tile meets
    settings.min_s1_observations, else 0.
    """
    return s1_observation_count(tile, year).gte(settings.min_s1_observations).int()


def s2_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """Full-year median composite, Cloud Score+ masked."""
    start, end = _date_range(year)

    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    )
    ic = _join_cloud_score_plus(ic, geom, start, end)

    s2 = (
        ic.map(_mask_cloud_score_plus)
        .select(NATIVE_10M_BANDS + NATIVE_20M_BANDS)
        .map(_upsample_20m_bands_to_10m)
    )

    return s2.median()


def _upsample_20m_bands_to_10m(img: ee.Image) -> ee.Image:
    native_10m = img.select(NATIVE_10M_BANDS)
    native_20m = img.select(NATIVE_20M_BANDS).resample("bilinear")
    return native_10m.addBands(native_20m).copyProperties(img, img.propertyNames())


def s2_peak_ndvi(geom: ee.Geometry, year: int, *, north: bool) -> ee.Image:
    """Leaf-on NDVI, Cloud Score+ masked — same masking as everything
    else in this module, just on the tighter leaf-on window."""
    start, end = leaf_on_window(year, north=north)

    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    )
    ic = _join_cloud_score_plus(ic, geom, start, end)

    return (
        ic.map(_mask_cloud_score_plus)
        .map(_add_ndvi)
        .select(["NDVI"])
        .median()
    )


def s2_ndvi_trend(geom: ee.Geometry, years: list[int], *, north: bool) -> ee.Image:
    imgs = []
    for year in years:
        ndvi = s2_peak_ndvi(geom, year, north=north).rename("ndvi")
        yr = ee.Image.constant(year).toFloat().rename("year")
        imgs.append(ee.Image.cat([yr, ndvi]))
    fit = ee.ImageCollection(imgs).reduce(ee.Reducer.linearFit())
    return fit.select("scale").rename("ndvi_trend")


def s1_composite(geom: ee.Geometry, year: int) -> ee.Image:
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
    return med.addBands(med.select("VV").subtract(med.select("VH")).rename("VVVH"))


def build_year_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """Combined S1+S2 composite for a single year."""
    return s2_composite(geom, year).addBands(s1_composite(geom, year))


def submit_composite_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    """Submit one export task per year in settings.period_years."""
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
    