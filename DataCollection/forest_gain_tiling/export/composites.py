from __future__ import annotations

import ee
from config import settings
from export.year_chunks import chunk_name, chunk_years, prefix_bands
from gee_datasets.registry import Datasets

NATIVE_10M_BANDS = ["B2", "B3", "B4", "B8"]
NATIVE_20M_BANDS = ["B5", "B6", "B7", "B8A", "B11", "B12"]

CLOUD_SCORE_PLUS_BAND = "cs_cdf"
SCENE_CLOUDY_PCT_MAX = 30

SNOW_NDSI_THRESH = getattr(settings, "snow_ndsi_thresh", 0.4)
SNOW_NIR_MIN = getattr(settings, "snow_nir_min", 1100)  # B8 DN, excludes water
SNOW_GREEN_MIN = getattr(settings, "snow_green_min", 1000)  # B3 DN


def hemisphere_from_tile(min_lat: float, max_lat: float) -> bool:
    """True if the tile's centroid is in the northern hemisphere.

    Only used for the leaf-on NDVI trend signal
    """
    return (min_lat + max_lat) / 2.0 >= 0


def _join_cloud_score_plus(
    ic: ee.ImageCollection, geom, start: str, end: str
) -> ee.ImageCollection:
    """Link Cloud Score+ QA band onto each S2 image via
    ImageCollection.linkCollection (matched by system:index)."""
    DATASETS = Datasets()
    cs_col = (
        ee.ImageCollection(DATASETS.cloud_score_plus)
        .filterDate(start, end)
        .filterBounds(geom)
    )
    return ic.linkCollection(cs_col, [CLOUD_SCORE_PLUS_BAND])


def _snow_mask(img: ee.Image) -> ee.Image:
    """1 where the pixel looks like snow/ice: high NDSI (green vs SWIR),
    with NIR and green floors to exclude water and dark pixels."""
    green = img.select("B3")
    nir = img.select("B8")
    swir = img.select("B11")  # nearest at 20 m is fine for a binary mask
    ndsi = green.subtract(swir).divide(green.add(swir))
    return (
        ndsi.gt(SNOW_NDSI_THRESH)
        .And(nir.gt(SNOW_NIR_MIN))
        .And(green.gt(SNOW_GREEN_MIN))
    )


def _mask_clouds(
    img: ee.Image, threshold: float = settings.cloud_score_thresh
) -> ee.Image:
    return img.updateMask(img.select(CLOUD_SCORE_PLUS_BAND).gte(threshold))


def _mask_clouds_and_snow(
    img: ee.Image, threshold: float = settings.cloud_score_thresh
) -> ee.Image:
    """Clear (Cloud Score+) AND snow-free. Must run before any .select()
    that drops B3/B8/B11."""
    clear = img.select(CLOUD_SCORE_PLUS_BAND).gte(threshold).And(_snow_mask(img).Not())
    return img.updateMask(clear)


def _masked_s2(geom, start: str, end: str, *, snow: bool = True) -> ee.ImageCollection:
    """Single source of truth for S2 filtering + masking. All bands
    retained; callers select afterwards."""
    datasets = Datasets()
    ic = (
        ee.ImageCollection(datasets.sentinel_2)
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUDY_PCT_MAX))
    )
    ic = _join_cloud_score_plus(ic, geom, start, end)
    return ic.map(_mask_clouds_and_snow if snow else _mask_clouds)


def _add_ndvi(img: ee.Image) -> ee.Image:
    return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))


def _date_range(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year + 1}-01-01"


def leaf_on_window(year: int, *, north: bool) -> tuple[str, str]:
    """Leaf-on window: used only by s2_peak_ndvi / s2_ndvi_trend (the
    cheap viability signal), never for composites or validity."""
    if north:
        return f"{year}-05-01", f"{year}-09-30"
    return f"{year}-11-01", f"{year + 1}-03-31"


def s2_availability(geom, year: int, *, snow: bool = True) -> ee.Image:
    """Full-year per-pixel check: 1 where at least one clear (cloud-masked,
    and snow-masked if snow=True) S2 observation exists, else 0."""
    start, end = _date_range(year)
    ic = _masked_s2(geom, start, end, snow=snow).select(settings.s2_check_band)
    return ic.count().gt(0).unmask(0)


def combine_year_validity(per_year: list[ee.Image]) -> ee.Image:
    """Valid in EVERY year. This is the mask the export applies."""
    return ee.Image.cat(per_year).reduce(ee.Reducer.min()).rename("all_years_valid")


def s2_all_years_valid(geom, years) -> ee.Image:
    """Build the export's full_valid with this so the filter and export
    use identical masking."""
    return combine_year_validity([s2_availability(geom, y) for y in years])


def s1_observation_count(tile: ee.Feature, year: int) -> ee.Number:
    """Acquisition-level S1 observation count for one tile / AOI feature."""
    datasets = Datasets()
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection(datasets.sentinel_1)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(settings.s1_check_band)
    )
    return ic.filterBounds(tile.geometry()).size()


def s1_availability(tile: ee.Feature, year: int) -> ee.Number:
    """1 if the tile meets settings.min_s1_observations, else 0."""
    return s1_observation_count(tile, year).gte(settings.min_s1_observations).int()


def s2_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """Full-year median composite, cloud- and snow-masked."""
    start, end = _date_range(year)
    s2 = (
        _masked_s2(geom, start, end, snow=True)
        .select(NATIVE_10M_BANDS + NATIVE_20M_BANDS)
        .map(_upsample_20m_bands_to_10m)
    )
    return s2.median()


def valid_mask_from_composite(
    img: ee.Image, *, out_band_name: str = "s2_valid"
) -> ee.Image:
    """0/1 validity from a composite (1 where any band is unmasked).
    Pass an S2-only image so S1 can't make it look valid."""
    mask_any = img.mask().reduce(ee.Reducer.max()).rename(out_band_name)
    return mask_any.toFloat()


def _upsample_20m_bands_to_10m(img: ee.Image) -> ee.Image:
    native_10m = img.select(NATIVE_10M_BANDS)
    native_20m = img.select(NATIVE_20M_BANDS).resample("bilinear")
    return native_10m.addBands(native_20m).copyProperties(img, img.propertyNames())


def s2_peak_ndvi(
    geom: ee.Geometry, year: int, *, north: bool, snow: bool = False
) -> ee.Image:
    """Leaf-on NDVI. Cloud-masked only by default (cheap stage)."""
    start, end = leaf_on_window(year, north=north)
    return (
        _masked_s2(geom, start, end, snow=snow).map(_add_ndvi).select(["NDVI"]).median()
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
    datasets = Datasets()

    def _mask_edge(img):
        edge = img.lt(-30.0)
        return img.updateMask(img.mask().And(edge.Not()))

    med = (
        ee.ImageCollection(datasets.sentinel_1)
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


def build_year_s2_s1(geom: ee.Geometry, year: int) -> tuple[ee.Image, ee.Image]:
    """(s2_only, s1+s2) so S2 validity isn't contaminated by S1."""
    s2 = s2_composite(geom, year)
    return s2, s2.addBands(s1_composite(geom, year))


def build_year_composite(geom: ee.Geometry, year: int) -> ee.Image:
    """Combined S1+S2 composite for a single year."""
    return build_year_s2_s1(geom, year)[1]


def submit_composite_exports(
    geom: ee.Geometry,
    crs_transform: list[float],
    full_valid: ee.Image,
    tile_id: str,
) -> dict[str, ee.batch.Task]:
    """One task per chunk of settings.export_years_per_task years."""
    tasks: dict[str, ee.batch.Task] = {}

    for years in chunk_years(list(settings.years), settings.export_years_per_task):
        year_images = []
        for year in years:
            s2, combined = build_year_s2_s1(geom, year)
            image = combined.updateMask(full_valid).toFloat()

            # S2-only validity
            mask_img = valid_mask_from_composite(
                s2.updateMask(full_valid), out_band_name=f"s2_valid_{year}"
            ).toFloat()
            image = image.addBands(mask_img)
            year_images.append(prefix_bands(image, year) if len(years) > 1 else image)

        combined_img = ee.Image.cat(year_images)

        name = chunk_name("s1s2", years)
        key = f"composites/{name}"
        prefix = f"{tile_id}__composites__{name}"

        task = ee.batch.Export.image.toDrive(
            image=combined_img,
            description=prefix,
            folder=settings.drive_folder,
            fileNamePrefix=prefix,
            region=geom,
            crs=settings.crs_wkt,
            crsTransform=crs_transform,
            maxPixels=10_000_000_000_000,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
        )
        task.start()
        tasks[key] = task

    return tasks
