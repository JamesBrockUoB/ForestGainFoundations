from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets


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
    if year == 2016:
        return "2015-01-01", "2016-12-31"
    return f"{year}-01-01", f"{year}-12-31"


def s2_availability(geom: ee.Geometry, year: int) -> ee.Image:
    start, end = _date_range(year)
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .map(_mask_s2_scl)
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    )
    return ic.map(lambda i: i.mask().reduce(ee.Reducer.min())).reduce(ee.Reducer.max())


def s2_coverage_frac(geom: ee.Geometry, year: int) -> ee.Number:
    """Fraction of pixels in geom with at least one valid S2 observation in `year`."""
    valid = s2_availability(geom, year)
    stats = valid.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=settings.scale,
        crs=settings.crs,
        maxPixels=1_000_000_000,
    )
    return ee.Number(ee.Algorithms.If(stats.get("valid"), stats.get("valid"), 0))


def s2_composite(geom: ee.Geometry, year: int) -> ee.Image:  # noqa: ARG001
    start, end = _date_range(year)
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "NDVI", "EVI"]
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(_add_indices)
    )
    fallback = (
        ee.Image.constant([0] * 9)
        .rename([b + "_p25" for b in bands])
        .updateMask(ee.Image.constant(0))
    )
    reduced = ee.Image(
        ee.Algorithms.If(
            ic.size().eq(0), fallback, ic.reduce(ee.Reducer.percentile([25]))
        )
    )
    return reduced.select([b + "_p25" for b in bands], bands)


def s2_peak(geom: ee.Geometry, year: int, ds: Datasets) -> ee.Image:  # noqa: ARG001
    centroid = ee.Geometry(geom).centroid(maxError=1)
    north = ee.Number(centroid.coordinates().get(1)).gt(0)

    if year == 2016:
        start = ee.String(ee.Algorithms.If(north, "2015-05-01", "2015-11-01"))
        end = ee.String(ee.Algorithms.If(north, "2016-09-30", "2017-03-31"))
    else:
        start = ee.String(ee.Algorithms.If(north, f"{year}-05-01", f"{year}-11-01"))
        end = ee.String(ee.Algorithms.If(north, f"{year}-09-30", f"{year+1}-03-31"))

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(_add_indices)
        .select(["NDVI", "EVI"])
        .median()
    )


def s1_composite(geom: ee.Geometry, year: int) -> ee.Image:  # noqa: ARG001
    med = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .median()
    )
    return med.addBands(med.select("VV").divide(med.select("VH")).rename("VVVH"))


def dw_composite(geom: ee.Geometry, year: int, ds: Datasets) -> ee.Image:
    return (
        ds.dw.filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .select(["trees", "crops", "built"])
        .median()
    )


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
    "DW_trees",
    "DW_crops",
    "DW_built",
]


def build_timestep_stack(
    geom: ee.Geometry, year: int, prefix: str, ds: Datasets
) -> ee.Image:
    band_names = [f"{prefix}_{b}" for b in _BAND_SUFFIXES]
    return (
        s2_composite(geom, year)
        .addBands(s1_composite(geom, year))
        .addBands(dw_composite(geom, year, ds))
        .rename(band_names)
    )
