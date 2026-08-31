from __future__ import annotations

import ee
from config import settings
from export.composites import (
    hemisphere_from_tile,
    s1_availability,
    s2_availability,
    s2_ndvi_trend,
)
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer

NO_GAIN_SENTINEL = -9999.0

CHEAP_BAND_NAMES = [
    "gain_frac",
    "ndvi_trend",
]

if settings.period == "p1":
    CHEAP_BAND_NAMES.append("pseudo_gain_frac")

S2_BAND_NAMES = [f"s2_{y}" for y in settings.period_years]
S1_BAND_NAMES = [f"s1_{y}" for y in settings.period_years]
IMAGERY_BAND_NAMES = S2_BAND_NAMES + S1_BAND_NAMES


def split_by_hemisphere(tiles: list[dict]) -> tuple[list[dict], list[dict]]:
    north, south = [], []
    for t in tiles:
        if hemisphere_from_tile(t["min_lat"], t["max_lat"]):
            north.append(t)
        else:
            south.append(t)
    return north, south


def build_cheap_stats_image(
    geom: ee.Geometry,
    ds: Datasets,
    *,
    north: bool,
) -> ee.Image:
    gain_validated, gain_binary, _ = build_gain_layer(geom, ds)
    gain_mask = gain_validated.selfMask()

    ndvi_trend = (
        s2_ndvi_trend(geom, settings.period_years, north=north)
        .updateMask(gain_mask)
        .rename("ndvi_trend")
    )

    forty = ds.forty.clip(geom)

    forty_valid = (
        ee.Image.cat(
            [
                forty.select("TreeCropsAndAgroforestry"),
                forty.select("NaturallyRegeneratingForest"),
                forty.select("PlantationForest"),
                forty.select("PlantedForest"),
            ]
        )
        .reduce(ee.Reducer.sum())
        .gt(0)
    )

    pseudo_gain = gain_mask.And(forty_valid).rename("pseudo_gain")

    bands = [
        gain_binary.rename("gain_frac"),
        ndvi_trend,
    ]

    if settings.period == "p1":
        bands.append(pseudo_gain.rename("pseudo_gain_frac"))

    return ee.Image.cat(bands).clip(geom)


def build_imagery_stats_image(geom: ee.Geometry) -> ee.Image:
    """Full-year, Cloud Score+ masked S2 availability"""
    bands = [
        s2_availability(geom, year).rename(f"s2_{year}")
        for year in settings.period_years
    ]
    return ee.Image.cat(bands).clip(geom)


def tiles_to_feature_collection(tiles: list[dict]) -> ee.FeatureCollection:
    features = []
    for t in tiles:
        geom = ee.Geometry.Rectangle(
            [t["x_min_m"], t["y_min_m"], t["x_max_m"], t["y_max_m"]],
            proj=ee.Projection(settings.crs_wkt),
            geodesic=False,
        )
        features.append(ee.Feature(geom, {"tile_id": t["tile_id"]}))
    return ee.FeatureCollection(features)


def _reduce_tiles(
    stats: ee.Image,
    tiles: list[dict],
    band_names: list[str],
    *,
    tile_scale: int = 4,
) -> dict[str, dict[str, float]]:
    fc = tiles_to_feature_collection(tiles)
    reduced = stats.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=settings.scale,
        tileScale=tile_scale,
    )
    result = reduced.getInfo()
    out = {}
    for feature in result["features"]:
        props = feature["properties"]
        tile_id = props["tile_id"]
        out[tile_id] = {band: props.get(band) for band in band_names}
    return out


def fetch_cheap_stats(
    tiles: list[dict],
    ds: Datasets,
) -> dict[str, dict[str, float]]:
    """Split by hemisphere — NDVI trend is leaf-on."""
    north_tiles, south_tiles = split_by_hemisphere(tiles)
    out: dict[str, dict[str, float]] = {}

    for group, north in ((north_tiles, True), (south_tiles, False)):
        if not group:
            continue
        geom = tiles_to_feature_collection(group).geometry()
        stats = build_cheap_stats_image(geom, ds, north=north)
        out.update(_reduce_tiles(stats, group, CHEAP_BAND_NAMES, tile_scale=2))

    return out


def fetch_imagery_stats(tiles: list[dict]) -> dict[str, dict[str, float]]:
    """
    S2: full-year, Cloud Score+ masked pixel-level availability
    S1: acquisition-level availability per tile/year
    """
    fc = tiles_to_feature_collection(tiles)

    bands = [
        s2_availability(fc, year).rename(f"s2_{year}")
        for year in settings.period_years
    ]
    stats = ee.Image.cat(bands)

    out = _reduce_tiles(stats, tiles, S2_BAND_NAMES, tile_scale=8)
    for t in tiles:
        out.setdefault(t["tile_id"], {})

    for tile in tiles:
        tile_id = tile["tile_id"]
        tile_geom = ee.Geometry.Rectangle(
            [tile["x_min_m"], tile["y_min_m"], tile["x_max_m"], tile["y_max_m"]],
            proj=ee.Projection(settings.crs_wkt),
            geodesic=False,
        )
        tile_feature = ee.Feature(tile_geom, {"tile_id": tile_id})

        for year in settings.period_years:
            available = s1_availability(tile_feature, year)
            out[tile_id][f"s1_{year}"] = available.getInfo()

    return out
