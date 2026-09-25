from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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

S2_BAND_NAMES = [f"s2_{y}" for y in settings.years]
S1_BAND_NAMES = [f"s1_{y}" for y in settings.years]
IMAGERY_BAND_NAMES = S2_BAND_NAMES + S1_BAND_NAMES


def split_by_hemisphere(tiles: list[dict]) -> tuple[list[dict], list[dict]]:
    north, south = [], []
    for t in tiles:
        if hemisphere_from_tile(t["min_lat"], t["max_lat"]):
            north.append(t)
        else:
            south.append(t)
    return north, south


def check_tessera_coverage(tiles, logger=None):
    from geotessera import GeoTessera

    gt = GeoTessera()

    def check_one(tile):
        bbox = (tile["min_lon"], tile["min_lat"], tile["max_lon"], tile["max_lat"])
        missing_years = []
        for year in settings.years:
            blocks = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
            if not blocks:
                missing_years.append(year)
        return tile["tile_id"], missing_years

    covered_tiles = []
    missing_by_tile = {}
    tiles_by_id = {t["tile_id"]: t for t in tiles}

    with ThreadPoolExecutor(max_workers=8) as ex:
        for tile_id, missing_years in ex.map(check_one, tiles):
            if missing_years:
                missing_by_tile[tile_id] = missing_years
                if logger:
                    logger.debug(
                        f"TESSERA coverage missing | tile={tile_id} | years={missing_years}"
                    )
            else:
                covered_tiles.append(tiles_by_id[tile_id])

    return covered_tiles, missing_by_tile


def build_cheap_stats_image(
    geom: ee.Geometry,
    ds: Datasets,
    *,
    north: bool,
) -> ee.Image:
    gain_validated, gain_binary, _ = build_gain_layer(geom, ds)
    gain_mask = gain_validated.selfMask()

    ndvi_trend = (
        s2_ndvi_trend(geom, settings.years, north=north)
        .updateMask(gain_mask)
        .rename("ndvi_trend")
    )

    bands = [
        gain_binary.rename("gain_frac"),
        ndvi_trend,
    ]

    return ee.Image.cat(bands).clip(geom)


def build_imagery_stats_image(geom: ee.Geometry) -> ee.Image:
    """Full-year, Cloud Score+ masked S2 availability"""
    bands = [
        s2_availability(geom, year).rename(f"s2_{year}") for year in settings.years
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
    S2: full-year, Cloud Score+ masked pixel-level availability — one
        reduceRegions call for all tiles.
    S1: acquisition-level availability via composites.s1_availability,
        mapped server-side over the tile FeatureCollection and fetched
        with one getInfo() per year — instead of one getInfo() per
        tile per year.
    """
    fc = tiles_to_feature_collection(tiles)

    tile_images = []

    for tile in tiles:
        geom = ee.Geometry.Rectangle(
            [
                tile["x_min_m"],
                tile["y_min_m"],
                tile["x_max_m"],
                tile["y_max_m"],
            ],
            proj=ee.Projection(settings.crs_wkt),
            geodesic=False,
        )

        tile_image = ee.Image.cat(
            [
                s2_availability(geom, year).rename(f"s2_{year}")
                for year in settings.years
            ]
        ).clip(geom)

        tile_images.append(tile_image)

    stats = ee.ImageCollection(tile_images).mosaic()

    out = _reduce_tiles(
        stats,
        tiles,
        S2_BAND_NAMES,
        tile_scale=8,
    )

    def add_s1_stats(feature):
        for year in settings.years:
            feature = feature.set(
                f"s1_{year}",
                s1_availability(feature, year),
            )
        return feature

    s1_info = fc.map(add_s1_stats).getInfo()

    for feature in s1_info["features"]:
        props = feature["properties"]
        tile_id = props["tile_id"]

        out.setdefault(tile_id, {})

        for year in settings.years:
            out[tile_id][f"s1_{year}"] = props.get(f"s1_{year}")

    return out
