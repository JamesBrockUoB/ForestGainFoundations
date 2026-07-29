from __future__ import annotations

import ee
from config import settings
from export.composites import s1_availability, s2_availability, s2_peak_ndvi
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer

NO_GAIN_SENTINEL = -9999.0

CHEAP_BAND_NAMES = [
    "gain_frac",
    "ndvi_delta",
]

if settings.period == "p1":
    CHEAP_BAND_NAMES.append("pseudo_gain_frac")

# Per-year S2 + S1 availability bands for the active period (settings.period,
# fixed for the process lifetime via the PERIOD env var).
#   PERIOD=p1 -> s2_2017, s2_2018, s2_2019, s2_2020, s1_2017, ..., s1_2020
#   PERIOD=p2 -> s2_2020, ..., s2_2024, s1_2020, ..., s1_2024
S2_BAND_NAMES = [f"s2_{y}" for y in settings.period_years]
S1_BAND_NAMES = [f"s1_{y}" for y in settings.period_years]
IMAGERY_BAND_NAMES = S2_BAND_NAMES + S1_BAND_NAMES


def build_cheap_stats_image(
    geom: ee.Geometry,
    ds: Datasets,
) -> ee.Image:

    gain_validated, gain_binary, _ = build_gain_layer(geom, ds)

    gain_mask = gain_validated.selfMask()

    ndvi_delta = (
        s2_peak_ndvi(geom, settings.year_end)
        .subtract(s2_peak_ndvi(geom, settings.year_start))
        .updateMask(gain_mask)
        .rename("ndvi_delta")
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
        ndvi_delta,
    ]

    # Only needed for P1
    if settings.period == "p1":
        bands.append(pseudo_gain.rename("pseudo_gain_frac"))

    return ee.Image.cat(bands).clip(geom)


def build_imagery_stats_image(geom: ee.Geometry) -> ee.Image:
    """
    Per-year S2 + S1 valid-pixel coverage fraction bands for every calendar
    year in the active period (settings.period_years). Band naming:
    s2_<year>, s1_<year>.
    """
    bands = []
    for year in settings.period_years:
        bands.append(s2_availability(geom, year).rename(f"s2_{year}"))
        bands.append(s1_availability(geom, year).rename(f"s1_{year}"))
    return ee.Image.cat(bands).clip(geom)


def tiles_to_feature_collection(tiles: list[dict]) -> ee.FeatureCollection:
    features = []

    for t in tiles:
        geom = ee.Geometry.Rectangle(
            [
                t["x_min_m"],
                t["y_min_m"],
                t["x_max_m"],
                t["y_max_m"],
            ],
            proj=ee.Projection(settings.crs_wkt),
            geodesic=False,
        )

        features.append(
            ee.Feature(
                geom,
                {
                    "tile_id": t["tile_id"],
                },
            )
        )

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
    geom = tiles_to_feature_collection(tiles).geometry()

    stats = build_cheap_stats_image(geom, ds)

    return _reduce_tiles(
        stats,
        tiles,
        CHEAP_BAND_NAMES,
        tile_scale=2,
    )


def fetch_imagery_stats(
    tiles: list[dict],
) -> dict[str, dict[str, float]]:
    geom = tiles_to_feature_collection(tiles).geometry()

    stats = build_imagery_stats_image(geom)

    return _reduce_tiles(
        stats,
        tiles,
        IMAGERY_BAND_NAMES,
        tile_scale=8,
    )
