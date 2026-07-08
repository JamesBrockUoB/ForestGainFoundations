from __future__ import annotations

import ee
from config import settings
from export.composites import s1_availability, s2_availability, s2_peak
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer

NO_GAIN_SENTINEL = -9999.0

# canopy_mean only included when the active period's end year matches the
# canopy-height snapshot (see settings.canopy_height_available) — tracked
# when available, but see tile_filter.evaluate_cheap_stats for why it's
# reporting-only, not a rejection gate.
CHEAP_BAND_NAMES = ["gain_frac", "ndvi_delta"]
if settings.canopy_height_available:
    CHEAP_BAND_NAMES = CHEAP_BAND_NAMES + ["canopy_mean"]

# Per-year S2 + S1 availability bands for the active period (settings.period,
# fixed for the process lifetime via the PERIOD env var).
#   PERIOD=p1 -> s2_2017, s2_2018, s2_2019, s2_2020, s1_2017, ..., s1_2020
#   PERIOD=p2 -> s2_2020, ..., s2_2024, s1_2020, ..., s1_2024
S2_BAND_NAMES = [f"s2_{y}" for y in settings.period_years]
S1_BAND_NAMES = [f"s1_{y}" for y in settings.period_years]
IMAGERY_BAND_NAMES = S2_BAND_NAMES + S1_BAND_NAMES


def build_cheap_stats_image(geom: ee.Geometry, ds: Datasets) -> ee.Image:
    gain_validated, gain_binary = build_gain_layer(geom, ds)
    gm = gain_validated.selfMask()

    ndvi_delta = (
        s2_peak(geom, settings.year_end)
        .select("NDVI")
        .subtract(s2_peak(geom, settings.year_start).select("NDVI"))
        .updateMask(gm)
        .rename("ndvi_delta")
    )

    bands = [gain_binary.rename("gain_frac"), ndvi_delta]
    if settings.canopy_height_available:
        bands.append(ds.meta_ch.updateMask(gm).rename("canopy_mean"))

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


def aggregate_to_tile_grid(
    stats: ee.Image,
    *,
    origin_x: float,
    origin_y: float,
    zero_fill_bands: list[str],
    no_gain_bands: list[str] | None = None,
) -> ee.Image:
    sz = settings.tile_size_m
    crs_transform = [sz, 0, origin_x, 0, -sz, origin_y]
    no_gain_bands = no_gain_bands or []

    stats = stats.setDefaultProjection(crs=settings.crs_wkt, scale=settings.scale)

    aggregated = stats.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=False,
        maxPixels=int((sz / settings.scale) ** 2) + 1,
    ).reproject(crs=settings.crs_wkt, crsTransform=crs_transform)

    pieces = [aggregated.select(zero_fill_bands).unmask(0)]
    if no_gain_bands:
        pieces.append(aggregated.select(no_gain_bands).unmask(NO_GAIN_SENTINEL))

    return ee.Image.cat(pieces).select(zero_fill_bands + no_gain_bands)


def _fetch_grid(
    tile_grid: ee.Image,
    band_names: list[str],
    *,
    origin_x: float,
    origin_y: float,
    n_cols: int,
    n_rows: int,
    tile_scale: int = 4,
) -> dict[str, list]:
    region = ee.Geometry.Rectangle(
        [
            origin_x,
            origin_y - n_rows * settings.tile_size_m,
            origin_x + n_cols * settings.tile_size_m,
            origin_y,
        ],
        proj=ee.Projection(settings.crs_wkt),
        geodesic=False,
    )

    reducer = ee.Reducer.toList().forEachBand(tile_grid)

    result = tile_grid.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=settings.tile_size_m,
        maxPixels=1e13,
        tileScale=tile_scale,
    ).getInfo()

    out = {}
    for band in band_names:
        if band not in result:
            raise RuntimeError(
                f"reduceRegion result missing band '{band}'; "
                f"available keys={list(result.keys())}"
            )
        flat = result[band]
        if len(flat) != n_cols * n_rows:
            raise RuntimeError(
                f"band '{band}' length mismatch: expected {n_cols*n_rows} "
                f"values ({n_cols}x{n_rows}), got {len(flat)}"
            )
        out[band] = [flat[r * n_cols : (r + 1) * n_cols] for r in range(n_rows)]
    return out


def fetch_cheap_stats(
    aoi_geom: ee.Geometry,
    ds: Datasets,
    *,
    origin_x: float,
    origin_y: float,
    n_cols: int,
    n_rows: int,
) -> dict[str, list]:
    stats = build_cheap_stats_image(aoi_geom, ds)

    no_gain_bands = ["ndvi_delta"]
    if settings.canopy_height_available:
        no_gain_bands.append("canopy_mean")

    tile_grid = aggregate_to_tile_grid(
        stats,
        origin_x=origin_x,
        origin_y=origin_y,
        zero_fill_bands=["gain_frac"],
        no_gain_bands=no_gain_bands,
    )
    return _fetch_grid(
        tile_grid,
        CHEAP_BAND_NAMES,
        origin_x=origin_x,
        origin_y=origin_y,
        n_cols=n_cols,
        n_rows=n_rows,
        tile_scale=2,
    )


def fetch_imagery_stats(
    aoi_geom: ee.Geometry,
    *,
    origin_x: float,
    origin_y: float,
    n_cols: int,
    n_rows: int,
) -> dict[str, list]:
    stats = build_imagery_stats_image(aoi_geom)
    tile_grid = aggregate_to_tile_grid(
        stats,
        origin_x=origin_x,
        origin_y=origin_y,
        zero_fill_bands=IMAGERY_BAND_NAMES,
    )
    return _fetch_grid(
        tile_grid,
        IMAGERY_BAND_NAMES,
        origin_x=origin_x,
        origin_y=origin_y,
        n_cols=n_cols,
        n_rows=n_rows,
        tile_scale=8,
    )
