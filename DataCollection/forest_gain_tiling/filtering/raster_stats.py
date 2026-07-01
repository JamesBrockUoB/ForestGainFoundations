from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets
from export.composites import s2_availability, s2_peak
from labels.gain import build_gain_layer

NO_GAIN_SENTINEL = -9999.0

BAND_NAMES = [
    "gain_frac",
    "ndvi_delta",
    "canopy_mean",
    "s2_2017",
    "s2_2020",
    "s2_2025",
]


def build_tile_stats_image(geom: ee.Geometry, ds: Datasets) -> ee.Image:
    """
    Build a multi-band image at native 10m resolution where, after
    reduceResolution + reproject to the tile grid, each output pixel
    corresponds to one tile and encodes:

      gain_frac    - fraction of tile pixels with validated tree gain
      ndvi_delta   - mean NDVI(2020) - NDVI(2017) within gain pixels,
                      or NO_GAIN_SENTINEL if the tile has no gain pixels
      canopy_mean  - mean canopy height (m) within gain pixels,
                      or NO_GAIN_SENTINEL if the tile has no gain pixels
      s2_2017      - fraction of tile pixels with valid S2 obs in 2017
      s2_2020      - fraction of tile pixels with valid S2 obs in 2020
      s2_2025      - fraction of tile pixels with valid S2 obs in 2025
    """
    gain_validated, gain_binary = build_gain_layer(geom, ds)
    gm = gain_validated.selfMask()

    ndvi_delta = (
        s2_peak(geom, 2020, ds)
        .select("NDVI")
        .subtract(s2_peak(geom, 2017, ds).select("NDVI"))
        .updateMask(gm)
        .rename("ndvi_delta")
    )

    canopy = ds.meta_ch.updateMask(gm).rename("canopy_mean")

    s2_2017 = s2_availability(geom, 2017).rename("s2_2017")
    s2_2020 = s2_availability(geom, 2020).rename("s2_2020")
    s2_2025 = s2_availability(geom, 2025).rename("s2_2025")

    return ee.Image.cat(
        [
            gain_binary.rename("gain_frac"),
            ndvi_delta,
            canopy,
            s2_2017,
            s2_2020,
            s2_2025,
        ]
    ).clip(geom)


def aggregate_to_tile_grid(
    stats: ee.Image,
    *,
    origin_x: float,
    origin_y: float,
) -> ee.Image:
    """
    Aggregate a 10m-resolution multi-band image to the tile grid (one
    output pixel per tile), using a crsTransform anchored at
    (origin_x, origin_y) so output pixel boundaries align exactly with
    tile boundaries.

    origin_x/origin_y must be the x_min_m / y_max_m of the
    top-left-most tile in the region being sampled, i.e. a point that
    lies exactly on the global tile grid.

    A default projection is set on the input image at settings.scale
    (10m) before reduceResolution, as EE requires a fixed native
    resolution to aggregate from. Without this, reduceResolution raises
    a projection error on composite images built from multiple sources.

    ndvi_delta and canopy_mean are masked wherever a tile has zero
    gain pixels (reduceResolution excludes masked inputs from the
    mean and leaves the output pixel masked); these are unmasked to
    NO_GAIN_SENTINEL afterwards so sampleRectangle returns a value for
    every tile.
    """
    sz = settings.tile_size_m
    crs_transform = [sz, 0, origin_x, 0, -sz, origin_y]

    # Set default projection so reduceResolution knows the native resolution
    stats = stats.setDefaultProjection(
        crs=settings.crs_wkt,
        scale=settings.scale,
    )

    aggregated = stats.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=False,
        maxPixels=int((sz / settings.scale) ** 2) + 1,
    ).reproject(crs=settings.crs_wkt, crsTransform=crs_transform)

    no_gain_bands = aggregated.select(["ndvi_delta", "canopy_mean"]).unmask(
        NO_GAIN_SENTINEL
    )
    other_bands = aggregated.select(
        ["gain_frac", "s2_2017", "s2_2020", "s2_2025"]
    ).unmask(0)

    return ee.Image.cat([other_bands, no_gain_bands]).select(BAND_NAMES)


def fetch_tile_stats(
    aoi_geom: ee.Geometry,
    ds: Datasets,
    *,
    origin_x: float,
    origin_y: float,
    n_cols: int,
    n_rows: int,
) -> dict[str, list]:

    stats = build_tile_stats_image(aoi_geom, ds)
    tile_grid = aggregate_to_tile_grid(stats, origin_x=origin_x, origin_y=origin_y)

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

    result = tile_grid.reduceRegion(
        reducer=ee.Reducer.toList().repeat(len(BAND_NAMES)),
        geometry=region,
        scale=settings.tile_size_m,
        maxPixels=1e13,
        tileScale=4,
    ).getInfo()

    out = {}

    for i, band in enumerate(BAND_NAMES):
        flat = result[band]
        # reshape into (n_rows, n_cols)
        out[band] = [flat[r * n_cols : (r + 1) * n_cols] for r in range(n_rows)]

    return out
