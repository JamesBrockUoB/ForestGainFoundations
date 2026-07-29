from __future__ import annotations

import ee
from config import settings
from gee_datasets.registry import Datasets


def _cover_image(
    ds: Datasets,
    year: int,
) -> ee.Image:

    return ds.get_dt_cover(year)


def build_gain_layer(
    geom: ee.Geometry,
    ds: Datasets,
) -> tuple[ee.Image, ee.Image, ee.Image]:
    """
    Forest gain:

    1. Start:
        canopy <20%

    2. End:
        canopy >50%

    3. Intermediate years:
        monotonic forest establishment,
        allowing configured dropout tolerance

    Returns:
        validated gain mask
        binary gain raster
        final canopy confidence
    """

    cover_start = _cover_image(ds, settings.year_start).clip(geom)

    cover_end = _cover_image(ds, settings.year_end).clip(geom)

    forest_start = cover_start.lt(settings.non_tree_threshold_frac)

    forest_end = cover_end.gt(settings.min_tree_threshold_frac)

    gain_candidate = forest_start.And(forest_end)

    intervening_years = sorted(
        y
        for y in settings.period_years
        if y
        not in (
            settings.year_start,
            settings.year_end,
        )
    )

    ever_forest = ee.Image.constant(0)

    dropout_count = ee.Image.constant(0)

    for year in intervening_years:

        cover = _cover_image(
            ds,
            year,
        ).clip(geom)

        forest = cover.gt(settings.min_tree_threshold_frac)
        dropout = ever_forest.And(forest.Not())
        dropout_count = dropout_count.add(dropout)

        ever_forest = ever_forest.Or(forest)

    sustained = forest_end.And(
        dropout_count.lte(settings.gain_sustain_dropout_tolerance)
    )

    gain = gain_candidate.And(sustained)

    validated = gain.selfMask()

    forest_confidence = cover_end.updateMask(validated).rename("forest_confidence")

    return (
        validated.rename("validated_gain"),
        validated.unmask(0).rename("gain"),
        forest_confidence,
    )
