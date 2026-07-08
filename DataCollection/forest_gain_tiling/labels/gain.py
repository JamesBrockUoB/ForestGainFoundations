from __future__ import annotations

import ee
from config import settings
from gee_datasets.registry import Datasets

TREE_THRESHOLD = 50  # % canopy cover


def _cover_image(ds: Datasets, year: int) -> ee.Image:
    image = ds.dt_cover.get(year)
    if image is None:
        raise RuntimeError(
            f"Canopy-cover asset for year {year} is not available yet — "
            f"cannot run build_gain_layer for period={settings.period} "
            f"({settings.year_start}->{settings.year_end}) until it's "
            f"uploaded and wired into gee_datasets/registry.py."
        )
    return image


def build_gain_layer(
    geom: ee.Geometry,
    ds: Datasets,
) -> tuple[ee.Image, ee.Image]:
    """
    Forest gain defined as:
        canopy <50% in settings.year_start (not-yet-forest at period start)
        canopy >50% in settings.year_end (strict — no tolerance)
        AND, once a pixel is first observed as forest in some intervening
        year, it does not revert to non-forest afterward — tolerating up
        to settings.gain_sustain_dropout_tolerance such reversions among
        strictly-intermediate years only.

    Crucially, non-forest readings *before* a pixel's first forest
    crossing are not reversions and never consume tolerance budget — a
    pixel that stays below threshold for several years and then grows in
    late is a valid gain pixel, not noise. Only a *later* drop back below
    threshold, after having already read forest, counts as a dip.
    """
    cover_start = _cover_image(ds, settings.year_start).clip(geom)
    cover_end = _cover_image(ds, settings.year_end).clip(geom)

    forest_start = cover_start.gt(TREE_THRESHOLD)
    forest_end = cover_end.gt(TREE_THRESHOLD)

    gain_candidate = forest_start.Not().And(forest_end)

    # Strictly-intermediate years, chronological order matters here —
    # the cumulative "have we ever been forest yet" scan depends on it.
    intervening_years = sorted(
        y
        for y in settings.period_years
        if y not in (settings.year_start, settings.year_end)
    )

    ever_forest_before = ee.Image.constant(0)  # boolean-as-int, starts false
    non_forest_hits = ee.Image.constant(0)
    for year in intervening_years:
        forest_y = _cover_image(ds, year).clip(geom).gt(TREE_THRESHOLD)
        reverted = ever_forest_before.And(forest_y.Not())
        non_forest_hits = non_forest_hits.add(reverted)
        ever_forest_before = ever_forest_before.Or(forest_y)

    sustained = forest_end.And(
        non_forest_hits.lte(settings.gain_sustain_dropout_tolerance)
    )

    gain = gain_candidate.And(sustained)

    clean = gain.updateMask(gain).focal_max(1).focal_min(1)
    validated = clean.And(ds.esa_trees.clip(geom))

    return (
        validated,
        validated.unmask(0).rename("gain"),
    )
