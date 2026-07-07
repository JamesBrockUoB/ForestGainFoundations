from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets

TREE_THRESHOLD = 50  # % canopy cover

# Maps each period's boundary years to the Datasets attribute holding that
# year's canopy-cover mosaic. Extend this (and Datasets) once the 2020->2024
# cover asset is uploaded — do not point 2024 at the 2020 mosaic as a
# placeholder, since that would silently produce zero gain everywhere for p2
# rather than failing loudly.
_COVER_ATTR_BY_YEAR = {
    2017: "dt_cover_2017",
    2020: "dt_cover_2020",
    2024: "dt_cover_2024",
}


def _cover_image(ds: Datasets, year: int) -> ee.Image:
    attr = _COVER_ATTR_BY_YEAR.get(year)
    if attr is None:
        raise ValueError(
            f"No canopy-cover source configured for year {year}. "
            f"Known years: {sorted(_COVER_ATTR_BY_YEAR)}"
        )

    image = getattr(ds, attr)
    if image is None:
        raise RuntimeError(
            f"Canopy-cover asset for year {year} (Datasets.{attr}) is not "
            f"available yet — cannot run build_gain_layer for period="
            f"{settings.period} ({settings.year_start}->{settings.year_end}) "
            f"until it's uploaded and wired into datasets/registry.py."
        )
    return image


def build_gain_layer(
    geom: ee.Geometry,
    ds: Datasets,
) -> tuple[ee.Image, ee.Image]:
    """
    Forest gain defined as:
        canopy <50% in settings.year_start
        canopy >50% in settings.year_end

    using the deadtrees canopy cover mosaics for the active period.
    """
    cover_start = _cover_image(ds, settings.year_start).clip(geom)
    cover_end = _cover_image(ds, settings.year_end).clip(geom)

    forest_start = cover_start.gt(TREE_THRESHOLD)
    forest_end = cover_end.gt(TREE_THRESHOLD)

    gain = forest_start.Not().And(forest_end)

    clean = gain.updateMask(gain).focal_max(1).focal_min(1)

    validated = clean.And(ds.esa_trees.clip(geom))

    return (
        validated,
        validated.unmask(0).rename("gain"),
    )
