from __future__ import annotations

import ee
from datasets.registry import Datasets

TREE_THRESHOLD = 50  # % canopy cover


def build_gain_layer(
    geom: ee.Geometry,
    ds: Datasets,
) -> tuple[ee.Image, ee.Image]:
    """
    Forest gain defined as:
        canopy <50% in 2017
        canopy >50% in 2020

    using the deadtrees canopy cover mosaics.
    """

    cover_2017 = ds.dt_cover_2017.clip(geom)
    cover_2020 = ds.dt_cover_2020.clip(geom)

    forest_2017 = cover_2017.gt(TREE_THRESHOLD)
    forest_2020 = cover_2020.gt(TREE_THRESHOLD)

    gain = forest_2017.Not().And(forest_2020)

    clean = gain.updateMask(gain).focal_max(1).focal_min(1)

    validated = clean.And(ds.esa_trees.clip(geom))

    return (
        validated,
        validated.unmask(0).rename("gain"),
    )
