from __future__ import annotations

import ee

from datasets.registry import Datasets


def build_gain_layer(geom: ee.Geometry, ds: Datasets) -> tuple[ee.Image, ee.Image]:
    ones = ee.List.repeat(1, ds.tree_classes.length())

    tree_2015 = ds.glulc_2015.clip(geom).remap(ds.tree_classes, ones, 0)
    tree_2020 = ds.glulc_2020.clip(geom).remap(ds.tree_classes, ones, 0)

    gain = tree_2020.And(tree_2015.Not())
    clean = gain.updateMask(gain).focal_max(1).focal_min(1)
    validated = clean.And(ds.esa_trees.clip(geom))

    return validated, validated.unmask(0).rename("gain")
