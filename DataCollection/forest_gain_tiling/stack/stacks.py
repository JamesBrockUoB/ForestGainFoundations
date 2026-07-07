from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets
from export.composites import build_timestep_stack, s2_availability
from labels.pseudo import build_pseudo_labels


def build_full_valid(geom: ee.Geometry) -> ee.Image:
    return (
        s2_availability(geom, settings.year_start)
        .And(s2_availability(geom, settings.year_end))
        .selfMask()
        .rename("valid")
    )


def build_full_stack(
    geom: ee.Geometry,
    gain_validated: ee.Image,
    full_valid: ee.Image,
    ds: Datasets,
) -> ee.Image:
    fabdem = (
        ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
        .filterBounds(geom)
        .mosaic()
        .clip(geom)
    )
    slope = ee.Terrain.slope(fabdem)
    gain_height = ds.meta_ch.updateMask(gain_validated.selfMask()).rename(
        "canopy_gain_height"
    )

    stack = (
        build_timestep_stack(geom, settings.year_start, "T0", ds)
        .addBands(build_timestep_stack(geom, settings.year_end, "T1", ds))
        .addBands(fabdem.rename("DEM"))
        .addBands(slope.rename("slope"))
        .addBands(gain_height)
        .addBands(gain_validated.unmask(0).rename("gain_mask"))
    )

    # Pseudo-labels (ForTy) are a fixed 2020 snapshot — only valid for
    # periods ending in 2020. p2 exports omit these bands entirely rather
    # than including a mislabeled or flagged approximation.
    if settings.pseudo_labels_available:
        stack = stack.addBands(build_pseudo_labels(geom, gain_validated, ds))

    return stack.updateMask(full_valid).toFloat()
