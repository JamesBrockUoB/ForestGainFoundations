from __future__ import annotations

import ee
from datasets.registry import Datasets
from export.composites import build_timestep_stack, s2_availability
from labels.pseudo import build_pseudo_labels


def build_full_valid(geom: ee.Geometry) -> ee.Image:
    return (
        s2_availability(geom, 2017)
        .And(s2_availability(geom, 2020))
        .And(s2_availability(geom, 2025))
        .selfMask()
        .rename("valid")
    )


def build_full_stack(
    tile: dict,
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

    return (
        build_timestep_stack(geom, 2017, "T0", ds)
        .addBands(build_timestep_stack(geom, 2020, "T1", ds))
        .addBands(build_timestep_stack(geom, 2025, "T2", ds))
        .addBands(fabdem.rename("DEM"))
        .addBands(slope.rename("slope"))
        .addBands(gain_height)
        .addBands(ds.jrc.rename("jrc_forest_type"))
        .addBands(ds.nat_forest.rename("natural_forest_prob"))
        .addBands(gain_validated.unmask(0).rename("gain_mask"))
        .addBands(build_pseudo_labels(geom, gain_validated, slope, ds))
        .updateMask(full_valid)
        .toFloat()
    )
