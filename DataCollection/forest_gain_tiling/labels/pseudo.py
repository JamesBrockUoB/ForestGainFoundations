from __future__ import annotations

import ee

from datasets.registry import Datasets


def build_pseudo_labels(
    geom: ee.Geometry, gain_validated: ee.Image, slope: ee.Image, ds: Datasets
) -> ee.Image:
    gm = gain_validated.selfMask()

    jrc_planted = ds.jrc.eq(20).unmask(0).toFloat()
    jrc_natreg = ds.jrc.eq(1).unmask(0).toFloat()
    low_nat = ee.Image(1.0).subtract(ds.nat_forest)

    ch_std = (
        ds.meta_ch.updateMask(gm)
        .unmask(0)
        .reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(3, "pixels"))
        .divide(10)
        .min(1.0)
    )
    ch_uni = ee.Image(1.0).subtract(ch_std)

    def dw_mean(y0: str, y1: str, band: str) -> ee.Image:
        return ds.dw.filterDate(y0, y1).filterBounds(geom).select(band).mean().unmask(0)

    dw_trees_pre = dw_mean("2015-01-01", "2016-12-31", "trees")
    dw_crops_pre = dw_mean("2015-01-01", "2016-12-31", "crops")
    dw_crops_post = dw_mean("2020-01-01", "2020-12-31", "crops")

    annual = [dw_mean(f"{y}-01-01", f"{y}-12-31", "trees") for y in range(2016, 2021)]
    dw_stack = ee.ImageCollection(annual).toBands()
    dw_slope = dw_stack.reduce(ee.Reducer.linearFit()).select("scale").max(0).min(1.0)
    dw_std = dw_stack.reduce(ee.Reducer.stdDev())

    # Band 0: AGROCROP
    s_agro = (
        ds.gem_treecrop.updateMask(gm)
        .unmask(0)
        .multiply(ds.esa_crop.toFloat())
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(dw_crops_pre.multiply(0.4))
            .add(dw_crops_post.multiply(0.4))
            .min(2.0)
        )
        .rename("score_agrocrop")
    )

    # Band 1: NAT_REGEN
    s_nat = (
        ds.nat_forest.multiply(dw_trees_pre)
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(jrc_natreg.multiply(0.5))
            .add(dw_std.multiply(2.0).min(0.5))
            .min(2.0)
        )
        .rename("score_nat_regen")
    )

    # Band 2: PLANTATION
    s_plant = (
        jrc_planted.multiply(ch_uni)
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(low_nat.multiply(0.5))
            .add(dw_trees_pre.multiply(0.3))
            .min(2.0)
        )
        .rename("score_plantation")
    )

    # Band 3: RESTORATION
    s_rest = (
        jrc_planted.multiply(ch_std)
        .multiply(low_nat)
        .pow(ee.Image(1.0).divide(3))
        .multiply(
            ee.Image(1.0)
            .add(dw_slope.multiply(0.5))
            .add(slope.divide(30).min(1.0).multiply(0.3))
            .min(2.0)
        )
        .rename("score_restoration")
    )

    scores = ee.Image.cat([s_agro, s_nat, s_plant, s_rest])
    dominant = scores.toArray().argmax().arrayGet(0).rename("dominant_class").toFloat()
    confidence = (
        scores.reduce(ee.Reducer.max())
        .divide(scores.reduce(ee.Reducer.sum()).max(1e-6))
        .rename("label_confidence")
    )

    return ee.Image.cat([scores, dominant, confidence]).toFloat()
