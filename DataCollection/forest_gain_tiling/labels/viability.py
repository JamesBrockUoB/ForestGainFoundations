from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets
from export.composites import s2_peak


def score_viability(
    geom: ee.Geometry, gain_validated: ee.Image, ds: Datasets
) -> dict[str, float]:
    gm = gain_validated.selfMask()

    ndvi_d = (
        s2_peak(geom, 2020, ds)
        .select("NDVI")
        .subtract(s2_peak(geom, 2017, ds).select("NDVI"))
        .updateMask(gm)
    )

    nd_stats = ndvi_d.reduceRegion(
        ee.Reducer.median(), geom, settings.scale, settings.crs_wkt, maxPixels=1e13
    )
    ch_stats = ds.meta_ch.updateMask(gm).reduceRegion(
        ee.Reducer.mean(), geom, settings.scale, settings.crs_wkt, maxPixels=1e13
    )

    nd_val = ee.Number(ee.Algorithms.If(nd_stats.get("NDVI"), nd_stats.get("NDVI"), 0))
    ch_val = ee.Number(
        ee.Algorithms.If(ch_stats.get("cover_code"), ch_stats.get("cover_code"), 0)
    )

    return {
        "ndvi_delta": nd_val.getInfo(),
        "gain_canopy_mean": ch_val.getInfo(),
    }
