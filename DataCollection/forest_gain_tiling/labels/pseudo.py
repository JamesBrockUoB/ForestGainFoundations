from __future__ import annotations

import ee
from datasets.registry import Datasets

# ForTy quantized probability bands (uint8, 0-250) and their scale to [0,1].
# https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_forest_typology_2020_v1_0_collection
_FORTY_SCALE = 0.004  # 0-250 -> 0.0-1.0 probability


def build_pseudo_labels(
    geom: ee.Geometry, gain_validated: ee.Image, ds: Datasets
) -> ee.Image:
    """
    Pseudo-labels sourced directly from ForTy's per-class probability
    map, restricted to validated gain pixels.

    ForTy's 6-class typology is remapped onto this project's 4-class
    PseudoLabel enum (band order intentionally matches enum values so
    dominant_class == PseudoLabel.value):

        0 AGROCROP    <- TreeCropsAndAgroforestry
        1 NAT_REGEN   <- NaturallyRegeneratingForest
        2 PLANTATION  <- PlantationForest   (rotation <= 40y, intensive)
        3 RESTORATION <- PlantedForest      (rotation > 40y; best proxy
                                              available for restoration
                                              plantings)

    ForTy is a fixed 2020 snapshot — callers should only invoke this for
    periods where settings.pseudo_labels_available is True (see
    stack/stacks.py, which gates the call site rather than this function
    gating itself).
    """
    gm = gain_validated.selfMask()
    forty = ds.forty.clip(geom).updateMask(gm)

    scores = ee.Image.cat(
        [
            forty.select("TreeCropsAndAgroforestry")
            .multiply(_FORTY_SCALE)
            .rename("score_agrocrop"),
            forty.select("NaturallyRegeneratingForest")
            .multiply(_FORTY_SCALE)
            .rename("score_nat_regen"),
            forty.select("PlantationForest")
            .multiply(_FORTY_SCALE)
            .rename("score_plantation"),
            forty.select("PlantedForest")
            .multiply(_FORTY_SCALE)
            .rename("score_restoration"),
        ]
    )

    dominant = (
        scores.toArray().arrayArgmax().arrayGet(0).rename("dominant_class").toFloat()
    )
    total = scores.reduce(ee.Reducer.sum()).max(1e-6)
    confidence = (
        scores.reduce(ee.Reducer.max()).divide(total).rename("label_confidence")
    )

    return ee.Image.cat([scores, dominant, confidence]).toFloat()
