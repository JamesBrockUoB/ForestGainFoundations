from __future__ import annotations

import ee
from gee_datasets.registry import Datasets

# ForTy quantized probability bands (uint8, 0-250) and their scale to [0,1].
# https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_forest_typology_2020_v1_0_collection
_FORTY_SCALE = 0.004  # 0-250 -> 0.0-1.0 probability


def build_pseudo_labels(
    geom: ee.Geometry, gain_confidence: ee.Image, ds: Datasets
) -> ee.Image:
    """
    Build pseudo labels from ForTy probabilities over validated gain pixels.

    ForTy coverage is independent from gain coverage: only pixels with
    actual ForTy probabilities receive pseudo labels.
    """
    forty = ds.forty.clip(geom)

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
            .rename("score_planted"),
        ]
    )

    # Pixels where ForTy has no information
    valid_forty = scores.reduce(ee.Reducer.sum()).gt(0)

    scores = scores.updateMask(valid_forty)

    dominant = (
        scores.toArray().arrayArgmax().arrayGet(0).rename("dominant_class").toFloat()
    )

    total = scores.reduce(ee.Reducer.sum())

    confidence = (
        scores.reduce(ee.Reducer.max()).divide(total).rename("label_confidence")
    )

    # Only gain pixels AND ForTy-covered pixels get pseudo labels.
    # But do not force ForTy coverage to match gain coverage.
    output_mask = gain_confidence.selfMask().And(valid_forty)

    return (
        ee.Image.cat(
            [
                scores,
                dominant,
                confidence,
            ]
        )
        .updateMask(output_mask)
        .unmask(-9999)
        .toFloat()
    )
