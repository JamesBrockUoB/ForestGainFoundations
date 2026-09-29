from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
from config import settings
from enums import ESRI_RAW_VALUES, ESRI_REMAPPED_VALUES, DWClass, ESRIClass
from gee_datasets.registry import Datasets
from rasterio.io import MemoryFile
from tiling.grid import tile_geom

SOIL_BANDS = ["soc", "clay_pct", "ph"]

ERA5_YEARLY_BANDS = [
    "precip_sum",
    "temp_mean",
    "temp_min",
    "temp_max",
]


def _fetch_soil(geom: ee.Geometry) -> dict[str, float | None]:
    datasets = Datasets()
    soil = (
        ee.Image.cat(
            [
                ee.Image(datasets.soilgrids_soc).select("soc_0-5cm_mean"),
                ee.Image(datasets.soilgrids_clay).select("clay_0-5cm_mean"),
                ee.Image(datasets.soilgrids_ph).select("phh2o_0-5cm_mean"),
            ]
        )
        .rename(SOIL_BANDS)
        .divide(10.0)
    )

    stats = soil.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        crs=settings.crs_wkt,
        scale=250,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()

    return {
        band: (float(stats[band]) if stats.get(band) is not None else None)
        for band in SOIL_BANDS
    }


def _fetch_year_climate(geom: ee.Geometry, year: int) -> dict[str, Any]:
    datasets = Datasets()
    ic = (
        ee.ImageCollection(datasets.era5_land_monthly)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(
            [
                "total_precipitation_sum",
                "temperature_2m",
                "temperature_2m_min",
                "temperature_2m_max",
            ]
        )
    )

    stacked = ee.Image.cat(
        [
            ic.select("total_precipitation_sum")
            .sum()
            .max(0)
            .multiply(1000)
            .rename("precip_sum"),
            ic.select("temperature_2m").mean().subtract(273.15).rename("temp_mean"),
            ic.select("temperature_2m_min").min().subtract(273.15).rename("temp_min"),
            ic.select("temperature_2m_max").max().subtract(273.15).rename("temp_max"),
        ]
    )

    point = geom.centroid(1)

    raw_stats = stacked.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=11132,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()

    if all(raw_stats.get(b) is not None for b in ERA5_YEARLY_BANDS):
        return {
            **{band: float(raw_stats[band]) for band in ERA5_YEARLY_BANDS},
            "climate_source": "ERA5_LAND",
        }

    for radius_px in (3, 6, 10):
        filled = stacked.focal_mean(
            radius=radius_px, kernelType="square", units="pixels"
        )
        stats = filled.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,
            bestEffort=True,
            maxPixels=1_000_000_000,
        ).getInfo()
        if all(stats.get(b) is not None for b in ERA5_YEARLY_BANDS):
            return {
                **{band: float(stats[band]) for band in ERA5_YEARLY_BANDS},
                "climate_source": f"ERA5_LAND_FILLED_r{radius_px}",
            }

    return {
        **{band: None for band in ERA5_YEARLY_BANDS},
        "climate_source": "MISSING",
    }


def _fetch_yearly_climate(
    geom: ee.Geometry, years: list[int]
) -> dict[str, dict[str, float | None]]:
    return {str(year): _fetch_year_climate(geom, year) for year in years}


def _dw_label_mode(geom: ee.Geometry, year: int) -> ee.Image:
    """Modal (most frequent) Dynamic World label per pixel across the
    calendar year — same logic as generate_aois.py's _dw_label_mode,
    reused here at tile scale."""
    datasets = Datasets()
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    dw = (
        ee.ImageCollection(datasets.dynamic_world)
        .filterDate(start, end)
        .filterBounds(geom)
    )
    return dw.select("label").reduce(ee.Reducer.mode()).rename("dw_label")


def _dw_label_confidence_image(geom: ee.Geometry, year: int) -> ee.Image:
    """
    Two-band composite over the calendar year:
      - dw_label: modal (most frequent) DW class per pixel — same
        definition as _dw_label_mode, used here inline so both bands
        come from one filtered collection rather than two.
      - dw_confidence: mean, across the year, of each image's top
        probability (i.e. the probability of whichever class won that
        image's argmax) — the natural per-pixel confidence for the
        label DW actually assigned, not a fixed single-band proxy.
    """
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    dw = (
        ee.ImageCollection(datasets.dynamic_world)
        .filterDate(start, end)
        .filterBounds(geom)
    )

    label = dw.select("label").reduce(ee.Reducer.mode()).toInt32().rename("dw_label")

    dw_prob_bands = [c.name for c in DWClass]
    top_prob_per_image = dw.select(dw_prob_bands).map(
        lambda img: img.reduce(ee.Reducer.max()).rename("top_prob")
    )
    confidence = top_prob_per_image.mean().toFloat().rename("dw_confidence")

    return label.addBands(confidence)


def _esri_lulc_image(geom: ee.Geometry, year: int) -> ee.Image:
    """
    Single-band classified image for `year` from ESRI/Impact Observatory's
    10m Annual LULC (v3, 2017-2024) time series, with the collection's raw,
    non-contiguous class values remapped to the dense 1-9 range in
    ESRIClass.
    """
    datasets = Datasets()
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(datasets.esri_lulc)
        .filterDate(start, end)
        .filterBounds(geom)
        .mosaic()
        .select(0)
        .remap(ESRI_RAW_VALUES, ESRI_REMAPPED_VALUES)
        .toInt32()
        .rename("esri_label")
    )


def _fetch_dt_dw_agreement(
    geom: ee.Geometry, year_start: int
) -> dict[str, float | None]:
    """
    Fraction of DT-non-forest pixels (at year_start) that Dynamic World
    also calls non-forest (label not in {trees, flooded_vegetation}) —
    same definition used at AOI level in generate_aois.py's
    _build_gee_datasets/process_batch, just reduced over one tile
    geometry instead of a FeatureCollection.
    """
    datasets = Datasets()
    dt_cover_start = (
        datasets.get_dt_cover(year_start)
        .select([0])
        .divide(2.55)
        .rename("tree_cover_pct")
    )
    forest_start = dt_cover_start.gt(settings.non_tree_threshold_frac).unmask(0)
    dt_non_forest_start = forest_start.Not().rename("dt_non_forest_start")

    dw_label_start = _dw_label_mode(geom, year_start)
    dw_forest_like = dw_label_start.eq(DWClass.trees).Or(
        dw_label_start.eq(DWClass.flooded_vegetation)
    )
    dw_agrees_non_forest = dt_non_forest_start.And(dw_forest_like.Not()).rename(
        "dt_non_forest_start_dw_agrees"
    )

    qa_img = dt_non_forest_start.addBands(dw_agrees_non_forest)

    stats = qa_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=settings.scale,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()

    non_forest_px = float(stats.get("dt_non_forest_start", 0.0) or 0.0)
    agree_px = float(stats.get("dt_non_forest_start_dw_agrees", 0.0) or 0.0)

    return {
        "dt_dw_agreement_frac": (
            agree_px / non_forest_px if non_forest_px > 0 else None
        ),
        "dt_non_forest_px": non_forest_px,
        "dt_dw_agree_px": agree_px,
    }


def _fetch_dt_esri_agreement(
    geom: ee.Geometry, year_start: int
) -> dict[str, float | None]:
    """
    Fraction of DT-non-forest pixels (at year_start) that ESRI's 10m LULC
    also calls non-forest (class not in {trees, flooded_vegetation}) —
    the same check as _fetch_dt_dw_agreement, run against ESRI as a
    second, independently-trained land-cover product.
    """
    datasets = Datasets()
    dt_cover_start = (
        datasets.get_dt_cover(year_start)
        .select([0])
        .divide(2.55)
        .rename("tree_cover_pct")
    )
    forest_start = dt_cover_start.gt(settings.non_tree_threshold_frac).unmask(0)
    dt_non_forest_start = forest_start.Not().rename("dt_non_forest_start")

    esri_label_start = _esri_lulc_image(geom, year_start)
    esri_forest_like = esri_label_start.eq(ESRIClass.trees).Or(
        esri_label_start.eq(ESRIClass.flooded_vegetation)
    )
    esri_agrees_non_forest = dt_non_forest_start.And(esri_forest_like.Not()).rename(
        "dt_non_forest_start_esri_agrees"
    )

    qa_img = dt_non_forest_start.addBands(esri_agrees_non_forest)

    stats = qa_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=settings.scale,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()

    non_forest_px = float(stats.get("dt_non_forest_start", 0.0) or 0.0)
    agree_px = float(stats.get("dt_non_forest_start_esri_agrees", 0.0) or 0.0)

    return {
        "dt_esri_agreement_frac": (
            agree_px / non_forest_px if non_forest_px > 0 else None
        ),
        "dt_non_forest_px": non_forest_px,
        "dt_esri_agree_px": agree_px,
    }


def _grid_from_raster(src) -> dict:
    """
    Build an ee.data.computePixels grid spec matching src's exact
    transform/dimensions, so the fetched EE array lines up pixel-for-pixel
    with the local gain array without any resampling.
    """
    t = src.transform
    return {
        "dimensions": {"width": src.width, "height": src.height},
        "affineTransform": {
            "scaleX": t.a,
            "shearX": t.b,
            "translateX": t.c,
            "shearY": t.d,
            "scaleY": t.e,
            "translateY": t.f,
        },
        "crsWkt": settings.crs_wkt,
    }


def _fetch_dw_transition_stats(
    geom: ee.Geometry, year: int, gain_mask: np.ndarray, src
) -> dict[str, Any]:
    """
    Among gain pixels only: percentage breakdown by Dynamic World class
    at `year`, and the mean DW confidence for each class present. Helps
    characterise what gain pixels actually transitioned INTO (e.g. is
    "gain" mostly landing on trees vs. shrub_and_scrub vs. crops), which
    plain gain_pct alone doesn't tell you.
    """
    dw_img = _dw_label_confidence_image(geom, year)
    grid = _grid_from_raster(src)

    arr = ee.data.computePixels(
        {"expression": dw_img, "fileFormat": "NUMPY_NDARRAY", "grid": grid}
    )

    dw_label = arr["dw_label"]
    dw_confidence = arr["dw_confidence"]

    n_gain = int(gain_mask.sum())
    if n_gain == 0:
        return {"n_gain_pixels": 0, "classes": {}}

    labels_in_gain = dw_label[gain_mask]
    confidence_in_gain = dw_confidence[gain_mask]

    classes: dict[str, Any] = {}
    for class_id in np.unique(labels_in_gain):
        class_mask = labels_in_gain == class_id
        try:
            name = DWClass(int(class_id)).name
        except ValueError:
            name = f"unknown_{class_id}"
        classes[name] = {
            "pct_of_gain_pixels": float(class_mask.sum()) / n_gain * 100,
            "mean_confidence": float(confidence_in_gain[class_mask].mean()),
        }

    return {"n_gain_pixels": n_gain, "classes": classes}


def _fetch_esri_transition_stats(
    geom: ee.Geometry, year: int, gain_mask: np.ndarray, src
) -> dict[str, Any]:
    """
    Among gain pixels only: percentage breakdown by ESRI LULC class at
    `year`. Mirrors _fetch_dw_transition_stats as a second reference for
    what gain pixels transitioned into, but ESRI has no per-pixel
    confidence band — it's one classified mosaic per year, not an
    ensemble/probability collection like DW — so there's no
    mean_confidence field here.
    """
    esri_img = _esri_lulc_image(geom, year)
    grid = _grid_from_raster(src)

    arr = ee.data.computePixels(
        {"expression": esri_img, "fileFormat": "NUMPY_NDARRAY", "grid": grid}
    )

    esri_label = arr["esri_label"]

    n_gain = int(gain_mask.sum())
    if n_gain == 0:
        return {"n_gain_pixels": 0, "classes": {}}

    labels_in_gain = esri_label[gain_mask]

    classes: dict[str, Any] = {}
    for class_id in np.unique(labels_in_gain):
        class_mask = labels_in_gain == class_id
        try:
            name = ESRIClass(int(class_id)).name
        except ValueError:
            name = f"unknown_{class_id}"
        classes[name] = {
            "pct_of_gain_pixels": float(class_mask.sum()) / n_gain * 100,
        }

    return {"n_gain_pixels": n_gain, "classes": classes}


def _compute_tile_metadata(
    tile: dict,
    gain_confidence_bytes: bytes,
) -> dict[str, Any]:
    with MemoryFile(gain_confidence_bytes) as memfile, memfile.open() as src:
        gain = src.read(1)
        gain_pixels = np.isfinite(gain) & (gain > 0)

        metadata: dict[str, Any] = {
            "tile_id": tile["tile_id"],
            "biome": tile.get("biome"),
            "region": tile.get("region"),
            "country": tile.get("country"),
            "bounds": {
                "crs": "EPSG:6933",
                "x_min_m": tile["x_min_m"],
                "y_min_m": tile["y_min_m"],
                "x_max_m": tile["x_max_m"],
                "y_max_m": tile["y_max_m"],
                "min_lon": tile["min_lon"],
                "min_lat": tile["min_lat"],
                "max_lon": tile["max_lon"],
                "max_lat": tile["max_lat"],
            },
            "gain_pct": float(gain_pixels.mean()) * 100,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            metadata["soil"] = _fetch_soil(tile_geom(tile))
        except Exception as exc:
            metadata["soil"] = None
            metadata["soil_error"] = str(exc)

        try:
            metadata["climate_yearly"] = _fetch_yearly_climate(
                tile_geom(tile), settings.years
            )
        except Exception as exc:
            metadata["climate_yearly"] = None
            metadata["climate_yearly_error"] = str(exc)

        try:
            metadata["dt_dw_qa"] = _fetch_dt_dw_agreement(
                tile_geom(tile), min(settings.years)
            )
        except Exception as exc:
            metadata["dt_dw_qa"] = None
            metadata["dt_dw_qa_error"] = str(exc)

        try:
            metadata["dw_transition"] = _fetch_dw_transition_stats(
                tile_geom(tile), max(settings.years), gain_pixels, src
            )
        except Exception as exc:
            metadata["dw_transition"] = None
            metadata["dw_transition_error"] = str(exc)

        try:
            metadata["dt_esri_qa"] = _fetch_dt_esri_agreement(
                tile_geom(tile), min(settings.years)
            )
        except Exception as exc:
            metadata["dt_esri_qa"] = None
            metadata["dt_esri_qa_error"] = str(exc)

        try:
            metadata["esri_transition"] = _fetch_esri_transition_stats(
                tile_geom(tile), max(settings.years), gain_pixels, src
            )
        except Exception as exc:
            metadata["esri_transition"] = None
            metadata["esri_transition_error"] = str(exc)

    return metadata


def write_tile_metadata(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    gain_confidence_bytes: bytes | None = None,
) -> None:
    if gain_confidence_bytes is None:
        gain_confidence_bytes = (
            output_dir / "labels" / "gain_confidence.tif"
        ).read_bytes()

    metadata = _compute_tile_metadata(tile, gain_confidence_bytes)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"{tile['tile_id']} | metadata written")
