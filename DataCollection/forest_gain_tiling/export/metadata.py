from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
from config import settings
from rasterio.io import MemoryFile
from tiling.grid import tile_geom

SOIL_BANDS = ["soc", "clay_pct", "ph"]
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

ERA5_YEARLY_BANDS = [
    "precip_sum",
    "temp_mean",
    "temp_min",
    "temp_max",
]


def _fetch_soil(geom: ee.Geometry) -> dict[str, float | None]:
    soil = (
        ee.Image.cat(
            [
                ee.Image("projects/soilgrids-isric/soc_mean").select("soc_0-5cm_mean"),
                ee.Image("projects/soilgrids-isric/clay_mean").select(
                    "clay_0-5cm_mean"
                ),
                ee.Image("projects/soilgrids-isric/phh2o_mean").select(
                    "phh2o_0-5cm_mean"
                ),
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
    ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
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
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    dw = ee.ImageCollection(DW_COLLECTION).filterDate(start, end).filterBounds(geom)
    return dw.select("label").reduce(ee.Reducer.mode()).rename("dw_label")


def _fetch_dt_dw_agreement(
    geom: ee.Geometry, year_start: int
) -> dict[str, float | None]:
    """
    Fraction of DT-non-forest pixels (at year_start) that Dynamic World
    also calls non-forest (label != 1, "trees") — same definition used
    at AOI level in generate_aois.py's _build_gee_datasets/process_batch,
    just reduced over one tile geometry instead of a FeatureCollection.
    """
    dt_cover_start = (
        ee.Image(f"projects/symbolic-base-346316/assets/dt_tree_cover_{year_start}_v2")
        .select([0])
        .divide(2.55)
        .rename("tree_cover_pct")
    )
    forest_start = dt_cover_start.gt(settings.non_tree_threshold_frac).unmask(0)
    dt_non_forest_start = forest_start.Not().rename("dt_non_forest_start")

    dw_label_start = _dw_label_mode(geom, year_start)
    dw_forest_like = dw_label_start.eq(1).Or(dw_label_start.eq(3))
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
