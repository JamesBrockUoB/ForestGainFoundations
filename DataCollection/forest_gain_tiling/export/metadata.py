from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
import rasterio
from config import settings
from enums import PseudoLabel
from tiling.grid import tile_geom

SOIL_BANDS = ["soc", "clay_pct", "ph"]
ERA5_YEARLY_BANDS = [
    "precip_sum",
    "precip_min",
    "precip_max",
    "temp_mean",
    "temp_min",
    "temp_max",
    "lai_high_veg_mean",
    "lai_high_veg_min",
    "lai_high_veg_max",
    "lai_low_veg_mean",
    "lai_low_veg_min",
    "lai_low_veg_max",
]


def _fetch_soil(geom: ee.Geometry) -> dict[str, float | None]:
    """
    Tile-level soil scalars — SoilGrids (250m) is coarser than or
    comparable to tile size, so a per-pixel raster would be
    interpolation artifact rather than real spatial signal. Static,
    single value, no period dependency.

    NOTE: SoilGrids band scale factors are NOT yet verified against the
    projects/soilgrids-isric asset docs -- treat these as provisional
    until checked against .getInfo() band descriptions.
    """
    soil = ee.Image.cat(
        [
            ee.Image("projects/soilgrids-isric/soc_mean").select("soc_0-5cm_mean"),
            ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean"),
            ee.Image("projects/soilgrids-isric/phh2o_mean").select("phh2o_0-5cm_mean"),
        ]
    ).rename(SOIL_BANDS)

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


def _fetch_year_climate(geom: ee.Geometry, year: int) -> dict[str, float | None]:
    """
    One year of ERA5-Land Monthly Aggregated stats -- precipitation,
    2m air temperature, and leaf area index (high/low vegetation),
    each as mean/min/max across the year's 12 monthly values. The
    dataset's own precomputed _min/_max bands are genuine within-month
    extremes (not derived from already-smoothed monthly means), so
    min-of-mins / max-of-maxes across the 12 months preserves real
    extreme events rather than smoothing them away.

    temperature_2m bands are Kelvin; converted to Celsius after
    reduction (linear, so order doesn't matter). LAI is a unitless
    area-fraction index, no conversion needed.
    """
    ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(
            [
                "total_precipitation_sum",
                "total_precipitation_min",
                "total_precipitation_max",
                "temperature_2m",
                "temperature_2m_min",
                "temperature_2m_max",
                "leaf_area_index_high_vegetation",
                "leaf_area_index_high_vegetation_min",
                "leaf_area_index_high_vegetation_max",
                "leaf_area_index_low_vegetation",
                "leaf_area_index_low_vegetation_min",
                "leaf_area_index_low_vegetation_max",
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
            ic.select("total_precipitation_min")
            .min()
            .max(0)
            .multiply(1000)
            .rename("precip_min"),
            ic.select("total_precipitation_max")
            .max()
            .max(0)
            .multiply(1000)
            .rename("precip_max"),
            ic.select("temperature_2m").mean().subtract(273.15).rename("temp_mean"),
            ic.select("temperature_2m_min").min().subtract(273.15).rename("temp_min"),
            ic.select("temperature_2m_max").max().subtract(273.15).rename("temp_max"),
            ic.select("leaf_area_index_high_vegetation")
            .mean()
            .rename("lai_high_veg_mean"),
            ic.select("leaf_area_index_high_vegetation_min")
            .min()
            .rename("lai_high_veg_min"),
            ic.select("leaf_area_index_high_vegetation_max")
            .max()
            .rename("lai_high_veg_max"),
            ic.select("leaf_area_index_low_vegetation")
            .mean()
            .rename("lai_low_veg_mean"),
            ic.select("leaf_area_index_low_vegetation_min")
            .min()
            .rename("lai_low_veg_min"),
            ic.select("leaf_area_index_low_vegetation_max")
            .max()
            .rename("lai_low_veg_max"),
        ]
    )

    point = geom.centroid(1)

    stats = stacked.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=11132,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()

    return {
        band: (float(stats[band]) if stats.get(band) is not None else None)
        for band in ERA5_YEARLY_BANDS
    }


def _fetch_yearly_climate(
    geom: ee.Geometry, period_years: list[int]
) -> dict[str, dict[str, float | None]]:
    return {str(year): _fetch_year_climate(geom, year) for year in period_years}


def _compute_tile_metadata(tile: dict, output_dir: Path) -> dict[str, Any]:
    with rasterio.open(output_dir / "labels" / "gain_confidence.tif") as src:
        gain = src.read(1)
    valid = ~np.isnan(gain)
    gain_frac = float(np.nansum(gain) / valid.sum()) if valid.sum() > 0 else None

    metadata: dict[str, Any] = {
        "tile_id": tile["tile_id"],
        "period": tile.get("period"),
        "biome": tile.get("biome"),
        "region": tile.get("region"),
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
        "gain_frac": gain_frac,
        "full_valid_coverage_frac": float(valid.sum() / gain.size),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        metadata["soil"] = _fetch_soil(tile_geom(tile))
    except Exception as exc:
        metadata["soil"] = None
        metadata["soil_error"] = str(exc)

    try:
        metadata["climate_yearly"] = _fetch_yearly_climate(
            tile_geom(tile), settings.period_years
        )
    except Exception as exc:
        metadata["climate_yearly"] = None
        metadata["climate_yearly_error"] = str(exc)

    pseudo_path = output_dir / "labels" / "pseudo_labels.tif"

    if pseudo_path.exists():
        with rasterio.open(pseudo_path) as src:
            pseudo = src.read()

        dominant, confidence = pseudo[4], pseudo[5]

        gain_bool = np.nan_to_num(gain, nan=0.0) > 0

        pseudo_valid = gain_bool & np.isfinite(dominant) & np.isfinite(confidence)

        class_names = PseudoLabel._member_names_

        if pseudo_valid.any():
            vals = dominant[pseudo_valid].astype(int)
            conf = confidence[pseudo_valid]

            counts = np.bincount(vals, minlength=len(class_names))

            metadata["pseudo_labels"] = {
                "class_pixel_counts": {n: int(c) for n, c in zip(class_names, counts)},
                "dominant_class": class_names[int(counts.argmax())],
                "mean_confidence": float(np.mean(conf)),
                "labelled_gain_pixel_fraction": float(
                    pseudo_valid.sum() / gain_bool.sum()
                ),
            }
        else:
            metadata["pseudo_labels"] = {
                "class_pixel_counts": {},
                "dominant_class": None,
                "mean_confidence": None,
                "labelled_gain_pixel_fraction": 0.0,
            }

    slope_path = output_dir / "static" / "slope.tif"
    if slope_path.exists():
        with rasterio.open(slope_path) as src:
            s = src.read(1)
        s_valid = s[~np.isnan(s)]
        metadata["slope_deg"] = {
            "mean": float(s_valid.mean()) if s_valid.size else None,
            "p90": float(np.percentile(s_valid, 90)) if s_valid.size else None,
        }

    return metadata


def write_tile_metadata(tile: dict, output_dir: Path, logger: logging.Logger) -> None:
    metadata = _compute_tile_metadata(tile, output_dir)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"{tile['tile_id']} | metadata written")
