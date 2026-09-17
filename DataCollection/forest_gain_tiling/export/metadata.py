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
    geom: ee.Geometry, period_years: list[int]
) -> dict[str, dict[str, float | None]]:
    return {str(year): _fetch_year_climate(geom, year) for year in period_years}


def _compute_tile_metadata(
    tile: dict,
    gain_confidence_bytes: bytes,
    pseudo_labels_bytes: bytes | None,
) -> dict[str, Any]:
    with MemoryFile(gain_confidence_bytes) as memfile, memfile.open() as src:
        gain = src.read(1)

    gain_pixels = np.isfinite(gain) & (gain > 0)

    metadata: dict[str, Any] = {
        "tile_id": tile["tile_id"],
        "period": tile.get("period"),
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
            tile_geom(tile), settings.period_years
        )
    except Exception as exc:
        metadata["climate_yearly"] = None
        metadata["climate_yearly_error"] = str(exc)

    if pseudo_labels_bytes is not None:
        with MemoryFile(pseudo_labels_bytes) as memfile, memfile.open() as src:
            pseudo = src.read()

        dominant, confidence = pseudo[4], pseudo[5]
        gain_bool = np.nan_to_num(gain, nan=0.0) > 0
        labelled = (dominant != -9999) & (confidence != -9999)
        pseudo_valid = (
            gain_bool & labelled & np.isfinite(dominant) & np.isfinite(confidence)
        )

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

    return metadata


def write_tile_metadata(
    tile: dict,
    output_dir: Path,
    logger: logging.Logger,
    gain_confidence_bytes: bytes | None = None,
    pseudo_labels_bytes: bytes | None = None,
) -> None:
    if gain_confidence_bytes is None:
        gain_confidence_bytes = (
            output_dir / "labels" / "gain_confidence.tif"
        ).read_bytes()

    if pseudo_labels_bytes is None:
        pseudo_path = output_dir / "labels" / "pseudo_labels.tif"
        if pseudo_path.exists():
            pseudo_labels_bytes = pseudo_path.read_bytes()

    metadata = _compute_tile_metadata(tile, gain_confidence_bytes, pseudo_labels_bytes)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"{tile['tile_id']} | metadata written")
