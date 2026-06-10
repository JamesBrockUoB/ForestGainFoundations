from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Generator

import ee
import numpy as np
from config import settings
from enums import TileStatus
from tqdm import tqdm


def _snap(coord_m: float, *, down: bool) -> float:
    fn = math.floor if down else math.ceil
    return fn(coord_m / settings.tile_size_m) * settings.tile_size_m


def _aoi_to_3857(aoi: dict) -> tuple[float, float, float, float]:
    R = 6_378_137.0

    def lon2x(lon: float) -> float:
        return R * math.radians(lon)

    def lat2y(lat: float) -> float:
        return R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    return (
        lon2x(aoi["minLon"]),
        lat2y(aoi["minLat"]),
        lon2x(aoi["maxLon"]),
        lat2y(aoi["maxLat"]),
    )


def _x2lon(x: float) -> float:
    return math.degrees(x / 6_378_137.0)


def _y2lat(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(y / 6_378_137.0)) - math.pi / 2)


def tile_geom(tile: dict) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [tile["min_lon"], tile["min_lat"], tile["max_lon"], tile["max_lat"]],
        proj=ee.Projection("EPSG:4326"),
        geodesic=False,
    )


def crs_transform(tile: dict) -> list[float]:
    s = settings.scale
    return [s, 0, tile["x_min_m"], 0, -s, tile["y_max_m"]]


def build_global_grid(
    valid_aois: list[dict], logger: logging.Logger
) -> Generator[dict[str, Any], None, None]:
    """
    Build and yield tiles from global grid as a generator.
    Never materialises the full grid - yields one tile at a time.
    Database handles deduplication via INSERT OR IGNORE.
    """
    logger.info("Projecting AOI bounds to EPSG:3857…")
    aoi_bounds_m = [_aoi_to_3857(a) for a in valid_aois]

    sz = settings.tile_size_m
    global_xmin = _snap(min(b[0] for b in aoi_bounds_m), down=True)
    global_ymin = _snap(min(b[1] for b in aoi_bounds_m), down=True)
    global_xmax = _snap(max(b[2] for b in aoi_bounds_m), down=False)
    global_ymax = _snap(max(b[3] for b in aoi_bounds_m), down=False)

    n_cols = round((global_xmax - global_xmin) / sz)
    n_rows = round((global_ymax - global_ymin) / sz)
    logger.info(f"Grid: {n_cols} cols x {n_rows} rows = {n_cols * n_rows:,} candidates")

    min_overlap_area = settings.min_aoi_overlap_frac * sz * sz

    first_aoi = np.full((n_rows, n_cols), -1, dtype=np.int32)

    col_xmin = global_xmin + np.arange(n_cols) * sz
    col_xmax = col_xmin + sz
    row_ymin = global_ymin + np.arange(n_rows) * sz
    row_ymax = row_ymin + sz

    for aoi_idx, (_, (ax_min, ay_min, ax_max, ay_max)) in enumerate(
        tqdm(
            zip(valid_aois, aoi_bounds_m),
            total=len(valid_aois),
            desc="Building grid",
            unit="aoi",
        )
    ):
        ci_lo = max(0, math.floor((ax_min - global_xmin) / sz))
        ci_hi = min(n_cols, math.ceil((ax_max - global_xmin) / sz))
        ri_lo = max(0, math.floor((ay_min - global_ymin) / sz))
        ri_hi = min(n_rows, math.ceil((ay_max - global_ymin) / sz))
        if ci_lo >= ci_hi or ri_lo >= ri_hi:
            continue

        x_overlap = np.maximum(
            0,
            np.minimum(col_xmax[ci_lo:ci_hi], ax_max)
            - np.maximum(col_xmin[ci_lo:ci_hi], ax_min),
        )
        y_overlap = np.maximum(
            0,
            np.minimum(row_ymax[ri_lo:ri_hi], ay_max)
            - np.maximum(row_ymin[ri_lo:ri_hi], ay_min),
        )

        window = first_aoi[ri_lo:ri_hi, ci_lo:ci_hi]
        mask = (y_overlap[:, None] * x_overlap[None, :] >= min_overlap_area) & (
            window == -1
        )
        window[mask] = aoi_idx

    ri_arr, ci_arr = np.where(first_aoi >= 0)
    logger.info(f"Streaming {len(ri_arr):,} tiles…")

    x_mins = global_xmin + ci_arr * sz
    x_maxs = x_mins + sz
    y_mins = global_ymin + ri_arr * sz
    y_maxs = y_mins + sz

    for k in tqdm(range(len(ri_arr)), desc="Streaming tiles", unit="tile"):
        primary = valid_aois[first_aoi[ri_arr[k], ci_arr[k]]]
        xi = round(x_mins[k] / sz)
        yi = round(y_mins[k] / sz)

        yield {
            "tile_id": f"tile_{xi}_{yi}",
            "xi": xi,
            "yi": yi,
            "x_min_m": float(x_mins[k]),
            "y_min_m": float(y_mins[k]),
            "x_max_m": float(x_maxs[k]),
            "y_max_m": float(y_maxs[k]),
            "min_lon": _x2lon(float(x_mins[k])),
            "min_lat": _y2lat(float(y_mins[k])),
            "max_lon": _x2lon(float(x_maxs[k])),
            "max_lat": _y2lat(float(y_maxs[k])),
            "biome": primary.get("biome_name", "Unknown"),
            "region": primary.get("region", "Unknown"),
            "aoi_ids": [primary["id"]],
            "status": str(TileStatus.PENDING),
            "gee_task_id": None,
            "submitted_at": None,
            "completed_at": None,
            "rejection_reason": None,
            "error": None,
        }
