from __future__ import annotations

import logging
import math
from typing import Any, Generator

import ee
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

from config import settings
from enums import TileStatus

# Equal-area CRS for grid math — ensures every tile covers the same ground
# area regardless of latitude. EPSG:3857 (Web Mercator) does not preserve
# area, which previously caused tile counts to balloon at high latitudes
# and tile ground size to shrink toward the poles.
_GRID_CRS = settings.crs
_to_grid_crs = Transformer.from_crs("EPSG:4326", _GRID_CRS, always_xy=True)
_from_grid_crs = Transformer.from_crs(_GRID_CRS, "EPSG:4326", always_xy=True)


def _snap(coord_m: float, *, down: bool) -> float:
    fn = math.floor if down else math.ceil
    return fn(coord_m / settings.tile_size_m) * settings.tile_size_m


def _aoi_to_grid_crs(aoi: dict) -> tuple[float, float, float, float]:
    x_min, y_min = _to_grid_crs.transform(aoi["minLon"], aoi["minLat"])
    x_max, y_max = _to_grid_crs.transform(aoi["maxLon"], aoi["maxLat"])
    return (x_min, y_min, x_max, y_max)


def _xy2lonlat(x: float, y: float) -> tuple[float, float]:
    lon, lat = _from_grid_crs.transform(x, y)
    return lon, lat


def tile_geom(tile: dict) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [tile["x_min_m"], tile["y_min_m"], tile["x_max_m"], tile["y_max_m"]],
        proj=ee.Projection(settings.crs_wkt),
        geodesic=False,
    )


def crs_transform(tile: dict) -> list[float]:
    s = settings.scale
    return [s, 0, tile["x_min_m"], 0, -s, tile["y_max_m"]]


def build_grid(
    valid_aois: list[dict], logger: logging.Logger
) -> Generator[dict[str, Any], None, None]:
    """
    Build and yield tiles from global grid as a generator.
    Never materialises the full grid - yields one tile at a time.
    Database handles deduplication via INSERT OR IGNORE.

    Tiles are tagged with settings.period. `valid_aois` is expected to
    already be period-scoped (loaded from settings.valid_aois_path, e.g.
    data/aois/valid_aois_p1.json), since gain/imagery validity differs by
    period even for AOIs at the same grid location.

    tile_id encodes the period ("tile_{xi}_{yi}_{period}") because the same
    (xi, yi) grid cell is a legitimate, independently-valid tile in both p1
    and p2 — without the period suffix, planning p2 after p1 would collide
    on the primary key and silently produce zero new rows (INSERT OR
    IGNORE), rather than adding p2's tiles alongside p1's.
    """
    period = settings.period
    logger.info(f"Projecting AOI bounds to {_GRID_CRS} (equal-area) | period={period}…")
    aoi_bounds_m = [_aoi_to_grid_crs(a) for a in valid_aois]

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

        min_lon, min_lat = _xy2lonlat(float(x_mins[k]), float(y_mins[k]))
        max_lon, max_lat = _xy2lonlat(float(x_maxs[k]), float(y_maxs[k]))

        yield {
            "tile_id": f"tile_{xi}_{yi}_{period}",
            "xi": xi,
            "yi": yi,
            "x_min_m": float(x_mins[k]),
            "y_min_m": float(y_mins[k]),
            "x_max_m": float(x_maxs[k]),
            "y_max_m": float(y_maxs[k]),
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "biome": primary.get("biome_name", "Unknown"),
            "region": primary.get("region", "Unknown"),
            "country": primary.get("country", "Unknown"),
            "period": period,
            "aoi_ids": [primary["id"]],
            "status": str(TileStatus.PENDING),
            "gee_task_id": None,
            "submitted_at": None,
            "completed_at": None,
            "rejection_reason": None,
            "error": None,
        }


def assign_countries_to_aois(
    aois: list[dict], logger: logging.Logger | None = None, batch_size: int = 2000
) -> None:
    if not aois:
        return
    if logger is None:
        logger = logging.getLogger("gee.assign_country")

    countries_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

    for i in range(0, len(aois), batch_size):
        batch = aois[i : i + batch_size]
        features = []
        for a in batch:
            geom = ee.Geometry.Rectangle(
                [a["minLon"], a["minLat"], a["maxLon"], a["maxLat"]]
            )
            props = {"aoi_id": a["id"]}
            if "centroid_lon" in a and "centroid_lat" in a:
                props["centroid_lon"] = a["centroid_lon"]
                props["centroid_lat"] = a["centroid_lat"]
            features.append(ee.Feature(geom, props))

        fc = ee.FeatureCollection(features)

        def annotate(f):
            geom = f.geometry()
            centroid_lon = ee.Algorithms.If(
                f.get("centroid_lon"), f.get("centroid_lon"), None
            )
            centroid_lat = ee.Algorithms.If(
                f.get("centroid_lat"), f.get("centroid_lat"), None
            )

            def _centroid_country():
                pt = ee.Geometry.Point([centroid_lon, centroid_lat])
                c = countries_fc.filterBounds(pt).first()
                return ee.Algorithms.If(c, c.get("COUNTRY_NA"), None)

            centroid_country = ee.Algorithms.If(
                ee.Algorithms.IsEqual(centroid_lon, None), None, _centroid_country()
            )

            intersecting = countries_fc.filterBounds(geom)
            with_area = intersecting.map(
                lambda c: c.set("_area", c.geometry().intersection(geom, 1).area())
            )
            best = ee.Algorithms.If(
                with_area.size().gt(0), with_area.sort("_area", False).first(), None
            )
            majority_country = ee.Algorithms.If(
                best, ee.Feature(best).get("COUNTRY_NA"), None
            )

            country_raw = ee.Algorithms.If(
                ee.Algorithms.IsEqual(centroid_country, None),
                majority_country,
                centroid_country,
            )

            country_final = ee.String(
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(country_raw, None),
                    "Unknown",
                    ee.Algorithms.If(
                        ee.Algorithms.IsEqual(country_raw, "N/A"),
                        "Unknown",
                        ee.Algorithms.If(
                            ee.Algorithms.IsEqual(country_raw, ""),
                            "Unknown",
                            country_raw,
                        ),
                    ),
                )
            )

            return f.set("country", country_final)

        annotated = fc.map(annotate)
        info = annotated.getInfo()
        features_info = info.get("features", [])
        id_to_country = {}
        for feat in features_info:
            props = feat.get("properties", {})
            aoi_id = props.get("aoi_id")
            country = props.get("country")
            id_to_country[aoi_id] = country if country is not None else "Unknown"

        for a in batch:
            a_id = a["id"]
            a["country"] = id_to_country.get(a_id, "Unknown")

        logger.info(
            f"Annotated AOIs {i}..{i+len(batch)-1} with country (batch size={len(batch)})"
        )
