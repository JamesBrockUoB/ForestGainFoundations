"""Point-centred tile construction and single-tile filtering helpers."""

from __future__ import annotations

from typing import Any

from pyproj import Transformer

from config import settings
from filtering.raster_stats import fetch_cheap_stats, fetch_imagery_stats
from gee_datasets.registry import Datasets

_TO_GRID = Transformer.from_crs("EPSG:4326", settings.crs, always_xy=True)
_FROM_GRID = Transformer.from_crs(settings.crs, "EPSG:4326", always_xy=True)


def point_centred_tile(lon: float, lat: float, period: str) -> dict[str, Any]:
    """Create a 2.56 km EPSG:6933 square centred exactly on a clicked point.

    The point is not snapped to the production grid. Bounds remain in metres
    and produce the usual 256 x 256, 10 m affine export grid.
    """
    x_center, y_center = _TO_GRID.transform(lon, lat)
    half_size = settings.tile_size_m / 2
    x_min, x_max = x_center - half_size, x_center + half_size
    y_min, y_max = y_center - half_size, y_center + half_size
    corners = [
        _FROM_GRID.transform(x, y)
        for x, y in ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    ]
    lons, lats = zip(*corners)
    tile_id = f"inspect_{period}_{x_center:.3f}_{y_center:.3f}".replace(
        ".", "d"
    )

    return {
        "tile_id": tile_id,
        "period": period,
        "x_min_m": x_min,
        "y_min_m": y_min,
        "x_max_m": x_max,
        "y_max_m": y_max,
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
        "biome": "Inspector tile",
        "region": "Inspector tile",
        "country": "Inspector tile",
    }


def tile_corners_lonlat(tile: dict[str, Any]) -> list[tuple[float, float]]:
    """Return the inspector tile's closed WGS84 outline for map drawing."""
    corners = [
        _FROM_GRID.transform(x, y)
        for x, y in (
            (tile["x_min_m"], tile["y_min_m"]),
            (tile["x_min_m"], tile["y_max_m"]),
            (tile["x_max_m"], tile["y_max_m"]),
            (tile["x_max_m"], tile["y_min_m"]),
        )
    ]
    return corners + [corners[0]]


def fetch_tile_metrics(tile: dict[str, Any], ds: Datasets) -> dict[str, dict[str, float]]:
    """Fetch raw metrics once; the UI can then vary thresholds without re-fetching."""
    tile_id = tile["tile_id"]
    return {
        "cheap": fetch_cheap_stats([tile], ds)[tile_id],
        "imagery": fetch_imagery_stats([tile])[tile_id],
    }


def assess_metrics(
    metrics: dict[str, dict[str, float]],
    *,
    gain_pct_min: float,
    ndvi_trend_min: float,
    pseudo_gain_pct_min: float,
    imagery_min_valid_frac: float,
    pseudo_labels_available: bool,
) -> list[tuple[str, bool, str]]:
    """Apply adjustable UI thresholds without issuing an Earth Engine request."""
    cheap = metrics["cheap"]
    gain_pct = 100 * (cheap.get("gain_frac") or 0.0)
    rows = [
        (
            "Gain",
            gain_pct > 0 and gain_pct >= gain_pct_min,
            f"{gain_pct:.2f}% (min {gain_pct_min:.2f}%)",
        )
    ]
    trend = cheap.get("ndvi_trend")
    rows.append(
        (
            "NDVI trend",
            trend is not None and trend > ndvi_trend_min,
            f"{trend:.5f}" if trend is not None else "no valid gain pixels",
        )
    )
    if pseudo_labels_available:
        pseudo_pct = 100 * (cheap.get("pseudo_gain_frac") or 0.0)
        rows.append(
            (
                "ForTy coverage over gain",
                pseudo_pct >= pseudo_gain_pct_min,
                f"{pseudo_pct:.2f}% (min {pseudo_gain_pct_min:.2f}%)",
            )
        )
    for band, value in metrics["imagery"].items():
        rows.append(
            (
                band.upper(),
                value is not None and value >= imagery_min_valid_frac,
                f"{value:.1%}" if value is not None else "no coverage",
            )
        )
    return rows
