from __future__ import annotations

import logging
import time

from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from filtering.raster_stats import BAND_NAMES, NO_GAIN_SENTINEL, fetch_tile_stats
from registry.store import update_tile


def compute_aoi_extent(tiles: list[dict]) -> dict:
    """
    Given a list of tile dicts (with xi, yi, x_min_m, y_min_m), compute
    the bounding extent of this AOI's tiles in grid-index space, and
    the (origin_x, origin_y) anchor for the raster fetch.

    origin_x = x_min_m of the westmost column (xi_min)
    origin_y = y_min_m of the northmost row (yi_max) + tile_size_m
             = y_max_m of the northmost row
    """
    sz = settings.tile_size_m

    xi_min = min(t["xi"] for t in tiles)
    xi_max = max(t["xi"] for t in tiles)
    yi_min = min(t["yi"] for t in tiles)
    yi_max = max(t["yi"] for t in tiles)

    west_tile = next(t for t in tiles if t["xi"] == xi_min)
    origin_x = west_tile["x_min_m"]

    north_tile = next(t for t in tiles if t["yi"] == yi_max)
    origin_y = north_tile["y_min_m"] + sz

    n_cols = xi_max - xi_min + 1
    n_rows = yi_max - yi_min + 1

    return {
        "xi_min": xi_min,
        "xi_max": xi_max,
        "yi_min": yi_min,
        "yi_max": yi_max,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "n_cols": n_cols,
        "n_rows": n_rows,
    }


def aoi_bbox_geom(extent: dict):
    """Build an ee.Geometry.Rectangle covering this AOI's tile extent."""
    import ee

    sz = settings.tile_size_m
    return ee.Geometry.Rectangle(
        [
            extent["origin_x"],
            extent["origin_y"] - extent["n_rows"] * sz,
            extent["origin_x"] + extent["n_cols"] * sz,
            extent["origin_y"],
        ],
        proj=ee.Projection(settings.crs),
        geodesic=False,
    )


def evaluate_tile_stats(stats: dict[str, float]) -> tuple[str, str | None]:
    """
    Apply the cheap-filter thresholds to a single tile's aggregated stats.
    Returns (new_status, rejection_reason).
    """
    gain_pct = stats["gain_frac"] * 100.0

    if gain_pct < settings.gain_pct_min:
        return (
            str(TileStatus.REJECTED),
            f"gain_pct={gain_pct:.3f} < {settings.gain_pct_min}",
        )

    ndvi_delta = stats["ndvi_delta"]
    canopy_mean = stats["canopy_mean"]

    if ndvi_delta == NO_GAIN_SENTINEL or canopy_mean == NO_GAIN_SENTINEL:
        return (
            str(TileStatus.REJECTED),
            f"no_gain_pixels gain_pct={gain_pct:.3f}",
        )

    if ndvi_delta <= settings.ndvi_delta_min or canopy_mean < settings.gain_canopy_min:
        return (
            str(TileStatus.REJECTED),
            f"viability ndvi_delta={ndvi_delta:.4f} canopy_mean={canopy_mean:.3f}",
        )

    fracs = {
        "2016": stats["s2_2016"],
        "2020": stats["s2_2020"],
        "2025": stats["s2_2025"],
    }
    if any(f < settings.s2_min_valid_frac for f in fracs.values()):
        return str(TileStatus.REJECTED), f"s2_coverage={fracs}"

    return str(TileStatus.VALID), None


def filter_aoi(
    aoi_id: str, tiles: list[dict], ds: Datasets, logger: logging.Logger
) -> dict[str, int]:
    """
    Run the raster-batch filter for one AOI's pending tiles.
    Updates tile statuses in the DB. Returns counts of valid/rejected/skipped.
    """
    sz = settings.tile_size_m
    extent = compute_aoi_extent(tiles)
    n_rows = extent["n_rows"]
    n_cols = extent["n_cols"]
    geom = aoi_bbox_geom(extent)

    logger.info(
        f"  [{aoi_id}] extent: {n_cols}x{n_rows} tiles "
        f"({n_cols * n_cols} total) | "
        f"origin=({extent['origin_x']:.0f}, {extent['origin_y']:.0f})"
    )

    # --- Raster fetch ---
    t_fetch = time.time()
    try:
        raster = fetch_tile_stats(
            geom,
            ds,
            origin_x=extent["origin_x"],
            origin_y=extent["origin_y"],
            n_cols=n_cols,
            n_rows=n_rows,
        )
        logger.info(f"  [{aoi_id}] raster fetch: {time.time()-t_fetch:.1f}s")
    except Exception as exc:
        logger.error(
            f"  [{aoi_id}] raster fetch failed after {time.time()-t_fetch:.1f}s — {exc}"
        )
        for t in tiles:
            update_tile(t["tile_id"], status=TileStatus.FAILED, error=str(exc))
        return {"failed": len(tiles)}

    # --- Shape check ---
    for band in BAND_NAMES:
        arr = raster.get(band)
        if arr is None or len(arr) != n_rows or any(len(row) != n_cols for row in arr):
            msg = (
                f"raster shape mismatch band={band} "
                f"expected={n_rows}x{n_cols} "
                f"got={len(arr) if arr else 0}x{len(arr[0]) if arr and arr[0] else 0}"
            )
            logger.error(f"  [{aoi_id}] {msg}")
            for t in tiles:
                update_tile(t["tile_id"], status=TileStatus.FAILED, error=msg)
            return {"failed": len(tiles)}

    # --- Per-tile evaluation ---
    t_eval = time.time()
    counts: dict[str, int] = {"valid": 0, "rejected": 0}

    for t in tiles:
        xi_local = t["xi"] - extent["xi_min"]
        yi_local = t["yi"] - extent["yi_min"]
        row = (n_rows - 1) - yi_local
        col = xi_local

        stats = {band: raster[band][row][col] for band in BAND_NAMES}
        new_status, reason = evaluate_tile_stats(stats)

        if new_status == str(TileStatus.REJECTED):
            update_tile(
                t["tile_id"], status=TileStatus.REJECTED, rejection_reason=reason
            )
            counts["rejected"] += 1
        else:
            update_tile(t["tile_id"], status=TileStatus.VALID)
            counts["valid"] += 1

    logger.info(
        f"  [{aoi_id}] tile eval: {time.time()-t_eval:.1f}s | "
        f"valid={counts['valid']} rejected={counts['rejected']}"
    )

    return counts
