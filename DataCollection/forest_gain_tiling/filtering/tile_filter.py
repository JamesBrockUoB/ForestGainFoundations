from __future__ import annotations

import logging
import time

from config import settings
from enums import TileStatus
from filtering.raster_stats import (
    CHEAP_BAND_NAMES,
    IMAGERY_BAND_NAMES,
    NO_GAIN_SENTINEL,
    fetch_cheap_stats,
    fetch_imagery_stats,
)
from registry.store import update_tile


def compute_batch_extent(tiles: list[dict]) -> dict:
    """Bounding-box extent (in grid-index space) covering a batch of tiles."""
    sz = settings.tile_size_m

    xi_min = min(t["xi"] for t in tiles)
    xi_max = max(t["xi"] for t in tiles)
    yi_min = min(t["yi"] for t in tiles)
    yi_max = max(t["yi"] for t in tiles)

    west_tile = next(t for t in tiles if t["xi"] == xi_min)
    origin_x = west_tile["x_min_m"]
    north_tile = next(t for t in tiles if t["yi"] == yi_max)
    origin_y = north_tile["y_min_m"] + sz

    return {
        "xi_min": xi_min,
        "xi_max": xi_max,
        "yi_min": yi_min,
        "yi_max": yi_max,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "n_cols": xi_max - xi_min + 1,
        "n_rows": yi_max - yi_min + 1,
    }


def batch_bbox_geom(extent: dict):
    import ee

    sz = settings.tile_size_m
    return ee.Geometry.Rectangle(
        [
            extent["origin_x"],
            extent["origin_y"] - extent["n_rows"] * sz,
            extent["origin_x"] + extent["n_cols"] * sz,
            extent["origin_y"],
        ],
        proj=ee.Projection(settings.crs_wkt),
        geodesic=False,
    )


def _rowcol(t: dict, extent: dict) -> tuple[int, int]:
    row = (extent["n_rows"] - 1) - (t["yi"] - extent["yi_min"])
    col = t["xi"] - extent["xi_min"]
    return row, col


def _check_shape(
    raster: dict, band_names: list[str], n_rows: int, n_cols: int
) -> str | None:
    for band in band_names:
        arr = raster.get(band)
        if arr is None or len(arr) != n_rows or any(len(row) != n_cols for row in arr):
            got_r = len(arr) if arr else 0
            got_c = len(arr[0]) if arr and arr[0] else 0
            return f"raster shape mismatch band={band} expected={n_rows}x{n_cols} got={got_r}x{got_c}"
    return None


def evaluate_cheap_stats(
    stats: dict[str, float], logger: logging.Logger | None = None
) -> tuple[str, str | None]:
    gain_pct = stats["gain_frac"] * 100.0
    ndvi_delta = stats["ndvi_delta"]

    if gain_pct == 0.0:
        return str(TileStatus.REJECTED), "no_gain"

    if gain_pct < settings.gain_pct_min:
        return str(TileStatus.REJECTED), "low_gain_pct"

    if ndvi_delta <= settings.ndvi_delta_min:
        if logger:
            logger.debug(f"low_viability: ndvi_delta={ndvi_delta:.4f}")
        return str(TileStatus.REJECTED), "low_viability"

    return str(TileStatus.CHEAP_VALID), None


def evaluate_imagery_stats(
    stats: dict[str, float], logger: logging.Logger | None = None
) -> tuple[str, str | None]:
    """
    Requires every per-year S1 and S2 band to clear settings.imagery_min_valid_frac.
    """
    low = {b: v for b, v in stats.items() if v < settings.imagery_min_valid_frac}
    if low:
        if logger:
            logger.debug(f"low_imagery_coverage: {low}")
        return str(TileStatus.REJECTED), "low_imagery_coverage"
    return str(TileStatus.VALID), None


def filter_batch_cheap(
    tiles: list[dict], ds, logger: logging.Logger, batch_label: str = ""
) -> dict[str, int]:
    extent = compute_batch_extent(tiles)
    n_rows, n_cols = extent["n_rows"], extent["n_cols"]
    geom = batch_bbox_geom(extent)

    logger.debug(f"{batch_label} cheap: {len(tiles)} tiles")
    t0 = time.time()
    try:
        raster = fetch_cheap_stats(
            geom,
            ds,
            origin_x=extent["origin_x"],
            origin_y=extent["origin_y"],
            n_cols=n_cols,
            n_rows=n_rows,
        )
        logger.debug(f"  {batch_label} cheap fetch: {time.time()-t0:.1f}s")
    except Exception as exc:
        logger.error(
            f"  {batch_label} cheap fetch failed after {time.time()-t0:.1f}s — {exc}"
        )
        for t in tiles:
            update_tile(t["tile_id"], status=TileStatus.FAILED, error=str(exc))
        return {"failed": len(tiles)}

    shape_err = _check_shape(raster, CHEAP_BAND_NAMES, n_rows, n_cols)
    if shape_err:
        logger.error(f"  {batch_label} {shape_err}")
        for t in tiles:
            update_tile(t["tile_id"], status=TileStatus.FAILED, error=shape_err)
        return {"failed": len(tiles)}

    counts = {"cheap_valid": 0, "rejected": 0}
    for t in tiles:
        row, col = _rowcol(t, extent)
        stats = {b: raster[b][row][col] for b in CHEAP_BAND_NAMES}
        status, reason = evaluate_cheap_stats(stats, logger=logger)
        if status == str(TileStatus.REJECTED):
            update_tile(
                t["tile_id"], status=TileStatus.REJECTED, rejection_reason=reason
            )
            counts["rejected"] += 1
        else:
            update_tile(t["tile_id"], status=TileStatus.CHEAP_VALID)
            counts["cheap_valid"] += 1

    logger.debug(
        f"  {batch_label} cheap eval: cheap_valid={counts['cheap_valid']} rejected={counts['rejected']}"
    )
    return counts


def filter_batch_imagery(
    tiles: list[dict], logger: logging.Logger, batch_label: str = ""
) -> dict[str, int]:
    """Stage 2: per-year S1+S2 availability. CHEAP_VALID -> VALID | REJECTED | FAILED."""
    extent = compute_batch_extent(tiles)
    n_rows, n_cols = extent["n_rows"], extent["n_cols"]
    geom = batch_bbox_geom(extent)

    logger.debug(f"{batch_label} imagery: {len(tiles)} tiles")
    t0 = time.time()
    try:
        raster = fetch_imagery_stats(
            geom,
            origin_x=extent["origin_x"],
            origin_y=extent["origin_y"],
            n_cols=n_cols,
            n_rows=n_rows,
        )
        logger.debug(f"  {batch_label} imagery fetch: {time.time()-t0:.1f}s")
    except Exception as exc:
        logger.error(
            f"  {batch_label} imagery fetch failed after {time.time()-t0:.1f}s — {exc}"
        )
        for t in tiles:
            update_tile(t["tile_id"], status=TileStatus.FAILED, error=str(exc))
        return {"failed": len(tiles)}

    shape_err = _check_shape(raster, IMAGERY_BAND_NAMES, n_rows, n_cols)
    if shape_err:
        logger.error(f"  {batch_label} {shape_err}")
        for t in tiles:
            update_tile(t["tile_id"], status=TileStatus.FAILED, error=shape_err)
        return {"failed": len(tiles)}

    counts = {"valid": 0, "rejected": 0}
    for t in tiles:
        row, col = _rowcol(t, extent)
        stats = {b: raster[b][row][col] for b in IMAGERY_BAND_NAMES}
        status, reason = evaluate_imagery_stats(stats, logger=logger)
        if status == str(TileStatus.REJECTED):
            update_tile(
                t["tile_id"], status=TileStatus.REJECTED, rejection_reason=reason
            )
            counts["rejected"] += 1
        else:
            update_tile(t["tile_id"], status=TileStatus.VALID)
            counts["valid"] += 1

    logger.debug(
        f"  {batch_label} imagery eval: valid={counts['valid']} rejected={counts['rejected']}"
    )
    return counts
