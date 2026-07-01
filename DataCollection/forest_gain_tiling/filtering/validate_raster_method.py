"""
Validation script: compare the raster-batch filter method against the
original per-tile reduceRegion method for a single AOI.

Usage:
  python -m filtering.validate_raster_method --aoi-id <AOI_ID> [--max-tiles N]

Prints a per-tile diff table and a summary of agreement/disagreement.
Does NOT write any status changes to the DB.
"""

from __future__ import annotations

import argparse
import logging

import ee
from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from filtering.aoi_filter import (
    aoi_bbox_geom,
    compute_aoi_extent,
    evaluate_tile_stats,
)
from filtering.checks import filter_tile_with_stats
from filtering.raster_stats import BAND_NAMES, NO_GAIN_SENTINEL, fetch_tile_stats
from registry.store import _get_db


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("validate_raster")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        logger.addHandler(sh)
    return logger


def get_aoi_tiles(aoi_id: str, status: str | None = None) -> list[dict]:
    db = _get_db()
    with db._conn() as conn:
        query = """
            SELECT t.tile_id, t.xi, t.yi, t.x_min_m, t.y_min_m, t.x_max_m, t.y_max_m,
                   t.min_lon, t.min_lat, t.max_lon, t.max_lat, t.status
            FROM tiles t
            INNER JOIN tile_aois ta ON t.tile_id = ta.tile_id
            WHERE ta.aoi_id = ?
        """
        params: list = [aoi_id]
        if status is not None:
            query += " AND t.status = ?"
            params.append(status)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi-id", required=True)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--status",
        default=str(TileStatus.PENDING),
        help="Only compare tiles currently in this status (default: pending)",
    )
    args = parser.parse_args()

    logger = setup_logging()

    ee.Initialize(
        ee.ServiceAccountCredentials(None, settings.gee_credentials),
        project=settings.gee_project,
    )
    ds = Datasets()

    tiles = get_aoi_tiles(args.aoi_id, status=args.status)
    if not tiles:
        logger.error(f"No tiles found for AOI {args.aoi_id} with status={args.status}")
        return

    if args.max_tiles:
        tiles = tiles[: args.max_tiles]

    logger.info(f"AOI {args.aoi_id}: {len(tiles):,} tiles to compare")

    # --- Raster batch method ---
    extent = compute_aoi_extent(tiles)
    geom = aoi_bbox_geom(extent)

    logger.info(
        f"Raster extent: {extent['n_cols']}x{extent['n_rows']} tiles "
        f"(origin_x={extent['origin_x']}, origin_y={extent['origin_y']})"
    )

    raster = fetch_tile_stats(
        geom,
        ds,
        origin_x=extent["origin_x"],
        origin_y=extent["origin_y"],
        n_cols=extent["n_cols"],
        n_rows=extent["n_rows"],
    )

    n_rows = extent["n_rows"]
    n_cols = extent["n_cols"]

    for band in BAND_NAMES:
        arr = raster.get(band)
        if arr is None or len(arr) != n_rows or any(len(row) != n_cols for row in arr):
            logger.error(
                f"Raster shape mismatch for band {band}: "
                f"expected {n_rows}x{n_cols}, "
                f"got {len(arr) if arr else 0}x{len(arr[0]) if arr and arr[0] else 0}"
            )
            return

    # --- Compare per tile ---
    agree = 0
    disagree = 0

    header = (
        f"{'tile_id':<22} {'raster_status':<10} {'tile_status':<10} "
        f"{'r_gain%':>8} {'t_gain%':>8} "
        f"{'r_ndvi':>8} {'t_ndvi':>8} "
        f"{'r_canopy':>8} {'t_canopy':>8} "
        f"{'r_s2_17':>7} {'t_s2_17':>7} "
        f"{'r_s2_20':>7} {'t_s2_20':>7} "
        f"{'r_s2_25':>7} {'t_s2_25':>7}"
    )
    print(header)
    print("-" * len(header))

    for t in tiles:
        xi_local = t["xi"] - extent["xi_min"]
        yi_local = t["yi"] - extent["yi_min"]
        row = (n_rows - 1) - yi_local
        col = xi_local

        r_stats = {band: raster[band][row][col] for band in BAND_NAMES}
        r_status, r_reason = evaluate_tile_stats(r_stats)

        t_result = filter_tile_with_stats(t, ds)
        t_status = t_result["status"]

        match = "==" if r_status == t_status else "!!"
        if r_status == t_status:
            agree += 1
        else:
            disagree += 1

        def fmt_ndvi_canopy(r_val, t_val):
            r_str = "no_gain" if r_val == NO_GAIN_SENTINEL else f"{r_val:.3f}"
            t_str = f"{t_val:.3f}"
            return r_str, t_str

        r_ndvi_s, t_ndvi_s = fmt_ndvi_canopy(
            r_stats["ndvi_delta"], t_result["ndvi_delta"]
        )
        r_canopy_s, t_canopy_s = fmt_ndvi_canopy(
            r_stats["canopy_mean"], t_result["gain_canopy_mean"]
        )

        print(
            f"{t['tile_id']:<22} {r_status:<10} {t_status:<10} "
            f"{r_stats['gain_frac']*100:>8.3f} {t_result['gain_pct']:>8.3f} "
            f"{r_ndvi_s:>8} {t_ndvi_s:>8} "
            f"{r_canopy_s:>8} {t_canopy_s:>8} "
            f"{r_stats['s2_2017']:>7.3f} {t_result['s2_2017']:>7.3f} "
            f"{r_stats['s2_2020']:>7.3f} {t_result['s2_2020']:>7.3f} "
            f"{r_stats['s2_2025']:>7.3f} {t_result['s2_2025']:>7.3f} "
            f"{match}"
        )

        if r_status != t_status:
            logger.warning(
                f"DISAGREEMENT {t['tile_id']}: raster={r_status} ({r_reason}) "
                f"vs tile={t_status} ({t_result['reason']})"
            )

    print()
    logger.info(f"Agree: {agree}/{len(tiles)}  Disagree: {disagree}/{len(tiles)}")


if __name__ == "__main__":
    main()
