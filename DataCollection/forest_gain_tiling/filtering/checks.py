from __future__ import annotations

import ee
from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from export.composites import s2_coverage_frac
from labels.gain import build_gain_layer
from labels.viability import score_viability
from tiling.grid import crs_transform, tile_geom


def check_gain(
    geom: ee.Geometry, ct: list, ds: Datasets
) -> tuple[bool, str | None, ee.Image, ee.Image]:
    gain_validated, gain_binary = build_gain_layer(geom, ds)

    gain_stats = gain_binary.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=settings.scale,
        crs=settings.crs_wkt,
        crsTransform=ct,
        maxPixels=1_000_000_000,
    )
    gain_pct = (
        ee.Number(ee.Algorithms.If(gain_stats.get("gain"), gain_stats.get("gain"), 0))
        .multiply(100)
        .getInfo()
    )

    if gain_pct < settings.gain_pct_min:
        return (
            False,
            f"gain_pct={gain_pct:.3f} < {settings.gain_pct_min}",
            gain_validated,
            gain_binary,
        )

    return True, None, gain_validated, gain_binary


def check_viability(
    geom: ee.Geometry, gain_validated: ee.Image, ds: Datasets
) -> tuple[bool, str | None, dict]:
    viability = score_viability(geom, gain_validated, ds)

    if (
        viability["ndvi_delta"] <= settings.ndvi_delta_min
        or viability["gain_canopy_mean"] < settings.gain_canopy_min
    ):
        return False, f"viability={viability}", viability

    return True, None, viability


def check_s2_coverage(geom: ee.Geometry) -> tuple[bool, str | None, dict]:
    fracs = {
        year: s2_coverage_frac(geom, year).getInfo() for year in (2017, 2020, 2025)
    }

    if any(f < settings.s2_min_valid_frac for f in fracs.values()):
        return False, f"s2_coverage={fracs}", fracs

    return True, None, fracs


def filter_tile(tile: dict, ds: Datasets) -> tuple[str, str | None]:
    """
    Run cheap filters on a single tile via per-tile reduceRegion calls
    (original, slow method — used for validation against the raster
    batch method, and as a fallback for single-tile debugging).
    Returns (new_status, rejection_reason).
    """
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    ok, reason, gain_validated, _ = check_gain(geom, ct, ds)
    if not ok:
        return str(TileStatus.REJECTED), reason

    ok, reason, _ = check_viability(geom, gain_validated, ds)
    if not ok:
        return str(TileStatus.REJECTED), reason

    ok, reason, _ = check_s2_coverage(geom)
    if not ok:
        return str(TileStatus.REJECTED), reason

    return str(TileStatus.VALID), None


def filter_tile_with_stats(tile: dict, ds: Datasets) -> dict:
    """
    Like filter_tile, but returns the raw per-tile stats alongside the
    decision, for comparison against the raster-batch method.
    """
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    gain_validated, gain_binary = build_gain_layer(geom, ds)
    gain_stats = gain_binary.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=settings.scale,
        crs=settings.crs_wkt,
        crsTransform=ct,
        maxPixels=1_000_000_000,
    )
    gain_pct = (
        ee.Number(ee.Algorithms.If(gain_stats.get("gain"), gain_stats.get("gain"), 0))
        .multiply(100)
        .getInfo()
    )

    viability = score_viability(geom, gain_validated, ds)

    fracs = {
        year: s2_coverage_frac(geom, year).getInfo() for year in (2017, 2020, 2025)
    }

    status, reason = filter_tile(tile, ds)

    return {
        "tile_id": tile["tile_id"],
        "gain_pct": gain_pct,
        "ndvi_delta": viability["ndvi_delta"],
        "gain_canopy_mean": viability["gain_canopy_mean"],
        "s2_2017": fracs[2017],
        "s2_2020": fracs[2020],
        "s2_2025": fracs[2025],
        "status": status,
        "reason": reason,
    }
