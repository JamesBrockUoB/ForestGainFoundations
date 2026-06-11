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
        crs=settings.crs,
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
) -> tuple[bool, str | None]:
    viability = score_viability(geom, gain_validated, ds)

    if (
        viability["ndvi_delta"] <= settings.ndvi_delta_min
        or viability["gain_canopy_mean"] < settings.gain_canopy_min
    ):
        return False, f"viability={viability}"

    return True, None


def check_s2_coverage(geom: ee.Geometry) -> tuple[bool, str | None]:
    fracs = {
        year: s2_coverage_frac(geom, year).getInfo() for year in (2016, 2020, 2025)
    }

    if any(f < settings.s2_min_valid_frac for f in fracs.values()):
        return False, f"s2_coverage={fracs}"

    return True, None


def filter_tile(tile: dict, ds: Datasets) -> tuple[str, str | None]:
    """
    Run cheap filters on a single tile.
    Returns (new_status, rejection_reason).
    """
    geom = tile_geom(tile)
    ct = crs_transform(tile)

    ok, reason, gain_validated, _ = check_gain(geom, ct, ds)
    if not ok:
        return str(TileStatus.REJECTED), reason

    ok, reason = check_viability(geom, gain_validated, ds)
    if not ok:
        return str(TileStatus.REJECTED), reason

    ok, reason = check_s2_coverage(geom)
    if not ok:
        return str(TileStatus.REJECTED), reason

    return str(TileStatus.VALID), None
