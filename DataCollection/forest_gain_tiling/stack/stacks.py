from __future__ import annotations

import ee
from config import settings
from export.composites import s2_availability


def build_full_valid(geom: ee.Geometry) -> ee.Image:
    return (
        s2_availability(geom, settings.year_start)
        .And(s2_availability(geom, settings.year_end))
        .selfMask()
        .rename("valid")
    )
