from __future__ import annotations

import ee
from config import settings
from export.composites import s2_availability


def build_full_valid(geom: ee.Geometry) -> ee.Image:
    availabilities = [s2_availability(geom, year) for year in settings.period_years]
    valid = availabilities[0]
    for avail in availabilities[1:]:
        valid = valid.And(avail)
    return valid.selfMask().rename("valid")
