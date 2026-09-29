from enum import Enum, IntEnum


class TileStatus(str, Enum):
    PENDING = "pending"
    CHEAP_VALID = "cheap_valid"
    VALID = "valid"
    SUBMITTED = "submitted"
    COMPLETE = "complete"
    REJECTED = "rejected"
    FAILED = "failed"
    # GEE/AEE outputs done and pushed; at least one TESSERA year is missing.
    TESSERA_MISSING = "tessera_missing"

    def __str__(self) -> str:
        return self.value


# NOTE: TESSERA_MISSING is included so `run` never re-submits (and overwrites)
# tiles whose GEE/AEE exports are already done. Only `retry-tessera` or an
# explicit `reset` moves a tile out of this status. If anything treats
# "terminal" as "finished successfully", exclude TESSERA_MISSING there.
TERMINAL_STATUSES: frozenset["TileStatus"] = frozenset(
    {TileStatus.COMPLETE, TileStatus.REJECTED, TileStatus.TESSERA_MISSING}
)


class DWClass(IntEnum):
    """Dynamic World's 9 land cover classes — the label values in DW's
    `label` band (already an argmax over these same classes' probability
    bands) map 1:1 onto this enum, and each member's name is also the
    exact band name for that class's probability in DW's raw output."""

    water = 0
    trees = 1
    grass = 2
    flooded_vegetation = 3
    crops = 4
    shrub_and_scrub = 5
    built = 6
    bare = 7
    snow_and_ice = 8


class ESRIClass(IntEnum):
    """ESRI/Impact Observatory 10m Annual Land Cover classes (the 9-class
    v3 schema used by ESRI_Global-LULC_10m_TS, 2017-2024). Values are the
    remapped class values used by this code."""

    water = 1
    trees = 2
    flooded_vegetation = 3
    crops = 4
    built = 5
    bare = 6
    snow_and_ice = 7
    clouds = 8
    rangeland = 9


ESRI_RAW_VALUES = [1, 2, 4, 5, 7, 8, 9, 10, 11]
ESRI_REMAPPED_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
