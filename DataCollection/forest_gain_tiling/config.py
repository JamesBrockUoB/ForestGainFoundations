from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PERIOD_YEARS = {
    "p1": (2017, 2020),
    "p2": (2020, 2024),
}

# Canopy-height (Meta) is a single fixed 2020 snapshot with no per-period
# release — only meaningful/available for periods whose gain window ends
# in 2020. Tracked/exported as information when available, but never used
# to reject tiles (see filtering/tile_filter.py) — gating on it only for
# the period where it happens to align would make p1/p2 cheap-filter
# selection criteria inconsistent with each other.
_CANOPY_HEIGHT_AVAILABLE_PERIODS = {"p1"}

# ForTy (forest typology) is likewise a fixed 2020 snapshot. For p2, using
# it would mean labeling a gain event that happens after the snapshot was
# taken — not just less precise, but describing the wrong point in time.
# p2 exports omit pseudo-label bands entirely rather than including a
# mislabeled or flagged approximation.
_PSEUDO_LABELS_AVAILABLE_PERIODS = {"p1"}


@dataclass(frozen=True)
class Settings:
    gee_project: str = field(default_factory=lambda: os.getenv("GEE_PROJECT", ""))
    drive_folder: str = field(
        default_factory=lambda: os.getenv("DRIVE_FOLDER", "forest_gain_tiles")
    )
    drive_remote: str = field(
        default_factory=lambda: os.getenv("DRIVE_REMOTE", "gdrive")
    )
    gee_credentials: str = field(
        default_factory=lambda: str(
            _ROOT_DIR / os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
    )
    hpc_base: str | None = field(default_factory=lambda: os.getenv("HPC_BASE"))

    # Optional override for the Earth Engine OAuth refresh-token file path.
    # If unset, gee/auth.py falls back to the library default
    # (~/.config/earthengine/credentials) via ee.data.get_persistent_credentials().
    # Set this when the token file lives somewhere else on an HPC node —
    # e.g. because it was uploaded to a scratch/project directory rather
    # than the job user's home dir.
    ee_credentials_path: str | None = field(
        default_factory=lambda: os.getenv("EE_CREDENTIALS_PATH")
    )

    # Which validity interval this whole process operates under:
    # "p1" = 2017->2020, "p2" = 2020->2024. Read once from env at process
    # startup (same pattern as generate_aois.py) and fixed for the process
    # lifetime. Every command in main.py (plan/filter/run/status/audit/
    # reset) is scoped to tiles tagged with this period.
    period: str = field(default_factory=lambda: os.getenv("PERIOD", "p1"))

    registry_db_path: Path = field(
        default_factory=lambda: _DATA_DIR / "tiles" / "tile_registry.db"
    )
    log_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "logs"
    )

    tile_pixels: int = 128
    scale: int = 10
    crs: str = "EPSG:6933"
    crs_wkt: str = (
        'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
        'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
        'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],'
        'PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],'
        'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
        'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","6933"]]'
    )

    min_aoi_overlap_frac: float = 0.1
    gain_pct_min: float = 1.0
    ndvi_delta_min: float = 0.0

    # Coverage-check band subsets — a deliberately small, high-resolution
    # proxy for "is there usable imagery here at all". NOT the full band
    # set used for actual spectral analysis (see export/composites.py's
    # s2_composite / s1_composite, which keep the full band lists).
    s2_check_bands: tuple[str, ...] = ("B2", "B4", "B8")
    s1_check_bands: tuple[str, ...] = ("VV", "VH")

    # Per-year, per-sensor minimum valid-pixel fraction, checked for every
    # calendar year in the active period (see Settings.period_years).
    # Lowered from the old flat 0.95 S2-only threshold: several published
    # forest-gain pipelines tolerate materially higher residual cloud/
    # shadow presence per composite year than that. Tune via
    # IMAGERY_MIN_VALID_FRAC — treat 0.95 as overly conservative, not a
    # safe default.
    imagery_min_valid_frac: float = field(
        default_factory=lambda: float(os.getenv("IMAGERY_MIN_VALID_FRAC", "0.7"))
    )

    poll_interval: int = 30
    use_hpc: bool = field(default_factory=lambda: os.getenv("USE_HPC", "0") == "1")
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    filter_batch_size: int = 100

    def __post_init__(self) -> None:
        if self.period not in PERIOD_YEARS:
            raise ValueError(
                f"PERIOD must be one of {list(PERIOD_YEARS)}, got {self.period!r}"
            )

        # valid_aois_path depends on `period`, so it can't be a plain
        # default_factory field (a default_factory has no access to
        # sibling fields at construction time). Set it here instead.
        # object.__setattr__ is required because the dataclass is frozen.
        object.__setattr__(
            self,
            "valid_aois_path",
            _DATA_DIR / "aois" / f"valid_aois_{self.period}.json",
        )

    @property
    def year_start(self) -> int:
        return PERIOD_YEARS[self.period][0]

    @property
    def year_end(self) -> int:
        return PERIOD_YEARS[self.period][1]

    @property
    def period_years(self) -> list[int]:
        """e.g. p1 -> [2017,2018,2019,2020], p2 -> [2020,2021,2022,2023,2024]."""
        return list(range(self.year_start, self.year_end + 1))

    @property
    def tile_size_m(self) -> int:
        return self.tile_pixels * self.scale

    @property
    def hpc_path(self) -> str | None:
        if self.hpc_base:
            return f"{self.hpc_base}/{self.drive_folder}"
        return None

    @property
    def canopy_height_available(self) -> bool:
        return self.period in _CANOPY_HEIGHT_AVAILABLE_PERIODS

    @property
    def pseudo_labels_available(self) -> bool:
        return self.period in _PSEUDO_LABELS_AVAILABLE_PERIODS


settings = Settings()
