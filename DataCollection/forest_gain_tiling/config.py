from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

PERIOD_YEARS = {
    "p1": (2017, 2020),
    "p2": (2020, 2024),
}

# ForTy (forest typology) is likewise a fixed 2020 snapshot. For p2, using
# it would mean labeling a gain event that happens after the snapshot was
# taken — not just less precise, but describing the wrong point in time.
# p2 exports omit pseudo-label bands entirely rather than including a
# mislabeled or flagged approximation.
_PSEUDO_LABELS_AVAILABLE_PERIODS = {"p1"}


@dataclass(frozen=True)
class Settings:
    # Project root (forest_gain_tiling/../../) and its data/ subdir.
    # Public settings fields rather than module-private constants so any
    # module can reference settings.root_dir / settings.data_dir instead
    # of each recomputing Path(__file__).resolve().parents[N] itself.
    root_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )

    gee_project: str = field(default_factory=lambda: os.getenv("GEE_PROJECT", ""))
    drive_folder: str = field(
        default_factory=lambda: os.getenv("DRIVE_FOLDER", "forest_gain_tiles")
    )
    drive_remote: str = field(
        default_factory=lambda: os.getenv("DRIVE_REMOTE", "gdrive")
    )
    gee_credentials: str = field(default_factory=lambda: "")
    hpc_remote: str | None = field(default_factory=lambda: os.getenv("HPC_REMOTE"))

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

    registry_db_path: Path = field(default_factory=Path)
    log_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "logs"
    )

    tile_pixels: int = field(
        default_factory=lambda: int(os.getenv("TILE_PIXELS", "256"))
    )
    scale: int = field(default_factory=lambda: int(os.getenv("TILE_SCALE", "10")))

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
    ndvi_trend_min: float = 0.0
    gain_sustain_dropout_tolerance: int = 1
    min_pseudo_gain_frac: float = 0.5

    non_tree_threshold_frac: int = 20
    min_tree_threshold_frac: int = 50

    # Allowed post-crossing reversions to non-forest in intermediate years
    # (never applies to year_end). See labels/gain.py.

    # Coverage-check band proxy for "is there usable imagery here at all". NOT the full band
    # set used for actual export
    s2_check_band: str = "B2"
    min_s1_observations: int = 5

    # Per-year, per-sensor minimum valid-pixel fraction, checked for every
    # calendar year in the active period
    imagery_min_valid_frac: float = 1

    # Cloud Score+ cs_cdf threshold for S2 availability/export masking.
    cloud_score_thresh: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_SCORE_THRESH", "0.6"))
    )

    poll_interval: int = 30
    use_hpc: bool = field(default_factory=lambda: os.getenv("USE_HPC", "0") == "1")
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    aee_source: str = field(default_factory=lambda: os.getenv("AEE_SOURCE", "gee"))

    filter_batch_size: int = 100

    def __post_init__(self) -> None:
        if self.period not in PERIOD_YEARS:
            raise ValueError(
                f"PERIOD must be one of {list(PERIOD_YEARS)}, got {self.period!r}"
            )

        object.__setattr__(
            self,
            "gee_credentials",
            str(self.root_dir / os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")),
        )
        object.__setattr__(
            self,
            "registry_db_path",
            self.data_dir / "tiles" / "tile_registry.db",
        )
        object.__setattr__(
            self,
            "valid_aois_path",
            self.data_dir / "aois" / f"valid_aois_{self.period}.json",
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
        if self.hpc_remote:
            return f"{self.hpc_remote}/{self.drive_folder}"
        return None

    @property
    def pseudo_labels_available(self) -> bool:
        return self.period in _PSEUDO_LABELS_AVAILABLE_PERIODS


settings = Settings()
