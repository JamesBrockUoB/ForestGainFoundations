from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")


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

    non_tree_threshold_frac: int = 20
    min_tree_threshold_frac: int = 50

    # Allowed post-crossing reversions to non-forest in intermediate years
    # (never applies to year_end). See labels/gain.py.

    # Coverage-check band proxy for "is there usable imagery here at all". NOT the full band
    # set used for actual export
    s2_check_band: str = "B2"
    s1_check_band: str = "VV"
    min_s1_observations: int = 5

    # Per-year, per-sensor minimum valid-pixel fraction, checked for every
    # calendar year
    imagery_min_valid_frac: float = 0.99

    # Cloud Score+ cs_cdf threshold for S2 availability/export masking.
    cloud_score_thresh: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_SCORE_THRESH", "0.6"))
    )

    poll_interval: int = 30
    use_hpc: bool = field(default_factory=lambda: os.getenv("USE_HPC", "0") == "1")
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    filter_batch_size_cheap: int = 200
    filter_batch_size_imagery: int = 40

    tessera_year_timeout_s: int = 90

    tessera_max_concurrent_years: int = 4

    # explicitly for s1/s2/aee data
    # since we can only have 2 batch exports concurrently, 4 workers for half the years may be more efficient
    export_years_per_task: int = 4

    rclone_transfers: int = 8

    rclone_checkers: int = 8

    rclone_fast_list: bool = True

    rclone_verify_checksum: bool = False
    rclone_contimeout: str = "30s"

    rclone_timeout: str = "120s"

    rclone_low_level_retries: int = 10

    rclone_max_workers: int = 3

    def __post_init__(self) -> None:
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
            self.data_dir / "aois" / "valid_aois.json",
        )

    @property
    def year_start(self) -> int:
        return 2017

    @property
    def year_end(self) -> int:
        return 2024

    @property
    def years(self) -> list[int]:
        return list(range(self.year_start, self.year_end + 1))

    @property
    def tile_size_m(self) -> int:
        return self.tile_pixels * self.scale

    @property
    def hpc_path(self) -> str | None:
        if self.hpc_remote:
            return f"{self.hpc_remote}/{self.drive_folder}"
        return None


settings = Settings()
