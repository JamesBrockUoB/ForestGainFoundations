from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


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

    valid_aois_path: Path = field(
        default_factory=lambda: _DATA_DIR / "aois" / "valid_aois.json"
    )
    registry_db_path: Path = field(
        default_factory=lambda: _DATA_DIR / "tiles" / "tile_registry.db"
    )
    aoi_audit_path: Path = field(
        default_factory=lambda: _DATA_DIR / "tiles" / "aoi_tile_audit.json"
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
    gain_canopy_min: float = 1.0
    s2_min_valid_frac: float = 0.95

    poll_interval: int = 30
    use_hpc: bool = field(default_factory=lambda: os.getenv("USE_HPC", "0") == "1")
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    @property
    def tile_size_m(self) -> int:
        return self.tile_pixels * self.scale

    @property
    def hpc_path(self) -> str | None:
        if self.hpc_base:
            return f"{self.hpc_base}/{self.drive_folder}"
        return None


settings = Settings()
