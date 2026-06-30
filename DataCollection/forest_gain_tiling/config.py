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
    crs: str = "EPSG:3857"

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
