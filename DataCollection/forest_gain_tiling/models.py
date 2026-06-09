from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from enums import TileStatus


@dataclass
class TileEntry:
    tile_id: str
    xi: int
    yi: int
    x_min_m: float
    y_min_m: float
    x_max_m: float
    y_max_m: float
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    biome: str = "Unknown"
    region: str = "Unknown"
    aoi_ids: list[str] = field(default_factory=list)
    status: TileStatus = TileStatus.PENDING
    gee_task_id: str | None = None
    submitted_at: str | None = None
    completed_at: str | None = None
    rejection_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = str(self.status)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TileEntry:
        d = dict(d)
        d["status"] = TileStatus(d["status"])
        return cls(**d)


@dataclass
class AoiAuditEntry:
    biome: str
    region: str
    tile_counts: dict[str, int]
    total_tiles: int
    complete_tiles: int
    has_coverage: bool
