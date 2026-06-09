from enum import Enum


class TileStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    COMPLETE = "complete"
    REJECTED = "rejected"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


TERMINAL_STATUSES: frozenset["TileStatus"] = frozenset(
    {TileStatus.COMPLETE, TileStatus.REJECTED}
)


class PseudoLabel(int, Enum):
    AGROCROP = 0
    NAT_REGEN = 1
    PLANTATION = 2
    RESTORATION = 3
