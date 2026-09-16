from pathlib import Path

import numpy as np
import rasterio
import torch
from config import (
    BACKBONE_BAND_INDICES,
    PERIOD_YEARS,
    S1_BANDS,
    S2_BANDS,
    VALID_MASK_BAND_INDEX,
)
from torch.utils.data import Dataset


class MultiTemporalGainDataset(Dataset):
    """
    Dataset loading N-timestep S1/S2 composite stacks exported by GEE.

    `sources` controls which modalities are loaded:
        ("s1",)          -> S1 only
        ("s2",)          -> S2 only
        ("s1", "s2")     -> S1 + S2

    The channel count and S2 scaling are derived automatically from the
    selected bands.
    """

    BAND_GROUPS = {
        "s1": S1_BANDS,
        "s2": S2_BANDS,
    }

    def __init__(
        self,
        tile_dirs: list[Path],
        period: str = "p1",
        sources: tuple[str, ...] = ("s1", "s2"),
    ):
        self.tile_dirs = sorted(tile_dirs)
        self.period = period
        self.years = PERIOD_YEARS[period]

        invalid_sources = set(sources) - self.BAND_GROUPS.keys()
        if invalid_sources:
            raise ValueError(
                f"Unknown sources: {invalid_sources}. "
                f"Expected one or more of: {tuple(self.BAND_GROUPS)}"
            )

        if not sources:
            raise ValueError("At least one source must be selected")

        self.sources = sources

        # Resolve semantic band names -> 1-based GeoTIFF band indices.
        self.band_names = tuple(
            band for source in sources for band in self.BAND_GROUPS[source]
        )

        self.band_indices = [BACKBONE_BAND_INDICES[band] for band in self.band_names]

        # Track which loaded channels need S2 reflectance scaling.
        self.s2_channel_indices = [
            i for i, band in enumerate(self.band_names) if band in S2_BANDS
        ]

    def __len__(self):
        return len(self.tile_dirs)

    @property
    def num_channels(self) -> int:
        return len(self.band_indices)

    def _load_tif_bands(
        self,
        path: Path,
        channels: list[int],
    ) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read(channels)

        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        tile_dir = self.tile_dirs[idx]

        frames = []
        valid_masks = []

        for year in self.years:
            img_path = tile_dir / "composites" / f"s1s2_{year}.tif"

            if not img_path.exists():
                raise FileNotFoundError(f"Missing composite geotiff: {img_path}")

            # Load only the requested modalities.
            frame = self._load_tif_bands(
                img_path,
                self.band_indices,
            )

            # Scale S2 reflectance from 0-10000 -> 0-1.
            # S1 SAR channels are left unchanged.
            if self.s2_channel_indices:
                frame[self.s2_channel_indices] = np.clip(
                    frame[self.s2_channel_indices] / 10000.0,
                    0.0,
                    1.0,
                )

            frames.append(frame)

            # Extract embedded validity mask from Band 14.
            valid_mask = self._load_tif_bands(
                img_path,
                [VALID_MASK_BAND_INDEX],
            )[0]

            valid_masks.append(valid_mask)

        # Stack into (T, C, H, W).
        pixels = np.stack(frames, axis=0)
        pixels = np.nan_to_num(pixels, nan=0.0)

        # Pixel must be valid across all timesteps.
        combined_valid = np.prod(
            np.stack(valid_masks, axis=0),
            axis=0,
        )

        confidence = self._load_tif_bands(
            tile_dir / "labels" / "gain_confidence.tif",
            [1],
        )[0]

        # NaN confidence = no gain, zero weight.
        confidence = np.nan_to_num(
            confidence,
            nan=0.0,
        )

        # Binary gain-presence mask.
        gain_mask = (confidence > 0).astype(np.float32)

        # Soft confidence weight in [0, 1].
        gain_weight = np.clip(
            confidence / 100.0,
            0.0,
            1.0,
        ).astype(np.float32)

        return {
            "pixels": torch.from_numpy(pixels).float(),
            "gain_mask": torch.from_numpy(gain_mask).float(),
            "gain_weight": torch.from_numpy(gain_weight).float(),
            "gain_valid": torch.from_numpy(combined_valid).float(),
            "tile_id": tile_dir.name,
        }
