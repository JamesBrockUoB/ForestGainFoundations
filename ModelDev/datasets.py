from pathlib import Path

import numpy as np
import rasterio
import torch
from config import (
    BACKBONE_BAND_INDICES,
    S1_BANDS,
    S2_BANDS,
    VALID_MASK_BAND_INDEX,
    settings,
)
from torch.utils.data import Dataset


class MultiTemporalGainDataset(Dataset):
    BAND_GROUPS = {
        "s1": S1_BANDS,
        "s2": S2_BANDS,
    }

    def __init__(
        self,
        tile_dirs: list[Path],
        sources: tuple[str, ...] = ("s1", "s2"),
    ):
        self.tile_dirs = sorted(tile_dirs)
        self.years = settings.years

        invalid_sources = set(sources) - self.BAND_GROUPS.keys()
        if invalid_sources:
            raise ValueError(f"Unknown sources: {invalid_sources}")
        if not sources:
            raise ValueError("At least one source must be selected")

        self.sources = sources
        self.band_names = tuple(
            band for source in sources for band in self.BAND_GROUPS[source]
        )
        self.band_indices = [BACKBONE_BAND_INDICES[band] for band in self.band_names]

        # Classify band array positions for fast vectorised scaling
        self.s2_channel_indices = [
            i for i, b in enumerate(self.band_names) if b in S2_BANDS
        ]
        self.s1_channel_indices = [
            i for i, b in enumerate(self.band_names) if b in S1_BANDS
        ]

    def __len__(self):
        return len(self.tile_dirs)

    @property
    def num_channels(self) -> int:
        return len(self.band_indices)

    def _load_tif_bands(self, path: Path, channels: list[int]) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(channels).astype(np.float32)

    def _scale_physical_units(self, frame: np.ndarray) -> np.ndarray:
        """
        Scales optical reflectance and radar dB backscatter to [0.0, 1.0]
        using global physical boundaries (Sentinel Hub best practices).
        """
        scaled = np.empty_like(frame, dtype=np.float32)

        # 1. Sentinel-2: Convert DN (0-10000) to reflectance factor [0.0, 1.0]
        if self.s2_channel_indices:
            s2_data = frame[self.s2_channel_indices] / 10000.0
            scaled[self.s2_channel_indices] = np.clip(s2_data, 0.0, 1.0)

        # 2. Sentinel-1: Clip dB backscatter to [-35.0, 5.0] dB and scale linearly to [0.0, 1.0]
        if self.s1_channel_indices:
            s1_data = frame[self.s1_channel_indices]
            s1_clipped = np.clip(s1_data, -35.0, 5.0)
            scaled[self.s1_channel_indices] = (s1_clipped + 35.0) / 40.0

        return scaled

    def __getitem__(self, idx: int) -> dict:
        tile_dir = self.tile_dirs[idx]
        frames, valid_masks = [], []

        for year in self.years:
            img_path = tile_dir / "composites" / f"s1s2_{year}.tif"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing composite geotiff: {img_path}")

            # Read selected bands in exact channel order
            raw_frame = self._load_tif_bands(img_path, self.band_indices)

            # Apply physical normalization per modality
            norm_frame = self._scale_physical_units(raw_frame)
            frames.append(norm_frame)

            valid_mask = self._load_tif_bands(img_path, [VALID_MASK_BAND_INDEX])[0]
            valid_masks.append(valid_mask)

        pixels = np.stack(frames, axis=0)
        pixels = np.nan_to_num(pixels, nan=0.0)

        combined_valid = np.prod(np.stack(valid_masks, axis=0), axis=0)

        confidence = self._load_tif_bands(
            tile_dir / "labels" / "gain_confidence.tif", [1]
        )[0]
        confidence = np.nan_to_num(confidence, nan=0.0)

        gain_mask = (confidence > 0).astype(np.float32)
        gain_weight = np.clip(confidence / 100.0, 0.0, 1.0).astype(np.float32)

        return {
            "pixels": torch.from_numpy(pixels).float(),
            "gain_mask": torch.from_numpy(gain_mask).float(),
            "gain_weight": torch.from_numpy(gain_weight).float(),
            "gain_valid": torch.from_numpy(combined_valid).float(),
            "tile_id": tile_dir.name,
        }
