from pathlib import Path

import numpy as np
import rasterio
import torch
from config import BACKBONE_BAND_INDICES, PERIOD_YEARS, VALID_MASK_BAND_INDEX
from torch.utils.data import Dataset


class MultiTemporalGainDataset(Dataset):
    """
    Dataset loading N-timestep S1/S2 composite stacks exported by GEE.
    Extracts embedded validity mask directly from Band 14 (s2_valid_<year>).
    """

    def __init__(self, tile_dirs: list[Path], period: str = "p1"):
        self.tile_dirs = sorted(tile_dirs)
        self.period = period
        self.years = PERIOD_YEARS[period]
        self.band_indices = list(BACKBONE_BAND_INDICES.values())

    def __len__(self):
        return len(self.tile_dirs)

    def _load_tif_bands(self, path: Path, channels: list[int]) -> np.ndarray:
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

            # 1. Read model input features (S1 + S2 bands)
            frame = self._load_tif_bands(img_path, self.band_indices)  # (C, H, W)

            # S2 Reflectance scaling (0-10000 -> 0-1), leave SAR (VV, VH, VVVH) intact
            # Band positions 3 to 9 correspond to S2 optical bands in BACKBONE_BAND_INDICES
            frame[3:, :, :] = np.clip(frame[3:, :, :] / 10000.0, 0.0, 1.0)
            frames.append(frame)

            # 2. Extract embedded validity mask from Band 14
            valid_mask = self._load_tif_bands(img_path, [VALID_MASK_BAND_INDEX])[
                0
            ]  # (H, W)
            valid_masks.append(valid_mask)

        # Stack into (T, C, H, W)
        pixels = np.stack(frames, axis=0)
        pixels = np.nan_to_num(pixels, nan=0.0)

        # Combine temporal validity: pixel must be valid across all timesteps
        combined_valid = np.prod(np.stack(valid_masks, axis=0), axis=0)  # (H, W)

        confidence = self._load_tif_bands(
            tile_dir / "labels" / "gain_confidence.tif", [1]
        )[0]

        # Treat NaN confidence pixels as no-gain, zero weight
        confidence = np.nan_to_num(confidence, nan=0.0)

        # Derive binary presence mask
        gain_mask = (confidence > 0).astype(np.float32)

        # Derive soft confidence weight, normalized to 0-1
        gain_weight = np.clip(confidence / 100.0, 0.0, 1.0).astype(np.float32)

        return {
            "pixels": torch.from_numpy(pixels).float(),  # (T, C, H, W)
            "gain_mask": torch.from_numpy(gain_mask).float(),  # (H, W)
            "gain_weight": torch.from_numpy(gain_weight).float(),  # (H, W)
            "gain_valid": torch.from_numpy(combined_valid).float(),  # (H, W)
            "tile_id": tile_dir.name,
        }
