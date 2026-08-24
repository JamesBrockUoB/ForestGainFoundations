from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .config import (
    BACKBONE_BAND_INDICES,
    NUM_CLASSES,
    PERIOD_HAS_TYPOLOGY,
    PERIOD_YEARS,
)


class GainTileDataset(Dataset):
    """
    One item per tile: multi-year 6-band Prithvi-native stack plus weak
    labels. `period` ("p1" or "p2") determines both the year sequence
    (PERIOD_YEARS) and whether typology labels are even attempted
    (PERIOD_HAS_TYPOLOGY) -- p2 tiles never read pseudo_labels.tif, since
    that label source doesn't exist for p2 at all, not just at low
    coverage. A dataset instance is always single-period: p1 and p2 tiles
    have different T and can't be collated into the same batch, so build
    one dataset/dataloader per period.

    Expected layout:
        <tile_dir>/composites/s1s2_<year>.tif   (13-band composite)
        <tile_dir>/labels/pseudo_labels.tif      (band 5 = dominant class,
                                                   band 6 = confidence;
                                                   p1 only)
        <tile_dir>/labels/gain_confidence.tif    (continuous, 50-100 range,
                                                   NaN outside gain coverage;
                                                   both periods)
        <tile_dir>/embeddings/aee_<year>.tif    64-band embeddings product
        <tile_dir>/embeddings/tessera_<year>.tif 128-band embeddings product
        <tile_dir>/static/fabdem.tif             elevation map product
        <tile_dir>/static/slope.tif              slope data map
        <tile_dir>/static/protected_area.tif             protected area map
        <tile_dir>/metadata.json                 collection of tile-level data for stratification and feature use
    """

    def __init__(
        self,
        tile_dirs: list[Path],
        period: str,
        band_stats: dict[str, tuple[float, float]] | None = None,
    ):
        if period not in PERIOD_YEARS:
            raise ValueError(f"Unknown period '{period}'")
        self.tile_dirs = tile_dirs
        self.period = period
        self.years = PERIOD_YEARS[period]
        self.has_typology = PERIOD_HAS_TYPOLOGY[period]
        self.band_stats = band_stats or {b: (0.0, 1.0) for b in BACKBONE_BAND_INDICES}

    def __len__(self):
        return len(self.tile_dirs)

    @property
    def num_frames(self) -> int:
        return len(self.years)

    def _read_year(self, tile_dir: Path, year: int) -> np.ndarray:
        path = tile_dir / "composites" / f"s1s2_{year}.tif"
        with rasterio.open(path) as src:
            bands = []
            for name, idx in BACKBONE_BAND_INDICES.items():
                arr = src.read(idx).astype(np.float32)
                mean, std = self.band_stats[name]
                bands.append((arr - mean) / std)
            return np.stack(bands, axis=0)  # (C, H, W)

    def _read_typology(self, tile_dir: Path, gain_valid: np.ndarray):
        if not self.has_typology:
            return np.zeros(NUM_CLASSES, dtype=np.float32), 0.0

        with rasterio.open(tile_dir / "labels" / "pseudo_labels.tif") as src:
            pseudo = src.read()
        dominant = pseudo[4]
        confidence = pseudo[5]
        labelled = (dominant != -9999) & (confidence != -9999) & gain_valid

        if labelled.sum() == 0:
            return np.zeros(NUM_CLASSES, dtype=np.float32), 0.0

        counts = np.bincount(dominant[labelled].astype(np.int64), minlength=NUM_CLASSES)
        class_dist = counts / counts.sum()
        cls_weight = float(confidence[labelled].mean())
        return class_dist.astype(np.float32), cls_weight

    def __getitem__(self, i):
        tile_dir = self.tile_dirs[i]

        frames = [self._read_year(tile_dir, y) for y in self.years]
        stack = np.stack(frames, axis=0)  # (T, C, H, W)
        stack = np.transpose(stack, (1, 0, 2, 3))  # (C, T, H, W)
        pixels = torch.from_numpy(stack).float()

        with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
            gain_conf = src.read(1).astype(np.float32)
        gain_valid = ~np.isnan(gain_conf)
        gain_weight = np.clip(
            (np.nan_to_num(gain_conf, nan=50.0) - 50.0) / 50.0, 0.0, 1.0
        )
        gain_mask = gain_valid.astype(np.float32)

        class_dist, cls_weight = self._read_typology(tile_dir, gain_valid)

        return {
            "pixels": pixels,
            "gain_mask": torch.from_numpy(gain_mask),
            "gain_weight": torch.from_numpy(gain_weight),
            "gain_valid": torch.from_numpy(gain_valid.astype(np.float32)),
            "class_dist": torch.from_numpy(class_dist),
            "cls_weight": torch.tensor(cls_weight, dtype=torch.float32),
            "tile_id": tile_dir.name,
        }
