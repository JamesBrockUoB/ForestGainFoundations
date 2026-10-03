import hashlib
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from config import (
    BACKBONE_BAND_INDICES,
    NORM_CLIP,
    NORM_NAN_FILL,
    NORM_STATS,
    S1_BANDS,
    S2_BANDS,
    S2_SCALE,
    VALID_MASK_BAND_INDEX,
    YEARS,
)
from torch.utils.data import Dataset


def _bucket(name: str) -> float:
    """Stable value in [0, 1) from the folder name (md5, not Python's salted hash())."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10_000 / 10_000


def split_tile_dirs(data_dir, val_frac=0.1, test_frac=0.1, seed=0):
    """
    Returns (train, val, test).
    Test membership depends only on the tile's folder name, so it is fixed
    across seeds and stays fixed as new tiles are added.
    Train/val are split from the remainder with the seed.
    """
    tile_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    test = [d for d in tile_dirs if _bucket(d.name) < test_frac]
    rest = [d for d in tile_dirs if _bucket(d.name) >= test_frac]

    random.Random(seed).shuffle(rest)
    n_val = int(round(val_frac * len(tile_dirs)))
    return rest[n_val:], rest[:n_val], test


def read_physical(path: Path, band_names) -> np.ndarray:
    """Read bands as (C, H, W) float32 in physical units: S2 reflectance, S1 dB."""
    idx = [BACKBONE_BAND_INDICES[b] for b in band_names]
    with rasterio.open(path) as src:
        x = src.read(idx).astype(np.float32)
    for i, b in enumerate(band_names):
        if b in S2_BANDS:
            x[i] /= S2_SCALE
    return x


def normalize(x: np.ndarray, band_names) -> np.ndarray:
    """Per-band linear scaling (x - p1) / (p99 - p1). x: (..., C, H, W)."""
    out = np.empty(x.shape, dtype=np.float32)
    for i, b in enumerate(band_names):
        lo, hi = NORM_STATS[b]
        out[..., i, :, :] = (x[..., i, :, :] - lo) / (hi - lo)
    if NORM_CLIP is not None:
        np.clip(out, NORM_CLIP[0], NORM_CLIP[1], out=out)
    return np.nan_to_num(
        out, nan=NORM_NAN_FILL, posinf=NORM_NAN_FILL, neginf=NORM_NAN_FILL
    )


class MultiTemporalGainDataset(Dataset):
    BAND_GROUPS = {
        "s1": S1_BANDS,
        "s2": S2_BANDS,
    }

    def __init__(
        self,
        tile_dirs: list[Path],
        sources: tuple[str, ...] = ("s1", "s2"),
        augment: bool = False,
        crop_size: int | None = None,
        repeats: int = 1,
        gain_sigma: float = 0.03,
        frame_sigma: float = 0.01,
        frame_drop_p: float = 0.1,
        label_sigma: float = 0.0,
    ):
        """
        crop_size: random crop (train) / center crop (eval). None keeps full tile.
        repeats:   virtual dataset length multiplier (more steps per epoch).
        label_sigma: jitter on soft confidence weights (use only with soft loss).
        """
        if not sources:
            raise ValueError("At least one source must be selected")
        invalid_sources = set(sources) - self.BAND_GROUPS.keys()
        if invalid_sources:
            raise ValueError(f"Unknown sources: {invalid_sources}")

        self.tile_dirs = sorted(tile_dirs)
        self.years = YEARS
        self.sources = sources
        self.augment = augment
        self.crop_size = crop_size
        self.repeats = repeats
        self.gain_sigma = gain_sigma
        self.frame_sigma = frame_sigma
        self.frame_drop_p = frame_drop_p
        self.label_sigma = label_sigma

        self.band_names = tuple(
            band for source in sources for band in self.BAND_GROUPS[source]
        )

        missing = [b for b in self.band_names if b not in NORM_STATS]
        if missing:
            raise RuntimeError(
                f"No normalisation stats for {missing}. Run "
                "`python compute_norm_stats.py` and paste the output into "
                "NORM_STATS in config.py."
            )

        self.s2_channel_indices = [
            i for i, b in enumerate(self.band_names) if b in S2_BANDS
        ]

    def __len__(self):
        return len(self.tile_dirs) * self.repeats

    @property
    def num_channels(self) -> int:
        return len(self.band_names)

    # ------------------------------------------------------------------ crops
    def _crop(self, arrays, train: bool):
        """Same spatial crop for every array (last two dims are H, W)."""
        h, w = arrays[0].shape[-2:]
        s = self.crop_size
        if s is None or (h <= s and w <= s):
            return arrays
        if train:
            top, left = np.random.randint(0, h - s + 1), np.random.randint(0, w - s + 1)
        else:
            top, left = (h - s) // 2, (w - s) // 2
        return [a[..., top : top + s, left : left + s] for a in arrays]

    # ----------------------------------------------------------- augmentation
    def _augment(self, pixels, gain_mask, gain_weight, valid):
        """Runs in PHYSICAL units (before normalisation)."""
        T = pixels.shape[0]

        # Geometric: identical across frames and labels
        k, flip = np.random.randint(4), np.random.rand() < 0.5

        def geo(a):
            a = np.rot90(a, k, axes=(-2, -1))
            return np.ascontiguousarray(np.flip(a, -1) if flip else a)

        pixels, gain_mask, gain_weight, valid = map(
            geo, (pixels, gain_mask, gain_weight, valid)
        )

        # Spectral: per-sample gain x small per-frame gain on S2 reflectance
        s2 = self.s2_channel_indices
        if s2 and (self.gain_sigma > 0 or self.frame_sigma > 0):
            g = (1.0 + np.random.normal(0, self.gain_sigma)) * (
                1.0 + np.random.normal(0, self.frame_sigma, size=(T, 1, 1, 1))
            )
            pixels[:, s2] = pixels[:, s2] * g.astype(np.float32)

        # Interior frame dropout (keeps T fixed; never drops first/last year)
        for t in range(1, T - 1):
            if np.random.rand() < self.frame_drop_p:
                pixels[t] = pixels[t - 1]

        # Soft-label jitter (confidence weights only; binary mask untouched)
        if self.label_sigma > 0:
            gain_weight = np.clip(
                gain_weight + np.random.normal(0, self.label_sigma, gain_weight.shape),
                0.0,
                1.0,
            )

        return pixels, gain_mask, gain_weight.astype(np.float32), valid

    # ------------------------------------------------------------------- item
    def __getitem__(self, idx: int) -> dict:
        tile_dir = self.tile_dirs[idx % len(self.tile_dirs)]
        frames, valid_masks = [], []

        for year in self.years:
            img_path = tile_dir / "composites" / f"s1s2_{year}.tif"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing composite geotiff: {img_path}")

            frames.append(read_physical(img_path, self.band_names))

            with rasterio.open(img_path) as src:
                vm = src.read(VALID_MASK_BAND_INDEX).astype(np.float32)
            valid_masks.append(np.nan_to_num(vm, nan=0.0) > 0)

        pixels = np.stack(frames, axis=0)  # (T, C, H, W), physical units, may hold NaN
        combined_valid = np.all(np.stack(valid_masks, axis=0), axis=0).astype(
            np.float32
        )

        with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
            confidence = src.read(1).astype(np.float32)
        confidence = np.nan_to_num(confidence, nan=0.0)

        gain_mask = (confidence > 0).astype(np.float32)
        gain_weight = np.clip(confidence / 100.0, 0.0, 1.0).astype(np.float32)

        pixels, gain_mask, gain_weight, combined_valid = self._crop(
            [pixels, gain_mask, gain_weight, combined_valid], train=self.augment
        )

        if self.augment:
            pixels, gain_mask, gain_weight, combined_valid = self._augment(
                np.ascontiguousarray(pixels), gain_mask, gain_weight, combined_valid
            )

        pixels = normalize(pixels, self.band_names)

        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels)).float(),
            "gain_mask": torch.from_numpy(np.ascontiguousarray(gain_mask)).float(),
            "gain_weight": torch.from_numpy(np.ascontiguousarray(gain_weight)).float(),
            "gain_valid": torch.from_numpy(
                np.ascontiguousarray(combined_valid)
            ).float(),
            "tile_id": tile_dir.name,
        }
