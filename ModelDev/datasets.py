import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from config import (
    BAND_FILE_INDEX,
    EMBEDDING_ABS_MAX,
    EMBEDDING_SOURCES,
    NORM_CLIP,
    NORM_NAN_FILL,
    NORM_STATS,
    S2_BANDS,
    S2_SCALE,
    SOURCES,
    VALID_MASK_BAND_INDEX,
    YEARS,
)
from mask_cleaning import clean_labels
from torch.utils.data import Dataset


def split_tile_dirs(data_dir, val_frac=0.2, seed=0):
    tile_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    random.Random(seed).shuffle(tile_dirs)
    n_val = max(1, round(val_frac * len(tile_dirs)))
    return tile_dirs[n_val:], tile_dirs[:n_val]


def clean_gain_mask(gain_mask, gain_weight, min_px=0, fill_px=0):
    """Backwards-compatible wrapper (no confidence step, every pixel valid). See mask_cleaning.clean_labels."""
    mask, weight, _ = clean_labels(
        gain_mask,
        gain_weight,
        np.ones(gain_mask.shape, dtype=bool),
        min_px=min_px,
        fill_px=fill_px,
    )
    return mask, weight


def read_physical(path: Path, band_names, mask_nodata: bool = False) -> np.ndarray:
    """Read bands as (C, H, W) float32. S2 -> reflectance, S1 -> dB,
    embeddings -> raw values (scaled later by NORM_STATS)."""
    idx = [BAND_FILE_INDEX[b] for b in band_names]
    with rasterio.open(path) as src:
        x = src.read(idx).astype(np.float32)
        nodata = src.nodata
    if mask_nodata:
        if nodata is not None and not np.isnan(nodata):
            x[x == np.float32(nodata)] = np.nan
        # Corrupt values (inf, or finite but absurd): whole pixel becomes missing.
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(x) | (np.abs(x) > EMBEDDING_ABS_MAX)
        x[:, bad.any(axis=0)] = np.nan
    for i, b in enumerate(band_names):
        if b in S2_BANDS:
            x[i] /= S2_SCALE
    return x


def normalize(x: np.ndarray, band_names) -> np.ndarray:
    """S1/S2: (x - p1) / (p99 - p1), clipped, NaN -> NORM_NAN_FILL.
    Embeddings (no NORM_STATS entry): raw values, NaN -> 0."""
    out = x.astype(np.float32, copy=True)
    for i, b in enumerate(band_names):
        ch = out[..., i, :, :]
        if b in NORM_STATS:
            lo, hi = NORM_STATS[b]
            ch = np.clip((ch - lo) / (hi - lo), *NORM_CLIP)
            fill = NORM_NAN_FILL
        else:
            fill = 0.0
        out[..., i, :, :] = np.nan_to_num(ch, nan=fill, posinf=fill, neginf=fill)
    return out


class MultiTemporalGainDataset(Dataset):
    def __init__(
        self,
        tile_dirs: list[Path],
        sources: tuple[str, ...] = ("s1", "s2"),
        augment: bool = False,
        crop_size: int | None = None,
        repeats: int = 1,
        gain_sigma: float = 0.03,
        frame_sigma: float = 0.01,
        frame_drop_p: float = 0.0,
        label_sigma: float = 0.0,
        geo_aug: bool = True,
        min_gain_px: int = 0,
        fill_holes_px: int = 0,
        conf_filter: float | None = None,
        conf_harden: float | None = None,
        years: tuple[int, ...] | None = None,
    ):
        """
        years: frames fed to the model (default all of config.YEARS), e.g. (2017, 2024) for bi-temporal change detection.
        The valid-pixel mask is ALWAYS taken over every year in config.YEARS, so every year subset is trained and
        scored on exactly the same pixels.
        sources: any of s1, s2, alphaearth, tessera (channels are concatenated).
        crop_size: random crop (train) / center crop (eval). None keeps full tile.
        repeats:   virtual dataset length multiplier (more steps per epoch).
        label_sigma: jitter on soft confidence weights (use only with soft loss).
        geo_aug: random 90-degree rotation + flip (only used when augment=True).
        min_gain_px: drop gain patches smaller than this many pixels (0 = off).
        fill_holes_px: fill holes smaller than this many pixels inside gain (0 = off).
        conf_filter: ignore gain pixels whose confidence (0-1) is below this in loss and metrics (valid=0).
        conf_harden: like conf_filter, and retained gain pixels get weight 1.0 (matters for the soft loss only).
        All cleaning runs on the full tile (confidence step, then removal, then filling), before crop/augmentation.
        """
        if conf_filter is not None and conf_harden is not None:
            raise ValueError("conf_filter and conf_harden are mutually exclusive")
        for t in (conf_filter, conf_harden):
            if t is not None and not 0.0 < t <= 1.0:
                raise ValueError(f"confidence thresholds must be in (0, 1], got {t}")
        sources = tuple(sources)
        if not sources:
            raise ValueError("At least one source must be selected")
        invalid_sources = set(sources) - SOURCES.keys()
        if invalid_sources:
            raise ValueError(f"Unknown sources: {invalid_sources}")

        self.tile_dirs = sorted(tile_dirs)
        self.years = tuple(YEARS if years is None else sorted(years))
        unknown = set(self.years) - set(YEARS)
        if not self.years or unknown:
            raise ValueError(
                f"years must be a non-empty subset of {tuple(YEARS)}, got {years}"
            )
        self.sources = sources
        self.augment = augment
        self.crop_size = crop_size
        self.repeats = repeats
        self.gain_sigma = gain_sigma
        self.frame_sigma = frame_sigma
        self.frame_drop_p = frame_drop_p
        self.label_sigma = label_sigma
        self.geo_aug = geo_aug
        self.min_gain_px = min_gain_px
        self.fill_holes_px = fill_holes_px
        self.conf_filter = conf_filter
        self.conf_harden = conf_harden

        self.band_names = tuple(band for s in sources for band in SOURCES[s][1])

        # Spectral augmentation only applies to S2 reflectance channels
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
        if self.geo_aug:
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

        for year in YEARS:
            if year in self.years:
                parts = []
                for source in self.sources:
                    template, bands = SOURCES[source]
                    path = tile_dir / template.format(year=year)
                    if not path.exists():
                        raise FileNotFoundError(f"Missing {source} raster: {path}")
                    parts.append(
                        read_physical(
                            path, bands, mask_nodata=source in EMBEDDING_SOURCES
                        )
                    )
                frames.append(np.concatenate(parts, axis=0))

            # Valid mask comes from the S1/S2 composite, further restricted to
            # pixels TESSERA has an embedding for (it marks sea / uncovered
            # pixels as NaN). Applied whatever --sources and --years are, so every
            # input type and frame subset is trained and scored on identical pixels.
            mask_path = tile_dir / "composites" / f"s1s2_{year}.tif"
            with rasterio.open(mask_path) as src:
                vm = src.read(VALID_MASK_BAND_INDEX).astype(np.float32)
            valid_masks.append(np.nan_to_num(vm, nan=0.0) > 0)

            tessera_path = tile_dir / "embeddings" / f"tessera_{year}.tif"
            if tessera_path.exists():
                with rasterio.open(tessera_path) as src:
                    valid_masks.append(~np.isnan(src.read(1)))

        pixels = np.stack(frames, axis=0)  # (T, C, H, W), may hold NaN
        combined_valid = np.all(np.stack(valid_masks, axis=0), axis=0).astype(
            np.float32
        )

        with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
            confidence = src.read(1).astype(np.float32)
        confidence = np.nan_to_num(confidence, nan=0.0)

        gain_mask = (confidence > 0).astype(np.float32)
        gain_weight = np.clip(confidence / 100.0, 0.0, 1.0).astype(np.float32)
        if (
            self.min_gain_px > 1
            or self.fill_holes_px > 1
            or self.conf_filter is not None
            or self.conf_harden is not None
        ):
            gain_mask, gain_weight, valid = clean_labels(
                gain_mask,
                gain_weight,
                combined_valid > 0.5,
                conf_filter=self.conf_filter,
                conf_harden=self.conf_harden,
                min_px=self.min_gain_px,
                fill_px=self.fill_holes_px,
            )
            combined_valid = valid.astype(np.float32)

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
