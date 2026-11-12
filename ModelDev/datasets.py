import copy
import json
import os
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
    S1_BANDS,
    S2_BANDS,
    S2_SCALE,
    SOURCES,
    VALID_MASK_BAND_INDEX,
    YEARS,
)
from mask_cleaning import clean_labels
from torch.utils.data import Dataset

# Derived spectral indices (optional extra input channels, requested with indices=(...))
#
# Computed per pixel and per frame in PHYSICAL units (S2 reflectance 0-1, S1 dB), AFTER augmentation and cropping and
# BEFORE normalisation, so the augmented reflectance flows through to the indices. NaN inputs give NaN outputs, which
# normalize() then fills like any other S2/S1 channel. The channels are appended after the source channels and are
# named "idx_<name>". Dual-pol S1 only (VV, VH): RFDI needs HH, so it is not offered, and VH-VV is already the S1_VVVH channel.
#
#   ndvi   (B8 - B4) / (B8 + B4)
#   evi2   2.5 (B8 - B4) / (B8 + 2.4 B4 + 1)
#   nbr    (B8 - B12) / (B8 + B12)
#   ndmi   (B8 - B11) / (B8 + B11)
#   ndre   (B8 - B5) / (B8 + B5)
#   tcb / tcg / tcw   Tasseled Cap brightness / greenness / wetness, linear in B2 B3 B4 B8 B11 B12 (TCT_COEFFS)
#   rvi4s1 4 VH / (VV + VH) on linear power (S1 is stored in dB, converted here)
#
# Scaling: indices are not in NORM_STATS. normalize() uses (lo, hi) from index_stats.json ("norm") when present
# (write it with compute_index_stats.py, which uses TRAINING tiles only) and otherwise the rough fixed defaults below.
# ---------------------------------------------------------------------------------------------------------------------
# Rows: brightness, greenness, wetness. Columns: B2 (blue), B3 (green), B4 (red), B8 (NIR), B11 (SWIR1), B12 (SWIR2).
# Source: the tutorial at medium.com/@hee2013news (attributed there to Shi & Xu 2019). Reader reports say these do not
# match the paper's six-band values (they resemble Landsat-8 OLI coefficients), so check them before relying on them.
TCT_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
TCT_COEFFS = {
    "tcb": (0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863),
    "tcg": (-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800),
    "tcw": (0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572),
}

INDEX_BANDS = {
    "ndvi": ("B8", "B4"),
    "evi2": ("B8", "B4"),
    "nbr": ("B8", "B12"),
    "ndmi": ("B8", "B11"),
    "ndre": ("B8", "B5"),
    "tcb": TCT_BANDS,
    "tcg": TCT_BANDS,
    "tcw": TCT_BANDS,
    "di": TCT_BANDS,
    "rvi4s1": ("VV", "VH"),
}
INDEX_NAMES = tuple(INDEX_BANDS)

# Rough fixed (lo, hi) used only when index_stats.json has no "norm" entry for an index.
DEFAULT_INDEX_NORM = {
    "ndvi": (-0.1, 0.95),
    "evi2": (-0.05, 0.9),
    "nbr": (-0.2, 0.9),
    "ndmi": (-0.3, 0.7),
    "ndre": (-0.05, 0.6),
    "tcb": (0.0, 0.6),
    "tcg": (-0.2, 0.35),
    "tcw": (-0.35, 0.15),
    "di": (-3.0, 5.0),
    "rvi4s1": (0.1, 0.9),
}
INDEX_STATS_PATH = Path(
    os.environ.get("FG_INDEX_STATS", Path(__file__).with_name("index_stats.json"))
)


# Raw channels that get (x - lo) / (hi - lo) scaling from the stats file. Embeddings are not normalised.
NORMALISED_BANDS = tuple(S2_BANDS) + tuple(S1_BANDS)

_STATS_CACHE: dict = {}


def load_index_stats(path: Path = INDEX_STATS_PATH) -> dict:
    """{"bands": {name: [lo, hi]}, "norm": {index: [lo, hi]}, "forest_tc": {...}}; {} if the file is absent.
    Re-read only when the file's mtime changes; callers get a private copy."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return {}
    if key not in _STATS_CACHE:
        _STATS_CACHE.clear()
        _STATS_CACHE[key] = json.loads(path.read_text())
    return copy.deepcopy(_STATS_CACHE[key])


def band_norm(name: str, stats: dict | None = None) -> tuple[float, float]:
    stats = load_index_stats() if stats is None else stats
    try:
        lo, hi = stats["bands"][name]
    except KeyError:
        raise KeyError(
            f"No normalisation stats for band {name!r} in {INDEX_STATS_PATH}. "
            "train.py fits them automatically; or run compute_index_stats.py."
        ) from None
    return float(lo), float(hi)


def index_norm(name: str, stats: dict | None = None) -> tuple[float, float]:
    stats = load_index_stats() if stats is None else stats
    lo, hi = stats.get("norm", {}).get(name, DEFAULT_INDEX_NORM[name])
    return float(lo), float(hi)


def resolve_index_channels(band_names, indices) -> dict[str, int]:
    """Canonical band ('B8', 'VV', ...) -> channel position in band_names, for everything `indices` needs.
    Matches the last '_'-separated token of the config band name (so 'B8', 's2_B8' and 'S2_b8' all work), VV/VH by
    token anywhere in the name. Raises with the band list if something is missing."""
    unknown = set(indices) - set(INDEX_BANDS)
    if unknown:
        raise ValueError(
            f"Unknown indices {sorted(unknown)}; choose from {INDEX_NAMES}"
        )
    plan, missing = {}, []
    for canon in sorted({b for n in indices for b in INDEX_BANDS[n]}):
        if canon in ("VV", "VH"):
            hits = [
                i
                for i, b in enumerate(band_names)
                if b not in S2_BANDS and canon in b.upper().replace("-", "_").split("_")
            ]
        else:
            hits = [
                i
                for i, b in enumerate(band_names)
                if b in S2_BANDS and b.upper().split("_")[-1] == canon
            ]
        if len(hits) == 1:
            plan[canon] = hits[0]
        else:
            missing.append(f"{canon} ({'none' if not hits else 'ambiguous'})")
    if missing:
        raise ValueError(
            f"indices {tuple(indices)} need bands {', '.join(missing)} among the loaded channels {tuple(band_names)}. "
            "Add the s1/s2 source or adjust resolve_index_channels() to your config's band names."
        )
    return plan


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(den) > 1e-6, num / den, np.nan)


def compute_indices(
    pixels: np.ndarray, plan: dict[str, int], indices, stats=None
) -> np.ndarray:
    """pixels (T, C, H, W) in physical units -> (T, len(indices), H, W) float32."""
    ch = lambda canon: pixels[:, plan[canon]]  # noqa: E731
    stats = load_index_stats() if stats is None else stats
    out = []
    for n in indices:
        if n == "ndvi":
            v = _ratio(ch("B8") - ch("B4"), ch("B8") + ch("B4"))
        elif n == "evi2":
            v = 2.5 * _ratio(ch("B8") - ch("B4"), ch("B8") + 2.4 * ch("B4") + 1.0)
        elif n == "nbr":
            v = _ratio(ch("B8") - ch("B12"), ch("B8") + ch("B12"))
        elif n == "ndmi":
            v = _ratio(ch("B8") - ch("B11"), ch("B8") + ch("B11"))
        elif n == "ndre":
            v = _ratio(ch("B8") - ch("B5"), ch("B8") + ch("B5"))
        elif n in ("tcb", "tcg", "tcw", "di"):

            def tc(k):
                return sum(c * ch(b) for c, b in zip(TCT_COEFFS[k], TCT_BANDS))

            if n == "di":
                ft = stats.get("forest_tc")
                if not ft:
                    raise ValueError(
                        'index "di" needs mature-forest TC statistics: add a "forest_tc" entry with [mean, std] for '
                        f'"tcb", "tcg" and "tcw" to {INDEX_STATS_PATH}'
                    )
                z = {k: (tc(k) - ft[k][0]) / ft[k][1] for k in ("tcb", "tcg", "tcw")}
                v = z["tcb"] - (z["tcg"] + z["tcw"])
            else:
                v = tc(n)
        elif n == "rvi4s1":
            vv, vh = 10.0 ** (ch("VV") / 10.0), 10.0 ** (ch("VH") / 10.0)
            v = 4.0 * _ratio(vh, vv + vh)
        out.append(v.astype(np.float32))
    return np.stack(out, axis=1)


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
    """S1/S2 bands and derived indices ("idx_*"): (x - lo) / (hi - lo) with (lo, hi) from the stats file, clipped,
    NaN -> NORM_NAN_FILL. Embeddings: raw values, NaN -> 0."""
    out = x.astype(np.float32, copy=True)
    stats = (
        load_index_stats()
        if any(b in NORMALISED_BANDS or b.startswith("idx_") for b in band_names)
        else {}
    )
    for i, b in enumerate(band_names):
        ch = out[..., i, :, :]
        if b in NORMALISED_BANDS or b.startswith("idx_"):
            lo, hi = (
                band_norm(b, stats)
                if b in NORMALISED_BANDS
                else index_norm(b[4:], stats)
            )
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
        indices: tuple[str, ...] = (),
    ):
        """
        indices: derived spectral indices appended as extra channels ("idx_<name>"), see INDEX_NAMES. Computed after
        augmentation, before normalisation; the sources they need (s1 for rvi4s1, s2 for the rest) must be loaded.
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

        source_bands = tuple(band for s in sources for band in SOURCES[s][1])

        # Spectral augmentation only applies to S2 reflectance channels (index channels are added after it)
        self.s2_channel_indices = [
            i for i, b in enumerate(source_bands) if b in S2_BANDS
        ]

        self.indices = tuple(indices)
        self._index_plan = (
            resolve_index_channels(source_bands, self.indices) if self.indices else {}
        )
        if "di" in self.indices:  # fail at construction, not inside a DataLoader worker
            compute_indices(
                np.zeros((1, len(source_bands), 1, 1), np.float32),
                self._index_plan,
                ("di",),
            )
        self.band_names = source_bands + tuple(f"idx_{n}" for n in self.indices)

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

        if self.indices:
            pixels = np.concatenate(
                [pixels, compute_indices(pixels, self._index_plan, self.indices)],
                axis=1,
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
