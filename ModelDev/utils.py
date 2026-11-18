from pathlib import Path

import numpy as np
import rasterio
import torch
from sklearn.model_selection import train_test_split


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def auto_pos_weight(tile_dirs) -> float:
    """sqrt(negatives / positives) over the TRAINING tiles' gain labels.

    Positives follow the dataset's definition (gain_confidence > 0). Pixels with
    gain_valid == 0 are not excluded here (the loss ignores them), which is a
    negligible difference when imagery retention is ~99%.
    """
    pos = neg = 0
    for d in tile_dirs:
        with rasterio.open(Path(d) / "labels" / "gain_confidence.tif") as src:
            c = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
        n_pos = int((c > 0).sum())
        pos += n_pos
        neg += c.size - n_pos
    if pos == 0:
        raise ValueError("No positive gain pixels in the training tiles")
    return float(np.sqrt(neg / pos))


def _positive_fraction(tile_dir) -> float:
    with rasterio.open(Path(tile_dir) / "labels" / "gain_confidence.tif") as src:
        c = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
    return float((c > 0).mean())


def split_tiles(data_dir, val_frac=0.2, seed=0, n_bins=3):
    """Random train/val split over one directory or a list of them, stratified by each tile's
    positive-pixel fraction (binned into n_bins quantile groups, computed over all tiles pooled).
    With several directories the strata are also split per directory, so every dataset appears
    in both train and val. Returns (train_dirs, val_dirs)."""
    data_dirs = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
    data_dirs = list(
        dict.fromkeys(str(Path(d).resolve()) for d in data_dirs)
    )  # drop repeats

    dirs, origin = [], []
    for i, root in enumerate(data_dirs):
        found = sorted(
            p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("tile_")
        )
        print(f"  {root}: {len(found)} tiles")
        dirs += found
        origin += [i] * len(found)
    if not dirs:
        raise ValueError(f"No tile_* directories found in {data_dirs}")

    frac = np.array([_positive_fraction(d) for d in dirs])
    edges = np.unique(np.quantile(frac, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.digitize(frac, edges)
    # one dataset: identical to the old behaviour; several: stratify by (dataset, bin) first
    strata = [bins + (n_bins + 1) * np.array(origin), bins, None]
    if len(data_dirs) == 1:
        strata = strata[1:]
    for s in strata:
        try:
            return train_test_split(
                dirs, test_size=val_frac, random_state=seed, stratify=s
            )
        except ValueError:  # a stratum too small: try the coarser one
            continue
