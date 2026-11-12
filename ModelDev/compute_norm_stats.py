"""One-off: compute per-band p1/p99 from the TRAIN split and print a config snippet."""

import argparse

import numpy as np
import rasterio
from config import S1_BANDS, S2_BANDS, VALID_MASK_BAND_INDEX, YEARS
from datasets import read_physical
from utils import split_tiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../DataCollection/data/forest_gain_tiles")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--pct", type=float, nargs=2, default=(1.0, 99.0))
    args = p.parse_args()

    # Same split as train.py, train tiles only (no val leakage)
    train_dirs, _ = split_tiles(
        args.data_dir, val_frac=args.val_frac, seed=args.split_seed
    )

    band_names = tuple(S2_BANDS) + tuple(S1_BANDS)
    samples = {b: [] for b in band_names}

    for d in train_dirs:
        for year in YEARS:  # pool all years: captures temporal variability
            path = d / "composites" / f"s1s2_{year}.tif"
            x = read_physical(path, band_names)
            with rasterio.open(path) as src:
                valid = src.read(VALID_MASK_BAND_INDEX) > 0  # NaN -> False

            for i, b in enumerate(band_names):
                v = x[i][valid]
                samples[b].append(v[np.isfinite(v)])

    print(
        f"# {len(train_dirs)} tiles x {len(YEARS)} years, all pixels, "
        f"percentiles {tuple(args.pct)}"
    )
    print("NORM_STATS: dict[str, tuple[float, float]] = {")
    for b in band_names:
        lo, hi = np.percentile(np.concatenate(samples[b]), args.pct)
        print(f'    "{b}": ({lo:.6f}, {hi:.6f}),')
    print("}")


if __name__ == "__main__":
    main()
