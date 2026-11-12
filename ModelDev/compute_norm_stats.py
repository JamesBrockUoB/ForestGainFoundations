"""One-off: compute per-band p1/p99 from the TRAIN split and print a config snippet."""

import argparse
import random

import numpy as np
import rasterio
from config import S1_BANDS, S2_BANDS, VALID_MASK_BAND_INDEX, YEARS
from datasets import read_physical, split_tile_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../DataCollection/data/test_tiles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tiles", type=int, default=200)
    p.add_argument("--stride", type=int, default=2, help="pixel subsampling")
    p.add_argument("--pct", type=float, nargs=2, default=(1.0, 99.0))
    args = p.parse_args()

    # Same split as training, train tiles only (no val leakage)
    train_dirs, _ = split_tile_dirs(args.data_dir, seed=args.seed)
    dirs = random.Random(args.seed).sample(
        train_dirs, min(args.max_tiles, len(train_dirs))
    )

    band_names = tuple(S2_BANDS) + tuple(S1_BANDS)
    samples = {b: [] for b in band_names}
    s = args.stride

    for d in dirs:
        for year in YEARS:  # pool all years: captures temporal variability
            path = d / "composites" / f"s1s2_{year}.tif"
            x = read_physical(path, band_names)[:, ::s, ::s]
            with rasterio.open(path) as src:
                valid = src.read(VALID_MASK_BAND_INDEX)[::s, ::s] > 0  # NaN -> False

            for i, b in enumerate(band_names):
                v = x[i][valid]
                samples[b].append(v[np.isfinite(v)])

    print(
        f"# {len(dirs)} tiles x {len(YEARS)} years, stride {s}, "
        f"percentiles {tuple(args.pct)}"
    )
    print("NORM_STATS: dict[str, tuple[float, float]] = {")
    for b in band_names:
        lo, hi = np.percentile(np.concatenate(samples[b]), args.pct)
        print(f'    "{b}": ({lo:.6f}, {hi:.6f}),')
    print("}")


if __name__ == "__main__":
    main()
