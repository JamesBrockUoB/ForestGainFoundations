"""Fit the (lo, hi) = (p1, p99) scaling for the raw S1/S2 bands and the derived indices from TRAINING tiles only, and
store it in index_stats.json ("bands" and "norm" blocks). "forest_tc" is preserved.

    python compute_index_stats.py --data_dir ../DataCollection/data/forest_gain_tiles --val_frac 0.2 --split_seed 0

Uses every training tile and every pixel (no subsampling).
train.py calls ensure_norm_stats() and fits whatever is missing, so running this by hand is only needed to refit.
"di" needs a "forest_tc" entry in the file before it can be fitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from config import SOURCES, VALID_MASK_BAND_INDEX, YEARS
from datasets import (
    INDEX_NAMES,
    INDEX_STATS_PATH,
    NORMALISED_BANDS,
    compute_indices,
    load_index_stats,
    read_physical,
    resolve_index_channels,
)
from utils import split_tiles


def fit_band_stats(
    train_dirs,
    bands,
    pct=(1.0, 99.0),
    out=INDEX_STATS_PATH,
) -> dict:
    """p1/p99 of raw S1/S2 bands over valid pixels of all years, merged into the "bands" block of `out`."""
    bands = list(dict.fromkeys(bands))
    train_dirs = list(train_dirs)
    template = SOURCES["s1"][0]  # S1 and S2 bands share one composite file
    print(
        f"fitting band stats on {len(train_dirs)} training tiles (all pixels): {bands}"
    )

    samples = {b: [] for b in bands}
    for tile in train_dirs:
        for year in YEARS:  # pool all years: captures temporal variability
            path = tile / template.format(year=year)
            x = read_physical(path, bands)
            with rasterio.open(path) as src:
                valid = src.read(VALID_MASK_BAND_INDEX) > 0  # NaN -> False
            for i, b in enumerate(bands):
                v = x[i][valid]
                samples[b].append(v[np.isfinite(v)])

    stats = load_index_stats(out)
    block = stats.setdefault("bands", {})
    for b in bands:
        allv = np.concatenate(samples[b])
        lo, hi = np.percentile(allv, pct)
        block[b] = [float(lo), float(hi)]
        print(f"{b:8s} p1={lo:11.6f} p99={hi:11.6f}  (n={allv.size})")
    Path(out).write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    return stats


def fit_index_stats(
    train_dirs,
    indices,
    sources=("s1", "s2"),
    pct=(1.0, 99.0),
    out=INDEX_STATS_PATH,
) -> dict:
    """p1/p99 of derived indices over all pixels, merged into the "norm" block of `out`. Reads the s1+s2 composites."""
    indices = list(indices)
    train_dirs = list(train_dirs)
    band_names = tuple(b for s in sources for b in SOURCES[s][1])
    plan = resolve_index_channels(band_names, indices)
    stats = load_index_stats(out)

    print(
        f"fitting index stats on {len(train_dirs)} training tiles (all pixels): {indices}"
    )

    samples = {n: [] for n in indices}
    for tile in train_dirs:
        for year in YEARS:
            parts = [
                read_physical(tile / SOURCES[s][0].format(year=year), SOURCES[s][1])
                for s in sources
            ]
            x = np.concatenate(parts, axis=0)[None]  # (1, C, H, W)
            v = compute_indices(x, plan, indices, stats=stats)[0]  # (K, H, W)
            flat = v.reshape(v.shape[0], -1)
            for i, n in enumerate(indices):
                vals = flat[i]
                samples[n].append(vals[np.isfinite(vals)])

    norm = stats.setdefault("norm", {})
    for n in indices:
        allv = np.concatenate(samples[n])
        lo, hi = np.percentile(allv, pct)
        norm[n] = [float(lo), float(hi)]
        print(f"{n:7s} p1={lo:9.4f} p99={hi:9.4f}  (n={allv.size})")
    Path(out).write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    return stats


def ensure_norm_stats(
    train_dirs, sources, indices=(), refit=False, out=INDEX_STATS_PATH
) -> dict:
    """Fit (from train_dirs only) whatever band / index scaling the run needs and the file lacks. Returns the stats."""
    stats = load_index_stats(out)
    want_bands = list(
        dict.fromkeys(
            b for s in sources for b in SOURCES[s][1] if b in NORMALISED_BANDS
        )
    )
    todo_bands = [b for b in want_bands if refit or b not in stats.get("bands", {})]
    todo_idx = [n for n in indices if refit or n not in stats.get("norm", {})]

    if todo_bands:
        fit_band_stats(train_dirs, todo_bands, out=out)
    if todo_idx:
        fit_index_stats(train_dirs, todo_idx, out=out)
    if not (todo_bands or todo_idx):
        print(f"normalisation stats found in {out}")
    return load_index_stats(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="../DataCollection/data/forest_gain_tiles")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--sources", nargs="+", default=["s1", "s2"])
    p.add_argument(
        "--indices", nargs="+", default=[i for i in INDEX_NAMES if i != "di"]
    )
    p.add_argument("--out", default=str(INDEX_STATS_PATH))
    args = p.parse_args()

    train_dirs, _ = split_tiles(
        args.data_dir, val_frac=args.val_frac, seed=args.split_seed
    )
    ensure_norm_stats(
        train_dirs, tuple(args.sources), tuple(args.indices), refit=True, out=args.out
    )


if __name__ == "__main__":
    main()
