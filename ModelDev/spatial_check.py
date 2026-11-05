"""How close are the validation and holdout tiles to the training tiles?  Run from ModelDev (reads only raster headers):

    python spatial_check.py --data_dir ../DataCollection/data/test_tiles --holdout_dir ../DataCollection/data/holdout_tiles

Reproduces the training split (utils.split_tiles with the same val_frac / split_seed) and reports, in km, the distance from
each validation tile and each holdout tile to its NEAREST training tile (train-to-train nearest-neighbour distance is the
baseline). Tiles are 2.56 km wide, so distances under ~3 km mean touching tiles.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

THRESHOLDS_KM = (3, 10, 25, 50, 100)


def centroid_km(tile_dir):
    """Tile centre in EPSG:3035, in km."""
    with rasterio.open(Path(tile_dir) / "labels" / "gain_confidence.tif") as src:
        b, crs = src.bounds, src.crs
    x, y = (b.left + b.right) / 2, (b.bottom + b.top) / 2
    if crs is not None and crs.to_epsg() != 3035:
        from pyproj import Transformer

        x, y = Transformer.from_crs(crs, "EPSG:3035", always_xy=True).transform(x, y)
    return x / 1000.0, y / 1000.0


def nn_km(query, ref, same=False):
    """Distance (km) from every query point to its nearest reference point (same=True: skip the point itself)."""
    d, _ = cKDTree(ref).query(query, k=2 if same else 1)
    return d[:, 1] if same else d


def report(name, d):
    q = np.quantile(d, [0.0, 0.1, 0.5, 0.9])
    share = "  ".join(f"<{t}km {100 * np.mean(d < t):4.0f}%" for t in THRESHOLDS_KM)
    print(
        f"{name:<22} n={len(d):<4} nearest train tile (km): min {q[0]:7.1f}  p10 {q[1]:7.1f}  median {q[2]:7.1f}  p90 {q[3]:7.1f}"
    )
    print(f"{'':<22} share with a training tile closer than: {share}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="../DataCollection/data/test_tiles")
    p.add_argument("--holdout_dir", default="../DataCollection/data/holdout_tiles")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    a = p.parse_args()

    from utils import split_tiles

    train, val = split_tiles(a.data_dir, val_frac=a.val_frac, seed=a.split_seed)
    hold = sorted(
        d
        for d in Path(a.holdout_dir).iterdir()
        if d.is_dir() and d.name.startswith("tile_")
    )
    tr, va, ho = (np.array([centroid_km(d) for d in s]) for s in (train, val, hold))
    print(f"train {len(tr)}  val {len(va)}  holdout {len(ho)} tiles\n")
    report("train -> train (base)", nn_km(tr, tr, same=True))
    report("val -> train", nn_km(va, tr))
    report("holdout -> train", nn_km(ho, tr))


if __name__ == "__main__":
    main()
