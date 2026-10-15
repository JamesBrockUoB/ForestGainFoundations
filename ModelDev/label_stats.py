"""How much would label cleaning change? Run before training to pick values.

    python label_stats.py ../DataCollection/data/test_tiles
    python label_stats.py <tiles_dir> --mmu 2 4 9 25 --fill 2 9 --min_tile_gain 1.0

Reports, over all tiles (10 m pixels: 100 px = 1 ha):
  1. per min_gain_px: gain patches and gain pixels removed
  2. per fill_holes_px: holes filled and pixels added
  3. patch size vs label confidence: are the small patches also low-confidence?
  4. per min_gain_px: tiles whose gain % falls below --min_tile_gain (your cheap-stage
     gain_pct_min) or to zero once small patches are removed
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

p = argparse.ArgumentParser()
p.add_argument("tiles_dir")
p.add_argument("--mmu", type=int, nargs="+", default=[2, 4, 9, 25, 50, 100])
p.add_argument("--fill", type=int, nargs="+", default=[2, 4, 9, 25])
p.add_argument(
    "--min_tile_gain",
    type=float,
    default=1.0,
    help="gain_pct_min, in %% of tile pixels",
)
a = p.parse_args()

ALL8 = np.ones((3, 3), dtype=bool)
patch_sizes, patch_conf, hole_sizes = [], [], []
tile_sizes, tile_px = [], []
total_px = 0
for d in sorted(Path(a.tiles_dir).glob("tile_*")):
    with rasterio.open(d / "labels" / "gain_confidence.tif") as src:
        c = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
    m = c > 0
    total_px += int(m.sum())
    lab, n = ndimage.label(m, structure=ALL8)
    sizes = np.bincount(lab.ravel())[1:]
    patch_sizes.append(sizes)
    tile_sizes.append(sizes)
    tile_px.append(m.size)
    if n:
        patch_conf.append(np.asarray(ndimage.mean(c, lab, np.arange(1, n + 1))))
    else:
        patch_conf.append(np.zeros(0))
    lab, _ = ndimage.label(~m)
    hole_sizes.append(np.bincount(lab.ravel())[1:])  # includes the big background

ps = np.concatenate(patch_sizes)
pc = np.concatenate(patch_conf)
hs = np.concatenate(hole_sizes)
print(
    f"{total_px:,} gain px in {len(ps):,} patches (8-connected) across {len(tile_px)} tiles; "
    f"median patch {np.median(ps):.0f} px"
)

print("\n1) min_gain_px  patches removed   gain px removed")
for t in a.mmu:
    s = ps[ps < t]
    print(f"{t:>14}  {len(s) / len(ps):>14.1%}  {s.sum() / total_px:>15.1%}")

print("\n2) fill_holes_px  holes filled   gain px added")
for t in a.fill:
    s = hs[hs < t]
    print(f"{t:>15}  {len(s):>12,}  {s.sum() / total_px:>13.1%}")

print(
    "\n3) patch size    patches   gain px share   mean conf (px-weighted)   median patch conf"
)
edges = [1, 2, 3, 5, 9, 25, 100, np.inf]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (ps >= lo) & (ps < hi)
    if not sel.any():
        continue
    px = ps[sel].sum()
    label = (
        f"{lo}"
        if hi == lo + 1
        else (f"{lo}+" if hi == np.inf else f"{lo}-{int(hi) - 1}")
    )
    print(
        f"{label:>13}  {sel.sum():>9,}  {px / total_px:>13.1%}  "
        f"{np.average(pc[sel], weights=ps[sel]):>22.1f}  {np.median(pc[sel]):>17.1f}"
    )

print(
    f"\n4) tiles below {a.min_tile_gain:g}% gain after removing patches (raw: "
    f"{sum(s.sum() / n * 100 < a.min_tile_gain for s, n in zip(tile_sizes, tile_px))} of {len(tile_px)})"
)
for t in a.mmu:
    pct = np.array([s[s >= t].sum() / n * 100 for s, n in zip(tile_sizes, tile_px)])

    below = np.where(pct < a.min_tile_gain)[0]

    print(f"{t:>14}  {len(below):>21}  {int((pct == 0).sum()):>22}")

    if t == 2:
        tile_paths = sorted(Path(a.tiles_dir).glob("tile_*"))
        low_tiles = [
            (d.name, p) for d, p in zip(tile_paths, pct) if p < a.min_tile_gain
        ]
        for name, p in low_tiles:
            print(f"   tile: {name} — gain percentage: {p:.2f}%")
