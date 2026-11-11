# Tile-level feature analysis

This folder contains a simple unsupervised analysis pipeline for tile-level feature comparison.

## Step 1: build tile feature tables

From the repo root:

```bash
python analysis/tile_features.py \
  --data-dir /path/to/tile_root \
  --out-dir analysis \
  --sources s1 s2 alphaearth tessera \
  --mode all_valid

python analysis/tile_features.py \
  --data-dir /path/to/tile_root \
  --out-dir analysis \
  --sources s1 s2 alphaearth tessera \
  --mode gain_only