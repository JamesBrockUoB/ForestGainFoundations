#!/usr/bin/env python3
"""
Compare biome labels against cluster labels.

Usage:
    python cluster_biome_alignment.py \
      --input ./analysis_results/all_valid_clusters/all_valid_clustered_tiles.csv \
      --tiles-dir ../data/holdout_tiles \
      --out-dir ./analysis_results/all_valid_clusters/biome_alignment

    python cluster_biome_alignment.py \
      --input ./analysis_results/gain_clusters/gain_only_clustered_tiles.csv \
      --tiles-dir ../data/holdout_tiles \
      --out-dir ./analysis_results/gain_clusters/biome_alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

BIOME_CANDIDATES = ("biome", "ecosystem", "land_cover", "ecoregion")


def read_tile_metadata(tile_dir: Path) -> dict:
    meta_path = tile_dir / "metadata.json"
    if not meta_path.is_file():
        return {}

    try:
        with meta_path.open() as f:
            data = json.load(f)
    except Exception:
        return {}

    return data


def load_cluster_df(csv_path: Path, tiles_dir: Path | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"tile_id", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df = df[["tile_id", "cluster"]].copy()
    df["tile_id"] = df["tile_id"].astype(str)
    df["cluster"] = df["cluster"].astype(str)

    # If biome is present, use it.
    meta_col = None
    for candidate in BIOME_CANDIDATES:
        if candidate in df.columns:
            meta_col = candidate
            break

    if meta_col is None and tiles_dir is not None:
        biomes = []
        for _, row in df.iterrows():
            tile_dir = tiles_dir / row["tile_id"]
            metadata = read_tile_metadata(tile_dir)
            biome = None

            for candidate in BIOME_CANDIDATES:
                val = metadata.get(candidate)
                if val is not None and str(val).strip() != "":
                    biome = str(val)
                    break

            if biome is None:
                biome = "unknown"

            biomes.append(biome)

        df["biome"] = biomes
    elif meta_col is not None:
        df["biome"] = df[meta_col].fillna("unknown").astype(str)

    if "biome" not in df.columns:
        raise ValueError(
            "No biome column was found in the CSV, and metadata.json does not "
            "contain biome-like fields."
        )

    df["biome"] = df["biome"].fillna("unknown").astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cluster labels against biome labels."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tiles-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()

    df = load_cluster_df(args.input, args.tiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This is the actual alignment score
    ami = adjusted_mutual_info_score(df["biome"], df["cluster"])
    nmi = normalized_mutual_info_score(df["biome"], df["cluster"])

    print(f"AMI between biome and cluster: {ami:.4f}")
    print(f"NMI between biome and cluster: {nmi:.4f}")

    # Cluster x biome contingency table
    contingency = pd.crosstab(df["cluster"], df["biome"])
    contingency.to_csv(out_dir / "cluster_biome_contingency.csv")

    # Top biome distribution per cluster
    cluster_biome_share = (
        contingency.div(contingency.sum(axis=1), axis=0).fillna(0.0).sort_index()
    )
    cluster_biome_share.to_csv(out_dir / "cluster_biome_share.csv")

    # Print the biggest biome associations by cluster
    print("\nTop biome composition by cluster:")
    for cluster in sorted(cluster_biome_share.index, key=lambda x: str(x)):
        row = cluster_biome_share.loc[cluster]
        top = row.sort_values(ascending=False).head(5)
        print(f"cluster {cluster}: {top.to_dict()}")

    # Heatmap of cluster x biome
    plt.figure(
        figsize=(
            max(8, 0.25 * contingency.shape[1] + 2),
            max(6, 0.35 * contingency.shape[0] + 2),
        )
    )
    sns.heatmap(
        contingency,
        cmap="Blues",
        annot=True,
        fmt="d",
        cbar_kws={"label": "tile count"},
    )
    plt.title("Cluster vs biome")
    plt.xlabel("Biome")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_biome_heatmap.png", dpi=200)
    plt.close()

    # Save summary JSON
    summary = {
        "adjusted_mutual_information": float(ami),
        "normalized_mutual_information": float(nmi),
        "n_tiles": int(len(df)),
        "n_clusters": int(df["cluster"].nunique()),
        "n_biomes": int(df["biome"].nunique()),
    }
    import json

    with open(out_dir / "cluster_biome_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved")
    print(f"  {out_dir / 'cluster_biome_contingency.csv'}")
    print(f"  {out_dir / 'cluster_biome_share.csv'}")
    print(f"  {out_dir / 'cluster_biome_heatmap.png'}")
    print(f"  {out_dir / 'cluster_biome_summary.json'}")


if __name__ == "__main__":
    main()
