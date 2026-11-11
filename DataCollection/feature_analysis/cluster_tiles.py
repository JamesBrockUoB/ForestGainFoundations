#!/usr/bin/env python3
"""Reduce, cluster, and visualise tile-level feature tables.

Metadata such as country, biome, and bounds is expected to already be
in the feature CSV, read from each tile's metadata.json during feature
extraction. Metadata is used for plot annotations only, never for clustering.

Examples:
    python DataCollection/feature_analysis/cluster_tiles.py \
        --input DataCollection/feature_analysis/output/tile_all_valid_features.csv \
        --out-dir DataCollection/feature_analysis/output/all_valid_clusters \
        --feature-set all_valid

    python DataCollection/feature_analysis/cluster_tiles.py \
        --input DataCollection/feature_analysis/output/tile_gain_only_features.csv \
        --out-dir DataCollection/feature_analysis/output/gain_clusters \
        --feature-set gain_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

METADATA_COLUMNS = {
    "tile_id",
    "biome",
    "country",
    "gain_pct_metadata",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "n_valid_pixels",
    "valid_fraction",
}

GAIN_SUMMARY_COLUMNS = {
    "gain_n_pixels",
    "gain_fraction",
    "gain_conf_mean",
    "gain_conf_p10",
    "gain_conf_p90",
}


def load_feature_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Feature CSV has no rows: {path}")
    if "tile_id" not in df.columns:
        raise ValueError(f"{path} must contain a 'tile_id' column.")

    # Ensure metadata is treated as metadata, not numeric features.
    df["tile_id"] = df["tile_id"].astype(str)
    return df


def infer_feature_set(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "gain" in name:
        return "gain_only"
    return "all_valid"


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Choose measurement features, excluding metadata and prevalence/counts."""
    excluded = set(METADATA_COLUMNS) | set(GAIN_SUMMARY_COLUMNS)

    numeric_columns = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    if feature_set == "all_valid":
        # The all-valid extraction table may also contain gain-pixel summaries.
        # Exclude them so this clustering describes the full valid tile area.
        columns = [col for col in numeric_columns if not col.startswith("gain_")]
    elif feature_set == "gain_only":
        # Gain-only trajectory columns are named gain_<year>_<source>_<band>_<stat>.
        # Keep these but not counts, fractions, or confidence summaries.
        columns = [
            col
            for col in numeric_columns
            if col.startswith("gain_") and col not in GAIN_SUMMARY_COLUMNS
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if not columns:
        raise ValueError(
            f"No feature columns found for feature set '{feature_set}'. "
            "Check the input CSV and --feature-set."
        )

    # Remove columns that are entirely missing or constant; neither can help
    # distinguish tiles and constant columns can cause scaling problems.
    usable = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) < 2:
            continue
        usable.append(col)

    if not usable:
        raise ValueError(
            f"No varying, non-empty feature columns remain for '{feature_set}'."
        )

    return usable


def choose_k(
    X: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int,
) -> tuple[int, dict[int, float]]:
    """Choose k by silhouette score in PCA space; return k and all scores."""
    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 tiles to perform clustering.")

    upper = min(max_k, n_samples - 1)
    lower = max(2, min_k)
    if upper < lower:
        raise ValueError(f"Cannot test k={lower}..{upper} with {n_samples} tiles.")

    scores: dict[int, float] = {}
    for k in range(lower, upper + 1):
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state,
        ).fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue

        # Silhouette is expensive for large tile collections; use a fixed,
        # reproducible sample when there are many tiles.
        sample_size = min(2000, n_samples)
        try:
            score = silhouette_score(
                X,
                labels,
                sample_size=sample_size if sample_size < n_samples else None,
                random_state=random_state,
            )
        except ValueError:
            continue
        scores[k] = float(score)

    if not scores:
        raise ValueError("Could not calculate a valid silhouette score for any k.")

    best_k = max(scores, key=scores.get)
    return best_k, scores


def save_scatter(
    result: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    output_path: Path,
    *,
    legend_title: str | None = None,
) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=result,
        x=x,
        y=y,
        hue=hue,
        palette="tab10" if hue == "cluster" else "tab20",
        s=65,
        linewidth=0,
    )
    plt.title(title)
    plt.legend(title=legend_title or hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def cluster_and_plot(
    df: pd.DataFrame,
    out_dir: Path,
    label: str,
    feature_set: str,
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = select_feature_columns(df, feature_set)
    raw_features = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    # Impute missing summaries with that feature's median, not zero. Zero may
    # be a real signal value and otherwise creates artificial distances.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(raw_features)

    # Standardise the chosen features. Since this table contains many bands,
    # also save feature/source profiles for interpretation rather than treating
    # the resulting cluster labels as ecological classes.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed).astype(np.float32)

    n_samples, n_features = X_scaled.shape
    if n_samples < 3:
        raise ValueError(f"Need at least 3 tiles; got {n_samples}.")

    n_components = min(20, n_samples - 1, n_features)
    if n_components < 2:
        raise ValueError(
            f"Need at least 2 usable PCA components; feature matrix is "
            f"{n_samples} tiles x {n_features} features."
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_scaled)
    clustering_components = X_pca[:, : min(10, X_pca.shape[1])]

    k, silhouette_scores = choose_k(
        clustering_components,
        min_k=min_k,
        max_k=max_k,
        random_state=random_state,
    )

    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=random_state,
    )
    cluster_labels = kmeans.fit_predict(clustering_components)

    # UMAP is for visualisation only; clustering is performed in PCA space.
    n_neighbors = min(15, n_samples - 1)
    reducer = UMAP(
        n_neighbors=max(2, n_neighbors),
        min_dist=0.1,
        n_components=2,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(clustering_components)

    result = df.copy()
    result["cluster"] = cluster_labels
    result["umap_1"] = embedding[:, 0]
    result["umap_2"] = embedding[:, 1]
    result["pca_1"] = X_pca[:, 0]
    result["pca_2"] = X_pca[:, 1]

    result_path = out_dir / f"{label}_clustered_tiles.csv"
    result.to_csv(result_path, index=False)

    # PCA and UMAP cluster plots.
    save_scatter(
        result,
        "pca_1",
        "pca_2",
        "cluster",
        f"{label}: PCA coloured by cluster",
        out_dir / f"{label}_pca_clusters.png",
    )
    save_scatter(
        result,
        "umap_1",
        "umap_2",
        "cluster",
        f"{label}: UMAP coloured by cluster",
        out_dir / f"{label}_umap_clusters.png",
    )

    # Metadata overlays are descriptive only; metadata never enters X_scaled.
    for metadata_col in ("country", "biome"):
        if metadata_col not in result.columns:
            continue
        categories = result[metadata_col].fillna("Unknown").astype(str)
        if categories.nunique() > 25:
            print(
                f"Skipping UMAP plot by {metadata_col}: "
                f"{categories.nunique()} categories."
            )
            continue

        plot_df = result.copy()
        plot_df[metadata_col] = categories
        save_scatter(
            plot_df,
            "umap_1",
            "umap_2",
            metadata_col,
            f"{label}: UMAP coloured by {metadata_col}",
            out_dir / f"{label}_umap_{metadata_col}.png",
            legend_title=metadata_col,
        )

    # Cluster profiles in original, imputed feature units. This is descriptive,
    # not a causal or supervised feature-importance measure.
    feature_frame = pd.DataFrame(X_imputed, columns=feature_columns)
    feature_frame["cluster"] = cluster_labels
    cluster_means = feature_frame.groupby("cluster")[feature_columns].mean()
    overall_mean = feature_frame[feature_columns].mean()
    overall_std = feature_frame[feature_columns].std(ddof=0).replace(0, 1.0)
    cluster_z = (cluster_means - overall_mean) / overall_std

    # Heatmap uses features that vary most between cluster means.
    top_features = (
        cluster_z.var(axis=0).sort_values(ascending=False).head(30).index.tolist()
    )
    cluster_z[top_features].to_csv(out_dir / f"{label}_cluster_feature_zscores.csv")

    plt.figure(figsize=(15, max(5, 0.35 * len(top_features) + 2)))
    sns.heatmap(
        cluster_z[top_features],
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
    )
    plt.title(f"{label}: cluster feature profiles (z-score; top varying features)")
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_cluster_feature_heatmap.png", dpi=200)
    plt.close()

    # PCA variance and component loadings help identify which input features
    # drive broad directions of variation.
    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance_df.to_csv(out_dir / f"{label}_pca_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.index.name = "feature"
    loadings.to_csv(out_dir / f"{label}_pca_loadings.csv")

    plt.figure(figsize=(9, 5))
    plt.plot(
        variance_df["component"],
        variance_df["cumulative_explained_variance"],
        marker="o",
    )
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"{label}: PCA explained variance")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_pca_variance.png", dpi=200)
    plt.close()

    # Cluster sizes and candidate k scores.
    counts = (
        result["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .to_frame("n_tiles")
    )
    counts.to_csv(out_dir / f"{label}_cluster_counts.csv")

    scores_path = out_dir / f"{label}_silhouette_scores.json"
    scores_path.write_text(json.dumps(silhouette_scores, indent=2))

    # Per-cluster deviations, including both positive and negative directions.
    top_feature_summary = []
    for cluster_id in sorted(cluster_means.index):
        deviations = cluster_z.loc[cluster_id]
        ranked = deviations.reindex(
            deviations.abs().sort_values(ascending=False).index
        ).head(20)

        top_feature_summary.append(
            {
                "cluster": int(cluster_id),
                "n_tiles": int((cluster_labels == cluster_id).sum()),
                "features": [
                    {
                        "name": feature,
                        "cluster_zscore": float(score),
                    }
                    for feature, score in ranked.items()
                ],
            }
        )

    (out_dir / f"{label}_cluster_feature_summary.json").write_text(
        json.dumps(top_feature_summary, indent=2)
    )

    # Run configuration makes the chosen features and preprocessing auditable.
    config = {
        "input_feature_set": feature_set,
        "n_tiles": int(n_samples),
        "n_input_features": int(n_features),
        "features": feature_columns,
        "imputation": "per-feature median",
        "scaling": "StandardScaler",
        "pca_components": int(n_components),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "clustering": "KMeans on first up-to-10 PCA components",
        "selected_k": int(k),
        "silhouette_scores": silhouette_scores,
        "umap": {
            "n_neighbors": int(max(2, n_neighbors)),
            "min_dist": 0.1,
            "random_state": random_state,
        },
        "metadata_used_for_clustering": False,
    }
    (out_dir / f"{label}_analysis.json").write_text(json.dumps(config, indent=2))

    print(f"[{label}] tiles={n_samples}, features={n_features}, clusters={k}")
    print(f"Cluster assignments: {result_path}")
    print(f"Outputs: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA, UMAP, and clustering for tile feature tables."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Feature CSV generated by tile_features.py.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for cluster labels, tables, and plots.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Output filename prefix (default: inferred from input filename).",
    )
    parser.add_argument(
        "--feature-set",
        choices=("auto", "all_valid", "gain_only"),
        default="auto",
        help=(
            "Use all-valid features or gain-only trajectory features. "
            "By default inferred from the input filename."
        ),
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.min_k < 2:
        parser.error("--min-k must be at least 2")
    if args.max_k < args.min_k:
        parser.error("--max-k must be greater than or equal to --min-k")

    df = load_feature_csv(args.input)
    feature_set = (
        infer_feature_set(args.input)
        if args.feature_set == "auto"
        else args.feature_set
    )
    label = args.label or Path(args.input).stem.replace("tile_", "").replace(
        "_features", ""
    )

    cluster_and_plot(
        df=df,
        out_dir=Path(args.out_dir),
        label=label,
        feature_set=feature_set,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
