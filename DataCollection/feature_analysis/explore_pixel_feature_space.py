#!/usr/bin/env python3

"""
Explore pixel-level forest-gain feature spaces.

Analyses:
    1. Sentinel-1
    2. Sentinel-2
    3. AlphaEarth
    4. Tessera
    5. Combined multimodal features

Method:
    - Fit StandardScaler on gain pixels only.
    - Fit PCA on gain pixels only.
    - Transform unlabelled pixels using the same scaler and PCA.
    - Fit UMAP on a reproducible subset of gain pixels in PCA space.
    - Transform all gain and unlabelled pixels using that UMAP model.

Outputs:
    feature_space_exploration/
        s1/
        s2/
        alphaearth/
        tessera/
        combined/
        exploration_summary.json

Requires:
    numpy
    pandas
    matplotlib
    scikit-learn
    umap-learn
    joblib
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path.home() / "Documents" / "ForestGainFoundations"

INPUT_DIR = PROJECT_DIR / "DataCollection" / "feature_analysis" / "pixel_typology_data"

OUTPUT_DIR = (
    PROJECT_DIR / "DataCollection" / "feature_analysis" / "feature_space_exploration"
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RANDOM_SEED = 42

# Number of principal components retained.
N_PCA_COMPONENTS = 30

# UMAP is fitted on at most this many gain pixels.
UMAP_FIT_SAMPLE_SIZE = 10000

# Number of PCA dimensions supplied to UMAP.
UMAP_INPUT_COMPONENTS = 20

UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# Plotting limits.
MAX_PLOT_GAIN = 6000
MAX_PLOT_UNLABELLED = 3000

# Individual modalities and their combined representation.
MODALITIES = [
    "s1",
    "s2",
    "alphaearth",
    "tessera",
    "combined",
]


# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------


def save_explained_variance_plot(
    pca,
    output_path,
    modality,
):
    """Save PCA explained and cumulative variance."""

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(9, 5))

    components = np.arange(1, len(explained) + 1)

    ax.bar(
        components,
        explained,
        alpha=0.7,
        label="Individual explained variance",
    )

    ax.plot(
        components,
        cumulative,
        marker="o",
        markersize=3,
        label="Cumulative explained variance",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"{modality}: PCA explained variance")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sample_plot_indices(
    n,
    maximum,
    rng,
):
    """Select reproducible indices for plotting."""

    if n <= maximum:
        return np.arange(n)

    return np.sort(
        rng.choice(
            n,
            size=maximum,
            replace=False,
        )
    )


def save_embedding_plot(
    gain_coords,
    unlabelled_coords,
    output_path,
    title,
    xlabel,
    ylabel,
    rng,
):
    """Plot gain and unlabelled samples in a 2D embedding."""

    gain_idx = sample_plot_indices(
        len(gain_coords),
        MAX_PLOT_GAIN,
        rng,
    )

    unlabelled_idx = sample_plot_indices(
        len(unlabelled_coords),
        MAX_PLOT_UNLABELLED,
        rng,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    if len(unlabelled_idx):
        ax.scatter(
            unlabelled_coords[unlabelled_idx, 0],
            unlabelled_coords[unlabelled_idx, 1],
            s=5,
            alpha=0.25,
            label="Unlabelled candidates",
        )

    if len(gain_idx):
        ax.scatter(
            gain_coords[gain_idx, 0],
            gain_coords[gain_idx, 1],
            s=5,
            alpha=0.45,
            label="Gain",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=3)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------


def get_feature_indices(feature_names, modality):
    """Select feature columns belonging to a modality."""

    if modality == "combined":
        return np.arange(len(feature_names))

    prefix = modality + "_"

    indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]

    return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------


def analyse_modality(
    X,
    metadata,
    feature_names,
    modality,
    rng,
):
    print("\n" + "=" * 65)
    print(f"ANALYSING: {modality.upper()}")
    print("=" * 65)

    modality_dir = OUTPUT_DIR / modality
    modality_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Select features.
    # -----------------------------------------------------

    feature_indices = get_feature_indices(
        feature_names,
        modality,
    )

    selected_names = [feature_names[i] for i in feature_indices]

    if len(feature_indices) == 0:
        raise RuntimeError(f"No features found for {modality}")

    X_modality = X[:, feature_indices]

    print(f"Number of features: {X_modality.shape[1]}")

    # -----------------------------------------------------
    # Separate gain and unlabelled pixels.
    # -----------------------------------------------------

    gain_mask = metadata["sample_type"].values == "gain"

    unlabelled_mask = metadata["sample_type"].values == "unlabelled"

    X_gain = X_modality[gain_mask]
    X_unlabelled = X_modality[unlabelled_mask]

    print(f"Gain pixels: {len(X_gain):,}")
    print(f"Unlabelled pixels: {len(X_unlabelled):,}")

    if len(X_gain) < 3:
        raise RuntimeError(f"Not enough gain pixels for {modality}")

    if not np.isfinite(X_gain).all():
        raise ValueError(f"Non-finite gain features in {modality}")

    if not np.isfinite(X_unlabelled).all():
        raise ValueError(f"Non-finite unlabelled features in {modality}")

    # -----------------------------------------------------
    # Standardisation.
    # Fit on gain pixels only.
    # -----------------------------------------------------

    print("Fitting StandardScaler...")

    scaler = StandardScaler()

    X_gain_scaled = scaler.fit_transform(X_gain).astype(np.float32)

    X_unlabelled_scaled = scaler.transform(X_unlabelled).astype(np.float32)

    joblib.dump(
        scaler,
        modality_dir / "scaler.joblib",
    )

    # -----------------------------------------------------
    # PCA.
    # -----------------------------------------------------

    n_components = min(
        N_PCA_COMPONENTS,
        X_gain_scaled.shape[1],
        X_gain_scaled.shape[0] - 1,
    )

    print(f"Fitting PCA with {n_components} components...")

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )

    X_gain_pca = pca.fit_transform(X_gain_scaled).astype(np.float32)

    X_unlabelled_pca = pca.transform(X_unlabelled_scaled).astype(np.float32)

    joblib.dump(
        pca,
        modality_dir / "pca.joblib",
    )

    # -----------------------------------------------------
    # Save PCA variance plot.
    # -----------------------------------------------------

    save_explained_variance_plot(
        pca,
        modality_dir / "pca_explained_variance.png",
        modality,
    )

    # -----------------------------------------------------
    # Save PCA coordinates.
    # -----------------------------------------------------

    pca_gain_metadata = metadata.loc[gain_mask].copy().reset_index(drop=True)

    pca_unlabelled_metadata = (
        metadata.loc[unlabelled_mask].copy().reset_index(drop=True)
    )

    pca_gain_metadata["PC1"] = X_gain_pca[:, 0]
    pca_gain_metadata["PC2"] = X_gain_pca[:, 1]

    pca_unlabelled_metadata["PC1"] = X_unlabelled_pca[:, 0]

    pca_unlabelled_metadata["PC2"] = X_unlabelled_pca[:, 1]

    pca_gain_metadata["sample_type"] = "gain"
    pca_unlabelled_metadata["sample_type"] = "unlabelled"

    pca_coordinates = pd.concat(
        [
            pca_gain_metadata,
            pca_unlabelled_metadata,
        ],
        ignore_index=True,
    )

    pca_coordinates.to_csv(
        modality_dir / "pca_coordinates.csv",
        index=False,
    )

    save_embedding_plot(
        X_gain_pca[:, :2],
        X_unlabelled_pca[:, :2],
        modality_dir / "pca_scatter.png",
        f"{modality}: PCA feature space",
        "PC1",
        "PC2",
        rng,
    )

    # -----------------------------------------------------
    # Save PCA transformed matrices.
    # -----------------------------------------------------

    np.savez_compressed(
        modality_dir / "pca_features.npz",
        X_gain=X_gain_pca,
        X_unlabelled=X_unlabelled_pca,
    )

    # -----------------------------------------------------
    # UMAP.
    # -----------------------------------------------------

    try:
        import umap

    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Run: " "pip install umap-learn"
        ) from exc

    umap_dimensions = min(
        UMAP_INPUT_COMPONENTS,
        X_gain_pca.shape[1],
    )

    X_gain_umap_input = X_gain_pca[:, :umap_dimensions]

    X_unlabelled_umap_input = X_unlabelled_pca[:, :umap_dimensions]

    # Fit UMAP on a reproducible subset of gain pixels.
    fit_idx = sample_plot_indices(
        len(X_gain_umap_input),
        UMAP_FIT_SAMPLE_SIZE,
        rng,
    )

    print(f"Fitting UMAP using {len(fit_idx):,} gain pixels...")

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
        n_jobs=1,
        low_memory=True,
    )

    reducer.fit(X_gain_umap_input[fit_idx])

    # Transform all gain pixels and unlabelled candidates.
    print("Transforming gain pixels into UMAP space...")

    X_gain_umap = reducer.transform(X_gain_umap_input).astype(np.float32)

    print("Transforming unlabelled pixels into UMAP space...")

    X_unlabelled_umap = reducer.transform(X_unlabelled_umap_input).astype(np.float32)

    joblib.dump(
        reducer,
        modality_dir / "umap.joblib",
    )

    # -----------------------------------------------------
    # Save UMAP coordinates.
    # -----------------------------------------------------

    umap_gain_metadata = metadata.loc[gain_mask].copy().reset_index(drop=True)

    umap_unlabelled_metadata = (
        metadata.loc[unlabelled_mask].copy().reset_index(drop=True)
    )

    umap_gain_metadata["UMAP1"] = X_gain_umap[:, 0]
    umap_gain_metadata["UMAP2"] = X_gain_umap[:, 1]

    umap_unlabelled_metadata["UMAP1"] = X_unlabelled_umap[:, 0]

    umap_unlabelled_metadata["UMAP2"] = X_unlabelled_umap[:, 1]

    umap_gain_metadata["sample_type"] = "gain"
    umap_unlabelled_metadata["sample_type"] = "unlabelled"

    umap_coordinates = pd.concat(
        [
            umap_gain_metadata,
            umap_unlabelled_metadata,
        ],
        ignore_index=True,
    )

    umap_coordinates.to_csv(
        modality_dir / "umap_coordinates.csv",
        index=False,
    )

    save_embedding_plot(
        X_gain_umap,
        X_unlabelled_umap,
        modality_dir / "umap_scatter.png",
        f"{modality}: UMAP feature space",
        "UMAP1",
        "UMAP2",
        rng,
    )

    np.savez_compressed(
        modality_dir / "umap_features.npz",
        X_gain=X_gain_umap,
        X_unlabelled=X_unlabelled_umap,
    )

    # -----------------------------------------------------
    # Feature loading summary.
    # -----------------------------------------------------

    loading_rows = []

    for component_idx, component in enumerate(pca.components_):

        top_indices = np.argsort(np.abs(component))[::-1][:20]

        for rank, feature_idx in enumerate(
            top_indices,
            start=1,
        ):

            loading_rows.append(
                {
                    "component": component_idx + 1,
                    "rank": rank,
                    "feature": selected_names[feature_idx],
                    "loading": float(component[feature_idx]),
                    "absolute_loading": float(abs(component[feature_idx])),
                }
            )

    pd.DataFrame(loading_rows).to_csv(
        modality_dir / "pca_top_loadings.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Summary.
    # -----------------------------------------------------

    summary = {
        "modality": modality,
        "n_features": int(X_modality.shape[1]),
        "n_gain_pixels": int(len(X_gain)),
        "n_unlabelled_pixels": int(len(X_unlabelled)),
        "n_pca_components": int(n_components),
        "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
        "pca_cumulative_variance": [
            float(x) for x in np.cumsum(pca.explained_variance_ratio_)
        ],
        "umap_fit_samples": int(len(fit_idx)),
        "umap_input_dimensions": int(umap_dimensions),
        "umap_neighbors": UMAP_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
    }

    with open(
        modality_dir / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(f"Completed {modality.upper()}: " f"{X_modality.shape[1]} features")

    return summary


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    print("=" * 65)
    print("PIXEL FEATURE SPACE EXPLORATION")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    features_path = INPUT_DIR / "pixel_features.npz"
    metadata_path = INPUT_DIR / "pixel_metadata.csv"
    names_path = INPUT_DIR / "feature_names.json"

    for path in [
        features_path,
        metadata_path,
        names_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading extracted features...")

    with np.load(features_path) as data:

        X = data["X"]

    metadata = pd.read_csv(metadata_path)

    with open(names_path) as f:
        feature_names = json.load(f)

    if len(X) != len(metadata):
        raise ValueError("Feature matrix and metadata row counts differ.")

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and feature name counts differ.")

    if "sample_type" not in metadata.columns:
        raise ValueError("Metadata must contain sample_type.")

    print(f"Pixels: {len(X):,}")
    print(f"Features: {X.shape[1]:,}")

    print("\nSample counts:")
    print(metadata["sample_type"].value_counts().to_string())

    gain_count = (metadata["sample_type"] == "gain").sum()

    unlabelled_count = (metadata["sample_type"] == "unlabelled").sum()

    if gain_count < 3:
        raise RuntimeError("At least three gain pixels are required.")

    rng = np.random.default_rng(RANDOM_SEED)

    summaries = {}

    for modality in MODALITIES:

        try:
            summary = analyse_modality(
                X,
                metadata,
                feature_names,
                modality,
                rng,
            )

            summaries[modality] = summary

        except Exception as exc:

            warnings.warn(f"Failed to analyse {modality}: {exc}")

            summaries[modality] = {"error": str(exc)}

    # -----------------------------------------------------
    # Overall summary.
    # -----------------------------------------------------

    overall_summary = {
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "n_pixels": int(len(X)),
        "n_gain_pixels": int(gain_count),
        "n_unlabelled_pixels": int(unlabelled_count),
        "n_features": int(X.shape[1]),
        "random_seed": RANDOM_SEED,
        "method": (
            "StandardScaler and PCA fitted on gain pixels only. "
            "UMAP fitted on a subset of gain pixels in PCA space. "
            "Unlabelled candidates transformed afterwards."
        ),
        "modalities": summaries,
    }

    with open(
        OUTPUT_DIR / "exploration_summary.json",
        "w",
    ) as f:

        json.dump(
            overall_summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("EXPLORATION COMPLETE")
    print("=" * 65)

    print(f"\nResults saved to:\n{OUTPUT_DIR}")

    print("\nModality results:")

    for modality, result in summaries.items():

        if "error" in result:
            print(f"  {modality}: FAILED - {result['error']}")
        else:
            cumulative = result["pca_cumulative_variance"]
            print(
                f"  {modality}: "
                f"{result['n_features']} features, "
                f"30-PC cumulative variance: "
                f"{cumulative[-1]:.3f}"
            )


if __name__ == "__main__":
    main()
