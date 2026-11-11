"""IoU / confidence / mask-shape analysis from the downloaded tifs ONLY: no checkpoint, no dataset, no imagery.

For every tile in --tif_dir it reads <tile_id>_prob.tif (predicted probability) and <tile_id>_gt.tif (ground truth: the
tile's labels/gain_confidence.tif; gain = value > 0, NaN = no gain). Valid pixels = finite probability. Output files are the same as
mask_analysis.py: tile_analysis.csv, iou_correlations.csv, ranked_plots/, correlation_plots/.

    python mask_analysis_tifs.py --tif_dir <model folder>/tifs --out_dir analysis/<model>
"""

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------
# Spatial component statistics
# ---------------------------------------------------------


def component_stats(mask):
    """
    Calculate connected-component statistics on a 2D boolean mask.

    Uses 8-connectivity, matching the convention used by
    clean_gain_mask.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {mask.shape}")

    labels, n_components = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )

    if n_components == 0:
        return {
            "n_components": 0,
            "mean_component_size": 0.0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
        }

    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max())
    total = int(sizes.sum())

    return {
        "n_components": int(n_components),
        "mean_component_size": float(sizes.mean()),
        "largest_component_size": largest,
        "largest_component_fraction": (float(largest / total) if total > 0 else 0.0),
    }


# ---------------------------------------------------------
# Per-tile analysis
# ---------------------------------------------------------


def safe_mean(values):
    """Return NaN for empty arrays instead of raising a warning."""
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(values.mean())


def analyse_tile(probs, gt, valid, threshold):
    """
    Calculate per-tile performance, mask characteristics,
    and prediction-confidence statistics.

    probs, gt and valid must all be 2D arrays.
    """
    probs = np.asarray(probs, dtype=np.float32)
    gt = np.asarray(gt, dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    if probs.shape != gt.shape or probs.shape != valid.shape:
        raise ValueError(
            "Shape mismatch: "
            f"probabilities={probs.shape}, "
            f"ground truth={gt.shape}, "
            f"valid={valid.shape}"
        )

    # Restrict all pixel-level metrics to valid pixels.
    p = probs[valid]
    t = gt[valid]

    if p.size == 0:
        raise ValueError("Tile contains no valid pixels.")

    # Binary prediction.
    pred = p > threshold

    # Confusion matrix.
    tp = int(np.sum(pred & t))
    fp = int(np.sum(pred & ~t))
    fn = int(np.sum(~pred & t))
    tn = int(np.sum(~pred & ~t))

    # Performance metrics.
    iou_denom = tp + fp + fn
    precision_denom = tp + fp
    recall_denom = tp + fn

    iou = tp / iou_denom if iou_denom > 0 else 0.0
    precision = tp / precision_denom if precision_denom > 0 else 0.0
    recall = tp / recall_denom if recall_denom > 0 else 0.0

    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom > 0 else 0.0

    # Pixel-level mask characteristics.
    n_valid = int(p.size)
    gt_gain_pixels = int(t.sum())
    predicted_gain_pixels = int(pred.sum())

    gt_area_pct = 100 * gt_gain_pixels / n_valid
    predicted_area_pct = 100 * predicted_gain_pixels / n_valid

    if gt_gain_pixels > 0:
        area_ratio = predicted_gain_pixels / gt_gain_pixels
    else:
        # Undefined when there are no ground-truth gain pixels.
        area_ratio = np.nan

    # Model probability and confidence.
    confidence = np.maximum(p, 1.0 - p)
    correct = pred == t
    incorrect = ~correct

    # Probability on true gain and background pixels.
    probability_gt_gain = p[t]
    probability_gt_background = p[~t]

    # Confidence assigned to true positives, false positives
    # and false negatives.
    probability_tp = p[pred & t]
    probability_fp = p[pred & ~t]
    probability_fn = 1.0 - p[~pred & t]

    # Brier score: lower is better.
    brier_score = float(np.mean((p - t.astype(np.float32)) ** 2))

    # IMPORTANT:
    # Preserve the original 2D spatial layout when calculating
    # connected components. Do not flatten and then label.
    gt_stats = component_stats(gt & valid)
    pred_stats = component_stats((probs > threshold) & valid)

    results = {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "valid_pixels": n_valid,
        "gt_gain_pixels": gt_gain_pixels,
        "predicted_gain_pixels": predicted_gain_pixels,
        "gain_area_pct": float(gt_area_pct),
        "predicted_gain_area_pct": float(predicted_area_pct),
        "predicted_to_true_area_ratio": float(area_ratio),
        "mean_probability": safe_mean(p),
        "mean_predicted_positive_probability": safe_mean(p[pred]),
        "mean_confidence": safe_mean(confidence),
        "mean_confidence_correct": safe_mean(confidence[correct]),
        "mean_confidence_incorrect": safe_mean(confidence[incorrect]),
        "mean_probability_gt_gain": safe_mean(probability_gt_gain),
        "mean_probability_gt_background": safe_mean(probability_gt_background),
        "mean_probability_tp": safe_mean(probability_tp),
        "mean_confidence_fp": safe_mean(probability_fp),
        "mean_confidence_fn": safe_mean(probability_fn),
        "brier_score": brier_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    # Add ground-truth component statistics.
    for key, value in gt_stats.items():
        results[f"gt_{key}"] = value

    # Add predicted component statistics.
    for key, value in pred_stats.items():
        results[f"pred_{key}"] = value

    return results


# ---------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    # Ground-truth mask characteristics
    "gain_area_pct",
    "gt_n_components",
    "gt_mean_component_size",
    "gt_largest_component_size",
    "gt_largest_component_fraction",
    # Predicted mask characteristics
    "pred_n_components",
    "pred_mean_component_size",
    "pred_largest_component_size",
    "pred_largest_component_fraction",
    # Probability distribution
    "mean_probability",
    "mean_confidence",
    "mean_probability_gt_gain",
    "mean_probability_gt_background",
]


def calculate_correlations(df):
    """Calculate Spearman correlations between features and IoU."""
    results = []

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue

        subset = df[["iou", column]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(subset) < 3:
            continue

        # Constant columns cannot produce a meaningful correlation.
        if subset["iou"].nunique() < 2 or subset[column].nunique() < 2:
            continue

        rho, p_value = spearmanr(
            subset["iou"],
            subset[column],
        )

        results.append(
            {
                "feature": column,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "n_tiles": int(len(subset)),
            }
        )

    correlations = pd.DataFrame(results)

    if not correlations.empty:
        correlations["abs_rho"] = correlations["spearman_rho"].abs()
        correlations = correlations.sort_values(
            "abs_rho",
            ascending=False,
        )

    return correlations


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------


def make_correlation_plots(df, correlations, output_dir, top_n=8):
    """Create scatter plots for the strongest correlations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if correlations.empty:
        print("No usable correlations to plot.")
        return

    for _, row in correlations.head(top_n).iterrows():
        feature = row["feature"]

        subset = df[["iou", feature]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(subset) < 3:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            subset[feature],
            subset["iou"],
            alpha=0.7,
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("IoU")
        ax.set_title(
            f"IoU vs {feature}\n"
            f"Spearman rho = {row['spearman_rho']:.3f}, "
            f"p = {row['p_value']:.3g}"
        )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"iou_vs_{feature}.png"
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tif_dir",
        type=Path,
        required=True,
        help="Folder with <tile>_prob.tif and <tile>_gt.tif.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="Diagnostic PNGs (default: <tif_dir>/../plots).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the binary prediction.",
    )
    args = parser.parse_args()

    if not args.tif_dir.exists():
        raise FileNotFoundError(f"Path does not exist: {args.tif_dir}")
    if args.plot_dir is None:
        args.plot_dir = args.tif_dir.parent / "plots"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked_plot_dir = args.out_dir / "ranked_plots"
    correlation_plot_dir = args.out_dir / "correlation_plots"
    ranked_plot_dir.mkdir(parents=True, exist_ok=True)
    correlation_plot_dir.mkdir(parents=True, exist_ok=True)

    prob_files = sorted(args.tif_dir.glob("*_prob.tif"))
    print(
        f"Found {len(prob_files)} probability tifs in {args.tif_dir}; threshold {args.threshold}"
    )
    rows = []
    for probability_path in prob_files:
        tile_id = probability_path.name[: -len("_prob.tif")]
        gt_path = args.tif_dir / f"{tile_id}_gt.tif"
        if not gt_path.exists():
            print(f"WARNING: no ground truth for {tile_id}: {gt_path.name}")
            continue
        with rasterio.open(probability_path) as src:
            probs = src.read(1).astype(np.float32)
        with rasterio.open(gt_path) as src:
            gt_raw = src.read(1).astype(np.float32)
        if probs.shape != gt_raw.shape:
            raise ValueError(
                f"Shape mismatch for {tile_id}: probabilities={probs.shape}, ground truth={gt_raw.shape}"
            )
        valid = np.isfinite(
            probs
        )  # NaN in the ground truth means 'no gain' (as in utils.py), not 'invalid'

        gt = np.nan_to_num(gt_raw, nan=0.0) > 50
        probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)

        results = analyse_tile(
            probs=probs, gt=gt, valid=valid, threshold=args.threshold
        )
        results["tile_id"] = tile_id
        results["valid_fraction"] = float(valid.mean())
        rows.append(results)
        print(
            f"{tile_id}: IoU={results['iou']:.4f}, GT gain={results['gain_area_pct']:.2f}%, "
            f"predicted gain={results['predicted_gain_area_pct']:.2f}%, valid={100 * valid.mean():.1f}%"
        )

    if not rows:
        raise RuntimeError(
            "No tiles were analysed: check the *_prob.tif / *_gt.tif names in --tif_dir."
        )

    # -----------------------------------------------------
    # Save tile analysis and rank by IoU.
    # -----------------------------------------------------

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "iou",
        ascending=True,
    ).reset_index(drop=True)

    df.insert(
        0,
        "rank",
        np.arange(1, len(df) + 1),
    )

    csv_path = args.out_dir / "tile_analysis.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved tile analysis to: {csv_path}")

    # -----------------------------------------------------
    # Copy existing diagnostic plots into ranked folders.
    # -----------------------------------------------------

    if args.plot_dir.exists():
        plot_files = list(args.plot_dir.glob("*.png"))

        for _, row in df.iterrows():
            tile_id = str(row["tile_id"])
            rank = int(row["rank"])
            iou = float(row["iou"])

            # Match plots whose filename contains the tile ID.
            matching_plots = [path for path in plot_files if tile_id in path.stem]

            for plot_path in matching_plots:
                destination = (
                    ranked_plot_dir / f"{rank:03d}_IoU_{iou:.4f}_{plot_path.name}"
                )

                shutil.copy2(plot_path, destination)

    else:
        print(f"WARNING: Plot directory does not exist: {args.plot_dir}")

    # -----------------------------------------------------
    # Correlation analysis.
    # -----------------------------------------------------

    correlations = calculate_correlations(df)

    correlation_csv = args.out_dir / "iou_correlations.csv"

    correlations.to_csv(
        correlation_csv,
        index=False,
    )

    print(f"\nSaved correlations to: {correlation_csv}")

    if correlations.empty:
        print("No usable correlations could be calculated.")
    else:
        print("\nStrongest Spearman correlations with IoU:")
        print(
            correlations[
                [
                    "feature",
                    "spearman_rho",
                    "p_value",
                    "n_tiles",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # -----------------------------------------------------
    # Correlation plots.
    # -----------------------------------------------------

    make_correlation_plots(
        df,
        correlations,
        correlation_plot_dir,
        top_n=20,
    )

    # -----------------------------------------------------
    # Summary.
    # -----------------------------------------------------

    print("\nAnalysis complete.")
    print(f"Tiles analysed: {len(df)}")
    print(f"Mean IoU: {df['iou'].mean():.4f}")
    print(f"Median IoU: {df['iou'].median():.4f}")
    print(f"Minimum IoU: {df['iou'].min():.4f}")
    print(f"Maximum IoU: {df['iou'].max():.4f}")

    print(f"\nOutputs saved in: {args.out_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {correlation_csv.name}")
    print("  - ranked_plots/")
    print("  - correlation_plots/")


if __name__ == "__main__":
    main()
