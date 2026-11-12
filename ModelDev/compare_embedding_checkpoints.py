#!/usr/bin/env python3
"""
Compare forest-change checkpoints using each checkpoint's own trained-on sources.

Uses the project's existing:
  - config.py
  - datasets.MultiTemporalGainDataset
  - lightning_module.GainDetectionTask
  - utils.get_device

Each checkpoint's `hyper_parameters.sources` determines its own inputs. All models
are evaluated on the same tile directories and target gain labels. The dataset's
`gain_valid` mask is used for scoring.

Example:
  python compare_embedding_checkpoints.py \
    --test_root ../DataCollection/data/holdout_tiles \
    --model alphaearth=checkpoints/alphaearth.ckpt \
    --model tessera=checkpoints/tessera.ckpt \
    --out_dir embedding_comparison \
    --threshold 0.5

Run from the project directory so local imports (config, datasets, lightning_module,
utils) resolve correctly.

Outputs:
  overall_metrics.csv                 micro and macro metrics by model
  per_tile_metrics_long.csv           one row per model/tile
  per_tile_metrics_wide.csv           side-by-side tile metrics
  paired_deltas_vs_first_model.csv    tile-level metric deltas vs first --model
  predictions/<tile>__<model>_prob.tif
  predictions/<tile>__<model>_mask.tif
  comparisons/<tile>__<modelA>_vs_<modelB>.png
  comparisons/<tile>__<modelA>_vs_<modelB>_errors.tif

Error raster codes:
  0 = valid true negative for both
  1 = true positive for both
  2 = A true positive, B false negative
  3 = B true positive, A false negative
  4 = false positive for both
  5 = A-only false positive
  6 = B-only false positive
  7 = false negative for both
  255 = invalid/nodata

Important:
- Use identical threshold-selection rules for every model. Ideally choose thresholds
  on validation data, not by optimizing on holdout labels.
- If models use different source modalities, this is a comparison of their
  end-to-end modality/training configurations, not a controlled experiment that
  isolates the causal effect of embeddings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from utils import get_device


def parse_model(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT_PATH")
    name, checkpoint = value.split("=", 1)
    name, checkpoint = name.strip(), checkpoint.strip()
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT_PATH")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe_name, Path(checkpoint)


def checkpoint_sources(checkpoint_path: Path) -> tuple[str, ...]:
    """Read source list from Lightning checkpoint; support legacy S1+S2 checkpoints."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sources = ckpt.get("hyper_parameters", {}).get("sources", ("s1", "s2"))
    sources = tuple(sources)
    if not sources:
        raise ValueError(f"{checkpoint_path} has an empty `sources` hyperparameter")
    return sources


def confusion_metrics(truth, pred, valid):
    y = truth[valid].astype(bool)
    p = pred[valid].astype(bool)
    tp = int(np.count_nonzero(y & p))
    fp = int(np.count_nonzero(~y & p))
    tn = int(np.count_nonzero(~y & ~p))
    fn = int(np.count_nonzero(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "valid_pixels": int(valid.sum()),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1,
        "IoU": iou, "accuracy": accuracy,
    }


def get_truth_and_valid(sample):
    truth = sample["gain_mask"].detach().cpu().numpy() > 0.5
    valid = sample["gain_valid"].detach().cpu().numpy() > 0.5
    return np.squeeze(truth).astype(bool), np.squeeze(valid).astype(bool)


def raster_profile(tile_dir: Path, shape):
    """Use the project's reference S1/S2 raster and account for center crop."""
    ref = tile_dir / "composites" / "s1s2_2017.tif"
    with rasterio.open(ref) as src:
        profile = src.profile.copy()
        h, w = shape
        if (h, w) != (src.height, src.width):
            if h > src.height or w > src.width:
                raise ValueError(f"Prediction shape {shape} exceeds reference raster {(src.height, src.width)}")
            top, left = (src.height - h) // 2, (src.width - w) // 2
            profile.update(
                height=h,
                width=w,
                transform=src.transform * rasterio.Affine.translation(left, top),
            )
    profile.update(count=1, compress="deflate")
    return profile


def write_raster(path, arr, profile, dtype, nodata=None):
    p = profile.copy()
    p.update(dtype=dtype, count=1)
    if nodata is not None:
        p["nodata"] = nodata
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def safe_tile_key(tile_dir: Path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tile_dir.name)


def save_pairwise_comparison(path_png, path_tif, truth, valid, pred_a, pred_b,
                             name_a, name_b, profile):
    both_tp = truth & pred_a & pred_b & valid
    a_only_tp = truth & pred_a & ~pred_b & valid
    b_only_tp = truth & ~pred_a & pred_b & valid
    both_fn = truth & ~pred_a & ~pred_b & valid
    both_fp = ~truth & pred_a & pred_b & valid
    a_only_fp = ~truth & pred_a & ~pred_b & valid
    b_only_fp = ~truth & ~pred_a & pred_b & valid

    codes = np.zeros(truth.shape, dtype=np.uint8)
    codes[both_tp] = 1
    codes[a_only_tp] = 2
    codes[b_only_tp] = 3
    codes[both_fp] = 4
    codes[a_only_fp] = 5
    codes[b_only_fp] = 6
    codes[both_fn] = 7
    codes[~valid] = 255
    write_raster(path_tif, codes, profile, "uint8", nodata=255)

    rgb = np.full((*truth.shape, 3), 0.94, dtype=np.float32)
    rgb[~valid] = (0.55, 0.55, 0.55)
    rgb[both_tp] = (0.15, 0.65, 0.25)  # green
    rgb[a_only_tp] = (0.10, 0.35, 0.95)  # blue
    rgb[b_only_tp] = (1.00, 0.42, 0.05)  # orange
    rgb[both_fn] = (0.25, 0.25, 0.25)  # dark gray
    rgb[both_fp] = (0.65, 0.25, 0.75)  # purple
    rgb[a_only_fp] = (0.10, 0.75, 0.85)  # cyan
    rgb[b_only_fp] = (0.95, 0.70, 0.10)  # yellow

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    axes[0].imshow(truth & valid, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred_a & valid, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(name_a)
    axes[2].imshow(pred_b & valid, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(name_b)
    axes[3].imshow(rgb)
    axes[3].set_title("Paired error map")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{name_a} vs {name_b}\n"
        "Green both TP | Blue A TP/B FN | Orange B TP/A FN | "
        "Purple shared FP | Cyan A-only FP | Yellow B-only FP | "
        "Dark gray shared FN | Light gray TN | Gray invalid",
        fontsize=9,
    )
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def micro_summary(df, model_names):
    rows = []
    for name in model_names:
        sub = df[df["model"] == name]
        if sub.empty:
            continue
        tp, fp, tn, fn = (int(sub[k].sum()) for k in ("TP", "FP", "TN", "FN"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        rows.append({
            "model": name, "tiles_evaluated": int(sub["tile"].nunique()),
            "valid_pixels": int(sub["valid_pixels"].sum()),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "micro_precision": precision, "micro_recall": recall,
            "micro_F1": f1, "micro_IoU": iou, "micro_accuracy": accuracy,
            "macro_precision": sub["precision"].mean(),
            "macro_recall": sub["recall"].mean(),
            "macro_F1": sub["F1"].mean(),
            "macro_IoU": sub["IoU"].mean(),
            "macro_accuracy": sub["accuracy"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_root", required=True, help="Directory containing tile_* folders")
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="Repeat: --model NAME=checkpoint.ckpt. First model is the delta baseline.")
    parser.add_argument("--out_dir", default="embedding_comparison")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--device", default=None, help="Defaults to project utils.get_device()")
    parser.add_argument("--min_gain_px", type=int, default=0)
    parser.add_argument("--fill_holes_px", type=int, default=0)
    parser.add_argument("--max_plots", type=int, default=0,
                        help="Limit pairwise PNGs per tile; 0 means all tiles.")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    model_names = [n for n, _ in args.model]
    if len(set(model_names)) != len(model_names):
        parser.error("Model names must be unique")
    for name, ckpt in args.model:
        if not ckpt.is_file():
            parser.error(f"Checkpoint for {name} not found: {ckpt}")

    root = Path(args.test_root)
    tile_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("tile_"))
    if not tile_dirs:
        parser.error(f"No tile_* directories found in {root}")

    device = args.device or get_device()
    out = Path(args.out_dir)
    pred_dir = out / "predictions"
    cmp_dir = out / "comparisons"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    sources_by_model = {}
    for name, ckpt in args.model:
        sources = checkpoint_sources(ckpt)
        sources_by_model[name] = sources
        print(f"Loading {name}: sources={sources}, checkpoint={ckpt}")
        task = GainDetectionTask.load_from_checkpoint(str(ckpt), map_location=device)
        task.eval()
        task.freeze()
        task.to(device)
        models[name] = task

    # A separate dataset per model ensures its checkpoint gets its own source
    # channels. Target and validity masks come from the same dataset implementation.
    datasets = {
        name: MultiTemporalGainDataset(
            tile_dirs,
            sources=sources_by_model[name],
            crop_size=args.crop_size,
            min_gain_px=args.min_gain_px,
            fill_holes_px=args.fill_holes_px,
        )
        for name in model_names
    }

    rows, errors = [], []
    pairwise_counts = {}
    pairs = list(itertools.combinations(model_names, 2))

    for tile_idx, tile_dir in enumerate(tile_dirs):
        tile_id = tile_dir.name
        key = safe_tile_key(tile_dir)
        print(f"[{tile_idx + 1}/{len(tile_dirs)}] {tile_id}")
        per_model = {}
        truth = valid = profile = None

        for name in model_names:
            try:
                sample = datasets[name][tile_idx]
                gt, vm = get_truth_and_valid(sample)
                if truth is None:
                    truth, valid = gt, vm
                    profile = raster_profile(tile_dir, truth.shape)
                else:
                    # Labels and valid coverage must match across source-specific datasets.
                    if gt.shape != truth.shape or not np.array_equal(gt, truth):
                        raise ValueError("Target gain mask differs between model datasets")
                    if vm.shape != valid.shape or not np.array_equal(vm, valid):
                        raise ValueError(
                            "gain_valid differs between model datasets. "
                            "Check dataset/source validity logic before comparing."
                        )

                x = sample["pixels"].unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = models[name](x)
                    probs = torch.sigmoid(logits)
                    if probs.ndim == 4 and probs.shape[1] == 1:
                        probs = probs[:, 0]
                    probs = probs.squeeze().detach().float().cpu().numpy()
                if probs.ndim != 2 or probs.shape != gt.shape:
                    raise ValueError(f"Output shape {probs.shape}, expected {gt.shape}")
                pred = probs >= args.threshold
                per_model[name] = pred

                write_raster(pred_dir / f"{key}__{name}_prob.tif",
                             probs, profile, "float32")
                binary_out = np.where(vm, pred.astype(np.uint8), 255).astype(np.uint8)
                write_raster(pred_dir / f"{key}__{name}_mask.tif",
                             binary_out, profile, "uint8", nodata=255)

                row = {"tile": tile_id, "model": name, **confusion_metrics(gt, pred, vm)}
                row["sources"] = ",".join(sources_by_model[name])
                rows.append(row)
            except Exception as exc:
                errors.append({"tile": tile_id, "model": name, "error": repr(exc)})
                traceback.print_exc()

        if truth is None or valid is None or profile is None:
            continue
        for a, b in pairs:
            if a not in per_model or b not in per_model:
                continue
            pa, pb = per_model[a], per_model[b]
            pair_key = f"{a}__vs__{b}"
            counts = {
                "A_TP_B_FN": int(np.count_nonzero(truth & pa & ~pb & valid)),
                "B_TP_A_FN": int(np.count_nonzero(truth & ~pa & pb & valid)),
                "A_only_FP": int(np.count_nonzero(~truth & pa & ~pb & valid)),
                "B_only_FP": int(np.count_nonzero(~truth & ~pa & pb & valid)),
                "both_FN": int(np.count_nonzero(truth & ~pa & ~pb & valid)),
                "both_FP": int(np.count_nonzero(~truth & pa & pb & valid)),
            }
            pairwise_counts.setdefault(pair_key, []).append({"tile": tile_id, **counts})
            if args.max_plots == 0 or tile_idx < args.max_plots:
                save_pairwise_comparison(
                    cmp_dir / f"{key}__{pair_key}.png",
                    cmp_dir / f"{key}__{pair_key}_errors.tif",
                    truth, valid, pa, pb, a, b, profile,
                )

    if not rows:
        raise SystemExit("No successful predictions. Check the checkpoint and dataset errors.")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_tile_metrics_long.csv", index=False)
    micro_summary(df, model_names).to_csv(out / "overall_metrics.csv", index=False)

    metrics_to_widen = ["F1", "IoU", "precision", "recall", "TP", "FP", "TN", "FN"]
    wide = df.pivot(index="tile", columns="model", values=metrics_to_widen)
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    wide.reset_index().to_csv(out / "per_tile_metrics_wide.csv", index=False)

    # Paired deltas only for tiles on which both models succeeded.
    deltas = []
    baseline = model_names[0]
    for other in model_names[1:]:
        pair_df = df[df.model.isin([baseline, other])].pivot(
            index="tile", columns="model", values=["F1", "IoU", "precision", "recall"]
        )
        for metric in ("F1", "IoU", "precision", "recall"):
            if baseline in pair_df[metric].columns and other in pair_df[metric].columns:
                both = pair_df[metric][[baseline, other]].dropna()
                for tile, values in both.iterrows():
                    deltas.append({
                        "tile": tile, "baseline": baseline, "comparison": other,
                        "metric": metric, "baseline_value": values[baseline],
                        "comparison_value": values[other],
                        "comparison_minus_baseline": values[other] - values[baseline],
                    })
    pd.DataFrame(deltas).to_csv(out / "paired_deltas_vs_first_model.csv", index=False)

    pair_summary = []
    for pair_key, entries in pairwise_counts.items():
        d = pd.DataFrame(entries)
        summary = {"pair": pair_key, "tiles_compared": len(d)}
        for col in d.columns:
            if col != "tile":
                summary[f"total_{col}"] = int(d[col].sum())
        pair_summary.append(summary)
        d.to_csv(cmp_dir / f"{pair_key}_tile_error_counts.csv", index=False)
    pd.DataFrame(pair_summary).to_csv(out / "pairwise_error_summary.csv", index=False)

    summary_json = {
        "tiles_found": len(tile_dirs),
        "models": {
            name: {"checkpoint": str(ckpt), "sources": sources_by_model[name]}
            for name, ckpt in args.model
        },
        "threshold": args.threshold,
        "crop_size": args.crop_size,
        "min_gain_px": args.min_gain_px,
        "fill_holes_px": args.fill_holes_px,
        "device": str(device),
        "failed_inferences": errors,
        "notes": [
            "Each checkpoint was evaluated using its own checkpoint hyperparameter sources.",
            "All model datasets were required to return identical gain labels and valid masks.",
            "Micro metrics pool valid pixels; macro metrics average per-tile metrics equally.",
            "Embedding comparisons measure complete modality/training configurations.",
        ],
    }
    (out / "run_summary.json").write_text(json.dumps(summary_json, indent=2, default=str))
    if errors:
        pd.DataFrame(errors).to_csv(out / "inference_errors.csv", index=False)

    print("\nOverall metrics:")
    print(pd.read_csv(out / "overall_metrics.csv").to_string(index=False))
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
