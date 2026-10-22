"""Evaluate threshold and morphology post-processing for one checkpoint.

Positive-mask ``accuracy`` below means precision: the share of retained predicted
gain pixels that are true gain. Overall pixel accuracy is intentionally also
reported, but is not used for coverage selection because it is dominated by the
no-gain class in this task.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from datasets import MultiTemporalGainDataset
from eval import _checkpoint_sources
from lightning_module import GainDetectionTask
from utils import get_device


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
MORPH_SIZES = (1, 2, 5, 10, 25)
MIN_PRECISIONS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def remove_small(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    small = np.bincount(labels.ravel()) < size
    small[0] = False
    return mask & ~small[labels]


def fill_small(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    labels, _ = ndimage.label(~mask)  # 4-connected background components
    small = np.bincount(labels.ravel()) <= size
    small[0] = False
    # A connected background region touching an image boundary is exterior,
    # not a hole; never fill it even when processing small crops.
    boundary = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    small[boundary] = False
    return mask | small[labels]


def scores(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p, t = pred[valid], target[valid]
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "pixel_accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "coverage_pct": 100 * (tp + fp) / max(1, valid.sum()),
    }


def predict(checkpoint: str, data_dir: str, batch_size: int, num_workers: int):
    device = get_device()
    task = GainDetectionTask.load_from_checkpoint(checkpoint, map_location=device).eval().to(device)
    dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir() and p.name.startswith("tile_"))
    if not dirs:
        raise ValueError(f"No tile_* directories in {data_dir}")
    dataset = MultiTemporalGainDataset(dirs, sources=_checkpoint_sources(task))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, targets, valids = [], [], []
    with torch.no_grad():
        for batch in loader:
            prob = torch.sigmoid(task(batch["pixels"].to(device))).cpu().numpy()
            probs.append(prob)
            targets.append(batch["gain_mask"].numpy() > 0.5)
            valids.append(batch["gain_valid"].numpy() > 0.5)
    return np.concatenate(probs), np.concatenate(targets), np.concatenate(valids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--data-dir", default="../DataCollection/data/holdout_tiles")
    p.add_argument("--output", default="mask_sweep.csv")
    p.add_argument("--threshold", type=float, default=0.50, help="Base threshold for removal/filling sweeps.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    probs, target, valid = predict(args.checkpoint, args.data_dir, args.batch_size, args.num_workers)
    rows: list[dict] = []
    # Threshold sweep and a precision-constrained coverage frontier use raw masks.
    for threshold in THRESHOLDS:
        row = {"experiment": "threshold", "setting": threshold, "threshold": threshold}
        row.update(scores(probs >= threshold, target, valid))
        rows.append(row)
    for size in MORPH_SIZES:
        for name, fn in (("remove", remove_small), ("fill", fill_small)):
            pred = np.stack([fn(prob >= args.threshold, size) for prob in probs])
            row = {"experiment": name, "setting": size, "threshold": args.threshold}
            row.update(scores(pred, target, valid))
            rows.append(row)
    threshold_rows = [r for r in rows if r["experiment"] == "threshold"]
    for min_precision in MIN_PRECISIONS:
        eligible = [r for r in threshold_rows if r["precision"] >= min_precision]
        if eligible:
            best = max(eligible, key=lambda r: r["coverage_pct"])
            row = {"experiment": "coverage_at_precision", "setting": min_precision,
                   "threshold": best["threshold"], **{k: best[k] for k in best if k not in {"experiment", "setting", "threshold"}}}
        else:
            row = {"experiment": "coverage_at_precision", "setting": min_precision,
                   "threshold": None, "precision": None, "recall": None, "f1": None,
                   "iou": None, "pixel_accuracy": None, "coverage_pct": 0.0}
        rows.append(row)

    fields = list(rows[0])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {out}")


if __name__ == "__main__":
    main()
