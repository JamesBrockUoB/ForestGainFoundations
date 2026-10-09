from pathlib import Path

import rasterio
import torch
import torchmetrics
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader
from utils import get_device


def _checkpoint_sources(task) -> tuple[str, ...]:
    """Input sources the checkpoint was trained on (old checkpoints = s1+s2)."""
    return tuple(task.hparams.get("sources", ("s1", "s2")))


def evaluate_checkpoint(
    checkpoint_path: str,
    test_dir: str | Path,
    threshold: float = 0.5,
    crop_size: int | None = None,
    batch_size: int = 4,
    num_workers: int = 2,
):
    test_dir = Path(test_dir)
    test_dirs = sorted(
        p for p in test_dir.iterdir() if p.is_dir() and p.name.startswith("tile_")
    )

    if not test_dirs:
        raise ValueError(f"No tile_* directories found in {test_dir}")

    print(f"Evaluating {len(test_dirs)} tiles from {test_dir}")

    device = get_device()

    task = GainDetectionTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    task.eval()
    task.freeze()
    task.to(device)

    sources = _checkpoint_sources(task)
    print(f"sources: {sources}")

    test_ds = MultiTemporalGainDataset(test_dirs, sources=sources, crop_size=crop_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    kw = {"task": "binary", "threshold": threshold}
    metrics = {
        "f1": torchmetrics.F1Score(**kw).to(device),
        "iou": torchmetrics.JaccardIndex(**kw).to(device),
        "precision": torchmetrics.Precision(**kw).to(device),
        "recall": torchmetrics.Recall(**kw).to(device),
    }

    with torch.no_grad():
        for batch in test_loader:
            x = batch["pixels"].to(device=device, dtype=torch.float32)
            logits = task(x)
            probs = torch.sigmoid(logits)

            if probs.ndim == 4 and probs.shape[1] == 1:
                probs = probs.squeeze(1)

            # Move valid mask and targets to device (mps)
            valid = (batch["gain_valid"] > 0.5).to(device)
            targets = (batch["gain_mask"] > 0.5).int().to(device)

            p = probs[valid]
            t = targets[valid]

            if t.numel() == 0:
                continue

            for metric in metrics.values():
                metric.update(p, t)

    results = {name: metric.compute().item() for name, metric in metrics.items()}
    print(" | ".join(f"{k}: {v:.3f}" for k, v in results.items()))
    return results


def generate_gain_map(
    checkpoint_path: str,
    tile_dir: Path,
    output_tif: Path,
    crop_size: int | None = None,
):
    """Gain probability GeoTIFF for one tile (center crop if crop_size is set)."""
    device = get_device()
    task = GainDetectionTask.load_from_checkpoint(checkpoint_path, map_location=device)
    task.eval()
    task.to(device)

    ds = MultiTemporalGainDataset(
        [tile_dir], sources=_checkpoint_sources(task), crop_size=crop_size
    )
    pixels = (
        ds[0]["pixels"]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    with torch.no_grad():
        probs = torch.sigmoid(task(pixels)).squeeze().cpu().numpy()  # (H, W)

    ref_tif = tile_dir / "composites" / f"s1s2_{ds.years[0]}.tif"
    with rasterio.open(ref_tif) as src:
        profile = src.profile.copy()
        h, w = probs.shape
        if (h, w) != (src.height, src.width):
            top, left = (src.height - h) // 2, (src.width - w) // 2
            profile.update(
                height=h,
                width=w,
                transform=src.transform * rasterio.Affine.translation(left, top),
            )

    profile.update(count=1, dtype=rasterio.float32)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(probs, 1)

    print(f"Saved probability gain map to {output_tif}")
