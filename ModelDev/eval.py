from pathlib import Path

import rasterio
import torch
import torchmetrics
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_checkpoint(
    checkpoint_path: str,
    test_dirs: list[Path],
    threshold: float = 0.5,
    crop_size: int | None = None,
):
    device = _device()
    task = GainDetectionTask.load_from_checkpoint(checkpoint_path, map_location=device)
    task.eval()
    task.freeze()
    task.to(device)

    test_ds = MultiTemporalGainDataset(test_dirs, crop_size=crop_size)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    kw = {"task": "binary", "threshold": threshold}
    f1 = torchmetrics.F1Score(**kw)
    iou = torchmetrics.JaccardIndex(**kw)
    prec = torchmetrics.Precision(**kw)
    rec = torchmetrics.Recall(**kw)

    with torch.no_grad():
        for batch in test_loader:
            logits = task(batch["pixels"].to(device))
            probs = torch.sigmoid(logits).cpu()

            if probs.ndim == 4 and probs.shape[1] == 1:
                probs = probs.squeeze(1)

            valid = batch["gain_valid"] > 0.5
            targets = (batch["gain_mask"] > 0.5).int()

            p = probs[valid]
            t = targets[valid]

            if t.numel() == 0:
                continue

            # TorchMetrics applies the probability threshold itself.
            for metric in (f1, iou, prec, rec):
                metric.update(p, t)

    results = {
        "f1": f1.compute().item(),
        "iou": iou.compute().item(),
        "precision": prec.compute().item(),
        "recall": rec.compute().item(),
    }

    print(" | ".join(f"{k}: {v:.3f}" for k, v in results.items()))
    return results


def generate_gain_map(
    checkpoint_path: str,
    tile_dir: Path,
    output_tif: Path,
    crop_size: int | None = None,
):
    """Gain probability GeoTIFF for one tile (center crop if crop_size is set)."""
    device = _device()
    task = GainDetectionTask.load_from_checkpoint(checkpoint_path, map_location=device)
    task.eval()
    task.to(device)

    ds = MultiTemporalGainDataset([tile_dir], crop_size=crop_size)
    pixels = ds[0]["pixels"].unsqueeze(0).to(device)  # (1, T, C, H, W)

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
