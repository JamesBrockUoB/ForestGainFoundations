from pathlib import Path

import rasterio
import torch
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader


def evaluate_checkpoint(checkpoint_path: str, test_dirs: list[Path]):
    task = GainDetectionTask.load_from_checkpoint(checkpoint_path)
    task.eval()
    task.freeze()

    test_ds = MultiTemporalGainDataset(test_dirs)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for batch in test_loader:
            pixels = batch["pixels"]
            gain_mask = batch["gain_mask"]
            gain_valid = batch["gain_valid"]

            logits = task(pixels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            valid = gain_valid.bool()
            correct = (preds[valid] == gain_mask[valid]).sum().item()
            total_correct += correct
            total_pixels += valid.sum().item()

    accuracy = total_correct / max(total_pixels, 1)
    print(f"Test Pixel Accuracy (Valid Pixels): {accuracy * 100:.2f}%")
    return accuracy


def generate_gain_map(checkpoint_path: str, tile_dir: Path, output_tif: Path):
    """Generates continuous gain probability GeoTIFF map for a given tile."""
    task = GainDetectionTask.load_from_checkpoint(checkpoint_path)
    task.eval()

    ds = MultiTemporalGainDataset([tile_dir])
    sample = ds[0]
    pixels = sample["pixels"].unsqueeze(0)  # (1, T, C, H, W)

    with torch.no_grad():
        logits = task(pixels)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (H, W)

    ref_tif = tile_dir / f"s1s2_{ds.years[0]}.tif"
    with rasterio.open(ref_tif) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype=rasterio.float32)

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(probs, 1)

    print(f"Saved probability gain map to {output_tif}")
