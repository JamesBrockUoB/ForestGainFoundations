# predict_masks.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask


def main(ckpt, test_root, out_dir, threshold=0.5, max_plots=20):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    task = GainDetectionTask.load_from_checkpoint(ckpt, map_location=device)
    task.eval().to(device)

    tile_dirs = sorted(p for p in Path(test_root).iterdir() if p.is_dir())[-10:]
    ds = MultiTemporalGainDataset(tile_dirs)
    out = Path(out_dir)
    (out / "tifs").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rgb_idx = [ds.band_names.index(b) for b in ("B4", "B3", "B2")]
    tp = fp = fn = 0

    for i, tile_dir in enumerate(tile_dirs):
        s = ds[i]
        with torch.no_grad():
            logits = task(s["pixels"].unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = probs > threshold
        gt = s["gain_mask"].numpy() > 0.5
        valid = s["gain_valid"].numpy() > 0.5

        tp += int((pred & gt & valid).sum())
        fp += int((pred & ~gt & valid).sum())
        fn += int((~pred & gt & valid).sum())

        # GeoTIFFs: probability + binary mask (invalid pixels set to 0)
        ref = tile_dir / "composites" / f"s1s2_{ds.years[0]}.tif"
        with rasterio.open(ref) as src:
            prof = src.profile.copy()
        prof.update(count=1, nodata=None)
        tid = tile_dir.name
        with rasterio.open(
            out / "tifs" / f"{tid}_prob.tif", "w", **{**prof, "dtype": "float32"}
        ) as dst:
            dst.write(probs.astype(np.float32), 1)
        with rasterio.open(
            out / "tifs" / f"{tid}_mask.tif", "w", **{**prof, "dtype": "uint8"}
        ) as dst:
            dst.write((pred & valid).astype(np.uint8), 1)

        if i < max_plots:
            px = s["pixels"].numpy()  # T, C, H, W

            def rgb(t):
                a = px[t][rgb_idx].transpose(1, 2, 0)
                lo, hi = np.percentile(a, [2, 98])
                return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            ax[0].imshow(rgb(0))
            ax[0].set_title(f"{ds.years[0]} RGB")
            ax[1].imshow(rgb(-1))
            ax[1].set_title(f"{ds.years[-1]} RGB")
            ax[2].imshow(gt & valid, cmap="gray")
            ax[2].set_title("label")
            ax[3].imshow(probs, vmin=0, vmax=1, cmap="viridis")
            ax[3].set_title("prob")
            ax[4].imshow(pred & valid, cmap="gray")
            ax[4].set_title(f"pred > {threshold}")
            for a in ax:
                a.axis("off")
            fig.suptitle(tid)
            plt.tight_layout()
            plt.savefig(out / "plots" / f"{tid}.png", dpi=100)
            plt.close(fig)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    print(
        f"{len(tile_dirs)} tiles | IoU {iou:.3f} | precision {prec:.3f} | recall {rec:.3f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--out_dir", default="predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    main(a.ckpt, a.test_root, a.out_dir, a.threshold)
