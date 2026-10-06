# predict_masks.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from datasets import MultiTemporalGainDataset
from eval import evaluate_checkpoint, generate_gain_map


def main(ckpt, test_root, out_dir, threshold=0.5, max_plots=20, crop_size=None):
    out = Path(out_dir)
    (out / "tifs").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Running Evaluation Metrics ===")
    evaluate_checkpoint(
        checkpoint_path=ckpt,
        test_dir=test_root,
        threshold=threshold,
        crop_size=crop_size,
    )

    tile_dirs = sorted(
        p
        for p in Path(test_root).iterdir()
        if p.is_dir() and p.name.startswith("tile_")
    )

    if not tile_dirs:
        raise ValueError(f"No tile_* directories found in {test_root}")

    print(f"\n=== Step 2: Generating Gain Maps and Plots ({len(tile_dirs)} tiles) ===")
    ds = MultiTemporalGainDataset(tile_dirs, crop_size=crop_size)
    rgb_idx = [ds.band_names.index(b) for b in ("B4", "B3", "B2")]

    for i, tile_dir in enumerate(tile_dirs):
        tid = tile_dir.name
        prob_path = out / "tifs" / f"{tid}_prob.tif"
        mask_path = out / "tifs" / f"{tid}_mask.tif"

        # 1. Generate probability map GeoTIFF via eval.py helper
        generate_gain_map(
            checkpoint_path=ckpt,
            tile_dir=tile_dir,
            output_tif=prob_path,
            crop_size=crop_size,
        )

        # 2. Read back probability array to derive binary mask and plots
        with rasterio.open(prob_path) as src:
            probs = src.read(1)

        s = ds[i]
        pred = probs > threshold
        gt = s["gain_mask"].numpy() > 0.5
        valid = s["gain_valid"].numpy() > 0.5

        # 3. Save binary mask TIF matching spatial profile
        ref_tif = tile_dir / "composites" / f"s1s2_{ds.years[0]}.tif"
        with rasterio.open(ref_tif) as src:
            prof = src.profile.copy()
            h, w = probs.shape
            if (h, w) != (src.height, src.width):
                top, left = (src.height - h) // 2, (src.width - w) // 2
                prof.update(
                    height=h,
                    width=w,
                    transform=src.transform * rasterio.Affine.translation(left, top),
                )

        prof.update(count=1, dtype=np.uint8, nodata=None)
        with rasterio.open(mask_path, "w", **prof) as dst:
            dst.write((pred & valid).astype(np.uint8), 1)

        # 4. Diagnostic Plots
        if i < max_plots:
            px = s["pixels"].numpy()

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--out_dir", default="predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_plots", type=int, default=20)
    p.add_argument("--crop_size", type=int, default=None)

    a = p.parse_args()

    main(
        ckpt=a.ckpt,
        test_root=a.test_root,
        out_dir=a.out_dir,
        threshold=a.threshold,
        max_plots=a.max_plots,
        crop_size=a.crop_size,
    )
