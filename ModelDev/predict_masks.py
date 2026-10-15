# predict_masks.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from config import EMBEDDING_SOURCES, SOURCES
from datasets import MultiTemporalGainDataset
from eval import evaluate_checkpoint, generate_gain_map


def checkpoint_sources(ckpt) -> tuple[str, ...]:
    """Input sources the checkpoint was trained on (old checkpoints = s1 + s2)."""
    hp = torch.load(ckpt, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {}
    )
    return tuple(hp.get("sources", ("s1", "s2")))


def source_slice(sources, name):
    """Channel slice of one source inside the concatenated input."""
    start = 0
    for s in sources:
        n = len(SOURCES[s][1])
        if s == name:
            return slice(start, start + n)
        start += n
    raise KeyError(name)


def _stretch(a, lo_hi=None):
    lo, hi = lo_hi if lo_hi is not None else np.percentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def make_renderer(px, sources, name):
    """Return (render(t) -> HxWx3 image, description) for one trained-on source.
    px: (T, C, H, W) normalised model input."""
    sl = source_slice(sources, name)
    bands = list(SOURCES[name][1])
    x = px[:, sl]  # (T, c, H, W)

    if name == "s2":  # true colour
        idx = [bands.index(b) for b in ("B4", "B3", "B2")]
        return (lambda t: _stretch(x[t][idx].transpose(1, 2, 0))), "RGB"

    if name in EMBEDDING_SOURCES:  # PCA of the embedding, fitted on first+last frame
        T, c, H, W = x.shape
        flat = np.concatenate(
            [x[0].reshape(c, -1), x[-1].reshape(c, -1)], axis=1
        ).T  # (2HW, c)
        mu = flat.mean(0)
        _, _, vt = np.linalg.svd(
            flat[:: max(1, len(flat) // 20000)] - mu, full_matrices=False
        )
        comp = vt[:3]
        proj = {
            t: ((x[t].reshape(c, -1).T - mu) @ comp.T).reshape(H, W, 3)
            for t in (0, T - 1)
        }
        both = np.concatenate([proj[0], proj[T - 1]])
        lo_hi = np.percentile(
            both, [2, 98]
        )  # same stretch both years so colours compare
        return (lambda t: _stretch(proj[0 if t == 0 else T - 1], lo_hi)), "PCA(3)"

    # SAR / anything else: false colour from the first two bands (R=b0, G=b1, B=b0-b1)
    b0, b1 = x[:, 0], x[:, min(1, x.shape[1] - 1)]
    stack = np.stack([b0, b1, b0 - b1], axis=-1)  # (T, H, W, 3)
    lo_hi = np.percentile(stack, [2, 98])
    return (
        lambda t: _stretch(stack[t], lo_hi)
    ), f"{bands[0]}/{bands[min(1, len(bands) - 1)]}"


def main(
    ckpt, test_root, out_dir, threshold=0.5, max_plots=20, crop_size=None, show=None
):
    out = Path(out_dir)
    (out / "tifs").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    sources = checkpoint_sources(ckpt)
    show = list(show) if show else ["s2" if "s2" in sources else sources[0]]
    bad = [s for s in show if s not in sources]
    if bad:
        raise ValueError(f"--show {bad}: model was trained on {list(sources)}")
    print(f"model sources: {sources} | showing: {show}")

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
    ds = MultiTemporalGainDataset(tile_dirs, sources=sources, crop_size=crop_size)

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

        # 4. Diagnostic plots: start/end imagery for each chosen source, then label/prob/pred
        if i < max_plots:
            px = s["pixels"].numpy()
            ncols = 2 * len(show) + 3
            fig, ax = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

            k = 0
            for name in show:
                render, desc = make_renderer(px, sources, name)
                ax[k].imshow(render(0))
                ax[k].set_title(f"{ds.years[0]} {name} {desc}")
                ax[k + 1].imshow(render(-1))
                ax[k + 1].set_title(f"{ds.years[-1]} {name} {desc}")
                k += 2

            ax[k].imshow(gt & valid, cmap="gray")
            ax[k].set_title("label")
            ax[k + 1].imshow(probs, vmin=0, vmax=1, cmap="viridis")
            ax[k + 1].set_title("prob")
            ax[k + 2].imshow(pred & valid, cmap="gray")
            ax[k + 2].set_title(f"pred > {threshold}")

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
    p.add_argument(
        "--show",
        nargs="+",
        default=None,
        help="which of the model's training sources to display next to the results "
        "(e.g. s2, s1, tessera; several allowed). Default: s2 if the model used it, else its first source.",
    )

    a = p.parse_args()

    main(
        ckpt=a.ckpt,
        test_root=a.test_root,
        out_dir=a.out_dir,
        threshold=a.threshold,
        max_plots=a.max_plots,
        crop_size=a.crop_size,
        show=a.show,
    )
