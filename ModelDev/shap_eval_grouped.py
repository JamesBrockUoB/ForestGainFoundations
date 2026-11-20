"""Grouped GradientExplainer (expected gradients) attributions for TSViT gain segmentation.

Extends the previous shap_eval.py to report:
  1. Source-level importance: S1, S2, AlphaEarth, Tessera (and derived indices separately).
  2. Individual band / embedding-dimension importance.
  3. Temporal importance by year, including source-by-year percentages.

IMPORTANT: Use the same dataset arguments and model hyperparameters as training.
The script uses the validation split produced by split_tile_dirs; it does not use
holdout_tiles unless you explicitly point --data-dir there (do not do that for
model interpretation if the holdout is being kept untouched).

Example:
  python shap_eval_grouped.py --ckpt /path/to/best.pt \
      --data-dir /path/to/DataCollection/data/test_tiles \
      --sources s1 s2 alphaearth tessera --n-tiles 50 \
      --region true_gain --out shap_out

Edit the two project imports below only if your module paths differ.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import shap
import torch

from config import EMBEDDING_SOURCES, NORM_NAN_FILL, SOURCES
from datasets import NORMALISED_BANDS, MultiTemporalGainDataset, split_tile_dirs
from models.tsvit import TSViT


class GainScore(torch.nn.Module):
    """Return mean gain logit over a fixed pixel region: (B,T,C,H,W) -> (B,1)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.region = None

    def forward(self, x):
        logits = self.model(x).reshape(x.shape[0], -1)
        region = self.region.reshape(1, -1).float().to(logits.device)
        return (logits * region).sum(1, keepdim=True) / region.sum().clamp(min=1)


class ZeroSkip(torch.nn.Module):
    """Replace the full-resolution skip branch to isolate the transformer path."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, z):
        b, _, h, w = z.shape
        return z.new_zeros(b, self.channels, h, w)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, device, in_channels, args):
    model = TSViT(
        in_channels=in_channels,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth_temporal=args.depth_temporal,
        depth_spatial=args.depth_spatial,
        num_heads=args.num_heads,
        max_frames=args.max_frames,
        skip_channels=args.skip_channels,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    if state and all(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model


def build_feature_and_source_groups(ds):
    """Return individual feature groups and source groups as channel-index lists."""
    feature_groups = {}
    source_groups = {}
    offset = 0

    for source in ds.sources:
        if source not in SOURCES:
            raise KeyError(
                f"Source {source!r} is not present in config.SOURCES. Available: {list(SOURCES)}"
            )
        bands = SOURCES[source][1]
        indices = list(range(offset, offset + len(bands)))
        source_groups[source] = indices

        if source in EMBEDDING_SOURCES:
            for j, _band in enumerate(bands):
                feature_groups[f"{source}_{j:03d}"] = [offset + j]
        else:
            for j, band in enumerate(bands):
                feature_groups[str(band)] = [offset + j]
        offset += len(bands)

    # Dataset-derived indices are channels after the configured source channels.
    derived_names = list(ds.band_names[offset:])
    if derived_names:
        derived_indices = list(range(offset, offset + len(derived_names)))
        source_groups["derived_indices"] = derived_indices
        for j, band in enumerate(derived_names):
            feature_groups[str(band)] = [offset + j]

    # Fail early if config channel definitions and dataset channel count disagree.
    all_indices = [i for group in source_groups.values() for i in group]
    if sorted(all_indices) != list(range(ds.num_channels)):
        raise ValueError(
            "Could not map every dataset channel to exactly one source. "
            f"Mapped {len(all_indices)} of {ds.num_channels} channels. "
            "Check SOURCES, EMBEDDING_SOURCES and ds.band_names."
        )
    return feature_groups, source_groups


def make_baseline(kind, ds, k, T, C, H, W, device):
    """Return baseline tensor (N,T,C,H,W) and number of dataset tiles consumed."""
    if kind == "tiles":
        if k < 1:
            raise ValueError("--n-baseline must be at least 1 when --baseline tiles")
        if len(ds) <= k:
            raise ValueError(
                f"Validation dataset has {len(ds)} tiles, but --n-baseline={k}; "
                "need at least one additional tile to explain."
            )
        xs = [ds[i]["pixels"].unsqueeze(0) for i in range(k)]
        return torch.cat(xs).to(device), k

    if kind == "zeros":
        return torch.zeros(1, T, C, H, W, device=device), 0

    # 'missing': use the same fill values as the data pipeline.
    fill = torch.tensor(
        [
            float(NORM_NAN_FILL)
            if (band in NORMALISED_BANDS or band.startswith("idx_"))
            else 0.0
            for band in ds.band_names
        ],
        device=device,
    )
    baseline = fill.view(1, 1, C, 1, 1).expand(1, T, C, H, W).contiguous()
    return baseline, 0


def first_output(shap_values, x):
    """Handle SHAP versions returning list outputs or a trailing output dimension."""
    if isinstance(shap_values, list):
        return shap_values[0]
    values = np.asarray(shap_values)
    return values[..., 0] if values.ndim == x.ndim + 1 else values


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="shap_out")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")

    # These must match the training dataset configuration.
    p.add_argument("--sources", nargs="+", default=["s1", "s2", "alphaearth", "tessera"])
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--indices", nargs="*", default=[])
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--conf-filter", type=float, default=None)
    p.add_argument("--conf-harden", type=float, default=None)
    p.add_argument("--min-gain-px", type=int, default=0)
    p.add_argument("--fill-holes-px", type=int, default=0)

    # SHAP configuration.
    p.add_argument("--n-tiles", type=int, default=50)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--baseline", choices=["missing", "zeros", "tiles"], default="missing")
    p.add_argument("--n-baseline", type=int, default=4)
    p.add_argument("--region", choices=["true_gain", "pred_gain"], default="true_gain")
    p.add_argument("--aggregate", choices=["all", "region"], default="region")
    p.add_argument("--zero-skip", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # Model hyperparameters must match the checkpoint architecture.
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth-temporal", type=int, default=2)
    p.add_argument("--depth-spatial", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--skip-channels", type=int, default=32)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}")

    # Validation split only. Keep the separate holdout directory untouched.
    _, val_dirs = split_tile_dirs(args.data_dir, args.val_frac, args.split_seed)
    ds = MultiTemporalGainDataset(
        val_dirs,
        sources=tuple(args.sources),
        augment=False,
        crop_size=args.crop_size,
        years=tuple(args.years) if args.years else None,
        indices=tuple(args.indices),
        conf_filter=args.conf_filter,
        conf_harden=args.conf_harden,
        min_gain_px=args.min_gain_px,
        fill_holes_px=args.fill_holes_px,
    )

    if len(ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check --data-dir and split settings.")

    sample = ds[0]["pixels"]
    T, C, H, W = sample.shape
    if T > args.max_frames:
        raise ValueError(f"Dataset has T={T} frames, exceeding --max-frames={args.max_frames}.")
    print(f"Validation tiles: {len(ds)} | input T={T}, C={C}, H={H}, W={W} | years={ds.years}")

    feature_groups, source_groups = build_feature_and_source_groups(ds)
    feature_names = list(feature_groups)
    source_names = list(source_groups)

    model = load_model(args.ckpt, device, C, args)
    if args.zero_skip and getattr(model, "skip_branch", None) is not None:
        model.skip_branch = ZeroSkip(args.skip_channels).to(device)
        print("Skip branch zeroed: attributions isolate the transformer path.")

    wrapper = GainScore(model).to(device).eval()
    baseline, n_consumed = make_baseline(
        args.baseline, ds, args.n_baseline, T, C, H, W, device
    )

    # The explainer's initial forward pass needs a defined region.
    wrapper.region = torch.ones((H, W), dtype=torch.bool, device=device)
    explainer = shap.GradientExplainer(wrapper, baseline, batch_size=1)

    feature_abs = np.zeros((T, C), dtype=np.float64)
    feature_signed = np.zeros((T, C), dtype=np.float64)
    n_processed = 0

    for i in range(n_consumed, len(ds)):
        if n_processed >= args.n_tiles:
            break
        item = ds[i]
        x = item["pixels"].unsqueeze(0).to(device).float()
        valid = item["gain_valid"] > 0.5

        if args.region == "true_gain":
            region = (item["gain_mask"] > 0.5) & valid
        else:
            with torch.no_grad():
                region = (model(x).reshape(H, W).cpu() > 0) & valid.cpu()

        if int(region.sum()) == 0:
            print(f"Skipping {item['tile_id']}: selected region is empty.")
            continue

        wrapper.region = region.to(device)
        shap_values = explainer.shap_values(x, nsamples=args.nsamples)
        sv = np.asarray(first_output(shap_values, x))[0]  # expected shape (T,C,H,W)
        if sv.shape != (T, C, H, W):
            raise RuntimeError(f"Unexpected SHAP array shape {sv.shape}; expected {(T, C, H, W)}")

        with torch.no_grad():
            fx = wrapper(x).item()
            f0 = wrapper(baseline).mean().item()
        print(
            f"[{n_processed + 1}/{args.n_tiles}] {item['tile_id']}: "
            f"sum(attr)={sv.sum():.4f} | f(x)-f(base)={fx - f0:.4f}"
        )

        if args.aggregate == "region":
            mask = region.numpy()
            feature_abs += np.abs(sv[..., mask]).mean(axis=-1)
            feature_signed += sv[..., mask].mean(axis=-1)
        else:
            feature_abs += np.abs(sv).mean(axis=(-1, -2))
            feature_signed += sv.mean(axis=(-1, -2))

        n_processed += 1
        del sv, shap_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_processed == 0:
        raise RuntimeError("No tiles with a non-empty explanation region were processed.")

    feature_abs /= n_processed
    feature_signed /= n_processed

    # Individual feature arrays: (T,C). Source arrays: (T,S).
    source_abs = np.stack(
        [feature_abs[:, idx].sum(axis=1) for idx in source_groups.values()], axis=1
    )
    source_signed = np.stack(
        [feature_signed[:, idx].sum(axis=1) for idx in source_groups.values()], axis=1
    )
    source_mean_abs_per_feature = np.stack(
        [feature_abs[:, idx].mean(axis=1) for idx in source_groups.values()], axis=1
    )

    # Percentages are computed from absolute attribution magnitudes.
    source_total = source_abs.sum(axis=1, keepdims=True)
    source_pct_by_year = np.divide(
        source_abs, source_total, out=np.zeros_like(source_abs), where=source_total > 0
    ) * 100.0
    overall_source_abs = source_abs.sum(axis=0)
    overall_total = overall_source_abs.sum()
    overall_source_pct = (
        overall_source_abs / overall_total * 100.0 if overall_total > 0
        else np.zeros_like(overall_source_abs)
    )

    # Persist arrays and metadata.
    np.savez_compressed(
        os.path.join(args.out, "shap_importance.npz"),
        feature_abs=feature_abs,
        feature_signed=feature_signed,
        feature_names=np.array(ds.band_names),
        source_abs=source_abs,
        source_signed=source_signed,
        source_mean_abs_per_feature=source_mean_abs_per_feature,
        source_names=np.array(source_names),
        source_pct_by_year=source_pct_by_year,
        overall_source_pct=overall_source_pct,
        years=np.array(ds.years),
        n_tiles=n_processed,
    )

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "n_tiles_used": n_processed,
                   "source_names": source_names, "feature_count": C}, f, indent=2)

    # Source-level CSV: both total source attribution and per-feature average.
    source_rows = []
    for j, name in enumerate(source_names):
        source_rows.append([
            name,
            len(source_groups[name]),
            float(overall_source_abs[j]),
            float(overall_source_pct[j]),
            float(source_abs[:, j].mean()),
            float(source_mean_abs_per_feature[:, j].mean()),
            float(source_signed[:, j].sum()),
        ])
    source_rows.sort(key=lambda row: row[2], reverse=True)
    write_csv(
        os.path.join(args.out, "source_importance.csv"),
        ["source", "n_features", "total_abs_attribution", "overall_pct", 
         "mean_abs_attribution_per_year", "mean_abs_per_feature_per_year", "total_signed_attribution"],
        source_rows,
    )

    # Source-by-year CSV.
    year_rows = []
    for t, year in enumerate(ds.years):
        for j, name in enumerate(source_names):
            year_rows.append([
                int(year), name, float(source_abs[t, j]),
                float(source_pct_by_year[t, j]),
                float(source_mean_abs_per_feature[t, j]),
                float(source_signed[t, j]),
            ])
    write_csv(
        os.path.join(args.out, "source_year_importance.csv"),
        ["year", "source", "total_abs_attribution", "pct_of_year_total",
         "mean_abs_per_feature", "signed_attribution"],
        year_rows,
    )

    # Individual feature CSV, ordered by total absolute attribution.
    feature_rows = []
    for c, name in enumerate(ds.band_names):
        feature_rows.append([
            str(name), float(feature_abs[:, c].sum()),
            float(feature_abs[:, c].mean()), float(feature_signed[:, c].sum()),
        ])
    feature_rows.sort(key=lambda row: row[1], reverse=True)
    write_csv(
        os.path.join(args.out, "individual_feature_importance.csv"),
        ["feature", "total_abs_attribution", "mean_abs_attribution_per_year", "total_signed_attribution"],
        feature_rows,
    )

    # Individual feature-by-year CSV.
    feature_year_rows = []
    for t, year in enumerate(ds.years):
        for c, name in enumerate(ds.band_names):
            feature_year_rows.append([
                int(year), str(name), float(feature_abs[t, c]), float(feature_signed[t, c])
            ])
    write_csv(
        os.path.join(args.out, "feature_year_importance.csv"),
        ["year", "feature", "abs_attribution", "signed_attribution"],
        feature_year_rows,
    )

    print("\nSource importance (ranked by total |attribution|):")
    for row in source_rows:
        print(f"  {row[0]:<16} {row[3]:6.2f}%  total_abs={row[2]:.5e}  n_features={row[1]}")

    print("\nTemporal importance (all sources):")
    for t, year in enumerate(ds.years):
        print(f"  {year}: {source_abs[t].sum():.5e}")

    print("\nTop 20 individual features:")
    for row in feature_rows[:20]:
        print(f"  {row[0]:<20} abs={row[1]:.5e}")

    # Plots are optional; CSV and NPZ outputs are always written.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Overall source importance.
        order = np.argsort(-overall_source_abs)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.65 * len(source_names))))
        ax.barh([source_names[j] for j in order[::-1]], overall_source_abs[order[::-1]])
        ax.set_xlabel("Sum of mean absolute SHAP attribution")
        ax.set_title("TSViT input importance by source")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "source_importance.png"), dpi=180)
        plt.close(fig)

        # Percentage contribution by year.
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(T)
        for j, name in enumerate(source_names):
            ax.bar([str(y) for y in ds.years], source_pct_by_year[:, j], bottom=bottom, label=name)
            bottom += source_pct_by_year[:, j]
        ax.set_ylabel("Share of absolute attribution (%)")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 100)
        ax.set_title("Relative source contribution by year")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "source_temporal_percentages.png"), dpi=180)
        plt.close(fig)

        # Source/year heatmap.
        fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(source_names))))
        im = ax.imshow(source_abs.T, aspect="auto")
        ax.set_yticks(range(len(source_names)), source_names)
        ax.set_xticks(range(T), [str(y) for y in ds.years])
        ax.set_xlabel("Year")
        ax.set_title("Absolute SHAP attribution by source and year")
        fig.colorbar(im, ax=ax, label="Mean absolute attribution")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "source_year_heatmap.png"), dpi=180)
        plt.close(fig)

        # Top individual feature heatmap.
        top_n = min(30, C)
        top_idx = np.argsort(-feature_abs.sum(axis=0))[:top_n]
        fig, ax = plt.subplots(figsize=(12, max(5, 0.25 * top_n)))
        im = ax.imshow(feature_abs[:, top_idx].T, aspect="auto")
        ax.set_yticks(range(top_n), [str(ds.band_names[c]) for c in top_idx], fontsize=7)
        ax.set_xticks(range(T), [str(y) for y in ds.years])
        ax.set_xlabel("Year")
        ax.set_title(f"Top {top_n} individual features: absolute SHAP attribution")
        fig.colorbar(im, ax=ax, label="Mean absolute attribution")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "top_features_year_heatmap.png"), dpi=180)
        plt.close(fig)

        print(f"\nPlots saved in: {args.out}")
    except ImportError:
        print("matplotlib not installed; skipped plots. CSV and NPZ outputs were saved.")

    print(f"\nCompleted. Results saved to: {Path(args.out).resolve()}")
    print(f"Explained {n_processed} validation tiles; baseline tiles consumed: {n_consumed}.")


if __name__ == "__main__":
    main()
