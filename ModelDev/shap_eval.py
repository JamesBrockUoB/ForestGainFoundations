"""GradientExplainer (expected gradients) attributions for TSViT gain segmentation,
using MultiTemporalGainDataset on the validation split.

Edit the two import lines marked <-- if your module names differ.

Example:
    python shap_eval.py --ckpt best.pt --data-dir tiles --sources s1 s2 alphaearth tessera \
        --n-tiles 50 --region true_gain --out shap_out

Use the SAME --sources/--years/--indices/--conf-*/--min-gain-px/--fill-holes-px/
--val-frac/--split-seed as training, or the split, channels and labels will differ.
"""

import argparse
import json
import os

import numpy as np
import shap
import torch
from config import EMBEDDING_SOURCES, NORM_NAN_FILL, SOURCES
from datasets import (
    NORMALISED_BANDS,
    MultiTemporalGainDataset,
    split_tile_dirs,
)
from models.tsvit import TSViT  # <--


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GainScore(torch.nn.Module):
    """(B,T,C,H,W) -> (B,1): mean gain logit over a fixed pixel region."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.region = None  # (H,W) bool, set per tile

    def forward(self, x):
        logits = self.model(x).reshape(x.shape[0], -1)  # (B,H,W) or (B,1,H,W)
        r = self.region.reshape(1, -1).float().to(logits.device)
        return (logits * r).sum(1, keepdim=True) / r.sum().clamp(min=1)


class ZeroSkip(torch.nn.Module):
    """Replaces model.skip_branch so only the transformer path is explained."""

    def __init__(self, ch):
        super().__init__()
        self.ch = ch

    def forward(self, z):
        b, _, h, w = z.shape
        return z.new_zeros(b, self.ch, h, w)


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

    # Extract model weights from common checkpoint formats.
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    # Remove distributed-training prefix.
    state = {k.removeprefix("module."): v for k, v in checkpoint.items()}

    # Remove wrapper prefix if checkpoint was saved from a
    # model containing the TSViT under self.model.
    if state and all(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items()}

    # Strict loading ensures the architecture really matches.
    model.load_state_dict(state, strict=True)

    model.eval().requires_grad_(False).to(device)

    return model


def build_groups(ds) -> dict[str, list[int]]:
    groups = {}
    off = 0

    for s in ds.sources:
        bands = SOURCES[s][1]

        if s in EMBEDDING_SOURCES:
            # Preserve individual embedding dimensions
            for j, band in enumerate(bands):
                groups[f"{s}_{j:03d}"] = [off + j]
        else:
            for j, band in enumerate(bands):
                groups[band] = [off + j]

        off += len(bands)

    # Add derived indices
    for j, band in enumerate(ds.band_names[off:]):
        groups[band] = [off + j]

    return groups


def make_baseline(kind, ds, val_dirs, k, T, H, W, device):
    """Returns (baseline (N,T,C,H,W) tensor, number of val tiles consumed)."""
    C = ds.num_channels
    if kind == "tiles":
        xs = [ds[i]["pixels"].unsqueeze(0) for i in range(k)]
        return torch.cat(xs).to(device), k
    if kind == "zeros":
        return torch.zeros(1, T, C, H, W, device=device), 0
    # 'missing': what the pipeline writes for missing data
    fill = torch.tensor(
        [
            (
                float(NORM_NAN_FILL)
                if (b in NORMALISED_BANDS or b.startswith("idx_"))
                else 0.0
            )
            for b in ds.band_names
        ],
        device=device,
    )
    base = fill.view(1, 1, C, 1, 1).expand(1, T, C, H, W).contiguous()
    return base, 0


# ----------------------------------------------------------------------------
def first_output(sv, x):
    """shap versions differ: list of arrays vs one array with a trailing output dim."""
    if isinstance(sv, list):
        return sv[0]
    return sv[..., 0] if sv.ndim == x.ndim + 1 else sv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="shap_out")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    # dataset: match training
    p.add_argument(
        "--sources", nargs="+", default=["s1", "s2", "alphaearth", "tessera"]
    )
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--indices", nargs="*", default=[])
    p.add_argument("--crop-size", type=int, default=None, help="None = full tile")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--conf-filter", type=float, default=None)
    p.add_argument("--conf-harden", type=float, default=None)
    p.add_argument("--min-gain-px", type=int, default=0)
    p.add_argument("--fill-holes-px", type=int, default=0)
    # explainer
    p.add_argument("--n-tiles", type=int, default=50)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument(
        "--baseline", choices=["missing", "zeros", "tiles"], default="missing"
    )
    p.add_argument(
        "--n-baseline", type=int, default=4, help="tiles used when --baseline tiles"
    )
    p.add_argument("--region", choices=["true_gain", "pred_gain"], default="true_gain")
    p.add_argument(
        "--aggregate",
        choices=["all", "region"],
        default="region",
        help="average attributions over all pixels or only the explained region",
    )
    p.add_argument(
        "--zero-skip",
        action="store_true",
        help="zero the full-res skip branch to isolate the transformer path",
    )
    p.add_argument("--seed", type=int, default=0)
    # model hyper-parameters (must match the checkpoint)
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
    print(f"device: {device}")

    # Validation tiles only (holdout stays untouched)
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
    s0 = ds[0]["pixels"]
    T, C, H, W = s0.shape
    assert T <= args.max_frames, f"T={T} > --max-frames {args.max_frames}"
    print(f"val tiles: {len(ds)}  input: T={T} C={C} H={H} W={W}  years={ds.years}")

    groups = build_groups(ds)
    names = list(groups)

    model = load_model(args.ckpt, device, C, args)
    if args.zero_skip and model.skip_branch is not None:
        model.skip_branch = ZeroSkip(args.skip_channels).to(device)
    wrapper = GainScore(model).to(device).eval()

    baseline, n_used = make_baseline(
        args.baseline, ds, val_dirs, args.n_baseline, T, H, W, device
    )

    # SHAP calls the model during explainer initialisation.
    # Use a temporary all-pixel region for that initial call.
    # The actual region is assigned separately for each tile below.
    wrapper.region = torch.ones((H, W), dtype=torch.bool, device=device)

    explainer = shap.GradientExplainer(wrapper, baseline, batch_size=1)

    imp_abs = np.zeros((T, C))
    imp_sum = np.zeros((T, C))
    n = 0
    for i in range(n_used, len(ds)):
        if n >= args.n_tiles:
            break
        item = ds[i]
        x = item["pixels"].unsqueeze(0).to(device).float()  # fp32, no autocast
        valid = item["gain_valid"] > 0.5

        if args.region == "true_gain":
            region = (item["gain_mask"] > 0.5) & valid
        else:
            with torch.no_grad():
                region = (model(x).reshape(H, W).cpu() > 0) & valid
        if int(region.sum()) == 0:
            continue
        wrapper.region = region.to(device)

        sv = explainer.shap_values(x, nsamples=args.nsamples)
        sv = np.asarray(first_output(sv, x))[0]  # (T,C,H,W)

        with torch.no_grad():
            fx, f0 = wrapper(x).item(), wrapper(baseline).mean().item()
        print(
            f"[{n + 1}/{args.n_tiles}] {item['tile_id']}: "
            f"sum(attr)={sv.sum():.4f}  f(x)-f(base)={fx - f0:.4f}"
        )

        if args.aggregate == "region":
            r = region.numpy()
            imp_abs += np.abs(sv[..., r]).mean(-1)
            imp_sum += sv[..., r].mean(-1)
        else:
            imp_abs += np.abs(sv).mean(axis=(-1, -2))
            imp_sum += sv.mean(axis=(-1, -2))
        n += 1
        del sv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n == 0:
        raise RuntimeError("no tiles with a non-empty region were processed")
    imp_abs /= n
    imp_sum /= n

    g_abs = np.stack(
        [imp_abs[:, idx].sum(1) for idx in groups.values()], axis=1
    )  # (T,G)
    g_sum = np.stack([imp_sum[:, idx].sum(1) for idx in groups.values()], axis=1)

    np.savez(
        os.path.join(args.out, "shap_importance.npz"),
        imp_abs=imp_abs,
        imp_sum=imp_sum,
        grouped_abs=g_abs,
        grouped_sum=g_sum,
        group_names=np.array(names),
        band_names=np.array(ds.band_names),
        years=np.array(ds.years),
        n_tiles=n,
    )
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "n_tiles_used": n}, f, indent=2)

    order = np.argsort(-g_abs.sum(0))
    print("\nTop groups by total |attribution|:")
    for j in order[:15]:
        print(
            f"  {names[j]:<14} abs={g_abs[:, j].sum():.4e}  signed={g_sum[:, j].sum():+.4e}"
        )
    print(
        "\nPer-timestep total |attribution| (watch first/last: skip-branch shortcut):"
    )
    for t, y in enumerate(ds.years):
        print(f"  {y}: {g_abs[t].sum():.4e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sel = order  # most important groups first
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * len(sel))))
        for ax, M, title, cmap in (
            (axes[0], g_abs[:, sel].T, "mean |attribution|", "viridis"),
            (axes[1], g_sum[:, sel].T, "mean signed attribution", "RdBu_r"),
        ):
            kw = {}
            if cmap == "RdBu_r":
                v = np.abs(M).max() or 1.0
                kw = dict(vmin=-v, vmax=v)
            im = ax.imshow(M, aspect="auto", cmap=cmap, **kw)
            ax.set_xticks(range(T), [str(y) for y in ds.years])
            ax.set_yticks(range(len(sel)), [names[j] for j in sel], fontsize=7)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "shap_heatmap.png"), dpi=150)
        print(f"\nsaved heatmap to {args.out}/shap_heatmap.png")
    except ImportError:
        print("matplotlib not installed: skipped heatmap")

    print(f"saved arrays to {args.out}/shap_importance.npz  ({n} tiles)")


if __name__ == "__main__":
    main()
