"""Matched BCE-based controls for OHEM and learned pixel hardness.

This suite is standalone: it always compares BCE and BCE + Focal, and never
reads a prior loss-sweep result or carries settings from another experiment.
"""
from __future__ import annotations

from .common import build_manifest


HARDNESS_LAYERS = {
    "tsvit": ("decoder", "decoder.final_head"),
    "sits_scd": ("final_head",),
    "unet_lstm": ("out_head",),
}


def build_hardness(
    model="tsvit",
    group="hardness",
    seeds=(0, 1, 2),
    sources=("s1", "s2"),
    loss_names=("bce", "bce_focal"),
    layers=None,
    epochs=100,
    patience=20,
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=1e-2,
    data_dir="../DataCollection/data/forest_gain_tiles",
    holdout_dir="../DataCollection/data/holdout_tiles",
    split_seed=0,
):
    """Plan matched no-reweight, OHEM, and learned-hardness runs.

    Learned-hardness sweeps the offset ``c`` and auxiliary-loss weight ``alpha``
    for each applicable head. ``layers`` can override the model-specific defaults.
    No result/config from the losses suite is required.
    """
    if model not in HARDNESS_LAYERS:
        raise ValueError(f"Unsupported model {model!r}; expected one of {tuple(HARDNESS_LAYERS)}")
    layers = tuple(layers or HARDNESS_LAYERS[model])
    rows = []

    for loss_name in loss_names:
        # Matched controls: same recipe, split, seed, model, and optimizer.
        conditions = [("none", None, None, None)]
        conditions.extend(("ohem", None, threshold, None) for threshold in (0.7, 0.9))
        conditions.extend(
            ("hardness", layer, c, alpha)
            for layer in layers
            for c in (0.01, 0.1)
            for alpha in (0.001, 0.01, 0.05)
        )

        for mode, layer, value, alpha in conditions:
            if mode == "ohem":
                loss_opt = [f"ohem.thresh={value}"]
                variant = f"{loss_name}-ohem-t{value}"
            elif mode == "hardness":
                loss_opt = [f"hardness.c={value}", f"hardness.alpha={alpha}"]
                variant = f"{loss_name}-hardness-{layer}-c{value}-a{alpha}"
            else:
                loss_opt = None
                variant = f"{loss_name}-none"

            for seed in seeds:
                rows.append({
                    "suite": "hardness",
                    "model_type": model,
                    "sources": list(sources),
                    "seed": seed,
                    "loss_name": loss_name,
                    "reweight": mode,
                    "hl_layer": layer,
                    "loss_opt": loss_opt,
                    "epochs": epochs,
                    "patience": patience,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "data_dir": data_dir,
                    "holdout_dir": holdout_dir,
                    "split_seed": split_seed,
                    "skip_test_eval": True,
                    "val_metrics": True,
                    "wandb_group": group,
                    "wandb_tags": ["hardness", model],
                    "variant": variant,
                })

    return build_manifest(
        rows,
        "hardness",
        group,
        {
            "matched_controls": True,
            "losses": list(loss_names),
            "reweight_modes": ["none", "ohem", "hardness"],
            "hardness_c": [0.01, 0.1],
            "hardness_alpha": [0.001, 0.01, 0.05],
            "layers": list(layers),
            "seeds": list(seeds),
        },
    )
