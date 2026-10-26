"""Segmentation losses for rare positives.

    cfg = LossConfig(name="bce_focal_tversky", reweight="ohem").with_overrides(["ohem.thresh=0.9"])
    criterion = SegmentationLoss(cfg)
    loss = criterion(seg_logits, gain_mask, gain_weight, gain_valid)

Layout: config.py (all options and defaults), terms.py (BCE / focal / Tversky / Dice / Lovasz),
reweight.py (none / OHEM / learned hardness for the BCE term), this file (summing the terms).
"""

import torch.nn as nn

from .config import LOSS_NAMES, RECIPES, REWEIGHTS, LossConfig, parse_value
from .reweight import LearnedHardness, NoReweight, Ohem
from .terms import Bce, Dice, Focal, Lovasz, Pixels, Tversky, pixel_weights

__all__ = [
    "LOSS_NAMES",
    "RECIPES",
    "REWEIGHTS",
    "LossConfig",
    "SegmentationLoss",
    "pixel_weights",
    "parse_value",
]


def _reweighter(cfg: LossConfig):
    return {
        "none": lambda: NoReweight(),
        "ohem": lambda: Ohem(cfg.ohem),
        "hardness": lambda: LearnedHardness(cfg.hardness),
    }[cfg.reweight]()


def _term(name: str, cfg: LossConfig) -> nn.Module:
    if name == "bce":
        return Bce(_reweighter(cfg), cfg.pos_weight)
    if name == "focal":
        return Focal(cfg.focal, _reweighter(cfg))
    if name == "tversky":
        return Tversky(cfg.tversky)
    if name == "dice":
        return Dice(cfg.tversky)
    if name == "lovasz":
        return Lovasz(cfg.lovasz_per_image)
    raise ValueError(name)


class SegmentationLoss(nn.Module):
    """Sum of the terms of cfg.name; ignores pixels where gain_valid == 0. self.stats holds reweighter diagnostics
    (OHEM kept fraction, hardness of gain / background pixels) from the last forward."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.terms = nn.ModuleDict({name: _term(name, cfg) for name in cfg.terms})
        self.stats = {}

    def forward(
        self, seg_logits, gain_mask, gain_weight, gain_valid, hardness_logits=None
    ):
        self.stats = {}
        px = Pixels(seg_logits, gain_mask, gain_weight, gain_valid, self.cfg.confidence)
        if not px.any():
            return (px.logits2d * 0.0).sum()
        if hardness_logits is not None and self.cfg.reweight != "hardness":
            raise ValueError(
                "hardness_logits were given but cfg.reweight != 'hardness'."
            )

        total = 0.0
        for name, term in self.terms.items():
            if name in ("bce", "focal"):
                # BCE and focal both reduce per-pixel losses through the
                # configured reducer (plain mean, OHEM, or learned hardness).
                value, aux, stats = term(px, hardness_logits)
                if stats:
                    self.stats.update(stats)
                total = total + value + (aux if aux is not None else 0.0)
            else:
                total = total + term(px)
        return total
