"""How the BCE term's per-pixel losses are reduced: plain, OHEM, or learned pixel hardness.

Each reweighter has reduce(px, pix, hardness_logits) -> (loss, aux_loss | None, stats dict).
"""

import torch
import torch.nn as nn
from hardness import hardness_map

from .config import HardnessConfig, OhemConfig


def _weighted_mean(pix, w):
    return (pix * w).sum() / w.sum().clamp(min=1.0)


class NoReweight(nn.Module):
    def reduce(self, px, pix, hardness_logits=None):
        return _weighted_mean(pix, px.w), None, {}


class Ohem(nn.Module):
    """Online hard example mining as in mmsegmentation's OHEMPixelSampler, binary version. Training only."""

    def __init__(self, cfg: OhemConfig):
        super().__init__()
        self.cfg = cfg

    def reduce(self, px, pix, hardness_logits=None):
        if not self.training:
            return _weighted_mean(pix, px.w), None, {}
        keep = self._keep(px.x, px.y, pix, px.n_tiles)
        return (
            _weighted_mean(pix, px.w * keep.float()),
            None,
            {"ohem_kept": keep.float().mean()},
        )

    def _keep(self, x, y, pix, n_tiles):
        """bool keep-mask over the valid pixels."""
        with torch.no_grad():
            batch_kept = self.cfg.min_kept * n_tiles
            if self.cfg.thresh > 0:
                p = torch.sigmoid(x)
                p_true = torch.where(y > 0.5, p, 1.0 - p)
                sort_prob, _ = p_true.sort()
                min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)]
                return p_true < torch.clamp(min_threshold, min=self.cfg.thresh)
            keep = torch.zeros_like(pix, dtype=torch.bool)
            _, order = pix.detach().sort(descending=True)
            keep[order[:batch_kept]] = True
            return keep


class LearnedHardness(nn.Module):
    """Every pixel's BCE is weighted by its learned hardness (times the confidence weight) and the hardness head's own
    loss, alpha * (-sum H * BCE), is returned as aux. Gradients stay separate: H and BCE are detached in the other's loss.
    Without hardness_logits (validation) this is plain BCE. stats: hl_loss and the mean relative hardness of gain /
    background pixels (1.0 = the tile's average pixel)."""

    def __init__(self, cfg: HardnessConfig):
        super().__init__()
        self.cfg = cfg

    def reduce(self, px, pix, hardness_logits=None):
        if hardness_logits is None:
            return _weighted_mean(pix, px.w), None, {}
        v = px.valid.float()
        L = torch.zeros_like(px.logits2d).masked_scatter(
            px.valid, pix
        )  # back to (B, H, W); 0 on void pixels
        h = hardness_map(
            (
                hardness_logits.squeeze(1)
                if hardness_logits.dim() == 4
                else hardness_logits
            ),
            v,
            self.cfg.c,
        )
        ws = h.detach() * px.weight2d * v
        bce = (ws * L).sum() / ws.sum().clamp(min=1e-12)
        hl = -(h * L.detach()).sum(dim=(1, 2)).mean()
        with torch.no_grad():
            rel = h * v.sum(dim=(1, 2), keepdim=True)
            pos, neg = (px.mask2d > 0.5) & px.valid, (px.mask2d <= 0.5) & px.valid
            stats = {
                "hl_loss": hl.detach(),
                "hardness_gain": rel[pos].mean() if pos.any() else rel.new_zeros(()),
                "hardness_background": (
                    rel[neg].mean() if neg.any() else rel.new_zeros(())
                ),
            }
        return bce, self.cfg.alpha * hl, stats
