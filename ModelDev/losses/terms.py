"""Loss terms. Each term is an nn.Module taking a `Pixels` batch and returning a scalar; SegmentationLoss sums them."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FocalConfig, TverskyConfig


def _squeeze(t):
    return t.squeeze(1) if t.dim() == 4 and t.size(1) == 1 else t


def pixel_weights(gain_mask, gain_weight, loss_type):
    """Per-pixel weight shared by the soft loss and the soft evaluation.
    soft: gain pixels count with their confidence (gain_weight), background pixels count 1.  hard: 1 everywhere.
    """
    if loss_type == "soft":
        return torch.where(
            gain_mask > 0.5,
            gain_weight.float(),
            torch.ones_like(gain_weight, dtype=torch.float32),
        )
    return torch.ones_like(gain_weight, dtype=torch.float32)


class Pixels:
    """A batch seen two ways: tile-shaped (B, H, W) tensors, and flat vectors of the valid pixels (x, y, w)."""

    def __init__(self, seg_logits, gain_mask, gain_weight, gain_valid, confidence: str):
        self.logits2d, self.mask2d, gain_weight, valid = map(
            _squeeze, (seg_logits, gain_mask, gain_weight, gain_valid)
        )
        self.valid = valid.bool()
        self.weight2d = pixel_weights(self.mask2d, gain_weight, confidence)
        self.x = self.logits2d[self.valid]  # logits
        self.y = self.mask2d[self.valid].float()  # target (soft labels allowed)
        self.w = self.weight2d[self.valid]  # confidence weight
        self.n_tiles = self.logits2d.shape[0]

    def any(self) -> bool:
        return bool(self.valid.any())


# ---- Lovasz hinge (Berman et al. 2018; same as mmsegmentation) --------------------------------------
def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss w.r.t. sorted errors (Berman et al. 2018, Alg. 1)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss on flat valid pixels. logits [P] (raw), labels [P] in {0, 1}."""
    if len(labels) == 0:
        return logits.sum() * 0.0  # only void pixels: zero gradient
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.data]
    return torch.dot(F.relu(errors_sorted), lovasz_grad(gt_sorted))


# ---- terms ------------------------------------------------------------------------------------------
class Bce(nn.Module):
    """Weighted BCE. How pixels are re-weighted (none / OHEM / learned hardness) is the reweighter's job.
    forward returns (loss, aux_loss, stats); aux_loss is the reweighter's own training loss (hardness head) or None.
    """

    def __init__(self, reweighter, pos_weight=None):
        super().__init__()
        self.reweighter = reweighter
        # non-persistent: lives in the hyperparameters, keeps checkpoints free of loss state
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
            persistent=False,
        )

    def forward(self, px: Pixels, hardness_logits=None):
        pix = F.binary_cross_entropy_with_logits(
            px.x, px.y, pos_weight=self.pos_weight, reduction="none"
        )
        return self.reweighter.reduce(px, pix, hardness_logits)


class Focal(nn.Module):
    """Per-pixel binary focal loss, reduced by the configured reweighter.

    Returns the same (loss, auxiliary_loss, stats) tuple as Bce when a
    reweighter is supplied, so Focal can use OHEM consistently with BCE.
    """

    def __init__(self, cfg: FocalConfig, reweighter=None):
        super().__init__()
        self.cfg = cfg
        self.reweighter = reweighter

    def forward(self, px: Pixels, hardness_logits=None):
        ce = F.binary_cross_entropy_with_logits(px.x, px.y, reduction="none")
        pt = torch.exp(-ce)
        pix = (1.0 - pt) ** self.cfg.gamma * ce
        if self.cfg.alpha is not None:
            pix = pix * (self.cfg.alpha * px.y + (1.0 - self.cfg.alpha) * (1.0 - px.y))
        if self.reweighter is None:
            # Backward-compatible standalone use.
            return (pix * px.w).sum() / px.w.sum().clamp(min=1.0)
        return self.reweighter.reduce(px, pix, hardness_logits)


def _confusion(px: Pixels):
    p = torch.sigmoid(px.x)
    tp = (px.w * p * px.y).sum()
    fp = (px.w * p * (1 - px.y)).sum()
    fn = (px.w * (1 - p) * px.y).sum()
    return tp, fp, fn


class Tversky(nn.Module):
    """(1 - Tversky index) ** gamma; alpha weights false negatives, beta false positives."""

    def __init__(self, cfg: TverskyConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, px: Pixels):
        c = self.cfg
        tp, fp, fn = _confusion(px)
        index = (tp + c.smooth) / (tp + c.alpha * fn + c.beta * fp + c.smooth)
        return (1.0 - index).clamp(min=0.0) ** c.gamma


class Dice(nn.Module):
    """Soft Dice, 1 - 2TP / (2TP + FP + FN)."""

    def __init__(self, cfg: TverskyConfig):
        super().__init__()
        self.smooth = cfg.smooth

    def forward(self, px: Pixels):
        tp, fp, fn = _confusion(px)
        return 1.0 - (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)


class Lovasz(nn.Module):
    """Lovasz hinge on binary labels (ignores the soft confidence weights)."""

    def __init__(self, per_image: bool = False):
        super().__init__()
        self.per_image = per_image

    def forward(self, px: Pixels):
        logits, valid = px.logits2d, px.valid
        y = (px.mask2d > 0.5).float()
        if not self.per_image:
            return lovasz_hinge_flat(logits[valid], y[valid])
        per_tile = [
            lovasz_hinge_flat(logits[b][valid[b]], y[b][valid[b]])
            for b in range(logits.shape[0])
            if valid[b].any()
        ]
        return torch.stack(per_tile).mean()
