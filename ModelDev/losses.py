import torch
import torch.nn.functional as F


def weighted_bce_loss(
    seg_logits: torch.Tensor,
    gain_mask: torch.Tensor,
    gain_weight: torch.Tensor,
    gain_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel BCE weighted by pseudo-label confidence, restricted to
    valid (labelled-or-explicitly-negative) pixels. Assumes gain_valid
    covers both confidently-positive and confidently-negative pixels; if
    your masks are positive-only, you'll need a different negative-sampling
    strategy."""
    per_px = F.binary_cross_entropy_with_logits(seg_logits, gain_mask, reduction="none")
    weight = gain_weight * gain_valid
    denom = weight.sum().clamp(min=1.0)
    return (per_px * weight).sum() / denom


def weighted_soft_ce_loss(
    cls_logits: torch.Tensor, class_dist: torch.Tensor, cls_weight: torch.Tensor
) -> torch.Tensor:
    """Soft-label cross-entropy against a class distribution (not a hard
    one-hot), scaled per-tile by pseudo-label confidence. Degenerates to
    standard CE if class_dist happens to be one-hot. Only call this when
    cls_logits is not None -- i.e. never for a p2 (typology-free) model,
    see MultiTaskGainModel.include_classification_head."""
    log_probs = F.log_softmax(cls_logits, dim=-1)
    per_tile = -(class_dist * log_probs).sum(dim=-1)
    denom = cls_weight.sum().clamp(min=1e-6)
    return (per_tile * cls_weight).sum() / denom
