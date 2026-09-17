import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(
    seg_logits: torch.Tensor,
    gain_mask: torch.Tensor,
    gain_weight: torch.Tensor,
    gain_valid: torch.Tensor,
) -> torch.Tensor:
    """
    Per-pixel BCE weighted by pseudo-label confidence, restricted to valid pixels.

    Args:
        seg_logits: Model output logits of shape (B, H, W)
        gain_mask: Binary target mask of shape (B, H, W) (0 or 1)
        gain_weight: Soft confidence map of shape (B, H, W) in range [0, 1]
        gain_valid: Valid pixel indicator mask of shape (B, H, W) (0 or 1)
    """
    # Ensure shapes match
    if seg_logits.dim() == 4 and seg_logits.size(1) == 1:
        seg_logits = seg_logits.squeeze(1)

    per_px = F.binary_cross_entropy_with_logits(
        seg_logits, gain_mask.float(), reduction="none"
    )
    pixel_weight = torch.where(
        gain_mask.bool(), gain_weight, torch.ones_like(gain_weight)
    )
    weight = pixel_weight * gain_valid
    denom = weight.sum().clamp(min=1.0)
    return (per_px * weight).sum() / denom


class FocalWeightedBCELoss(nn.Module):
    """
    Optional Focal-weighted BCE for handling severe class imbalance in forest gain detection.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        seg_logits: torch.Tensor,
        gain_mask: torch.Tensor,
        gain_weight: torch.Tensor,
        gain_valid: torch.Tensor,
    ) -> torch.Tensor:
        if seg_logits.dim() == 4 and seg_logits.size(1) == 1:
            seg_logits = seg_logits.squeeze(1)

        bce = F.binary_cross_entropy_with_logits(
            seg_logits, gain_mask.float(), reduction="none"
        )
        p = torch.sigmoid(seg_logits)
        p_t = p * gain_mask + (1 - p) * (1 - gain_mask)
        focal_factor = (1 - p_t) ** self.gamma

        alpha_factor = self.alpha * gain_mask + (1 - self.alpha) * (1 - gain_mask)
        loss = alpha_factor * focal_factor * bce

        pixel_weight = torch.where(
            gain_mask.bool(), gain_weight, torch.ones_like(gain_weight)
        )
        weight = pixel_weight * gain_valid
        denom = weight.sum().clamp(min=1.0)
        return (loss * weight).sum() / denom
