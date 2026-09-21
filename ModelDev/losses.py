import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedDiceBCELoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE) and Soft Dice Loss for imbalanced segmentation.
    Supports both 'hard' (strict binary targets) and 'soft' (confidence-weighted targets) loss modes.
    Strictly ignores pixels where gain_valid == 0.
    """

    def __init__(self, loss_type: str = "hard", smooth: float = 1e-6):
        super().__init__()
        if loss_type not in ["hard", "soft"]:
            raise ValueError(
                f"Invalid loss_type '{loss_type}'. Must be 'hard' or 'soft'."
            )
        self.loss_type = loss_type
        self.smooth = smooth

    def forward(
        self,
        seg_logits: torch.Tensor,
        gain_mask: torch.Tensor,
        gain_weight: torch.Tensor,
        gain_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seg_logits: Model prediction logits (B, H, W) or (B, 1, H, W)
            gain_mask: Binary target ground truth (0 or 1) of shape matching seg_logits
            gain_weight: Pixel-level confidence map in range [0, 1]
            gain_valid: Binary mask (1 = valid pixel, 0 = invalid/cloud)
        """
        if seg_logits.dim() == 4 and seg_logits.size(1) == 1:
            seg_logits = seg_logits.squeeze(1)
        if gain_mask.dim() == 4 and gain_mask.size(1) == 1:
            gain_mask = gain_mask.squeeze(1)
        if gain_weight.dim() == 4 and gain_weight.size(1) == 1:
            gain_weight = gain_weight.squeeze(1)
        if gain_valid.dim() == 4 and gain_valid.size(1) == 1:
            gain_valid = gain_valid.squeeze(1)

        # Create strict valid boolean mask
        valid_mask = gain_valid.bool()

        # If no valid pixels exist in the entire batch, return zero loss with grad
        if not valid_mask.any():
            return (seg_logits * 0.0).sum()

        # Extract only valid pixels
        logits_v = seg_logits[valid_mask]
        targets_v = gain_mask[valid_mask].float()
        weights_v = gain_weight[valid_mask].float()

        # Determine effective pixel weight based on mode
        if self.loss_type == "soft":
            # Scale loss on positive pixels by pseudo-label confidence weight
            pixel_weights = torch.where(
                targets_v == 1.0, weights_v, torch.ones_like(weights_v)
            )
        else:  # 'hard' mode ignores continuous confidence weights
            pixel_weights = torch.ones_like(targets_v)

        # 1. Pixel-Weighted BCE Loss
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits_v, targets_v, reduction="none"
        )
        bce_loss = (bce_per_pixel * pixel_weights).sum() / pixel_weights.sum().clamp(
            min=1.0
        )

        # 2. Soft Dice Loss over valid pixels
        probs_v = torch.sigmoid(logits_v)

        # Apply confidence weights to Dice overlap if in soft mode
        probs_weighted = probs_v * pixel_weights
        targets_weighted = targets_v * pixel_weights

        intersection = (probs_weighted * targets_weighted).sum()
        cardinality = probs_weighted.sum() + targets_weighted.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        return bce_loss + dice_loss
