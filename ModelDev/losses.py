import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyBCELoss(nn.Module):
    """
    BCE + Focal Tversky for rare positives. Ignores pixels where gain_valid == 0.

    Defaults alpha=beta=0.5, gamma=1.0 make the Tversky term plain soft Dice:
    balanced false positives / false negatives and no focal exponent, so there is
    nothing to tune. The precision/recall trade-off is then chosen with the decision
    threshold on validation. alpha > beta penalises false negatives more than false
    positives; gamma < 1 focuses on hard examples.
    loss_type='soft' scales positive pixels by their confidence weight.
    pos_weight (optional float) up-weights positives in the BCE term.
    """

    def __init__(
        self,
        name: str = "bce_focal_tversky",
        loss_type: str = "hard",
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        pos_weight: float | None = None,
        smooth: float = 1e-6,
    ):
        super().__init__()
        if name not in ("bce", "dice", "focal_tversky", "bce_focal_tversky"):
            raise ValueError(f"Unsupported loss '{name}'.")
        if loss_type not in ("hard", "soft"):
            raise ValueError(f"Invalid loss_type '{loss_type}'. Use 'hard' or 'soft'.")
        self.name, self.loss_type = name, loss_type
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
        )

    @staticmethod
    def _squeeze(t):
        return t.squeeze(1) if t.dim() == 4 and t.size(1) == 1 else t

    def forward(self, seg_logits, gain_mask, gain_weight, gain_valid):
        seg_logits, gain_mask, gain_weight, gain_valid = map(
            self._squeeze, (seg_logits, gain_mask, gain_weight, gain_valid)
        )
        valid = gain_valid.bool()
        if not valid.any():
            return (seg_logits * 0.0).sum()

        x = seg_logits[valid]
        y = gain_mask[valid].float()
        w = gain_weight[valid].float()

        if self.loss_type == "soft":
            pw = torch.where(y == 1.0, w, torch.ones_like(w))
        else:
            pw = torch.ones_like(y)

        # BCE
        bce = F.binary_cross_entropy_with_logits(
            x, y, pos_weight=self.pos_weight, reduction="none"
        )
        bce = (bce * pw).sum() / pw.sum().clamp(min=1.0)

        # Focal Tversky
        p = torch.sigmoid(x)
        tp = (pw * p * y).sum()
        fp = (pw * p * (1 - y)).sum()
        fn = (pw * (1 - p) * y).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        focal_tversky = (1.0 - tversky).clamp(min=0.0) ** self.gamma

        if self.name == "bce":
            return bce
        if self.name == "dice":
            # Dice is the symmetric, non-focal special case of Tversky.
            return 1.0 - (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        if self.name == "focal_tversky":
            return focal_tversky
        return bce + focal_tversky
