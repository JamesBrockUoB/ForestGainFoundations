import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyBCELoss(nn.Module):
    """
    BCE + Focal Tversky for rare positives. Ignores pixels where gain_valid == 0.

    alpha > beta penalises false negatives more than false positives.
    loss_type='soft' scales positive pixels by their confidence weight.
    pos_weight (optional float) up-weights positives in the BCE term.
    """

    def __init__(
        self,
        loss_type: str = "hard",
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        pos_weight: float | None = None,
        smooth: float = 1e-6,
    ):
        super().__init__()
        if loss_type not in ("hard", "soft"):
            raise ValueError(f"Invalid loss_type '{loss_type}'. Use 'hard' or 'soft'.")
        self.loss_type = loss_type
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

        return bce + focal_tversky
