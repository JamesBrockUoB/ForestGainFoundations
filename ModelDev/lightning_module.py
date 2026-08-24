import pytorch_lightning as pl
import torch

from .config import PERIOD_HAS_TYPOLOGY
from .losses import weighted_bce_loss, weighted_soft_ce_loss
from .models import MultiTaskGainModel


class GainMultiTaskTask(pl.LightningModule):
    def __init__(
        self,
        period: str,
        seg_loss_weight=1.0,
        cls_loss_weight=1.0,
        backbone_lr=1e-4,
        head_lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "period": period,
                "seg_loss_weight": seg_loss_weight,
                "cls_loss_weight": cls_loss_weight,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
            }
        )
        self.has_typology = PERIOD_HAS_TYPOLOGY[period]
        self.model = MultiTaskGainModel(include_classification_head=self.has_typology)

    def forward(self, pixels):
        return self.model(pixels)

    def _step(self, batch, stage):
        seg_logits, cls_logits = self(batch["pixels"])
        batch_size = batch["pixels"].shape[0]

        seg_loss = weighted_bce_loss(
            seg_logits, batch["gain_mask"], batch["gain_weight"], batch["gain_valid"]
        )
        self.log(f"{stage}_seg_loss", seg_loss, prog_bar=True, batch_size=batch_size)

        if self.has_typology:
            cls_loss = weighted_soft_ce_loss(
                cls_logits, batch["class_dist"], batch["cls_weight"]
            )
            self.log(
                f"{stage}_cls_loss", cls_loss, prog_bar=True, batch_size=batch_size
            )
            total = (
                self.hparams.seg_loss_weight * seg_loss
                + self.hparams.cls_loss_weight * cls_loss
            )
        else:
            total = seg_loss

        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=batch_size)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Differential LR: LoRA params get a lower LR than the randomly
        # initialized heads, which need to learn from scratch.
        lora_params = [
            p for _, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        head_params = list(self.model.seg_head.parameters())
        if self.model.cls_head is not None:
            head_params += list(self.model.cls_head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.hparams.backbone_lr},
                {"params": head_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=0.05,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
