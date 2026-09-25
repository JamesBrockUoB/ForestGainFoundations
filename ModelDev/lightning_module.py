import pytorch_lightning as pl
import torch
import torchmetrics
from config import DEFAULT_IMAGE_SIZE, DEFAULT_LR, NUM_INPUT_CHANNELS
from losses import CombinedDiceBCELoss
from models import build_model


class GainDetectionTask(pl.LightningModule):
    """PyTorch Lightning Task for N-timestep satellite change/gain detection."""

    def __init__(
        self,
        model_type: str = "sits_scd",
        in_channels: int = NUM_INPUT_CHANNELS,
        img_size: int = DEFAULT_IMAGE_SIZE,
        lr: float = DEFAULT_LR,
        weight_decay: float = 1e-2,
        loss_type: str = "hard",
        eval_threshold: float = 0.50,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.eval_threshold = eval_threshold

        # Instantiate Network via models factory
        self.model = build_model(
            model_type=model_type,
            in_channels=in_channels,
            img_size=img_size,
            **model_kwargs,
        )

        # Dynamic Loss Strategy
        self.criterion = CombinedDiceBCELoss(loss_type=loss_type)

        metrics_kwargs = {"task": "binary", "threshold": self.eval_threshold}
        self.val_f1 = torchmetrics.F1Score(**metrics_kwargs)
        self.val_iou = torchmetrics.JaccardIndex(**metrics_kwargs)
        self.val_precision = torchmetrics.Precision(**metrics_kwargs)
        self.val_recall = torchmetrics.Recall(**metrics_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pixels"])
        loss = self.criterion(
            seg_logits=logits,
            gain_mask=batch["gain_mask"],
            gain_weight=batch["gain_weight"],
            gain_valid=batch["gain_valid"],
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["pixels"].shape[0],
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        logits = self(batch["pixels"])
        loss = self.criterion(
            seg_logits=logits,
            gain_mask=batch["gain_mask"],
            gain_weight=batch["gain_weight"],
            gain_valid=batch["gain_valid"],
        )

        probs = torch.sigmoid(logits)
        preds = (probs > self.eval_threshold).float()
        valid_indices = batch["gain_valid"] == 1.0

        if valid_indices.any():
            v_preds = preds[valid_indices]
            v_targets = batch["gain_mask"][valid_indices].int()

            self.val_f1.update(v_preds, v_targets)
            self.val_iou.update(v_preds, v_targets)
            self.val_precision.update(v_preds, v_targets)
            self.val_recall.update(v_preds, v_targets)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["pixels"].shape[0],
        )

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())

        self.val_f1.reset()
        self.val_iou.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
