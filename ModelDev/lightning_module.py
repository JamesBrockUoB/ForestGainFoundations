import pytorch_lightning as pl
import torch
import torchmetrics
from config import DEFAULT_IMAGE_SIZE, DEFAULT_LR, NUM_INPUT_CHANNELS
from losses import FocalTverskyBCELoss
from models import build_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class GainDetectionTask(pl.LightningModule):
    """PyTorch Lightning Task for N-timestep satellite change/gain detection."""

    def __init__(
        self,
        model_type: str = "sits_scd",
        in_channels: int = NUM_INPUT_CHANNELS,
        img_size: int = DEFAULT_IMAGE_SIZE,
        lr: float = DEFAULT_LR,
        weight_decay: float = 1e-2,
        warmup_frac: float = 0.05,
        loss_type: str = "hard",
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 0.75,
        pos_weight: float | None = None,
        eval_threshold: float = 0.50,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.eval_threshold = eval_threshold

        self.model = build_model(
            model_type=model_type,
            in_channels=in_channels,
            img_size=img_size,
            **model_kwargs,
        )

        self.criterion = FocalTverskyBCELoss(
            loss_type=loss_type,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            pos_weight=pos_weight,
        )

        metrics_kwargs = {
            "task": "binary",
            "threshold": self.eval_threshold,
        }
        self.val_f1 = torchmetrics.F1Score(**metrics_kwargs)
        self.val_iou = torchmetrics.JaccardIndex(**metrics_kwargs)
        self.val_precision = torchmetrics.Precision(**metrics_kwargs)
        self.val_recall = torchmetrics.Recall(**metrics_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, batch, logits):
        return self.criterion(
            seg_logits=logits,
            gain_mask=batch["gain_mask"],
            gain_weight=batch["gain_weight"],
            gain_valid=batch["gain_valid"],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pixels"])
        loss = self._loss(batch, logits)

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
        loss = self._loss(batch, logits)

        probs = torch.sigmoid(logits)

        if probs.ndim == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)

        valid = batch["gain_valid"] > 0.5

        if valid.any():
            v_probs = probs[valid]
            v_targets = (batch["gain_mask"][valid] > 0.5).int()

            self.val_f1.update(v_probs, v_targets)
            self.val_iou.update(v_probs, v_targets)
            self.val_precision.update(v_probs, v_targets)
            self.val_recall.update(v_probs, v_targets)

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

        for metric in (
            self.val_f1,
            self.val_iou,
            self.val_precision,
            self.val_recall,
        ):
            metric.reset()

    def configure_optimizers(self):
        # No weight decay on biases / norm parameters
        decay, no_decay = [], []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
        )

        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, int(self.warmup_frac * total))

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total - warmup),
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
