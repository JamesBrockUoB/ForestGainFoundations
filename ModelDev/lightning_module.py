import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from config import DEFAULT_IMAGE_SIZE, DEFAULT_LR, NUM_INPUT_CHANNELS, YEARS
from hardness import HardnessBranch
from losses import LossConfig, SegmentationLoss, pixel_weights
from metrics import summarise
from models import build_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class GainDetectionTask(pl.LightningModule):
    """PyTorch Lightning Task for N-timestep satellite change/gain detection.

    loss: a LossConfig as a dict (LossConfig.to_dict()). Checkpoints saved before the loss refactor stored the loss
    options as flat hyperparameters (loss_name, tversky_alpha, ohem, hl_c, ...); those are converted automatically.
    """

    def __init__(
        self,
        model_type: str = "sits_scd",
        in_channels: int = NUM_INPUT_CHANNELS,
        img_size: int = DEFAULT_IMAGE_SIZE,
        lr: float = DEFAULT_LR,
        weight_decay: float = 1e-2,
        warmup_frac: float = 0.05,
        loss: dict | None = None,
        eval_threshold: float = 0.50,
        sources: tuple[str, ...] = ("s1", "s2"),
        years: (
            tuple[int, ...] | None
        ) = None,  # frames the model sees (None = all of config.YEARS)
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if loss is None:  # checkpoint from before the loss refactor
            loss_cfg, model_kwargs = LossConfig.from_legacy(model_kwargs)
        else:
            loss_cfg = LossConfig.from_dict(loss)
        self.loss_cfg = loss_cfg

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.eval_threshold = eval_threshold
        self.loss_type = (
            loss_cfg.confidence
        )  # 'soft' -> validation metrics use the confidence weights (see metrics.py)
        self.sources = tuple(sources)  # stored so eval can rebuild the right dataset
        self.years = tuple(years) if years else tuple(YEARS)  # ditto (hparam `years`)

        self.model = build_model(
            model_type=model_type,
            in_channels=in_channels,
            img_size=img_size,
            **model_kwargs,
        )
        self.criterion = SegmentationLoss(loss_cfg)

        # Pixel hardness (arXiv 2305.08462): a branch that mirrors the segmentation head `hardness.layer` on its detached
        # inputs (see hardness.py). Training only; inference ignores it.
        self.hardness = loss_cfg.reweight == "hardness"
        self.hl_head = None
        if self.hardness:
            self.hl_head = HardnessBranch(
                self.model,
                loss_cfg.hardness.layer,
                torch.zeros(1, len(self.years), in_channels, img_size, img_size),
                head=loss_cfg.hardness.head,
            )

        # Per-tile (probs, targets, weights) of the valid val pixels, collected on the CPU (exact AP needs every prediction,
        # and this keeps it independent of MPS op coverage). All metrics are computed from these at the end of the epoch.
        self._val_samples: list[tuple[torch.Tensor, ...]] = []

    def on_load_checkpoint(self, checkpoint):
        # the loss holds no learned state; drop keys that older checkpoints stored (e.g. criterion.pos_weight)
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("criterion.")
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, batch, logits, hardness_logits=None):
        return self.criterion(
            seg_logits=logits,
            gain_mask=batch["gain_mask"],
            gain_weight=batch["gain_weight"],
            gain_valid=batch["gain_valid"],
            hardness_logits=hardness_logits,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        d = None
        if self.hardness:
            with self.hl_head.capture():
                logits = self(batch["pixels"])
            d = self.hl_head()
            if d.shape[-2:] != logits.shape[-2:]:
                d = F.interpolate(
                    d, size=logits.shape[-2:], mode="bilinear", align_corners=False
                )
        else:
            logits = self(batch["pixels"])
        loss = self._loss(batch, logits, d)
        if self.criterion.stats:  # hardness stats or the OHEM kept fraction
            for k, v in self.criterion.stats.items():
                self.log(
                    f"train_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch["pixels"].shape[0],
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
        loss = self._loss(batch, logits)

        probs = torch.sigmoid(logits)

        if probs.ndim == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)

        valid = batch["gain_valid"] > 0.5
        targets = (batch["gain_mask"] > 0.5).int()
        weights = pixel_weights(
            batch["gain_mask"], batch["gain_weight"], self.loss_type
        )

        for b in range(probs.shape[0]):
            vb = valid[b]
            if vb.any():
                self._val_samples.append(
                    (
                        probs[b][vb].detach().float().cpu(),
                        targets[b][vb].cpu(),
                        weights[b][vb].float().cpu(),
                    )
                )

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["pixels"].shape[0],
        )

    def on_validation_epoch_end(self):
        soft = self.loss_type == "soft"
        # hard: plain binary metrics. soft: every gain pixel counts with its confidence (same weights as the loss).
        res = summarise(self._val_samples, self.eval_threshold, weighted=soft)

        # val_ap: AP over all valid val pixels (compare with baseline.py's AP).
        # val_map: mean of per-tile AP over tiles that contain gain.
        # A random scorer has val_ap == val_prevalence.
        self.log("val_f1", res["f1"], prog_bar=True)
        self.log("val_iou", res["iou"], prog_bar=True)
        self.log("val_precision", res["precision"])
        self.log("val_recall", res["recall"])
        self.log("val_ap", res["ap"], prog_bar=True)
        self.log("val_map", res["map"], prog_bar=True)
        self.log("val_prevalence", res["prevalence"])

        if soft:  # unweighted versions, comparable with hard-loss runs
            for k, v in summarise(
                self._val_samples, self.eval_threshold, weighted=False
            ).items():
                self.log(f"val_bin_{k}", v)

        self._val_samples.clear()

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
