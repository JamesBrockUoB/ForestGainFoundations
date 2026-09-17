import os
from pathlib import Path

import pytorch_lightning as pl
import wandb
from config import NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader

WANDB_ENTITY = os.environ.get("WANDB_USERNAME")

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_iou"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "model_type": {"values": ["sits_scd", "unet_lstm", "tsvit"]},
        "batch_size": {"values": [4, 8, 16]},
    },
}


def sweep_train():
    wandb.init()
    config = wandb.config

    tile_root = Path("../DataCollection/data/test_tiles")
    tile_dirs = sorted(
        [p for p in tile_root.iterdir() if p.is_dir() and p.name.endswith("_p1")]
    )
    split = int(0.8 * len(tile_dirs))

    train_ds = MultiTemporalGainDataset(tile_dirs[:split], period="p1")
    val_ds = MultiTemporalGainDataset(tile_dirs[split:], period="p1")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=False,
        persistent_workers=True,
    )

    task = GainDetectionTask(
        model_type=config.model_type,
        in_channels=NUM_INPUT_CHANNELS,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # No project/entity args here on purpose: an agent-launched run is already
    # active from wandb.init() above, so this just attaches the PL logger to it
    # rather than starting a second, competing run.
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        precision="16-mixed",
        logger=pl.loggers.WandbLogger(),
    )

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_configuration,
        project="Forest-Gain-Hyperparameter-Sweep",
        entity=WANDB_ENTITY,
    )
    wandb.agent(sweep_id, function=sweep_train, count=10)
