import os

import pytorch_lightning as pl
import wandb
from config import DEFAULT_IMAGE_SIZE, NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset, split_tile_dirs
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader

WANDB_ENTITY = os.environ.get("WANDB_USERNAME")
DATA_DIR = "../DataCollection/data/test_tiles"

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

    # Test tiles are never used here, so sweeps can't leak into the held-out set
    train_dirs, val_dirs = split_tile_dirs(DATA_DIR, seed=0)

    train_ds = MultiTemporalGainDataset(train_dirs, augment=True)
    val_ds = MultiTemporalGainDataset(val_dirs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    task = GainDetectionTask(
        model_type=config.model_type,
        in_channels=NUM_INPUT_CHANNELS,
        img_size=DEFAULT_IMAGE_SIZE,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # No project/entity args: the run from wandb.init() is already active
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        precision="32-true",
        gradient_clip_val=1.0,
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
