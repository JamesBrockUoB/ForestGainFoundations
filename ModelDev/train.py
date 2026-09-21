import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import wandb
from config import NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader

WANDB_ENTITY = os.environ.get("WANDB_USERNAME")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Forest Gain Change Detection Model"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Loss criteria selection: 'hard' (binary targets) or 'soft' (confidence weighted).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="tsvit",
        choices=["sits_scd", "unet_lstm", "tsvit"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--data_dir", type=str, default="../DataCollection/data/test_tiles"
    )
    return parser.parse_args()


def train(args):
    tile_root = Path(args.data_dir)
    tile_dirs = sorted(
        [p for p in tile_root.iterdir() if p.is_dir() and p.name.endswith("_p1")]
    )
    split = int(0.8 * len(tile_dirs))

    train_ds = MultiTemporalGainDataset(tile_dirs[:split], period="p1")
    val_ds = MultiTemporalGainDataset(tile_dirs[split:], period="p1")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    task = GainDetectionTask(
        model_type=args.model_type,
        in_channels=NUM_INPUT_CHANNELS,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger=pl.loggers.WandbLogger(project="Forest-Gain-CD"),
    )

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    args = parse_args()
    train(args)
