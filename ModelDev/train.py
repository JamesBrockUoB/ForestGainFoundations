import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from config import NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


def run_training(
    tile_root: Path,
    period: str = "p1",
    model_type: str = "tsvit",
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 3e-4,
    project_name: str = "Forest-Gain-CD",
    entity: str | None = None,
):
    tile_dirs = sorted(
        [p for p in tile_root.iterdir() if p.is_dir() and p.name.endswith(f"_{period}")]
    )

    split = int(0.8 * len(tile_dirs))
    train_dirs, val_dirs = tile_dirs[:split], tile_dirs[split:]

    train_ds = MultiTemporalGainDataset(train_dirs, period=period)
    val_ds = MultiTemporalGainDataset(val_dirs, period=period)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    task = GainDetectionTask(
        model_type=model_type,
        in_channels=NUM_INPUT_CHANNELS,
        lr=lr,
    )

    # entity=None just falls back to your default W&B entity (your username),
    # same as before -- this only matters once you're logging to a team/org.
    # Note: WANDB_USERNAME only sets *entity* here -- it does not authenticate
    # you. Auth still comes from WANDB_API_KEY (or a cached `wandb login`).
    wandb_logger = WandbLogger(
        project=project_name,
        entity=entity or os.environ.get("WANDB_USERNAME"),
        name=f"{model_type}_{period}_run",
        config={
            "model_type": model_type,
            "period": period,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        },
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        filename=f"best-{model_type}-{period}-" + "{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=10,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
    )

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="../DataCollection/data/test_tiles"
    )
    parser.add_argument("--period", type=str, default="p1")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sits_scd", "unet_lstm", "tsvit"],
        default="tsvit",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/org). Defaults to $WANDB_USERNAME, then your personal account.",
    )
    args = parser.parse_args()

    run_training(
        tile_root=Path(args.data_dir),
        period=args.period,
        model_type=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        entity=args.wandb_entity,
    )
