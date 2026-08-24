from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import GainTileDataset
from .lightning_module import GainMultiTaskTask


def train_period(tile_dirs: list[Path], period: str, val_frac: float = 0.2):
    """Trains one standalone model for one period: p1 gets seg+cls, p2
    gets seg only. Call this once per period -- each call builds its own
    dataset, model, and trainer, with no weights shared between the two
    resulting models."""
    split = int((1 - val_frac) * len(tile_dirs))
    train_dirs, val_dirs = tile_dirs[:split], tile_dirs[split:]

    train_ds = GainTileDataset(train_dirs, period=period)
    val_ds = GainTileDataset(val_dirs, period=period)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    task = GainMultiTaskTask(period=period)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return task


if __name__ == "__main__":
    tile_root = Path("..DataCollection/data/test_tiles")
    p1_tile_dirs = sorted(p for p in tile_root.iterdir() if p.name.endswith("_p1"))
    p2_tile_dirs = sorted(p for p in tile_root.iterdir() if p.name.endswith("_p2"))

    p1_task = train_period(p1_tile_dirs, period="p1")
    p2_task = train_period(p2_tile_dirs, period="p2")
