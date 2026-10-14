import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio
from config import DEFAULT_IMAGE_SIZE, SOURCES
from datasets import MultiTemporalGainDataset, split_tile_dirs
from eval import evaluate_checkpoint
from lightning_module import GainDetectionTask
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader


def auto_pos_weight(tile_dirs) -> float:
    """sqrt(negatives / positives) over the TRAINING tiles' gain labels.

    Positives follow the dataset's definition (gain_confidence > 0). Pixels with
    gain_valid == 0 are not excluded here (the loss ignores them), which is a
    negligible difference when imagery retention is ~99%.
    """
    pos = neg = 0
    for d in tile_dirs:
        with rasterio.open(Path(d) / "labels" / "gain_confidence.tif") as src:
            c = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
        n_pos = int((c > 0).sum())
        pos += n_pos
        neg += c.size - n_pos
    if pos == 0:
        raise ValueError("No positive gain pixels in the training tiles")
    return float(np.sqrt(neg / pos))


def _pos_weight_arg(value: str):
    if value.lower() == "auto":
        return "auto"
    return float(value)


def parse_args():
    p = argparse.ArgumentParser(description="Train Forest Gain Change Detection Model")
    p.add_argument("--loss_type", type=str, default="hard", choices=["hard", "soft"])
    p.add_argument(
        "--model_type",
        type=str,
        default="tsvit",
        choices=["sits_scd", "unet_lstm", "tsvit"],
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["s1", "s2"],
        choices=list(SOURCES),
        help="Input products; channels are concatenated (e.g. alphaearth, "
        "tessera, s1 s2, s1 s2 alphaearth)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Stop if val_ap hasn't improved for this many epochs (0 = never stop early).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="../DataCollection/data/test_tiles")
    p.add_argument(
        "--holdout_dir", type=str, default="../DataCollection/data/holdout_tiles"
    )

    # Split
    p.add_argument("--val_frac", type=float, default=0.2)

    # Loss
    p.add_argument("--tversky_alpha", type=float, default=0.7)
    p.add_argument("--tversky_beta", type=float, default=0.3)
    p.add_argument("--tversky_gamma", type=float, default=0.75)
    p.add_argument(
        "--pos_weight",
        type=_pos_weight_arg,
        default=None,
        help="BCE positive weight: a number, 'auto' (sqrt(neg/pos) from the "
        "training labels) or omitted for no weighting.",
    )

    # Data
    p.add_argument(
        "--crop_size",
        type=int,
        default=0,
        help="Optional TRAINING-only random crop (0 = full tile, the default). "
        "Validation and test always use full tiles.",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Train samples per tile/epoch"
    )
    p.add_argument("--no_augment", action="store_true")
    return p.parse_args()


def train(args):
    pl.seed_everything(args.seed, workers=True)
    sources = tuple(args.sources)

    train_dirs, val_dirs = split_tile_dirs(
        args.data_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(f"train tiles: {len(train_dirs)} | val tiles: {len(val_dirs)} ")
    print(f"sources: {sources}")

    if args.pos_weight == "auto":
        args.pos_weight = auto_pos_weight(train_dirs)
        print(f"pos_weight (auto, sqrt(neg/pos)): {args.pos_weight:.2f}")

    run_name = "_".join(
        [
            args.model_type,
            "+".join(sources),
            args.loss_type,
            f"crop{args.crop_size or 'full'}",
            f"bs{args.batch_size}",
            f"lr{args.learning_rate:g}",
            f"wd{args.weight_decay:g}",
            f"rep{args.repeats}",
            *([f"pw{args.pos_weight:g}"] if args.pos_weight is not None else []),
            f"seed{args.seed}",
            datetime.now().strftime("%m%d-%H%M"),
        ]
    )
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    (ckpt_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    crop = args.crop_size or None
    img_size = crop or DEFAULT_IMAGE_SIZE

    train_ds = MultiTemporalGainDataset(
        train_dirs,
        sources=sources,
        augment=not args.no_augment,
        crop_size=crop,
        repeats=args.repeats,
        label_sigma=0.1 if args.loss_type == "soft" else 0.0,
    )
    # Validation always runs on full tiles (TSViT interpolates its position
    # embedding, so a model trained on crops still accepts them).
    val_ds = MultiTemporalGainDataset(val_dirs, sources=sources, crop_size=None)

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
        in_channels=train_ds.num_channels,
        img_size=img_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        loss_type=args.loss_type,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        pos_weight=args.pos_weight,
        sources=sources,
    )

    logger = pl.loggers.WandbLogger(project="Forest-Gain-CD")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="32-true",  # fp16-mixed is unreliable on MPS
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch}-{val_ap:.3f}",
                monitor="val_ap",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
            *(
                [EarlyStopping(monitor="val_ap", mode="max", patience=args.patience)]
                if args.patience > 0
                else []
            ),
        ],
    )

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)

    cb = trainer.checkpoint_callback
    best = cb.best_model_path
    print(f"best checkpoint: {best} | val_ap: {float(cb.best_model_score):.3f}")

    # Test set is touched once, with the checkpoint chosen on val
    if best:
        print("--- test set ---")
        results = evaluate_checkpoint(
            best,
            test_dir=Path(args.holdout_dir),
            crop_size=None,  # full tiles
        )
        logger.log_metrics({f"test_{k}": v for k, v in results.items()})


if __name__ == "__main__":
    train(parse_args())
