import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from config import DEFAULT_IMAGE_SIZE, NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset, split_tile_dirs
from eval import evaluate_checkpoint
from lightning_module import GainDetectionTask
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train Forest Gain Change Detection Model")
    p.add_argument("--loss_type", type=str, default="hard", choices=["hard", "soft"])
    p.add_argument(
        "--model_type",
        type=str,
        default="tsvit",
        choices=["sits_scd", "unet_lstm", "tsvit"],
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="../DataCollection/data/test_tiles")

    # Split
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)

    # Loss
    p.add_argument("--tversky_alpha", type=float, default=0.7)
    p.add_argument("--tversky_beta", type=float, default=0.3)
    p.add_argument("--tversky_gamma", type=float, default=0.75)
    p.add_argument("--pos_weight", type=float, default=None)

    # Data
    p.add_argument(
        "--crop_size",
        type=int,
        default=0,
        help="Random crop size for training (0 = full tile). Model img_size follows.",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Train samples per tile/epoch"
    )
    p.add_argument("--no_augment", action="store_true")
    return p.parse_args()


def train(args):
    pl.seed_everything(args.seed, workers=True)

    train_dirs, val_dirs, test_dirs = split_tile_dirs(
        args.data_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(
        f"train tiles: {len(train_dirs)} | val tiles: {len(val_dirs)} "
        f"| test tiles: {len(test_dirs)}"
    )

    run_name = f"{args.model_type}_{args.loss_type}_seed{args.seed}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Record the exact split next to the checkpoints (absolute paths)
    (ckpt_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": [str(p.resolve()) for p in train_dirs],
                "val": [str(p.resolve()) for p in val_dirs],
                "test": [str(p.resolve()) for p in test_dirs],
            },
            indent=2,
        )
    )

    crop = args.crop_size or None
    img_size = crop or DEFAULT_IMAGE_SIZE

    train_ds = MultiTemporalGainDataset(
        train_dirs,
        augment=not args.no_augment,
        crop_size=crop,
        repeats=args.repeats,
        label_sigma=0.1 if args.loss_type == "soft" else 0.0,
    )
    val_ds = MultiTemporalGainDataset(val_dirs, crop_size=crop)

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
        img_size=img_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        loss_type=args.loss_type,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        pos_weight=args.pos_weight,
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
                filename="{epoch}-{val_iou:.3f}",
                monitor="val_iou",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
        ],
    )

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)

    cb = trainer.checkpoint_callback
    best = cb.best_model_path
    print(f"best checkpoint: {best} | val_iou: {float(cb.best_model_score):.3f}")

    # Test set is touched once, with the checkpoint chosen on val
    if test_dirs and best:
        print("--- test set ---")
        results = evaluate_checkpoint(best, test_dirs, crop_size=crop)
        logger.log_metrics({f"test_{k}": v for k, v in results.items()})


if __name__ == "__main__":
    train(parse_args())
