import argparse
import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from config import DEFAULT_IMAGE_SIZE, SOURCES
from datasets import MultiTemporalGainDataset
from eval import evaluate_checkpoint
from lightning_module import GainDetectionTask
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from utils import auto_pos_weight, split_tiles


def _pos_weight_arg(value: str):
    if value.lower() == "auto":
        return "auto"
    return float(value)


def parse_args():
    p = argparse.ArgumentParser(description="Train Forest Gain Change Detection Model")
    p.add_argument("--loss_type", type=str, default="hard", choices=["hard", "soft"])
    p.add_argument(
        "--loss", dest="loss_name", default="bce_focal_tversky",
        choices=["bce", "dice", "focal_tversky", "bce_focal_tversky"],
        help="Objective; loss_type only controls hard/soft confidence labels.",
    )
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
    p.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for the train/val split only; --seed (model init, data order) does not change it.",
    )

    # Loss
    # Defaults 0.5 / 0.5 / 1.0 make the Tversky term plain soft Dice (balanced
    # FP/FN, no focal exponent), i.e. nothing to tune. The precision/recall
    # trade-off is chosen afterwards via the decision threshold on validation.
    p.add_argument("--tversky_alpha", type=float, default=0.5)
    p.add_argument("--tversky_beta", type=float, default=0.5)
    p.add_argument("--tversky_gamma", type=float, default=1.0)
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
    p.add_argument(
        "--min_gain_px",
        type=int,
        default=0,
        help="Label cleaning: drop gain patches smaller than this many pixels "
        "(10 m pixels: 100 px = 1 ha). Applied to train AND val labels. The holdout "
        "test is reported on raw labels (test_*) and on cleaned labels (test_clean_*).",
    )
    p.add_argument(
        "--fill_holes_px",
        type=int,
        default=0,
        help="Label cleaning: fill holes smaller than this many pixels inside gain.",
    )
    p.add_argument("--no_augment", action="store_true")
    # Architecture knobs. Ignored by architectures that do not use them.
    p.add_argument("--base_features", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--depth_temporal", type=int, default=2)
    p.add_argument("--depth_spatial", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--skip_channels", type=int, default=32)
    return p.parse_args()


def train(args):
    pl.seed_everything(args.seed, workers=True)
    sources = tuple(args.sources)

    train_dirs, val_dirs = split_tiles(
        args.data_dir, val_frac=args.val_frac, seed=args.split_seed
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
            args.loss_name,
            f"crop{args.crop_size or 'full'}",
            f"bs{args.batch_size}",
            f"lr{args.learning_rate:g}",
            f"wd{args.weight_decay:g}",
            f"rep{args.repeats}",
            *([f"mmu{args.min_gain_px}"] if args.min_gain_px else []),
            *([f"fill{args.fill_holes_px}"] if args.fill_holes_px else []),
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
        min_gain_px=args.min_gain_px,
        fill_holes_px=args.fill_holes_px,
    )
    # Validation always runs on full tiles (TSViT interpolates its position
    # embedding, so a model trained on crops still accepts them).
    val_ds = MultiTemporalGainDataset(
        val_dirs,
        sources=sources,
        crop_size=None,
        min_gain_px=args.min_gain_px,
        fill_holes_px=args.fill_holes_px,
    )

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
        loss_name=args.loss_name,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        pos_weight=args.pos_weight,
        sources=sources,
        base_features=args.base_features,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth_temporal=args.depth_temporal,
        depth_spatial=args.depth_spatial,
        dropout=args.dropout,
        skip_channels=args.skip_channels,
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

        # Same checkpoint against labels cleaned like the training labels
        if args.min_gain_px or args.fill_holes_px:
            print("--- test set, cleaned labels ---")
            clean = evaluate_checkpoint(
                best,
                test_dir=Path(args.holdout_dir),
                crop_size=None,
                min_gain_px=args.min_gain_px,
                fill_holes_px=args.fill_holes_px,
            )
            logger.log_metrics({f"test_clean_{k}": v for k, v in clean.items()})


if __name__ == "__main__":
    train(parse_args())
