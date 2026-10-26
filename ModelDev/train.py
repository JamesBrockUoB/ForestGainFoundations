import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from config import DEFAULT_IMAGE_SIZE, SOURCES, YEARS
from datasets import MultiTemporalGainDataset
from eval import evaluate_checkpoint
from lightning_module import GainDetectionTask
from losses import LOSS_NAMES, REWEIGHTS, LossConfig, parse_value
from losses.config import HardnessConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from utils import auto_pos_weight, split_tiles

# Architecture settings passed to every model (models ignore the ones they do not use). Override with --model_opt.
DEFAULT_MODEL_OPTS = {
    "base_features": 64,
    "num_heads": 4,
    "patch_size": 16,
    "embed_dim": 128,
    "depth_temporal": 2,
    "depth_spatial": 4,
    "dropout": 0.1,
    "skip_channels": 32,
}


def _pos_weight_arg(value: str):
    if value.lower() == "auto":
        return "auto"
    return float(value)


def parse_model_opts(opts) -> dict:
    """['embed_dim=96', 'dropout=0.2'] -> merged over DEFAULT_MODEL_OPTS."""
    merged = dict(DEFAULT_MODEL_OPTS)
    for opt in opts or ():
        key, eq, raw = opt.partition("=")
        if not eq:
            raise SystemExit(f"--model_opt '{opt}' must look like key=value")
        merged[key] = parse_value(raw)
    return merged


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Forest Gain Change Detection Model")

    g = p.add_argument_group("model")
    g.add_argument(
        "--model_type", default="tsvit", choices=["sits_scd", "unet_lstm", "tsvit"]
    )
    g.add_argument(
        "--model_opt",
        nargs="+",
        metavar="KEY=VAL",
        help=f"Architecture settings, e.g. embed_dim=96 depth_spatial=3. Defaults: {DEFAULT_MODEL_OPTS}",
    )

    g = p.add_argument_group("loss")
    g.add_argument(
        "--loss",
        dest="loss_name",
        default="bce_focal_tversky",
        choices=list(LOSS_NAMES),
    )
    g.add_argument(
        "--loss_type",
        default="hard",
        choices=["hard", "soft"],
        help="soft: gain pixels count with their confidence weight (loss and validation metrics).",
    )
    g.add_argument(
        "--pos_weight",
        type=_pos_weight_arg,
        default=None,
        help="BCE positive weight: a number, 'auto' (sqrt(neg/pos) from the training labels) or omitted.",
    )
    g.add_argument(
        "--reweight",
        default="none",
        choices=list(REWEIGHTS),
        help="Hard-pixel emphasis on the BCE/focal term: none, ohem (mmseg OHEM) or hardness (learned, arXiv 2305.08462).",
    )
    g.add_argument(
        "--hl_layer",
        help="--reweight hardness: the model's segmentation head (final decoder/classifier module) that the hardness branch mirrors.",
    )
    g.add_argument(
        "--loss_opt",
        nargs="+",
        metavar="KEY=VAL",
        help="Fine-tune loss settings with dotted keys, e.g. tversky.alpha=0.7 focal.gamma=3 ohem.thresh=0.9 hardness.c=0.01 lovasz_per_image=True.",
    )

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--learning_rate", type=float, default=3e-4)
    g.add_argument("--weight_decay", type=float, default=1e-2)
    g.add_argument("--warmup_frac", type=float, default=0.05)
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument(
        "--patience", type=int, default=20, help="Early stop on val_ap (0 = never)."
    )
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", default="../DataCollection/data/test_tiles")
    g.add_argument("--holdout_dir", default="../DataCollection/data/holdout_tiles")
    g.add_argument("--val_frac", type=float, default=0.2)
    g.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Train/val split only; --seed does not change it.",
    )
    g.add_argument(
        "--sources",
        nargs="+",
        default=["s1", "s2"],
        choices=list(SOURCES),
        help="Input products; channels are concatenated (e.g. alphaearth, tessera, s1 s2, s1 s2 alphaearth)",
    )
    g.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        choices=list(YEARS),
        help="Frames the model sees (default: all). --years 2017 2024 = bi-temporal. Scored pixels do not change.",
    )
    g.add_argument(
        "--crop_size",
        type=int,
        default=0,
        help="TRAINING-only random crop (0 = full tile).",
    )
    g.add_argument(
        "--repeats", type=int, default=1, help="Train samples per tile/epoch"
    )
    g.add_argument(
        "--min_gain_px",
        type=int,
        default=0,
        help="Label cleaning: drop gain patches smaller than this many pixels. Train AND val labels; holdout evaluation is optional.",
    )
    g.add_argument(
        "--fill_holes_px",
        type=int,
        default=0,
        help="Label cleaning: fill holes smaller than this inside gain.",
    )
    g.add_argument("--no_augment", action="store_true")
    g.add_argument(
        "--skip_test_eval",
        action="store_true",
        help="Do not evaluate holdout labels after training. Use during tuning/ablation; evaluate selected checkpoints once at the end.",
    )
    return p.parse_args(argv)


def build_loss_config(args, train_dirs) -> LossConfig:
    pos_weight = args.pos_weight
    if pos_weight == "auto":
        pos_weight = auto_pos_weight(train_dirs)
        print(f"pos_weight (auto, sqrt(neg/pos)): {pos_weight:.2f}")

    # Apply overrides to a valid initial config before enabling hardness.
    reweight = args.reweight
    initial_reweight = "none" if reweight == "hardness" else reweight

    cfg = LossConfig(
        name=args.loss_name,
        confidence=args.loss_type,
        pos_weight=pos_weight,
        reweight=initial_reweight,
        hardness=HardnessConfig(layer=args.hl_layer),
    ).with_overrides(args.loss_opt)

    # Set the requested reweight mode after overrides have been parsed.
    return LossConfig(
        name=cfg.name,
        confidence=cfg.confidence,
        pos_weight=cfg.pos_weight,
        lovasz_per_image=cfg.lovasz_per_image,
        reweight=reweight,
        tversky=cfg.tversky,
        focal=cfg.focal,
        ohem=cfg.ohem,
        hardness=cfg.hardness,
    )


def run_name(args, loss_cfg, years) -> str:
    slug = lambda opts: [
        re.sub(r"[^A-Za-z0-9.+-]", "", o.replace("=", "")) for o in opts or ()
    ]
    return "_".join(
        [
            args.model_type,
            "+".join(args.sources),
            loss_cfg.confidence,
            loss_cfg.name,
            *([f"rw-{loss_cfg.reweight}"] if loss_cfg.reweight != "none" else []),
            *slug(args.loss_opt),
            *slug(args.model_opt),
            *([f"y{len(years)}-{years[0]}-{years[-1]}"] if years else []),
            f"crop{args.crop_size or 'full'}",
            f"bs{args.batch_size}",
            f"lr{args.learning_rate:g}",
            f"wd{args.weight_decay:g}",
            f"rep{args.repeats}",
            *([f"mmu{args.min_gain_px}"] if args.min_gain_px else []),
            *([f"fill{args.fill_holes_px}"] if args.fill_holes_px else []),
            *(
                [f"pw{loss_cfg.pos_weight:g}"]
                if loss_cfg.pos_weight is not None
                else []
            ),
            f"seed{args.seed}",
            datetime.now().strftime("%m%d-%H%M%S"),
        ]
    )


def train(args):
    pl.seed_everything(args.seed, workers=True)
    sources = tuple(args.sources)
    years = tuple(sorted(args.years)) if args.years else None

    train_dirs, val_dirs = split_tiles(
        args.data_dir, val_frac=args.val_frac, seed=args.split_seed
    )
    print(f"train tiles: {len(train_dirs)} | val tiles: {len(val_dirs)} ")
    print(f"sources: {sources} | years: {years or tuple(YEARS)}")

    loss_cfg = build_loss_config(args, train_dirs)
    model_opts = parse_model_opts(args.model_opt)

    ckpt_dir = Path("checkpoints") / run_name(args, loss_cfg, years)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    (ckpt_dir / "args.json").write_text(
        json.dumps(
            {**vars(args), "loss_config": loss_cfg.to_dict(), "model_opts": model_opts},
            indent=2,
        )
    )

    crop = args.crop_size or None
    img_size = crop or DEFAULT_IMAGE_SIZE
    train_ds = MultiTemporalGainDataset(
        train_dirs,
        sources=sources,
        augment=not args.no_augment,
        crop_size=crop,
        repeats=args.repeats,
        label_sigma=0.1 if loss_cfg.confidence == "soft" else 0.0,
        min_gain_px=args.min_gain_px,
        fill_holes_px=args.fill_holes_px,
        years=years,
    )
    val_ds = MultiTemporalGainDataset(
        val_dirs,
        sources=sources,
        crop_size=None,
        min_gain_px=args.min_gain_px,
        fill_holes_px=args.fill_holes_px,
        years=years,
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
        loss=loss_cfg.to_dict(),
        years=years,
        sources=sources,
        **model_opts,
    )

    logger = pl.loggers.WandbLogger(project="Forest-Gain-CD")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="32-true",
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
    best_score = (
        float(cb.best_model_score) if cb.best_model_score is not None else float("nan")
    )
    print(f"best checkpoint: {best} | val_ap: {best_score:.3f}")

    # Holdout is not touched by experiment selection when --skip_test_eval is set.
    if best and not args.skip_test_eval:
        print("--- test set ---")
        results = evaluate_checkpoint(
            best, test_dir=Path(args.holdout_dir), crop_size=None
        )
        logger.log_metrics({f"test_{k}": v for k, v in results.items()})
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
    elif args.skip_test_eval:
        print("Skipping holdout evaluation (--skip_test_eval).")


if __name__ == "__main__":
    train(parse_args())
