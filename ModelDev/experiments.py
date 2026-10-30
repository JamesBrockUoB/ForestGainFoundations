"""Build and run reproducible Forest Gain experiment suites.

Examples
--------
python experiments.py benchmark --dry-run
python experiments.py benchmark --models tsvit --execute --trials 50 --study-name fg_tsvit --storage-path optuna_journal.log --worker-id 0
python experiments.py data-sources --model tsvit --execute
python experiments.py losses --model tsvit --execute --max-runs 12
python experiments.py hardness --model tsvit --dry-run
python experiments.py temporal --models tsvit sits_scd --seeds 0 1 2 --dry-run

Suites
  benchmark     lr / weight decay / size search, S1+S2 (Optuna TPE; parallel workers can share one study)
  data-sources  every input-source combination
  losses        bce, dice, focal Tversky, BCE+Tversky, focal, focal+OHEM, Lovasz, BCE+Lovasz, OHEM
  hardness      learned pixel hardness (offset c) vs no re-weighting vs OHEM, on BCE-based losses
  temporal      bi-temporal (first + last year) vs more frames vs the whole sequence; same scored pixels for all
  augmentation  repeats (compute-matched), augmentation families / strengths, frame dropout, crop size, label jitter
--seeds repeats every run per seed (seeds are the outer loop, so --max-runs still covers every config).

After a benchmark, confirm_top_k.py retrains the best trials over several seeds with holdout evaluation.

Every run is labelled for Weights & Biases: --wandb_group = one launch (suite + timestamp, or --group), --variant = the
seed-independent configuration label, --wandb_tags = suite / model. Aggregate over seeds with report.py.

Each invocation writes a JSON manifest containing exact ``train.py`` commands.
Review the manifest first; use ``--execute`` only when ready to train.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

YEAR_SETS = (  # frame subsets for the temporal suite; "all" = the whole sequence (config.YEARS)
    (2017, 2024),
    (2017, 2021, 2024),
    (2017, 2020, 2022, 2024),
    None,
)

SOURCE_SETS = (
    ("s1",),
    ("s2",),
    ("s1", "s2"),
    ("alphaearth",),
    ("tessera",),
    ("s1", "alphaearth"),
    ("s2", "alphaearth"),
    ("s1", "s2", "alphaearth"),
    ("s1", "tessera"),
    ("s2", "tessera"),
    ("s1", "s2", "tessera"),
    ("s1", "s2", "alphaearth", "tessera"),
)

BENCHMARK_MODELS = ("tsvit", "sits_scd", "unet_lstm")
BENCHMARK_LOSS = "bce_focal_tversky"


def _base(args: argparse.Namespace, model: str, sources: tuple[str, ...]) -> dict:
    return {
        "model_type": model,
        "sources": list(sources),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "holdout_dir": args.holdout_dir,
        "split_seed": args.split_seed,
        # Do not evaluate the final holdout during model/loss/source selection.
        "skip_test_eval": True,
        # Weights & Biases labelling
        "suite": args.suite,
        "wandb_group": args.group,
        "wandb_tags": [args.suite, model],
        **({"wandb_project": args.wandb_project} if args.wandb_project else {}),
        **({"loss_type": args.loss_type} if args.loss_type else {}),
    }


def _opts(**kv) -> list[str]:
    """{'tversky.alpha': 0.3} -> ['tversky.alpha=0.3'] for train.py's --loss_opt / --model_opt (None values skipped)."""
    return [f"{k}={v}" for k, v in kv.items() if v is not None]


def benchmark(args: argparse.Namespace) -> list[dict]:
    """Static preview configurations. Actual benchmark execution uses adaptive TPE search."""
    rows = []
    # Representative preview only; --execute uses the Bayesian/TPE objective below.
    for model in args.models or BENCHMARK_MODELS:
        rows.append(
            _base(args, model, ("s1", "s2"))
            | {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "model_opt": _opts(**_default_structure(model)),
            }
        )
    return rows


def _default_structure(model: str) -> dict:
    if model == "tsvit":
        return dict(
            embed_dim=128,
            num_heads=4,
            depth_temporal=2,
            depth_spatial=4,
            patch_size=16,
            dropout=0.1,
            skip_channels=32,
        )
    if model == "sits_scd":
        return dict(base_features=64, num_heads=4, dropout=0.1)
    return dict(base_features=64, skip_channels=32, dropout=0.1)


def _suggest_benchmark_trial(trial, model: str) -> dict:
    """Search optimiser and model-specific architectural parameters without redundant knobs.

    ``trial`` may be an Optuna Trial or a FixedTrial: confirm_top_k.py replays a stored trial's params through this
    function to rebuild its exact training configuration.
    """
    lr = trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.35)
    if model == "tsvit":
        embed_dim = trial.suggest_categorical("embed_dim", [96, 128, 160])
        # Restrict head counts to divisors of each candidate embedding dimension.
        # (All current candidates divide by 4 and 8, so the search space is static.)
        heads = [h for h in (4, 8) if embed_dim % h == 0]
        structure = dict(
            embed_dim=embed_dim,
            num_heads=trial.suggest_categorical("num_heads", heads),
            depth_temporal=trial.suggest_int("depth_temporal", 1, 4),
            depth_spatial=trial.suggest_int("depth_spatial", 2, 6),
            patch_size=trial.suggest_categorical("patch_size", [8, 16, 32]),
            dropout=dropout,
            skip_channels=trial.suggest_categorical("skip_channels", [16, 24, 32, 48]),
        )
    elif model == "sits_scd":
        structure = dict(
            base_features=trial.suggest_categorical("base_features", [32, 48, 64, 96]),
            num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
            dropout=dropout,
            depth_temporal=trial.suggest_int("depth_temporal", 1, 4),
            depth_spatial=trial.suggest_int("depth_spatial", 2, 6),
        )
    else:
        structure = dict(
            base_features=trial.suggest_categorical(
                "base_features", [24, 32, 48, 64, 96]
            ),
            skip_channels=trial.suggest_categorical("skip_channels", [16, 24, 32, 48]),
            dropout=dropout,
            depth_temporal=trial.suggest_int("depth_temporal", 1, 3),
        )
    return {"learning_rate": lr, "weight_decay": wd, "model_opt": _opts(**structure)}


def _make_storage(path: str):
    """Journal-file Optuna storage: safe for many concurrent workers on a shared filesystem."""
    import optuna

    try:  # Optuna >= 4
        from optuna.storages.journal import JournalFileBackend

        backend = JournalFileBackend(path)
    except ImportError:  # Optuna 3.x
        from optuna.storages import JournalFileStorage

        backend = JournalFileStorage(path)
    return optuna.storages.JournalStorage(backend)


def run_bayesian_benchmark(args: argparse.Namespace) -> None:
    """Parallel-safe TPE: many workers share one study; each asks for trials until the total budget is claimed.

    Run a single worker (no --storage-path) for the original serial behaviour, or launch several processes with the
    same --study-name / --storage-path and different --worker-id values (e.g. a SLURM array).
    """
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Bayesian benchmark needs Optuna: pip install optuna") from exc

    # multivariate / constant_liar are flagged experimental by Optuna; the warnings are harmless.
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models or BENCHMARK_MODELS
    if args.storage_path and len(models) != 1:
        raise SystemExit(
            "--storage-path/--study-name is per study: pass exactly one model in --models."
        )

    # Random start-up trials before TPE models the objective: about a third of the budget, between 5 and 24.
    startup = (
        args.startup_trials
        if args.startup_trials is not None
        else min(24, max(5, args.trials // 3))
    )

    all_results = []
    for model in models:
        study_name = (
            args.study_name
            or f"forestgain_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=_make_storage(args.storage_path) if args.storage_path else None,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                # Different seed per worker, otherwise workers can propose identical startup trials.
                seed=args.seed + args.worker_id,
                n_startup_trials=min(startup, args.trials),
                # Model interactions between parameters (e.g. best lr depends on model size).
                multivariate=True,
                # Treat in-flight trials as bad results so concurrent workers don't pile onto the same region.
                constant_liar=True,
            ),
        )
        model_results = []
        while True:
            # Counts running trials too, so workers stop once the whole budget is claimed.
            if len(study.get_trials(deepcopy=False)) >= args.trials:
                break
            trial = study.ask()
            if trial.number >= args.trials:
                # Another worker claimed the last slot between the check above and ask(): give this one back
                # without training, so the budget is exact.
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                break
            index = trial.number + 1  # global trial number across workers
            row = (
                _base(args, model, ("s1", "s2"))
                | _suggest_benchmark_trial(trial, model)
                | {
                    "seed": args.seed,
                    "loss_name": BENCHMARK_LOSS,
                    "wandb_group": study.study_name,
                    "variant": f"trial{index:02d}",
                }
            )
            # Store the exact training config so confirm_top_k.py can replay it.
            trial.set_user_attr("config", row)
            cmd = command(row)
            print(
                f"\n[{model} trial {index}/{args.trials} worker {args.worker_id}] {' '.join(cmd)}",
                flush=True,
            )
            proc = subprocess.run(
                cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            print(
                proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True
            )
            match = re.search(
                r"best checkpoint:.*?val_ap:\s*([0-9]*\.?[0-9]+)", proc.stdout
            )
            result = {
                "model": model,
                "trial": index,
                "worker": args.worker_id,
                "config": row,
                "command": cmd,
                "returncode": proc.returncode,
                "val_ap": float(match.group(1)) if match else None,
            }
            if proc.returncode != 0 or not match:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                result["error"] = (
                    "Training failed or best validation AP could not be parsed from train.py output."
                )
                if proc.returncode != 0:
                    print(
                        f"Trial {index} failed (exit {proc.returncode}); continuing.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Could not parse best val_ap; check train.py's final checkpoint line.",
                        file=sys.stderr,
                    )
            else:
                study.tell(trial, float(match.group(1)))
                print(f"Trial {index} val_ap={match.group(1)}", flush=True)
            model_results.append(result)
            (out / f"benchmark_{model}_worker{args.worker_id:02d}.json").write_text(
                json.dumps(
                    {
                        "suite": "benchmark",
                        "model": model,
                        "worker": args.worker_id,
                        "study": study.study_name,
                        "objective": "validation val_ap (max)",
                        "sampler": "Optuna TPE (multivariate, constant liar)",
                        "trials_requested": args.trials,
                        "startup_trials": min(startup, args.trials),
                        "results": model_results,
                    },
                    indent=2,
                )
            )
        all_results.extend(model_results)
        done = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        if done:
            print(
                f"\nBEST so far ({model}): trial {study.best_trial.number + 1}, "
                f"val_ap={study.best_value:.5f}, params={study.best_trial.params}"
            )
        else:
            print(
                f"\nNo successful trials for {model} yet; inspect the saved manifest and training logs.",
                file=sys.stderr,
            )
    summary = (
        out
        / f"benchmark_summary_worker{args.worker_id:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    summary.write_text(
        json.dumps(
            {
                "suite": "benchmark",
                "trials_per_model": args.trials,
                "worker": args.worker_id,
                "results": all_results,
            },
            indent=2,
        )
    )
    print(f"Wrote benchmark results to {summary}")


def data_sources(args: argparse.Namespace) -> list[dict]:
    return [
        _base(args, args.model, sources)
        | {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
        for sources in SOURCE_SETS
    ]


def _run(args: argparse.Namespace, **extra) -> dict:
    """One run of the chosen model / sources / lr / wd, plus the given overrides."""
    return _base(args, args.model, tuple(args.sources)) | {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        **extra,
    }


def losses(args: argparse.Namespace) -> list[dict]:
    rows = []
    for name in ("bce", "dice"):
        rows.append(_run(args, loss_name=name))
    # Tune both standalone Focal Tversky and BCE + Focal Tversky, using a
    # matched alpha/beta balance and focal exponent grid for both recipes.
    for name in ("focal_tversky", "bce_focal_tversky"):
        for alpha, gamma in itertools.product((0.3, 0.5, 0.7), (1.0, 1.33, 2.0)):
            opts = _opts(
                **{
                    "tversky.alpha": alpha,
                    "tversky.beta": round(1 - alpha, 2),
                    "tversky.gamma": gamma,
                }
            )
            rows.append(_run(args, loss_name=name, loss_opt=opts))
    # Plain focal loss: gamma 0 would be BCE; alpha 0.25 is the RetinaNet value
    for gamma, alpha in itertools.product((1.0, 2.0, 3.0), (None, 0.25)):
        opts = _opts(**{"focal.gamma": gamma, "focal.alpha": alpha})
        rows.append(_run(args, loss_name="focal", loss_opt=opts))
    # Lovasz hinge: pooled over the batch / per tile, alone and added to BCE
    for name, per_tile in itertools.product(("lovasz", "bce_lovasz"), (False, True)):
        rows.append(
            _run(
                args,
                loss_name=name,
                loss_opt=_opts(lovasz_per_image=True) if per_tile else None,
            )
        )
    # OHEM on BCE-based losses.
    for name, thresh in itertools.product(("bce", "bce_focal_tversky"), (0.7, 0.9)):
        rows.append(
            _run(
                args,
                loss_name=name,
                reweight="ohem",
                loss_opt=_opts(**{"ohem.thresh": thresh}),
            )
        )

    # Focal + OHEM: the Focal term must pass its per-pixel focal losses
    # to OHEM (this experiment config alone is not sufficient).
    for gamma, alpha, thresh in itertools.product(
        (1.0, 2.0, 3.0), (None, 0.25), (0.7, 0.9)
    ):
        opts = _opts(
            **{
                "focal.gamma": gamma,
                "focal.alpha": alpha,
                "ohem.thresh": thresh,
            }
        )
        rows.append(
            _run(
                args,
                loss_name="focal",
                reweight="ohem",
                loss_opt=opts,
            )
        )
    return rows


# Segmentation head the hardness branch mirrors, per model. TSViT's `decoder` is a multi-conv head, so the branch is a
# copy of it ("auto" -> copy); `decoder.final_head` is a lone 1x1 classifier, so the small default head is used.
# SITS-SCD / U-Net-LSTM only have a lone classifier (their decoder is several modules), so only the default head.
HARDNESS_LAYERS = {
    "tsvit": ("decoder", "decoder.final_head"),
    "sits_scd": ("final_head",),
    "unet_lstm": ("out_head",),
}


def hardness(args: argparse.Namespace) -> list[dict]:
    """Compare no reweighting, OHEM, and learned hardness on identical BCE-based recipes.
    Sweeps hardness offset c, auxiliary-loss alpha and (TSViT only) the segmentation head the branch mirrors.
    --hl-layer overrides the per-model layers above.
    """
    layers = (args.hl_layer,) if args.hl_layer else HARDNESS_LAYERS[args.model]
    rows = []
    for name in ("bce", "bce_focal_tversky"):
        # Matched controls: same recipe, split, seed, model and optimiser.
        rows.append(_run(args, loss_name=name, reweight="none"))
        for thresh in (0.7, 0.9):
            rows.append(
                _run(
                    args,
                    loss_name=name,
                    reweight="ohem",
                    loss_opt=_opts(**{"ohem.thresh": thresh}),
                )
            )
        for layer, c, alpha in itertools.product(
            layers, (0.01, 0.1), (0.001, 0.01, 0.05)
        ):
            rows.append(
                _run(
                    args,
                    loss_name=name,
                    reweight="hardness",
                    hl_layer=layer,
                    loss_opt=_opts(**{"hardness.c": c, "hardness.alpha": alpha}),
                )
            )
    return rows


def temporal(args: argparse.Namespace) -> list[dict]:
    """Same model, sources and loss; only the frames the model sees change.
    Scored pixels are identical for every subset (valid mask is taken over all years).
    """
    year_sets = (
        [tuple(sorted(y)) for y in args.year_sets] if args.year_sets else YEAR_SETS
    )
    rows = []
    for model in args.models or [args.model]:
        for years in year_sets:
            rows.append(
                _base(args, model, tuple(args.sources))
                | {
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "loss_name": args.loss,
                    **({"years": list(years)} if years else {}),
                }
            )
    return rows


def augmentation(args: argparse.Namespace) -> list[dict]:
    """Training-time augmentation / sampling. Everything else (model, sources, loss, lr, wd) is fixed.

    repeats multiplies the steps per epoch, so with --match-steps (default) epochs and patience are divided by repeats:
    every repeats value then gets the same number of optimiser steps, and only the *diversity* of samples differs.
    """

    def r(variant, repeats=1, **extra):
        k = repeats if args.match_steps else 1
        return _run(
            args,
            variant=variant,
            repeats=repeats,
            epochs=max(1, round(args.epochs / k)),
            patience=max(1, round(args.patience / k)) if args.patience else 0,
            **extra,
        )

    rows = [
        r("baseline")
    ]  # train.py defaults: rot90+flip, gain 0.03, frame 0.01, no frame drop, full tile, repeats 1
    # which augmentation family helps
    rows += [
        r("aug-off", no_augment=True),
        r("geo-only", gain_sigma=0.0, frame_sigma=0.0),
        r("spectral-only", no_geo_aug=True),
    ]
    # spectral strength (gain_sigma, frame_sigma)
    rows += [
        r(f"spectral-g{g}-f{f}", gain_sigma=g, frame_sigma=f)
        for g, f in ((0.1, 0.03), (0.2, 0.05))
    ]
    # interior-frame dropout
    rows += [r(f"framedrop{p}", frame_drop_p=p) for p in (0.1, 0.25)]
    # repeats: more augmented samples per tile per epoch (same total steps), plus an un-augmented control
    rows += [r(f"repeats{n}", repeats=n) for n in (2, 4, 8)]
    rows.append(r("repeats4-noaug", repeats=4, no_augment=True))
    # random training crops (val stays full tile); crop must be a multiple of the model's patch size
    rows += [r(f"crop{c}-repeats4", crop_size=c, repeats=4) for c in (128, 192)]
    # soft-label jitter only exists for the soft loss
    if args.loss_type == "soft":
        rows += [r(f"label-sigma{s}", label_sigma=s) for s in (0.0, 0.05, 0.1, 0.2)]
    return rows


def auto_variant(row: dict) -> str:
    """Seed-independent label for a configuration (used when a suite does not set `variant` itself)."""
    parts = [
        row.get("model_type"),
        "+".join(row.get("sources", [])),
        row.get("loss_name"),
        row.get("reweight") if row.get("reweight") not in (None, "none") else None,
        row.get("hl_layer"),
        *(row.get("loss_opt") or ()),
        "y" + "-".join(map(str, row["years"])) if row.get("years") else None,
    ]
    return "_".join(p for p in parts if p)


def command(row: dict) -> list[str]:
    cmd = [sys.executable, "train.py"]
    for key, value in row.items():
        flag = "--loss" if key == "loss_name" else "--" + key
        if value is None or value is False:
            continue  # unset option / switch off
        if value is True:
            cmd.append(flag)  # store_true switch
        elif isinstance(value, (list, tuple)):
            cmd += [flag, *map(str, value)]
        else:
            cmd += [flag, str(value)]
    return cmd


SUITES = {
    "benchmark": benchmark,
    "data-sources": data_sources,
    "losses": losses,
    "hardness": hardness,
    "temporal": temporal,
    "augmentation": augmentation,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=tuple(SUITES))
    p.add_argument(
        "--execute",
        action="store_true",
        help="Run commands (default only writes manifest).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands as well as writing manifest.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap for smoke tests / staged sweeps.",
    )
    p.add_argument("--output-dir", default="experiment_manifests")
    p.add_argument(
        "--model", default="tsvit", choices=("tsvit", "sits_scd", "unet_lstm")
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["s1", "s2"],
        choices=("s1", "s2", "alphaearth", "tessera"),
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Bayesian benchmark TOTAL trials PER MODEL, shared across parallel workers (default 50).",
    )
    p.add_argument(
        "--startup-trials",
        type=int,
        default=None,
        help="Random trials before TPE starts modelling (default: a third of --trials, between 5 and 24).",
    )
    p.add_argument(
        "--study-name",
        default=None,
        help="Shared Optuna study name (parallel workers must all use the same one).",
    )
    p.add_argument(
        "--storage-path",
        default=None,
        help="Optuna journal file on a shared filesystem; enables parallel workers.",
    )
    p.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Parallel worker index (used for sampler seed and output filenames).",
    )
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--data-dir", default="../DataCollection/data/forest_gain_tiles")
    p.add_argument("--holdout-dir", default="../DataCollection/data/holdout_tiles")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Repeat every run for each seed (default: --seed only).",
    )
    p.add_argument(
        "--loss-type",
        choices=("hard", "soft"),
        default=None,
        help="Pass --loss_type to every run (default: train.py's default).",
    )
    p.add_argument(
        "--loss",
        default="bce_focal_tversky",
        help="Loss for the temporal suite (the losses and hardness suites set their own).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=("tsvit", "sits_scd", "unet_lstm"),
        help="Temporal suite: models to compare (default: --model).",
    )
    p.add_argument(
        "--year-sets",
        nargs="+",
        type=lambda s: [int(y) for y in s.split(",")],
        help="Temporal suite: frame subsets as comma lists, e.g. 2017,2024 2017,2021,2024. "
        "Default: 2017,2024 | 2017,2021,2024 | 2017,2020,2022,2024 | all.",
    )
    p.add_argument(
        "--group",
        default=None,
        help="W&B group for every run of this launch (default: <suite>_<timestamp>).",
    )
    p.add_argument(
        "--wandb-project", default=None, help="W&B project (default: train.py's)."
    )
    p.add_argument(
        "--match-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augmentation suite: divide epochs/patience by repeats so all runs get equal optimiser steps.",
    )
    p.add_argument(
        "--hl-layer",
        default=None,
        help="Hardness suite: override the segmentation-head module the hardness branch mirrors "
        "(default per model: see HARDNESS_LAYERS).",
    )
    args = p.parse_args()
    args.group = (
        args.group or f"{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.suite == "benchmark":
        if args.trials < 1:
            p.error("--trials must be at least 1")
        if args.execute:
            run_bayesian_benchmark(args)
            return
        # A dry run reports the adaptive search plan; exact later trials depend on earlier val_ap results.
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        models = list(args.models or BENCHMARK_MODELS)
        manifest = {
            "suite": "benchmark",
            "sampler": "Optuna TPE Bayesian optimisation (multivariate, constant liar)",
            "objective": "maximise validation val_ap",
            "trials_per_model": args.trials,
            "startup_trials": (
                args.startup_trials
                if args.startup_trials is not None
                else min(24, max(5, args.trials // 3))
            ),
            "models": models,
            "sources": ["s1", "s2"],
            "loss": BENCHMARK_LOSS,
            "search_spaces": {
                "common": {
                    "learning_rate": "log-uniform [3e-5, 1e-3]",
                    "weight_decay": "log-uniform [1e-5, 5e-2]",
                    "dropout": "uniform [0, 0.35]",
                },
                "tsvit": {
                    "embed_dim": [96, 128, 160],
                    "num_heads": [4, 8],
                    "depth_temporal": [1, 4],
                    "depth_spatial": [2, 6],
                    "patch_size": [8, 16, 32],
                    "skip_channels": [16, 24, 32, 48],
                },
                "sits_scd": {
                    "base_features": [32, 48, 64, 96],
                    "num_heads": [2, 4, 8],
                    "depth_temporal": [1, 4],
                    "depth_spatial": [2, 6],
                },
                "unet_lstm": {
                    "base_features": [24, 32, 48, 64, 96],
                    "skip_channels": [16, 24, 32, 48],
                    "depth_temporal": [1, 3],
                },
            },
            "note": "This is adaptive: exact trial configs are chosen sequentially from prior validation AP results. Add --execute to run training.",
        }
        path = out / f"benchmark_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(manifest, indent=2))
        print(
            f"Wrote Bayesian benchmark plan ({args.trials} trials/model; {args.trials * len(models)} total) to {path}"
        )
        print("python experiments.py benchmark --execute --trials", args.trials)
        return

    rows = SUITES[args.suite](args)
    rows = [r | {"variant": r.get("variant") or auto_variant(r)} for r in rows]
    rows = [r | {"seed": seed} for seed in (args.seeds or [args.seed]) for r in rows]
    if args.max_runs:
        rows = rows[: args.max_runs]
    manifest = {
        "suite": args.suite,
        "wandb_group": args.group,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [{"config": r, "command": command(r)} for r in rows],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(rows)} runs to {path}")
    for item in manifest["runs"]:
        print(" ".join(item["command"]))
    if args.execute:
        for i, item in enumerate(manifest["runs"], 1):
            print(f"[{i}/{len(rows)}] {' '.join(item['command'])}", flush=True)
            subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
