"""Build and run reproducible Forest Gain experiment suites.

Examples
--------
python experiments.py benchmark --dry-run
python experiments.py data-sources --model tsvit --execute
python experiments.py losses --model tsvit --execute --max-runs 12
python experiments.py hardness --model tsvit --dry-run
python experiments.py temporal --models tsvit sits_scd --seeds 0 1 2 --dry-run

Suites
  benchmark     lr / weight decay / size search, S1+S2
  data-sources  every input-source combination
  losses        bce, dice, focal Tversky, BCE+Tversky, focal, focal+OHEM, Lovasz, BCE+Lovasz, OHEM
  hardness      learned pixel hardness (offset c) vs no re-weighting vs OHEM, on BCE-based losses
  temporal      bi-temporal (first + last year) vs more frames vs the whole sequence; same scored pixels for all
--seeds repeats every run per seed (seeds are the outer loop, so --max-runs still covers every config).

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
        **({"loss_type": args.loss_type} if args.loss_type else {}),
    }


def _opts(**kv) -> list[str]:
    """{'tversky.alpha': 0.3} -> ['tversky.alpha=0.3'] for train.py's --loss_opt / --model_opt (None values skipped)."""
    return [f"{k}={v}" for k, v in kv.items() if v is not None]


def benchmark(args: argparse.Namespace) -> list[dict]:
    """Static preview configurations. Actual benchmark execution uses adaptive TPE search."""
    rows = []
    # Representative preview only; --execute uses the Bayesian/TPE objective below.
    for model in ("tsvit", "sits_scd", "unet_lstm"):
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
    """Search optimiser and model-specific architectural parameters without redundant knobs."""
    lr = trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.35)
    if model == "tsvit":
        embed_dim = trial.suggest_categorical("embed_dim", [96, 128, 160])
        # Restrict head counts to divisors of each candidate embedding dimension.
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


def run_bayesian_benchmark(args: argparse.Namespace) -> None:
    """Sequential TPE optimisation: 10–25 adaptive trials per architecture, objective = best validation AP."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Bayesian benchmark needs Optuna: pip install optuna") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model in ("tsvit", "sits_scd", "unet_lstm"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed, n_startup_trials=min(5, args.trials)
            ),
            study_name=f"forestgain_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        model_results = []
        for index in range(args.trials):
            trial = study.ask()
            suggested = _suggest_benchmark_trial(trial, model)
            row = (
                _base(args, model, ("s1", "s2"))
                | suggested
                | {"seed": args.seed, "loss_name": "bce_focal_tversky"}
            )
            cmd = command(row)
            print(
                f"\n[{model} trial {index + 1}/{args.trials}] {' '.join(cmd)}",
                flush=True,
            )
            if not args.execute:
                print(
                    "Dry-run: adaptive Bayesian trials require --execute; no training launched."
                )
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
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
                "trial": index + 1,
                "config": row,
                "command": cmd,
                "returncode": proc.returncode,
                "val_ap": float(match.group(1)) if match else None,
            }
            model_results.append(result)
            if proc.returncode != 0 or not match:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                result["error"] = (
                    "Training failed or best validation AP could not be parsed from train.py output."
                )
                if proc.returncode != 0:
                    print(
                        f"Trial failed (exit {proc.returncode}); continuing to next trial.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Could not parse best val_ap; check train.py's final checkpoint line.",
                        file=sys.stderr,
                    )
            else:
                study.tell(trial, float(match.group(1)))
                result["best_so_far_val_ap"] = study.best_value
                print(
                    f"Trial val_ap={match.group(1)}; best so far={study.best_value:.5f}",
                    flush=True,
                )
            manifest_path = out / f"benchmark_{model}_adaptive.json"
            completed_now = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            best_info = None
            if completed_now:
                best_now = study.best_trial
                best_info = {
                    "number": best_now.number + 1,
                    "value": study.best_value,
                    "params": best_now.params,
                }
            manifest_path.write_text(
                json.dumps(
                    {
                        "suite": "benchmark",
                        "model": model,
                        "objective": "validation val_ap (max)",
                        "sampler": "Optuna TPE",
                        "trials_requested": args.trials,
                        "results": model_results,
                        "best_trial": best_info,
                    },
                    indent=2,
                )
            )
        all_results.extend(model_results)
        completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if args.execute and completed:
            best = study.best_trial
            print(
                f"\nBEST {model}: trial {best.number + 1}, validation AP={study.best_value:.5f}, params={best.params}"
            )
        elif args.execute:
            print(
                f"\nNo successful trials for {model}; inspect the saved manifest and training logs.",
                file=sys.stderr,
            )
    summary = out / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary.write_text(
        json.dumps(
            {
                "suite": "benchmark",
                "trials_per_model": args.trials,
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
        default=15,
        help="Bayesian benchmark trials PER MODEL (10–25; default 15).",
    )
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--data-dir", default="../DataCollection/data/test_tiles")
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
        "--hl-layer",
        default=None,
        help="Hardness suite: override the segmentation-head module the hardness branch mirrors "
        "(default per model: see HARDNESS_LAYERS).",
    )
    args = p.parse_args()
    if args.suite == "benchmark":
        if not 10 <= args.trials <= 25:
            p.error("--trials must be between 10 and 25 for the Bayesian benchmark")
        if args.execute:
            run_bayesian_benchmark(args)
            return
        # A dry run reports the adaptive search plan; exact later trials depend on earlier val_ap results.
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "suite": "benchmark",
            "sampler": "Optuna TPE Bayesian optimisation",
            "objective": "maximise validation val_ap",
            "trials_per_model": args.trials,
            "models": ["tsvit", "sits_scd", "unet_lstm"],
            "sources": ["s1", "s2"],
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
            f"Wrote Bayesian benchmark plan ({args.trials} trials/model; {args.trials * 3} total) to {path}"
        )
        print("python experiments.py benchmark --execute --trials", args.trials)
        return

    rows = SUITES[args.suite](args)
    rows = [r | {"seed": seed} for seed in (args.seeds or [args.seed]) for r in rows]
    if args.max_runs:
        rows = rows[: args.max_runs]
    manifest = {
        "suite": args.suite,
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
