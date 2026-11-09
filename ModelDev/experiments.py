"""Build and run reproducible Forest Gain experiment suites.

Examples
--------
python experiments.py benchmark --dry-run
python experiments.py benchmark --models tsvit --execute --trials 25 --search-seeds 0 1 2 --study-name fg_tsvit --storage-path optuna_journal.log --worker-id 0
python experiments.py data-sources --model tsvit --execute
python experiments.py data-sources --model tsvit --seeds 0 --loss bce_focal --learning-rate 3.5e-4 \
    --weight-decay 5.8e-3 --model-opt embed_dim=160 num_heads=4 depth_temporal=3 depth_spatial=2 patch_size=8 \
    dropout=0.22 skip_channels=24 --dry-run
python experiments.py losses --model tsvit --execute --max-runs 12
python experiments.py hardness --model tsvit --dry-run
python experiments.py temporal --models tsvit sits_scd --seeds 0 1 2 --dry-run
python experiments.py maskclean --model tsvit --seeds 0 1 2 --loss-type soft --dry-run
python experiments.py losses --model tsvit --confirm-from losses_tsvit_<job id> --seeds 0 1 2 --dry-run
python experiments.py benchmark --confirm --studies tsvit,fg_tsvit,j1.log sits_scd,fg_sits,j2.log unet_lstm,fg_unet,j3.log \
    --top-k 1 --stratify-by model --seeds 0 1 2 --dry-run

Suites
  benchmark     lr / weight decay / batch size / architecture search, S1+S2 (Optuna TPE; parallel workers can share one
                study). Every trial is trained with --search-seeds (default 0 1 2) and scored by the MEAN validation AP.
  data-sources  every input-source combination (use the best benchmark configuration: --learning-rate, --weight-decay,
                --loss, --model-opt; --source-index N runs only the N-th combination, for parallel SLURM tasks;
                --top-sources-from GROUP --top-k K retrains the K best combinations of a finished sweep over the
                seeds WITH holdout evaluation: the confirmation step)
  losses        bce, dice, focal Tversky, BCE+Tversky, focal, focal+OHEM, Lovasz, BCE+Lovasz, OHEM
  hardness      learned pixel hardness (offset c) vs no re-weighting vs OHEM, on BCE-based losses
  temporal      bi-temporal (first + last year) vs more frames vs the whole sequence; same scored pixels for all
  augmentation  repeats (compute-matched), augmentation families / strengths, frame dropout, crop size, label jitter
  maskclean     training-label cleaning (small-component removal, hole filling, confidence filtering / hardening)
                and combinations of them, with the model's confirmed configuration; see mask_cleaning.py.
                --mask-stage singles|combos|all
--seeds repeats every run per seed (seeds are the outer loop, so --max-runs still covers every config).
--model-opt passes architecture options to every run of data-sources, losses, hardness and augmentation (the temporal
suite compares several models, so it does not use it).

Confirmation and holdout
  Suites that sweep hyperparameter values select on validation AP, so their best configuration is retrained with
  holdout evaluation (the holdout stays untouched during the sweep):
    losses / hardness / maskclean   --confirm-from SWEEP_GROUP [--confirm-top-k K] [--confirm-index N]
                                    best configuration of every family (e.g. loss + reweight) of a finished sweep
    benchmark                       benchmark --confirm --studies MODEL,STUDY,JOURNAL ... [--top-k K --stratify-by model]
                                    [--confirm-index N]   best Optuna trials (by mean val AP), retrained over --seeds
                                    (one winner per architecture: --top-k 1 --stratify-by model)
    data-sources                    --top-sources-from SWEEP_GROUP --top-k K
  Suites that only compare settings (data-sources, temporal, augmentation) can log the holdout for every run with
  --eval-holdout. Choose configurations on validation AP; report holdout numbers next to it. confirm_top_k.py
  keeps the same select / run / aggregate commands (aggregate reads any W&B group).

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
import statistics
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from mask_cleaning import describe as describe_mask

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
    ("alphaearth", "tessera"),
    ("s1", "alphaearth", "tessera"),
    ("s2", "alphaearth", "tessera"),
)

BENCHMARK_MODELS = ("tsvit", "sits_scd", "unet_lstm")
BENCHMARK_LOSS = "bce_focal"  # must exist in losses.LOSS_NAMES


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
        # Do not evaluate the final holdout during model/loss/source selection (only confirmation runs do).
        "skip_test_eval": not getattr(args, "eval_holdout", False),
        # Log the full validation metric set (val_eval_*) for every run, not only val_ap.
        "val_metrics": True,
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
        return dict(
            base_features=64,
            num_heads=4,
            dropout=0.1,
            depth_temporal=1,
            convs_per_block=2,
        )
    return dict(
        base_features=64,
        dropout=0.1,
        depth_temporal=1,
        lstm_kernel=3,
        convs_per_block=2,
    )


def _suggest_benchmark_trial(trial, model: str) -> dict:
    """Search optimiser and model-specific architectural parameters.

    Every suggested value is consumed by the model.

    Categorical choice lists must stay static per parameter name,
    and embedding/base widths are divisible by all offered head counts.
    """
    lr = trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.35)

    if model == "tsvit":
        structure = dict(
            embed_dim=trial.suggest_categorical("embed_dim", [96, 128, 160, 192]),
            num_heads=trial.suggest_categorical("num_heads", [4, 8]),
            depth_temporal=trial.suggest_int("depth_temporal", 1, 4),
            depth_spatial=trial.suggest_int("depth_spatial", 2, 6),
            patch_size=trial.suggest_categorical("patch_size", [16, 32]),
            dropout=dropout,
            skip_channels=trial.suggest_categorical("skip_channels", [16, 24, 32, 48]),
        )
        batch_size = trial.suggest_categorical("batch_size", [8, 16])

    elif model == "sits_scd":
        structure = dict(
            base_features=trial.suggest_categorical("base_features", [32, 48, 64, 96]),
            num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
            dropout=dropout,
            depth_temporal=trial.suggest_int("depth_temporal", 1, 3),
            convs_per_block=trial.suggest_int("convs_per_block", 1, 3),
        )
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    else:  # unet_lstm
        structure = dict(
            base_features=trial.suggest_categorical(
                "base_features", [24, 32, 48, 64, 96]
            ),
            dropout=dropout,
            depth_temporal=trial.suggest_int("depth_temporal", 1, 3),
            lstm_kernel=trial.suggest_categorical("lstm_kernel", [3, 5]),
            convs_per_block=trial.suggest_int("convs_per_block", 1, 3),
        )
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    return {
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "model_opt": _opts(**structure),
    }


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


def load_confirmed_configs(path="experiment_manifests/confirmed_configs.json") -> dict:
    """Best confirmed configuration per model; {} if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        print(
            f"WARNING: {path} not found; temporal suite falls back to train.py defaults for every model.",
            file=sys.stderr,
        )
        return {}
    return json.loads(path.read_text())["models"]


def run_bayesian_benchmark(args: argparse.Namespace) -> None:
    """Parallel-safe TPE: many workers share one study; each asks for trials until the total budget is claimed.
    Each trial is trained once per seed in --search-seeds and the objective is the mean validation AP.

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

    search_seeds = tuple(args.search_seeds)
    if not search_seeds:
        raise SystemExit("--search-seeds needs at least one seed")

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
            # Store the exact training config so confirm_row can replay it.
            trial.set_user_attr("config", row)
            trial.set_user_attr("seeds", list(search_seeds))
            seed_runs, failed = [], None
            for seed in search_seeds:
                seed_row = row | {"seed": seed}
                cmd = command(seed_row)
                print(
                    f"\n[{model} trial {index}/{args.trials} seed {seed} worker {args.worker_id}] {' '.join(cmd)}",
                    flush=True,
                )
                proc = subprocess.run(
                    cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                print(
                    proc.stdout,
                    end="" if proc.stdout.endswith("\n") else "\n",
                    flush=True,
                )
                match = re.search(
                    r"best checkpoint:.*?val_ap:\s*([0-9]*\.?[0-9]+)", proc.stdout
                )
                seed_runs.append(
                    {
                        "seed": seed,
                        "command": cmd,
                        "returncode": proc.returncode,
                        "val_ap": float(match.group(1)) if match else None,
                    }
                )
                if proc.returncode != 0 or not match:
                    failed = (seed, proc.returncode)
                    break  # a trial needs every seed; do not spend compute on the rest
            aps = [r["val_ap"] for r in seed_runs if r["val_ap"] is not None]
            result = {
                "model": model,
                "trial": index,
                "worker": args.worker_id,
                "config": row,
                "seeds": seed_runs,
                "val_ap": statistics.fmean(aps) if failed is None else None,
                "val_ap_std": (statistics.pstdev(aps) if len(aps) > 1 else 0.0),
            }
            if failed is not None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                result["error"] = (
                    f"Seed {failed[0]} failed (exit {failed[1]}) or its best validation AP could not be parsed "
                    "from train.py output; trial discarded."
                )
                print(
                    f"Trial {index} failed at seed {failed[0]}; continuing.",
                    file=sys.stderr,
                )
            else:
                trial.set_user_attr("val_ap_per_seed", aps)
                study.tell(trial, result["val_ap"])
                print(
                    f"Trial {index} mean val_ap={result['val_ap']:.5f} over seeds {list(search_seeds)} "
                    f"(per seed {aps})",
                    flush=True,
                )
            model_results.append(result)
            (out / f"benchmark_{model}_worker{args.worker_id:02d}.json").write_text(
                json.dumps(
                    {
                        "suite": "benchmark",
                        "model": model,
                        "worker": args.worker_id,
                        "study": study.study_name,
                        "objective": f"mean validation val_ap over seeds {list(search_seeds)} (max)",
                        "search_seeds": list(search_seeds),
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
                f"mean val_ap={study.best_value:.5f}, params={study.best_trial.params}"
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


def _run_sources(run, model: str):
    """Source combination of a finished W&B run as a frozenset (from its config, else from its variant label)."""
    s = run.config.get("sources")
    if isinstance(s, (list, tuple)) and s:
        return frozenset(s)
    variant = run.config.get("variant") or run.name
    if variant.startswith(f"{model}_"):
        return frozenset(variant[len(model) + 1 :].split("_")[0].split("+"))
    return None


def top_source_sets(args: argparse.Namespace) -> list[tuple[str, ...]]:
    """Top --top-k source combinations of a FINISHED data-sources sweep, by mean validation AP over its seeds.

    Reads the sweep's W&B group (--top-sources-from). The ranking is deterministic (ties by SOURCE_SETS order), so
    every SLURM array task derives the same selection.
    """
    import statistics

    import wandb

    api = wandb.Api()
    project = args.wandb_project or "Forest-Gain-CD"
    lookup = {frozenset(s): s for s in SOURCE_SETS}
    by_set: dict[tuple[str, ...], list[float]] = {}
    for run in api.runs(
        f"{api.default_entity}/{project}", filters={"group": args.top_sources_from}
    ):
        if run.state != "finished":
            continue
        summary = dict(run.summary)
        val = summary.get("val_ap.max", summary.get("val_ap"))
        sources = lookup.get(_run_sources(run, args.model))
        if val is None or sources is None:
            continue
        by_set.setdefault(sources, []).append(float(val))
    if not by_set:
        raise SystemExit(
            f"No finished runs with a validation AP in W&B group '{args.top_sources_from}'."
        )
    ranked = sorted(
        by_set, key=lambda s: (-statistics.fmean(by_set[s]), SOURCE_SETS.index(s))
    )
    print(
        f"Source combinations in '{args.top_sources_from}' (mean validation AP over seeds):"
    )
    for s in ranked:
        print(
            f"  {'+'.join(s):<34} n={len(by_set[s])}  val_ap={statistics.fmean(by_set[s]):.4f}"
        )
    return ranked[: args.top_k]


# ---------------------------------------------------------------------------------------------------------------------
# Confirmation: retrain the best configuration of each family WITH holdout evaluation.
#
# Suites that sweep hyperparameter values (losses, hardness, maskclean, and the Optuna benchmark) select on validation
# AP, so their best configuration is partly the luckiest; they are confirmed by retraining the best of each family and
# scoring it on the holdout (--confirm-from GROUP; benchmark: --confirm --studies ...). Pure comparisons (data-sources,
# temporal, augmentation) do not select anything, so they log the holdout for every run directly (--eval-holdout).
# Holdout numbers are reported next to validation AP and must not be used to choose between configurations.
# ---------------------------------------------------------------------------------------------------------------------

# train.py arguments copied from the sweep's W&B run config into the confirmation command (exactly what the run used).
CONFIRM_TRAIN_KEYS = (
    "model_type", "sources", "years", "epochs", "patience", "batch_size", "data_dir", "holdout_dir", "split_seed",
    "val_frac", "learning_rate", "weight_decay", "warmup_frac", "model_opt", "loss_name", "loss_type", "pos_weight",
    "reweight", "hl_layer", "loss_opt", "crop_size", "repeats", "no_augment", "no_geo_aug", "gain_sigma",
    "frame_sigma", "frame_drop_p", "label_sigma", "min_gain_px", "fill_holes_px", "conf_filter", "conf_harden",
    "clean_val",
)  # fmt: skip
_ZERO_IS_DEFAULT = {"crop_size", "frame_drop_p", "min_gain_px", "fill_holes_px"}


def _loss_family(cfg: dict) -> tuple:
    return (cfg["loss_name"], cfg.get("reweight") or "none")


def _mask_family(cfg: dict) -> tuple:
    names = (
        ("conf_filter", "conf"),
        ("conf_harden", "harden"),
        ("min_gain_px", "rm"),
        ("fill_holes_px", "fill"),
    )
    return ("+".join(n for k, n in names if cfg.get(k)) or "baseline",)


# suite -> how its sweep runs are grouped into families (the best configuration of each family is confirmed)
CONFIRM_FAMILY = {
    "losses": _loss_family,  # bce, dice, focal_tversky, ..., with / without OHEM
    "hardness": _loss_family,  # bce / bce_focal_tversky x none | ohem | hardness
    "maskclean": _mask_family,  # baseline, rm, fill, conf, harden and their combinations
}


def top_wandb_configs(args: argparse.Namespace, family) -> list[tuple]:
    """Best configuration(s) of every family of a FINISHED sweep, by mean validation AP over its seeds.

    Reads the sweep's W&B group (--confirm-from; finished runs only). The ranking is deterministic (ties by variant
    name), so every SLURM array task derives the same selection. Returns (variant, mean, std, n, config) tuples.
    """
    import statistics

    import wandb

    api = wandb.Api()
    project = args.wandb_project or "Forest-Gain-CD"
    by_variant: dict[str, dict] = {}
    for run in api.runs(
        f"{api.default_entity}/{project}", filters={"group": args.confirm_from}
    ):
        if run.state != "finished":
            continue
        summary = dict(run.summary)
        val = summary.get("val_ap.max", summary.get("val_ap"))
        cfg = dict(run.config)
        variant = cfg.get("variant")
        if val is None or not variant or not cfg.get("loss_name"):
            continue
        by_variant.setdefault(variant, {"vals": [], "cfg": cfg})["vals"].append(
            float(val)
        )
    if not by_variant:
        raise SystemExit(
            f"No finished runs with a validation AP in W&B group '{args.confirm_from}'."
        )
    families: dict[tuple, list] = {}
    for variant, e in by_variant.items():
        vals = e["vals"]
        row = (
            variant,
            statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals),
            e["cfg"],
        )
        families.setdefault(family(e["cfg"]), []).append(row)
    selected = []
    print(f"Families in '{args.confirm_from}' (mean validation AP over seeds):")
    for key in sorted(families):
        ranked = sorted(families[key], key=lambda r: (-r[1], r[0]))
        print(f"\n{' / '.join(map(str, key))}  ({len(ranked)} configurations)")
        for i, (variant, mean, std, n, _) in enumerate(ranked):
            mark = "*" if i < args.confirm_top_k else " "
            print(f" {mark} n={n} val_ap={mean:.4f} +/- {std:.4f}  {variant}")
        selected += ranked[: args.confirm_top_k]
    return selected


def _take_index(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """--confirm-index N: only the N-th confirmation run (one per SLURM array task); [] if beyond the plan."""
    if args.confirm_index is None:
        return rows
    if not 0 <= args.confirm_index < len(rows):
        print(
            f"--confirm-index {args.confirm_index} is beyond the {len(rows)} planned confirmation runs; nothing to do."
        )
        return []
    return [rows[args.confirm_index]]


def wandb_confirm_rows(args: argparse.Namespace) -> list[dict]:
    """Confirmation runs for a W&B-ranked sweep (losses, hardness, maskclean): the best configuration of each family,
    rebuilt from the sweep run's own config, with the holdout evaluated (seeds are added by main).
    """
    rows = []
    for variant, _mean, _std, _n, cfg in top_wandb_configs(
        args, CONFIRM_FAMILY[args.suite]
    ):
        row = {
            k: cfg[k]
            for k in CONFIRM_TRAIN_KEYS
            if cfg.get(k) is not None
            and cfg.get(k) is not False
            and not (k in _ZERO_IS_DEFAULT and cfg.get(k) == 0)
        }
        rows.append(
            row
            | {
                "suite": args.suite,
                "wandb_group": args.group,
                "wandb_tags": [args.suite, "confirm", row.get("model_type", "")],
                "variant": f"confirm-{variant}",
                "skip_test_eval": False,  # evaluate the best checkpoint on the holdout tiles
                **({"wandb_project": args.wandb_project} if args.wandb_project else {}),
            }
        )
    return _take_index(rows, args)


# ---- benchmark confirmation (Optuna studies), formerly confirm_top_k.py --------------------------------------------
STRATIFY_ALIASES = {"loss": "loss_name"}


class Cand(NamedTuple):
    model: str
    study: str
    trial: object  # optuna FrozenTrial


def parse_studies(specs) -> list[tuple[str, str, str]]:
    out = []
    for spec in specs:
        parts = spec.split(",", 2)
        if len(parts) != 3 or parts[0] not in BENCHMARK_MODELS:
            raise SystemExit(
                f"--studies entry '{spec}' must look like MODEL,STUDY_NAME,JOURNAL_PATH "
                f"with MODEL in {BENCHMARK_MODELS}"
            )
        out.append((parts[0], parts[1], parts[2]))
    return out


def confirm_group(args) -> str:
    return args.group or "confirm_" + "__".join(
        name for _, name, _ in parse_studies(args.studies)
    )


def candidates(args) -> list[Cand]:
    """All completed trials of all studies, best validation AP first.

    Deterministic (ties by model, study, trial number), so every SLURM array task derives the same selection.
    """
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cands: list[Cand] = []
    for model, name, path in parse_studies(args.studies):
        study = optuna.load_study(study_name=name, storage=_make_storage(path))
        done = [
            t
            for t in study.get_trials(
                deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
            )
            if t.value is not None
        ]
        print(f"study {name}: {len(done)} completed trials", flush=True)
        cands += [Cand(model, name, t) for t in done]
    cands.sort(key=lambda c: (-c.trial.value, c.model, c.study, c.trial.number))
    return cands


def stratum_of(c: Cand, key: str) -> str:
    """Which pool a trial competes in for top-k selection."""
    if key == "none":
        return "all"
    if key == "model":
        return c.model
    name = STRATIFY_ALIASES.get(key, key)
    cfg = c.trial.user_attrs.get("config") or {}
    if name in cfg:
        return str(cfg[name])
    if name in c.trial.params:
        return str(c.trial.params[name])
    return "unknown"


def select_top(cands: list[Cand], top_k: int, stratify_by: str):
    """Top-k per stratum as [(stratum, Cand)], strata in sorted order, each best first."""
    strata: dict[str, list[Cand]] = {}
    for c in cands:  # cands is sorted best first, so each stratum is too
        strata.setdefault(stratum_of(c, stratify_by), []).append(c)
    top = []
    for s in sorted(strata):
        top += [(s, c) for c in strata[s][:top_k]]
    return top


def variant_label(c: Cand, args) -> str:
    """confirm-<model>-trialNN; uses the study name instead if one model has several studies (e.g. a loss search)."""
    models = [m for m, _, _ in parse_studies(args.studies)]
    key = c.model if len(set(models)) == len(models) else c.study
    return f"confirm-{key}-trial{c.trial.number + 1:02d}"


def confirm_row(c: Cand, args) -> dict:
    """Training config of a stored trial, re-labelled for confirmation and with holdout evaluation switched on."""
    import optuna

    model, trial = c.model, c.trial
    cfg = trial.user_attrs.get("config")
    if cfg is None:
        # Study created before configs were stored: replay the params through the search-space function.
        ns = argparse.Namespace(
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            holdout_dir=args.holdout_dir,
            split_seed=args.split_seed,
            suite="confirm",
            group=confirm_group(args),
            wandb_project=args.wandb_project,
            loss_type=None,
        )
        cfg = (
            _base(ns, model, ("s1", "s2"))
            | _suggest_benchmark_trial(optuna.trial.FixedTrial(trial.params), model)
            | {"loss_name": BENCHMARK_LOSS}
        )
    return dict(cfg) | {
        "suite": "confirm",
        "wandb_group": confirm_group(args),
        "wandb_tags": ["confirm", model],
        "variant": variant_label(c, args),
        "skip_test_eval": False,  # evaluate the best checkpoint on the holdout tiles
    }


def benchmark_confirm_plan(args):
    """(top, rows): top-k trials per stratum of the pooled studies and their confirmation rows (seeds added later)."""
    top = select_top(candidates(args), args.top_k, args.stratify_by)
    if not top:
        raise SystemExit("No completed trials in any study yet.")
    print(
        f"\nTop {args.top_k} per stratum (stratify-by '{args.stratify_by}'), mean validation AP over the search seeds:"
    )
    for s, c in top:
        print(
            f"  [{s}] {c.model:<10} trial {c.trial.number + 1:>3}  "
            f"mean val_ap={c.trial.value:.5f}  {c.trial.params}"
        )
    return top, [confirm_row(c, args) for _, c in top]


def benchmark_confirm_rows(args: argparse.Namespace) -> list[dict]:
    return _take_index(benchmark_confirm_plan(args)[1], args)


def data_sources(args: argparse.Namespace) -> list[dict]:
    """Every source combination with ONE fixed training configuration.

    Pass the best benchmark configuration of the chosen model (--learning-rate, --weight-decay, --loss, --model-opt),
    otherwise train.py's defaults are used for the architecture and the learning rate / weight decay defaults apply.
    """
    tuned = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_name": args.loss,
        **({"model_opt": args.model_opt} if args.model_opt else {}),
    }
    sets = top_source_sets(args) if args.top_sources_from else list(SOURCE_SETS)
    if args.source_index is not None:
        if not 0 <= args.source_index < len(sets):
            print(
                f"--source-index {args.source_index} is beyond the {len(sets)} planned source combinations; "
                "nothing to do."
            )
            return []
        sets = [sets[args.source_index]]  # one combination per SLURM array task
    rows = []
    for sources in sets:
        row = _base(args, args.model, sources) | tuned
        if (
            args.top_sources_from
        ):  # confirmation run: seed-independent label (holdout is evaluated, see main)
            row["variant"] = f"confirm-{args.model}-{'+'.join(sources)}"
        rows.append(row)
    return rows


def _run(args: argparse.Namespace, **extra) -> dict:
    """One run of the chosen model / sources / lr / wd (and --model-opt, if given), plus the given overrides."""
    return _base(args, args.model, tuple(args.sources)) | {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        **({"model_opt": args.model_opt} if args.model_opt else {}),
        **extra,
    }


CARRY_CANDIDATES = ("bce_focal", "bce_focal_tversky")


def pick_carry(args: argparse.Namespace) -> dict:
    """Better of the two BCE + focal-type losses (bce_focal, bce_focal_tversky) in a FINISHED base-stage sweep.

    Ranked by mean validation AP over seeds among un-reweighted runs; the winner's best loss_opt / pos_weight are
    returned too, so later stages (OHEM, hardness) build on its best setting. Deterministic (ties by variant name).
    """
    import statistics

    import wandb

    api = wandb.Api()
    project = args.wandb_project or "Forest-Gain-CD"
    by: dict[str, dict] = {}
    for run in api.runs(
        f"{api.default_entity}/{project}", filters={"group": args.carry_from}
    ):
        if run.state != "finished":
            continue
        cfg, summary = dict(run.config), dict(run.summary)
        val = summary.get("val_ap.max", summary.get("val_ap"))
        variant = cfg.get("variant")
        if (
            val is None
            or not variant
            or cfg.get("loss_name") not in CARRY_CANDIDATES
            or (cfg.get("reweight") or "none") != "none"
        ):
            continue
        by.setdefault(variant, {"vals": [], "cfg": cfg})["vals"].append(float(val))
    if not by:
        raise SystemExit(
            f"No finished {CARRY_CANDIDATES} runs with a validation AP in W&B group '{args.carry_from}'."
        )
    ranked = sorted(
        ((statistics.fmean(v["vals"]), name, v["cfg"]) for name, v in by.items()),
        key=lambda r: (-r[0], r[1]),
    )
    best: dict[str, tuple] = {}
    for mean, variant, cfg in ranked:  # best configuration of each candidate loss
        best.setdefault(cfg["loss_name"], (mean, variant, cfg))
    print(
        f"Carry-over candidates in '{args.carry_from}' (best configuration, mean validation AP over seeds):"
    )
    for name, (mean, variant, _) in best.items():
        print(f"  {name:<20} val_ap={mean:.4f}  {variant}")
    winner = max(best, key=lambda n: best[n][0])
    cfg = best[winner][2]
    print(f"-> carrying {winner} into the later stages")
    carry = {"loss_name": winner}
    if cfg.get("loss_opt"):
        carry["loss_opt"] = list(cfg["loss_opt"])
    if cfg.get("pos_weight") is not None:
        carry["pos_weight"] = cfg["pos_weight"]
    return carry


def resolve_carry(args: argparse.Namespace, required: bool) -> dict | None:
    """The loss added to the later stages: --carry-loss (manual) or the winner of --carry-from (W&B)."""
    if args.carry_loss:
        return {
            "loss_name": args.carry_loss,
            **({"loss_opt": list(args.carry_opt)} if args.carry_opt else {}),
        }
    if args.carry_from:
        return pick_carry(args)
    if required:
        raise SystemExit(
            "--loss-stage reweight needs --carry-from BASE_GROUP (or --carry-loss NAME [--carry-opt ...])."
        )
    return None


def losses(args: argparse.Namespace) -> list[dict]:
    """Loss sweep with the model's confirmed configuration (or train.py defaults), in two stages.

    base      un-reweighted families. Coordinate sweeps around each default (one parameter at a time) rather than the
              full grid, so the space is wider at about the same cost. Includes bce_focal (the default loss).
    reweight  OHEM on BCE, on Focal, and on the better of bce_focal / bce_focal_tversky from the finished base stage
              (--carry-from BASE_GROUP, or --carry-loss), using that winner's best parameters.
    """
    cfg = load_confirmed_configs().get(args.model)
    if cfg is None:
        print(
            f"WARNING: no confirmed configuration for {args.model}; using train.py defaults.",
            file=sys.stderr,
        )
        tuned = {}
    else:
        tuned = {
            "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "model_opt": cfg["model_opt"],
            **({"batch_size": cfg["batch_size"]} if cfg.get("batch_size") else {}),
        }

    def run(**kwargs):
        return _run(args, **(tuned | kwargs))

    def uniq(points):
        return list(dict.fromkeys(points))

    rows = []
    if args.loss_stage in ("base", "all"):
        # BCE / Dice; BCE positive-class weight (none vs sqrt(neg/pos))
        rows += [
            run(loss_name="bce"),
            run(loss_name="bce", pos_weight="auto"),
            run(loss_name="dice"),
        ]

        # Tversky-based: alpha at gamma 1, gamma at alpha 0.5 (beta = 1 - alpha)
        tversky = uniq(
            [(a, 1.0) for a in (0.3, 0.4, 0.5, 0.6, 0.7)]
            + [(0.5, g) for g in (0.75, 1.33, 1.67, 2.0)]
        )
        for name in ("focal_tversky", "bce_focal_tversky"):
            for alpha, gamma in tversky:
                opts = _opts(
                    **{
                        "tversky.alpha": alpha,
                        "tversky.beta": round(1 - alpha, 2),
                        "tversky.gamma": gamma,
                    }
                )
                rows.append(run(loss_name=name, loss_opt=opts))

        # Focal and BCE + Focal: gamma with alpha unset, alpha at gamma 2
        focal = uniq(
            [(g, None) for g in (0.5, 1.0, 2.0, 3.0, 5.0)]
            + [(2.0, a) for a in (0.25, 0.75)]
        )
        for name in ("focal", "bce_focal"):
            for gamma, alpha in focal:
                rows.append(
                    run(
                        loss_name=name,
                        loss_opt=_opts(**{"focal.gamma": gamma, "focal.alpha": alpha}),
                    )
                )
        rows.append(run(loss_name="bce_focal", pos_weight="auto"))

        # Lovasz hinge: pooled / per tile, alone / added to BCE
        for name, per_tile in itertools.product(
            ("lovasz", "bce_lovasz"), (False, True)
        ):
            rows.append(
                run(
                    loss_name=name,
                    loss_opt=_opts(lovasz_per_image=True) if per_tile else None,
                )
            )

    if args.loss_stage in ("reweight", "all"):
        carry = resolve_carry(args, required=True)
        carry_opts = carry.get("loss_opt", [])
        carry_extra = {
            k: v for k, v in carry.items() if k != "loss_opt"
        }  # loss_name (+ pos_weight)

        # OHEM threshold at the default min_kept, then min_kept at threshold 0.9
        ohem = [(t, 4096) for t in (0.6, 0.7, 0.8, 0.9)] + [(0.9, 1024), (0.9, 16384)]
        for t, k in ohem:
            o = _opts(**{"ohem.thresh": t, "ohem.min_kept": k})
            rows.append(run(loss_name="bce", reweight="ohem", loss_opt=o))
            rows.append(run(**carry_extra, reweight="ohem", loss_opt=carry_opts + o))

        # Focal + OHEM: gamma and alpha at threshold 0.9, threshold at the default focal
        pts = uniq(
            [(g, None, 0.9) for g in (1.0, 2.0, 3.0)]
            + [(2.0, a, 0.9) for a in (0.25, 0.75)]
            + [(2.0, None, t) for t in (0.6, 0.7, 0.8, 0.9)]
        )
        for gamma, alpha, t in pts:
            o = _opts(**{"focal.gamma": gamma, "focal.alpha": alpha, "ohem.thresh": t})
            rows.append(run(loss_name="focal", reweight="ohem", loss_opt=o))
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
    --hl-layer overrides the per-model layers above. The second loss is bce_focal, or the winner of the loss sweep
    when --carry-from / --carry-loss is given (with that winner's best parameters).
    """
    layers = (args.hl_layer,) if args.hl_layer else HARDNESS_LAYERS[args.model]
    carry = resolve_carry(args, required=False)
    second = carry["loss_name"] if carry else "bce_focal"
    carry_opts = carry.get("loss_opt", []) if carry else []
    carry_extra = {
        k: v for k, v in (carry or {}).items() if k not in ("loss_opt", "loss_name")
    }
    rows = []
    for name in ("bce", second):
        pre = carry_opts if name == second else []
        extra = carry_extra if name == second else {}
        # Matched controls: same recipe, split, seed, model and optimiser.
        rows.append(
            _run(args, loss_name=name, reweight="none", loss_opt=pre or None, **extra)
        )
        for thresh in (0.7, 0.9):
            rows.append(
                _run(
                    args,
                    loss_name=name,
                    reweight="ohem",
                    loss_opt=pre + _opts(**{"ohem.thresh": thresh}),
                    **extra,
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
                    loss_opt=pre + _opts(**{"hardness.c": c, "hardness.alpha": alpha}),
                    **extra,
                )
            )
    return rows


def temporal(args: argparse.Namespace) -> list[dict]:
    """Compare temporal frame subsets, using the confirmed configuration of each model if there is one,
    otherwise train.py's defaults (lr, weight decay, loss and model options are simply not passed).
    """
    configs = load_confirmed_configs()
    year_sets = (
        [tuple(sorted(y)) for y in args.year_sets] if args.year_sets else YEAR_SETS
    )
    rows = []
    for model in args.models or [args.model]:
        cfg = configs.get(model)
        if cfg is None:
            print(
                f"WARNING: no confirmed configuration for {model}; using train.py defaults.",
                file=sys.stderr,
            )
            tuned = {}
        else:
            tuned = {
                "learning_rate": cfg["learning_rate"],
                "weight_decay": cfg["weight_decay"],
                "loss_name": cfg["loss_name"],
                "model_opt": cfg["model_opt"],
                **({"batch_size": cfg["batch_size"]} if cfg.get("batch_size") else {}),
            }
        for years in year_sets:
            rows.append(
                _base(args, model, tuple(args.sources))
                | tuned
                | ({"years": list(years)} if years else {})
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


def maskclean(args: argparse.Namespace) -> list[dict]:
    """Train on cleaned labels with the model's confirmed configuration (experiment_manifests/confirmed_configs.json,
    as in the temporal suite; train.py defaults if there is none). Model, sources, loss, lr, wd are fixed.

    Cleaning is applied by datasets.py to the TRAINING labels via mask_cleaning.clean_labels (order: confidence
    step, small-component removal, hole filling). Early stopping and ``val_ap`` use the RAW validation masks, so every
    variant is selected on the same target and stays comparable (--mask-clean-val cleans them too). After training,
    train.py scores the best checkpoint on the validation tiles with raw labels (``val_eval_*``) and with the cleaned
    labels of the same variant (``val_eval_clean_*``), and the same for the holdout when it is evaluated
    (``test_*`` / ``test_clean_*``).
    Size-1 morphology is a no-op (identical to the baseline) and is skipped unless --mask-keep-noop is given.
    """
    cfg = load_confirmed_configs().get(args.model)
    if cfg is None:
        print(
            f"WARNING: no confirmed configuration for {args.model}; using train.py defaults.",
            file=sys.stderr,
        )
        tuned = {}
    else:
        tuned = {
            "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "loss_name": cfg["loss_name"],
            "model_opt": cfg["model_opt"],
            **({"batch_size": cfg["batch_size"]} if cfg.get("batch_size") else {}),
        }
    sizes = [n for n in args.mask_sizes if n > 1 or args.mask_keep_noop]
    confs = list(args.mask_confs)
    soft = args.loss_type == "soft"
    if not soft:
        print(
            "WARNING: --loss-type is not 'soft', so hardening equals filtering (the hard loss only uses the binary "
            "mask); harden runs are skipped. Use --loss-type soft to include them.",
            file=sys.stderr,
        )

    def r(conf=None, harden=None, rm=0, fill=0):
        extra = {}
        if conf is not None:
            extra["conf_filter"] = conf
        if harden is not None:
            extra["conf_harden"] = harden
        if rm:
            extra["min_gain_px"] = rm
        if fill:
            extra["fill_holes_px"] = fill
        return (
            _base(args, args.model, tuple(args.sources))
            | tuned
            | {"variant": describe_mask(conf, harden, rm, fill)}
            # same label jitter for every variant (harden needs binary labels), soft loss only
            | ({"label_sigma": args.mask_label_sigma} if soft else {})
            | ({"clean_val": True} if args.mask_clean_val and extra else {})
            | extra
        )

    rows = []
    if args.mask_stage in ("singles", "all"):
        rows.append(r())  # baseline: original masks
        rows += [r(rm=n) for n in sizes]
        rows += [r(fill=n) for n in sizes]
        rows += [r(conf=c) for c in confs]
        rows += [r(harden=c) for c in confs if soft]
    if args.mask_stage in ("combos", "all"):
        pairs = args.combo_morph  # (remove, fill) pairs
        rows += [r(rm=a, fill=b) for a, b in pairs]  # morphology only
        for c in args.combo_confs:  # confidence step + morphology
            for a, b in pairs:
                rows.append(r(conf=c, rm=a, fill=b))
                if soft:
                    rows.append(r(harden=c, rm=a, fill=b))
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
        f"pw{row['pos_weight']}" if row.get("pos_weight") is not None else None,
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
    "maskclean": maskclean,
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
    p.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Data-sources suite: run only the N-th source combination (0-11, order of SOURCE_SETS) so that "
        "the combinations can run as parallel SLURM array tasks (default: all of them, serially).",
    )
    p.add_argument(
        "--top-sources-from",
        default=None,
        metavar="WANDB_GROUP",
        help="Data-sources suite, confirmation step: train only the --top-k best source combinations (mean validation "
        "AP over seeds) of a FINISHED sweep in this W&B group, WITH holdout evaluation.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of source combinations for --top-sources-from.",
    )
    p.add_argument(
        "--confirm-from",
        "--top-losses-from",
        dest="confirm_from",
        default=None,
        metavar="WANDB_GROUP",
        help="Losses / hardness / maskclean, confirmation step: rank a FINISHED sweep in this W&B group by mean "
        "validation AP, and retrain the best configuration of every family WITH holdout evaluation.",
    )
    p.add_argument(
        "--confirm-top-k",
        "--losses-top-k",
        dest="confirm_top_k",
        type=int,
        default=1,
        help="Configurations per family for --confirm-from (default 1).",
    )
    p.add_argument(
        "--confirm-index",
        "--loss-index",
        dest="confirm_index",
        type=int,
        default=None,
        help="Confirmation: run only the N-th planned configuration (for parallel SLURM array tasks).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Benchmark suite: confirm the best Optuna trials (--studies, --top-k, --stratify-by) over --seeds with "
        "holdout evaluation, instead of running the search.",
    )
    p.add_argument(
        "--studies",
        nargs="+",
        default=None,
        metavar="MODEL,STUDY_NAME,JOURNAL",
        help="Benchmark --confirm: one entry per Optuna study; the trials of all of them are pooled.",
    )
    p.add_argument(
        "--stratify-by",
        default="none",
        metavar="none|model|KEY",
        help="Benchmark --confirm: 'none' = one global top-k, 'model' = top-k per architecture, or the top-k per "
        "distinct value of a config key / trial param (e.g. loss_name). --top-k is per stratum.",
    )
    p.add_argument(
        "--eval-holdout",
        action="store_true",
        help="Score each run's best checkpoint on the holdout tiles. For suites that only compare settings "
        "(data-sources, temporal, augmentation) and log the holdout for every run; never choose with it.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Bayesian benchmark TOTAL trials PER MODEL, shared across parallel workers (default 25).",
    )
    p.add_argument(
        "--search-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Benchmark search: every trial is trained once per seed; the objective is the mean validation AP "
        "(default 0 1 2, i.e. trials x 3 training runs per model).",
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
    p.add_argument(
        "--model-opt",
        nargs="+",
        default=None,
        metavar="KEY=VAL",
        help="Architecture options passed as --model_opt to every run of the data-sources, losses, hardness and "
        "augmentation suites, e.g. embed_dim=160 num_heads=4 (use the best benchmark configuration).",
    )
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
        default="bce_focal",
        help="Loss for the temporal and data-sources suites (the losses and hardness suites set their own).",
    )
    p.add_argument(
        "--loss-stage",
        choices=("base", "reweight", "all"),
        default="base",
        help="Losses suite: base = un-reweighted families (run first); reweight = OHEM stage built on the better of "
        "bce_focal / bce_focal_tversky (needs --carry-from or --carry-loss); all = both.",
    )
    p.add_argument(
        "--carry-from",
        default=None,
        metavar="BASE_GROUP",
        help="Losses (reweight stage) / hardness: W&B group of a FINISHED base-stage sweep; the better of bce_focal and "
        "bce_focal_tversky (by mean validation AP) and its best parameters are used.",
    )
    p.add_argument(
        "--carry-loss",
        default=None,
        help="Losses / hardness: name the carried loss manually.",
    )
    p.add_argument(
        "--carry-opt",
        nargs="+",
        default=None,
        metavar="KEY=VAL",
        help="Loss options for --carry-loss.",
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
    p.add_argument(
        "--mask-stage",
        choices=("singles", "combos", "all"),
        default="all",
        help="Maskclean suite: single-method runs, combined runs, or both (default all).",
    )
    p.add_argument(
        "--mask-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 25],
        help="Maskclean suite: sizes (px) for small-component removal and hole filling.",
    )
    p.add_argument(
        "--mask-confs",
        type=float,
        nargs="+",
        default=[round(0.5 + 0.05 * i, 2) for i in range(9)],
        help="Maskclean suite: confidence thresholds for filtering and hardening (default 0.50-0.90, step 0.05).",
    )
    p.add_argument(
        "--mask-keep-noop",
        action="store_true",
        help="Maskclean suite: also run size 1 (a no-op, identical to the baseline).",
    )
    p.add_argument(
        "--mask-clean-val",
        action="store_true",
        help="Maskclean suite: also clean the validation labels (--clean_val). Default: raw validation labels.",
    )
    p.add_argument(
        "--mask-label-sigma",
        type=float,
        default=0.0,
        help="Maskclean suite, soft loss: label jitter for every run (0 keeps hardened labels binary and the "
        "variants matched; train.py's soft default is 0.1).",
    )
    p.add_argument(
        "--combo-confs",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.8],
        help="Maskclean combos: confidence thresholds combined with morphology (filter and harden).",
    )
    p.add_argument(
        "--combo-morph",
        nargs="+",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        default=[(5, 5), (10, 10)],
        metavar="REMOVE,FILL",
        help="Maskclean combos: (remove, fill) size pairs, e.g. 5,5 10,10.",
    )
    args = p.parse_args()
    # Confirmation runs (and only those) evaluate the holdout tiles; every other suite keeps it untouched.
    args.eval_holdout = bool(
        args.eval_holdout
        or args.top_sources_from
        or args.confirm_from
        or (args.suite == "benchmark" and args.confirm)
    )
    args.group = (
        args.group or f"{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.confirm and args.suite != "benchmark":
        p.error(
            "--confirm is for the benchmark suite; use --confirm-from for losses, hardness and maskclean"
        )
    if args.confirm and not args.studies:
        p.error("--confirm needs --studies MODEL,STUDY_NAME,JOURNAL_PATH ...")
    if args.confirm_from and args.suite not in CONFIRM_FAMILY:
        p.error(
            f"--confirm-from supports {sorted(CONFIRM_FAMILY)}; data-sources has --top-sources-from"
        )
    if args.suite == "benchmark" and not args.confirm:
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
            "objective": f"maximise MEAN validation val_ap over seeds {list(args.search_seeds)}",
            "search_seeds": list(args.search_seeds),
            "trials_per_model": args.trials,
            "training_runs_total": args.trials * len(models) * len(args.search_seeds),
            "confirmation": "best trial per model retrained over seeds with holdout evaluation: "
            "benchmark --confirm --studies ... --top-k 1 --stratify-by model --seeds 0 1 2",
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
                    "dropout": "uniform [0.05, 0.35]",
                },
                "tsvit": {
                    "embed_dim": [96, 128, 160, 192],
                    "num_heads": [4, 8],
                    "depth_temporal": [1, 4],
                    "depth_spatial": [2, 6],
                    "patch_size": [16, 32],
                    "skip_channels": [16, 24, 32, 48],
                    "batch_size": [8, 16],
                },
                "sits_scd": {
                    "base_features": [32, 48, 64, 96],
                    "num_heads": [2, 4, 8],
                    "depth_temporal": [1, 3],
                    "convs_per_block": [1, 3],
                    "batch_size": [4, 8, 16],
                },
                "unet_lstm": {
                    "base_features": [24, 32, 48, 64, 96],
                    "depth_temporal": [1, 3],
                    "lstm_kernel": [3, 5],
                    "convs_per_block": [1, 3],
                    "batch_size": [4, 8, 16],
                },
            },
            "note": "This is adaptive: exact trial configs are chosen sequentially from prior validation AP results. Add --execute to run training.",
        }
        path = out / f"benchmark_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(manifest, indent=2))
        print(
            f"Wrote Bayesian benchmark plan ({args.trials} trials/model x {len(args.search_seeds)} seeds; "
            f"{args.trials * len(models) * len(args.search_seeds)} training runs total) to {path}"
        )
        print(
            "python experiments.py benchmark --execute --trials",
            args.trials,
            "--search-seeds",
            *args.search_seeds,
        )
        return

    if args.suite == "benchmark":  # --confirm
        rows = benchmark_confirm_rows(args)
    elif args.confirm_from:
        rows = wandb_confirm_rows(args)
    else:
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
