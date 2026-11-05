"""Confirm the best benchmark configurations: retrain the top-k over several seeds, with holdout evaluation.

The benchmark selects on single-seed validation AP and never touches the holdout tiles. This script is the next step.
It pools the completed trials of every benchmark study, ranks them by validation AP, picks the top-k (see
``--stratify-by``), trains each pick for every seed with ``--skip_test_eval`` OFF, so train.py scores each run's best
checkpoint on the holdout tiles, then aggregates.

Studies are given as ``MODEL,STUDY_NAME,JOURNAL_PATH`` (one per study), e.g.
``tsvit,forestgain_tsvit_123,experiment_manifests/benchmark_tsvit_123/optuna_journal.log``.

Selection (``--top-k`` is always PER STRATUM)
---------------------------------------------
--stratify-by none     (default) one pool: the global top-k over all studies, any mix of architectures
--stratify-by model    top-k within each architecture, e.g. --top-k 1 = the best trial of every architecture
--stratify-by <key>    top-k within each distinct value of <key>: a stored config key or an Optuna param of the trial,
                       e.g. ``loss_name`` (``loss`` is accepted as an alias). Trials without that key go to the
                       stratum "unknown".

Number of runs = (number of strata) x top-k x (number of seeds).

Modes
-----
select     print the selected trials and write the planned runs (add --execute to train them serially, for local use)
run        train ONE planned run, picked by --index (SLURM array task id); index 0 also writes the plan manifest
aggregate  read the finished runs from Weights & Biases: mean +/- std over seeds of val_ap and every test_* metric,
           ranked overall, plus the best configuration of every architecture

Examples
--------
python confirm_top_k.py select --studies tsvit,forestgain_tsvit_1,j1.log sits_scd,forestgain_sits_scd_2,j2.log \
    unet_lstm,forestgain_unet_lstm_3,j3.log --group confirm_demo --stratify-by model --top-k 1
python confirm_top_k.py run --studies ... --group confirm_demo --stratify-by model --top-k 1 --index 4 --execute
python confirm_top_k.py aggregate --group confirm_demo --out reports/confirm_demo

Selection rule: the final model is the configuration with the best MEAN VALIDATION AP over seeds. Holdout numbers are
reported next to it but must not be used to choose between the picks, otherwise the holdout is no longer held out.
Comparing architectures by benchmark val_ap is fair because every trial uses the same sources, loss, split and seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from experiments import (
    BENCHMARK_LOSS,
    BENCHMARK_MODELS,
    _base,
    _make_storage,
    _suggest_benchmark_trial,
    command,
)

# Short names accepted by --stratify-by for config / param keys.
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


def group_name(args) -> str:
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
    """Top-k per stratum as [(stratum, Cand)], strata in sorted order, each best first.

    The order is deterministic: run index = position in this list * n_seeds + seed position.
    """
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


def confirm_row(c: Cand, args, seed: int) -> dict:
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
            group=group_name(args),
            wandb_project=args.wandb_project,
            loss_type=None,
        )
        cfg = (
            _base(ns, model, ("s1", "s2"))
            | _suggest_benchmark_trial(optuna.trial.FixedTrial(trial.params), model)
            | {"loss_name": BENCHMARK_LOSS}
        )
    return dict(cfg) | {
        "seed": seed,
        "suite": "confirm",
        "wandb_group": group_name(args),
        "wandb_tags": ["confirm", model],
        "variant": variant_label(c, args),
        "skip_test_eval": False,  # evaluate the best checkpoint on the holdout tiles
    }


def build_plan(args):
    top = select_top(candidates(args), args.top_k, args.stratify_by)
    rows = [confirm_row(c, args, seed) for _, c in top for seed in args.seeds]
    return top, rows


def out_dir(args) -> Path:
    return (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiment_manifests") / group_name(args)
    )


def write_plan(args, top, rows) -> Path:
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "confirm_plan.json"
    path.write_text(
        json.dumps(
            {
                "wandb_group": group_name(args),
                "studies": args.studies,
                "stratify_by": args.stratify_by,
                "top_k_per_stratum": args.top_k,
                "top_trials": [
                    {
                        "stratum": s,
                        "model": c.model,
                        "study": c.study,
                        "trial": c.trial.number + 1,
                        "val_ap": c.trial.value,
                        "params": c.trial.params,
                    }
                    for s, c in top
                ],
                "seeds": args.seeds,
                "runs": [{"config": r, "command": command(r)} for r in rows],
            },
            indent=2,
        )
    )
    return path


def write_confirmed_configs(
    best_per_model,
    by_variant,
    runs,
    group,
    output_path="experiment_manifests/confirmed_configs.json",
):
    """Save the best confirmed configuration per architecture.

    Selection is based exclusively on mean validation AP across seeds.
    """

    models = {}

    for model, variant in best_per_model.items():
        records = by_variant[variant]

        # Select the highest-ranked configuration using validation AP.
        val_values = [r["val_ap"] for r in records if "val_ap" in r]

        if not val_values:
            continue

        # Find the corresponding W&B run configuration.
        run = next(r for r in runs if (r.config.get("variant") or r.name) == variant)

        cfg = dict(run.config)

        models[model] = {
            "variant": variant,
            "mean_val_ap": statistics.fmean(val_values),
            "learning_rate": cfg.get("learning_rate"),
            "weight_decay": cfg.get("weight_decay"),
            "loss_name": cfg.get("loss_name"),
            "model_opt": cfg.get("model_opt", []),
            "seeds": sorted(r["seed"] for r in records if r.get("seed") is not None),
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "wandb_group": group,
                "selection_metric": "mean_val_ap",
                "models": models,
            },
            indent=2,
        )
    )

    print(f"Updated confirmed configurations: {output}")


def cmd_select(args) -> None:
    top, rows = build_plan(args)
    if not top:
        raise SystemExit("No completed trials in any study yet.")
    print(
        f"\nTop {args.top_k} per stratum (stratify-by '{args.stratify_by}'), "
        f"single-seed validation AP:"
    )
    for s, c in top:
        print(
            f"  [{s}] {c.model:<10} trial {c.trial.number + 1:>3}  "
            f"val_ap={c.trial.value:.5f}  {c.trial.params}"
        )
    path = write_plan(args, top, rows)
    print(f"\n{len(rows)} confirmation runs -> {path}")
    for r in rows:
        print(" ".join(command(r)))
    if args.execute:
        for i, r in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}]", flush=True)
            subprocess.run(command(r), check=True)


def cmd_run(args) -> None:
    top, rows = build_plan(args)
    if args.index == 0 and rows:
        write_plan(args, top, rows)
    if args.index >= len(rows):
        print(
            f"Index {args.index} is beyond the {len(rows)} planned runs "
            f"({len(top)} selected trials); nothing to do."
        )
        return
    row = rows[args.index]
    cmd = command(row)
    stratum, c = top[args.index // len(args.seeds)]
    print(
        f"[confirm {args.index + 1}/{len(rows)}] [{stratum}] {c.model} trial {c.trial.number + 1} "
        f"(val_ap {c.trial.value:.5f}), seed {row['seed']}\n{' '.join(cmd)}",
        flush=True,
    )
    if args.execute:
        subprocess.run(cmd, check=True)


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    return (
        f"{mean:.4f} ± {statistics.stdev(values):.4f}"
        if len(values) > 1
        else f"{mean:.4f}"
    )


def cmd_aggregate(args) -> None:
    import wandb

    api = wandb.Api()
    entity = args.wandb_entity or api.default_entity
    group = args.group
    runs = [
        r
        for r in api.runs(f"{entity}/{args.wandb_project}", filters={"group": group})
        if r.state == "finished"
    ]
    if not runs:
        raise SystemExit(f"No finished runs in W&B group '{group}'.")

    records = []
    for r in runs:
        summary = dict(r.summary)
        metrics = {
            k: float(v)
            for k, v in summary.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and (k == "val_ap" or k.startswith("test_"))
        }
        # train.py logs the best-epoch validation AP as the summary key "val_ap.max".
        if isinstance(summary.get("val_ap.max"), (int, float)):
            metrics["val_ap"] = float(summary["val_ap.max"])
        records.append(
            {
                "variant": r.config.get("variant") or r.name,
                "model": r.config.get("model_type"),
                "seed": r.config.get("seed"),
                "run": r.name,
                "url": r.url,
                "checkpoint_dir": r.config.get("checkpoint_dir"),
                **metrics,
            }
        )
    keys = ["val_ap"] + sorted(
        {k for rec in records for k in rec if k.startswith("test_")}
    )

    by_variant: dict[str, list[dict]] = {}
    for rec in records:
        by_variant.setdefault(rec["variant"], []).append(rec)

    def mean_of(variant, key):
        vals = [rec[key] for rec in by_variant[variant] if key in rec]
        return statistics.fmean(vals) if vals else float("-inf")

    def cells_of(variant):
        return [_fmt([rec[k] for rec in by_variant[variant] if k in rec]) for k in keys]

    ranked = sorted(by_variant, key=lambda v: mean_of(v, "val_ap"), reverse=True)
    best = ranked[0]
    best_cfg = {
        k: next(
            (r.config.get(k) for r in runs if r.config.get("variant") == best), None
        )
        for k in ("model_type", "learning_rate", "weight_decay", "model_opts")
    }

    # Best configuration of every architecture (first hit in the ranking), for architecture comparison.
    best_per_model: dict[str, str] = {}
    for v in ranked:
        best_per_model.setdefault(by_variant[v][0].get("model") or "", v)

    write_confirmed_configs(
        best_per_model,
        by_variant,
        runs,
        group,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_keys = ["variant", "model", "seed", "run", "url", "checkpoint_dir", *keys]
    with open(f"{out}.runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(records, key=lambda x: (x["variant"], x["seed"] or 0)))

    lines = [
        f"# Confirmation runs: {group}",
        "",
        "Ranked by mean **validation** AP. Holdout (`test_*`) columns are reported for the final model; "
        "do not use them to choose between configurations.",
        "",
        "| variant | model | n | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 3),
    ]
    for v in ranked:
        mark = " (selected)" if v == best else ""
        model = by_variant[v][0].get("model") or ""
        lines.append(
            f"| {v}{mark} | {model} | {len(by_variant[v])} | "
            + " | ".join(cells_of(v))
            + " |"
        )
    lines += [
        "",
        "## Best configuration per architecture (by mean validation AP)",
        "",
        "| model | variant | n | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 3),
    ]
    for m, v in best_per_model.items():
        lines.append(
            f"| {m} | {v} | {len(by_variant[v])} | " + " | ".join(cells_of(v)) + " |"
        )
    lines += [
        "",
        f"Selected: **{best}**",
        "",
        "```json",
        json.dumps(best_cfg, indent=2),
        "```",
    ]
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    Path(f"{out}.json").write_text(
        json.dumps(
            {
                "group": group,
                "selected_variant": best,
                "selected_config": best_cfg,
                "selected_runs": by_variant[best],
                "best_per_model": best_per_model,
                "ranking": [
                    {
                        "variant": v,
                        "n": len(by_variant[v]),
                        **{
                            k: {
                                "mean": mean_of(v, k),
                                "values": [r[k] for r in by_variant[v] if k in r],
                            }
                            for k in keys
                        },
                    }
                    for v in ranked
                ],
            },
            indent=2,
        )
    )
    print("\n".join(lines))
    print(f"\nWrote {out}.md, {out}.json and {out}.runs.csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def plan_args(sp):
        sp.add_argument(
            "--studies",
            nargs="+",
            required=True,
            metavar="MODEL,STUDY_NAME,JOURNAL",
            help="One entry per study; trials from all of them are pooled.",
        )
        sp.add_argument(
            "--top-k",
            type=int,
            default=3,
            help="Top-k trials PER STRATUM (see --stratify-by).",
        )
        sp.add_argument(
            "--stratify-by",
            default="none",
            metavar="none|model|KEY",
            help="'none': one global top-k (default). 'model': top-k per architecture. "
            "Any other value: top-k per distinct value of that config key / trial param (e.g. loss_name).",
        )
        sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        sp.add_argument(
            "--group",
            default=None,
            help="W&B group (default: derived from study names).",
        )
        sp.add_argument("--wandb-project", default="Forest-Gain-CD")
        sp.add_argument(
            "--output-dir",
            default=None,
            help="Plan manifest directory (default: experiment_manifests/<group>).",
        )
        sp.add_argument("--execute", action="store_true")
        # Only used for studies created before trial configs were stored (must match the benchmark's values).
        sp.add_argument("--epochs", type=int, default=100)
        sp.add_argument("--patience", type=int, default=20)
        sp.add_argument("--batch-size", type=int, default=8)
        sp.add_argument(
            "--data-dir", default="../DataCollection/data/forest_gain_tiles"
        )
        sp.add_argument("--holdout-dir", default="../DataCollection/data/holdout_tiles")
        sp.add_argument("--split-seed", type=int, default=0)

    plan_args(sub.add_parser("select"))
    run = sub.add_parser("run")
    plan_args(run)
    run.add_argument(
        "--index",
        type=int,
        required=True,
        help="Which planned run to train: ranked trial-major, index = selection position * n_seeds + seed position.",
    )
    agg = sub.add_parser("aggregate")
    agg.add_argument(
        "--group", required=True, help="W&B group of the confirmation runs."
    )
    agg.add_argument("--wandb-project", default="Forest-Gain-CD")
    agg.add_argument("--wandb-entity", default=None)
    agg.add_argument("--out", default="reports/confirm")

    args = p.parse_args()
    {"select": cmd_select, "run": cmd_run, "aggregate": cmd_aggregate}[args.mode](args)


if __name__ == "__main__":
    main()
