"""Confirm the best benchmark configurations ACROSS ALL MODELS: retrain the global top-k over several seeds, with holdout evaluation.

The benchmark selects on single-seed validation AP and never touches the holdout tiles. This script is the next step.
It pools the completed trials of every benchmark study (one per architecture), takes the k best by validation AP
overall (default 3, so the winners can be any mix of architectures), trains each for every seed (default 3 x 3 = 9 runs
in total) with ``--skip_test_eval`` OFF, so train.py scores each run's best checkpoint on the holdout tiles, then
aggregates.

Studies are given as ``MODEL,STUDY_NAME,JOURNAL_PATH`` (one per architecture), e.g.
``tsvit,forestgain_tsvit_123,experiment_manifests/benchmark_tsvit_123/optuna_journal.log``.

Modes
-----
select     print the global top-k and write the planned runs (add --execute to train them serially, for local use)
run        train ONE planned run, picked by --index (SLURM array task id); index 0 also writes the plan manifest
aggregate  read the finished runs from Weights & Biases: mean +/- std over seeds of val_ap and every test_* metric

Examples
--------
python confirm_top_k.py select --studies tsvit,forestgain_tsvit_1,j1.log sits_scd,forestgain_sits_scd_2,j2.log \
    unet_lstm,forestgain_unet_lstm_3,j3.log --group confirm_demo
python confirm_top_k.py run --studies ... --group confirm_demo --index 4 --execute
python confirm_top_k.py aggregate --group confirm_demo --out reports/confirm_demo

Selection rule: the final model is the configuration with the best MEAN VALIDATION AP over seeds. Holdout numbers are
reported next to it but must not be used to choose between the top-k, otherwise the holdout is no longer held out.
Comparing architectures by benchmark val_ap is fair because every trial uses the same sources, loss, split and seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import warnings
from pathlib import Path

from experiments import (
    BENCHMARK_LOSS,
    BENCHMARK_MODELS,
    _base,
    _make_storage,
    _suggest_benchmark_trial,
    command,
)


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


def candidates(args):
    """All completed trials of all studies as (model, trial), best validation AP first.

    Deterministic (ties by model then trial number), so every SLURM array task derives the same top-k.
    """
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cands = []
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
        cands += [(model, t) for t in done]
    cands.sort(key=lambda c: (-c[1].value, c[0], c[1].number))
    return cands


def confirm_row(model: str, trial, args, seed: int) -> dict:
    """Training config of a stored trial, re-labelled for confirmation and with holdout evaluation switched on."""
    import optuna

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
        "variant": f"confirm-{model}-trial{trial.number + 1:02d}",
        "skip_test_eval": False,  # evaluate the best checkpoint on the holdout tiles
    }


def build_plan(args):
    top = candidates(args)[: args.top_k]
    rows = [confirm_row(m, t, args, seed) for m, t in top for seed in args.seeds]
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
                "top_trials": [
                    {
                        "model": m,
                        "trial": t.number + 1,
                        "val_ap": t.value,
                        "params": t.params,
                    }
                    for m, t in top
                ],
                "seeds": args.seeds,
                "runs": [{"config": r, "command": command(r)} for r in rows],
            },
            indent=2,
        )
    )
    return path


def cmd_select(args) -> None:
    top, rows = build_plan(args)
    if not top:
        raise SystemExit("No completed trials in any study yet.")
    print(f"\nGlobal top {len(top)} (single-seed validation AP, all architectures):")
    for m, t in top:
        print(f"  {m:<10} trial {t.number + 1:>3}  val_ap={t.value:.5f}  {t.params}")
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
            f"(only {len(top)} completed trials available); nothing to do."
        )
        return
    row = rows[args.index]
    cmd = command(row)
    model, t = top[args.index // len(args.seeds)]
    print(
        f"[confirm {args.index + 1}/{len(rows)}] {model} trial {t.number + 1} (val_ap {t.value:.5f}), "
        f"seed {row['seed']}\n{' '.join(cmd)}",
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

    ranked = sorted(by_variant, key=lambda v: mean_of(v, "val_ap"), reverse=True)
    best = ranked[0]
    best_cfg = {
        k: next(
            (r.config.get(k) for r in runs if r.config.get("variant") == best), None
        )
        for k in ("model_type", "learning_rate", "weight_decay", "model_opts")
    }

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
        cells = [_fmt([rec[k] for rec in by_variant[v] if k in rec]) for k in keys]
        mark = " (selected)" if v == best else ""
        model = by_variant[v][0].get("model") or ""
        lines.append(
            f"| {v}{mark} | {model} | {len(by_variant[v])} | "
            + " | ".join(cells)
            + " |"
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
            help="One entry per architecture; trials from all of them are pooled.",
        )
        sp.add_argument("--top-k", type=int, default=3, help="Global top-k trials.")
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
        help="Which planned run to train: ranked trial-major, index = rank * n_seeds + seed position.",
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
