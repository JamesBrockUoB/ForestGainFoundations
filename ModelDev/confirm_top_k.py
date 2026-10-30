"""Confirm the best benchmark configurations: retrain the top-k trials over several seeds, with holdout evaluation.

The benchmark selects on single-seed validation AP and never touches the holdout tiles. This script is the next step:
it takes the top-k completed trials of an Optuna study, trains each one for every seed (default 3 x 3 = 9 runs) with
``--skip_test_eval`` OFF, so train.py scores each run's best checkpoint on the holdout tiles, then aggregates.

Modes
-----
select     print the top-k trials and write the planned runs (add --execute to train them serially, for local use)
run        train ONE planned run, picked by --index (SLURM array task id); index 0 also writes the plan manifest
aggregate  read the finished runs from Weights & Biases: mean +/- std over seeds of val_ap and every test_* metric

Examples
--------
python confirm_top_k.py select --model tsvit --study-name forestgain_tsvit_123 \
    --storage-path experiment_manifests/benchmark_tsvit_123/optuna_journal.log
python confirm_top_k.py run --model tsvit --study-name ... --storage-path ... --index 4 --execute
python confirm_top_k.py aggregate --study-name forestgain_tsvit_123 --out reports/confirm_tsvit

Selection rule: the final model is the configuration with the best MEAN VALIDATION AP over seeds. Holdout numbers are
reported next to it but must not be used to choose between the top-k, otherwise the holdout is no longer held out.
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


def load_study(args):
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna.load_study(
        study_name=args.study_name, storage=_make_storage(args.storage_path)
    )


def group_name(args) -> str:
    return args.group or f"confirm_{args.study_name}"


def top_trials(study, k: int):
    """Top-k COMPLETE trials by validation AP. Deterministic (ties by trial number), so every SLURM task agrees."""
    import optuna

    done = [
        t
        for t in study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        if t.value is not None
    ]
    done.sort(key=lambda t: (-t.value, t.number))
    return done[:k]


def confirm_row(trial, args, seed: int) -> dict:
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
            _base(ns, args.model, ("s1", "s2"))
            | _suggest_benchmark_trial(
                optuna.trial.FixedTrial(trial.params), args.model
            )
            | {"loss_name": BENCHMARK_LOSS}
        )
    return dict(cfg) | {
        "seed": seed,
        "suite": "confirm",
        "wandb_group": group_name(args),
        "wandb_tags": ["confirm", args.model],
        "variant": f"confirm-trial{trial.number + 1:02d}",
        "skip_test_eval": False,  # evaluate the best checkpoint on the holdout tiles
    }


def build_plan(args):
    top = top_trials(load_study(args), args.top_k)
    rows = [confirm_row(t, args, seed) for t in top for seed in args.seeds]
    return top, rows


def write_plan(args, top, rows) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "confirm_plan.json"
    path.write_text(
        json.dumps(
            {
                "study": args.study_name,
                "wandb_group": group_name(args),
                "top_trials": [
                    {"trial": t.number + 1, "val_ap": t.value, "params": t.params}
                    for t in top
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
        raise SystemExit("No completed trials in the study yet.")
    print(f"Top {len(top)} of study {args.study_name} (single-seed validation AP):")
    for t in top:
        print(f"  trial {t.number + 1:>3}  val_ap={t.value:.5f}  {t.params}")
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
    t = top[args.index // len(args.seeds)]
    print(
        f"[confirm {args.index + 1}/{len(rows)}] trial {t.number + 1} (val_ap {t.value:.5f}), "
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
    group = group_name(args)
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
    best_run = by_variant[best][0]
    best_cfg = {
        k: next(
            (r.config.get(k) for r in runs if (r.config.get("variant") == best)), None
        )
        for k in ("model_type", "learning_rate", "weight_decay", "model_opts")
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_keys = ["variant", "seed", "run", "url", "checkpoint_dir", *keys]
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
        "| variant | n | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 2),
    ]
    for v in ranked:
        cells = [_fmt([rec[k] for rec in by_variant[v] if k in rec]) for k in keys]
        mark = " (selected)" if v == best else ""
        lines.append(
            f"| {v}{mark} | {len(by_variant[v])} | " + " | ".join(cells) + " |"
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

    def common(sp, plan: bool):
        sp.add_argument("--study-name", required=True)
        sp.add_argument(
            "--group",
            default=None,
            help="W&B group (default: confirm_<study name>).",
        )
        sp.add_argument("--wandb-project", default="Forest-Gain-CD")
        if not plan:
            return
        sp.add_argument("--model", required=True, choices=BENCHMARK_MODELS)
        sp.add_argument("--storage-path", required=True)
        sp.add_argument("--top-k", type=int, default=3)
        sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        sp.add_argument("--output-dir", default="experiment_manifests")
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

    common(sub.add_parser("select", help=cmd_select.__doc__), plan=True)
    run = sub.add_parser("run")
    common(run, plan=True)
    run.add_argument(
        "--index",
        type=int,
        required=True,
        help="Which planned run to train: trial-major, so index = rank * n_seeds + seed position.",
    )
    agg = sub.add_parser("aggregate")
    common(agg, plan=False)
    agg.add_argument("--wandb-entity", default=None)
    agg.add_argument("--out", default="reports/confirm")

    args = p.parse_args()
    {"select": cmd_select, "run": cmd_run, "aggregate": cmd_aggregate}[args.mode](args)


if __name__ == "__main__":
    main()
