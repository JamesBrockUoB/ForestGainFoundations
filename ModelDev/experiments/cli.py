"""CLI for planning, running, selecting, confirming, evaluating, and reporting."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from .common import (build_manifest, read_manifest, run_task, select_top_k,
                     wandb_runs, write_manifest)
from .losses import build_default_loss_manifest, build_tune_manifest
from .suites import SUITES, legacy_plan


def _plan(a):
    group = a.group or f"{a.suite}_{a.stage}_{a.model}"
    if a.suite == "losses" and a.stage == "defaults":
        manifest = build_default_loss_manifest(a.model, tuple(a.sources), group, tuple(a.seeds),
            {"epochs": a.epochs, "patience": a.patience, "batch_size": a.batch_size,
             "data_dir": a.data_dir, "holdout_dir": a.holdout_dir, "split_seed": a.split_seed,
             "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
             **({"model_opt": a.model_opt} if a.model_opt else {})})
    elif a.suite == "losses" and a.stage == "tune":
        if not a.selected_loss:
            raise SystemExit("--selected-loss is required for tune stage")
        base = {"epochs": a.epochs, "patience": a.patience, "batch_size": a.batch_size,
                "data_dir": a.data_dir, "holdout_dir": a.holdout_dir, "split_seed": a.split_seed,
                "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
                **({"model_opt": a.model_opt} if a.model_opt else {})}
        manifest = build_tune_manifest(a.model, tuple(a.sources), group, tuple(a.seeds), a.selected_loss, base)
    else:
        options = {"epochs": a.epochs, "patience": a.patience, "batch_size": a.batch_size,
                   "data_dir": a.data_dir, "holdout_dir": a.holdout_dir, "split_seed": a.split_seed,
                   "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
                   "model_opt": a.model_opt, "loss": a.loss, "loss_type": a.loss_type,
                   "loss_stage": a.stage, "mask_stage": a.mask_stage}
        manifest = legacy_plan(a.suite, model=a.model, sources=tuple(a.sources), seeds=tuple(a.seeds), group=group, overrides=options)
    path = Path(a.output)
    write_manifest(manifest, path)
    print(f"Plan: {a.suite}; tasks: {len(manifest['tasks'])}; manifest: {path}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    for task in manifest["tasks"]:
        print(f"{task['task_id']:04d}: {' '.join(task['command'])}")


def _run(a):
    manifest = read_manifest(a.manifest)
    indices = [a.task] if a.task is not None else range(len(manifest["tasks"]))
    for index in indices:
        code = run_task(a.manifest, index, a.cwd, a.log_dir)
        if code:
            raise SystemExit(code)


def _select(a):
    data = wandb_runs(a.project, a.group)
    selected = select_top_k(data, a.metric, a.top_k, a.seeds)
    if not selected:
        raise SystemExit("No variants have completed results for every required seed")
    for i, item in enumerate(selected, 1):
        print(f"{i}. {item['variant']}: mean {a.metric}={item['mean']:.6f}; per-seed={item['per_seed']}")
    result = {"source_group": a.group, "project": a.project, "metric": a.metric,
              "required_seeds": a.seeds, "selected": [
                  {"rank": i, "variant": x["variant"], "mean": x["mean"], "per_seed": x["per_seed"],
                   "runs": [{"id": r["id"], "url": r.get("url"), "config": r["config"]} for r in x["runs"]]}
                  for i, x in enumerate(selected, 1)]}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Selection saved to {a.output}")


def _confirm(a):
    selection = json.loads(Path(a.selection).read_text())
    configs = []
    for chosen in selection["selected"]:
        source_cfg = chosen["runs"][0]["config"]
        # Keep train-relevant fields only; remove W&B/internal values not accepted by train.py.
        cfg = {k: v for k, v in source_cfg.items() if not k.startswith("_")}
        cfg.update({"suite": selection.get("suite", source_cfg.get("suite", "confirm")),
                    "variant": f"confirm-{chosen['variant']}", "wandb_group": a.group,
                    "wandb_tags": ["confirm", source_cfg.get("model_type", "tsvit")],
                    "skip_test_eval": False, "val_metrics": True,
                    "confirmation_source_group": selection["source_group"]})
        for seed in a.seeds:
            configs.append({**cfg, "seed": seed})
    manifest = build_manifest(configs, "confirmation", a.group,
        {"source_selection": str(Path(a.selection)), "source_group": selection["source_group"],
         "top_k": len(selection["selected"]), "confirmation_seeds": a.seeds,
         "holdout_evaluation": True})
    write_manifest(manifest, a.output)
    print(f"Wrote {len(configs)} retrain-and-holdout tasks to {a.output}; run with: python -m experiments run {a.output}")


def _evaluate(a):
    command = [sys.executable, "eval.py", str(Path(a.checkpoint).resolve()), "--test_dir", a.test_dir]
    print(" ".join(command))
    subprocess.run(command, cwd=a.cwd, check=True)


def _report(a):
    rows = wandb_runs(a.project, a.group)
    fields = ["id", "name", "state", "group", "variant", "seed", "val_ap", "val_eval_ap", "val_eval_prevalence", "test_ap", "test_prevalence"]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in rows:
            cfg, summary = run["config"], run["summary"]
            writer.writerow({"id": run["id"], "name": run["name"], "state": run["state"],
                "group": a.group, "variant": cfg.get("variant"), "seed": cfg.get("seed"),
                **{key: summary.get(key, summary.get(f"{key}.max")) for key in fields[6:]}})
    print(f"Wrote {len(rows)} runs to {output}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments", description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="Write a reproducible manifest")
    plan.add_argument("suite", choices=SUITES)
    plan.add_argument("--stage", default="base", choices=("defaults", "tune", "base", "reweight", "all", "singles", "combos"))
    plan.add_argument("--model", default="tsvit", choices=("tsvit", "sits_scd", "unet_lstm"))
    plan.add_argument("--sources", nargs="+", default=["s1", "s2"])
    plan.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    plan.add_argument("--group"); plan.add_argument("--output", required=True)
    plan.add_argument("--epochs", type=int, default=100); plan.add_argument("--patience", type=int, default=20)
    plan.add_argument("--batch-size", type=int, default=8); plan.add_argument("--learning-rate", type=float, default=3e-4)
    plan.add_argument("--weight-decay", type=float, default=1e-2); plan.add_argument("--data-dir", default="../DataCollection/data/forest_gain_tiles")
    plan.add_argument("--holdout-dir", default="../DataCollection/data/holdout_tiles"); plan.add_argument("--split-seed", type=int, default=0)
    plan.add_argument("--model-opt", nargs="+"); plan.add_argument("--loss", default="bce_focal_tversky")
    plan.add_argument("--loss-type", choices=("hard", "soft")); plan.add_argument("--selected-loss")
    plan.add_argument("--mask-stage", default="all", choices=("singles", "combos", "all")); plan.set_defaults(func=_plan)
    run = sub.add_parser("run", help="Run all manifest tasks (or one task)")
    run.add_argument("manifest"); run.add_argument("--task", type=int); run.add_argument("--cwd", default=".")
    run.add_argument("--log-dir", default="logs"); run.set_defaults(func=_run)
    select = sub.add_parser("select", help="Select top-k using validation metrics from W&B")
    select.add_argument("--project", required=True); select.add_argument("--group", required=True)
    select.add_argument("--metric", default="val_ap"); select.add_argument("--top-k", type=int, default=3)
    select.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); select.add_argument("--output", required=True); select.set_defaults(func=_select)
    confirm = sub.add_parser("confirm", help="Create top-k x seed retrain-and-holdout manifest")
    confirm.add_argument("selection"); confirm.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    confirm.add_argument("--group", required=True); confirm.add_argument("--output", required=True); confirm.set_defaults(func=_confirm)
    ev = sub.add_parser("evaluate", help="Evaluate an existing checkpoint without retraining")
    ev.add_argument("--checkpoint", required=True); ev.add_argument("--test-dir", default="../DataCollection/data/holdout_tiles")
    ev.add_argument("--cwd", default="."); ev.set_defaults(func=_evaluate)
    report = sub.add_parser("report", help="Export W&B runs to CSV")
    report.add_argument("--project", required=True); report.add_argument("--group", required=True)
    report.add_argument("--output", required=True); report.set_defaults(func=_report)
    args = p.parse_args(argv); args.func(args)

if __name__ == "__main__":
    main()
