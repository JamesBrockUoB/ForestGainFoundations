"""Shared manifest, command, execution, selection, and checkpoint utilities."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Exactly the public training CLI options. Internal metadata must never leak into argv.
TRAIN_KEYS = {
    "model_type", "model_opt", "loss_name", "loss_type", "pos_weight", "reweight", "hl_layer", "loss_opt",
    "batch_size", "learning_rate", "weight_decay", "warmup_frac", "epochs", "patience", "seed",
    "data_dir", "holdout_dir", "val_frac", "split_seed", "sources", "years", "crop_size", "repeats",
    "min_gain_px", "fill_holes_px", "conf_filter", "conf_harden", "clean_val", "no_augment", "no_geo_aug",
    "gain_sigma", "frame_sigma", "frame_drop_p", "label_sigma", "val_metrics", "skip_test_eval",
    "wandb_project", "wandb_name", "wandb_group", "wandb_tags", "suite", "variant",
}


def train_command(config: dict[str, Any]) -> list[str]:
    """Translate training config to argv, retaining train.py's underscore option names."""
    command = [sys.executable, "train.py"]
    for key, value in config.items():
        if key not in TRAIN_KEYS or value is None or value is False:
            continue
        flag = "--loss" if key == "loss_name" else "--" + key
        if value is True:
            command.append(flag)
        elif isinstance(value, (tuple, list)):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    return command


def build_manifest(rows: Iterable[dict[str, Any]], suite: str, group: str,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    tasks = [{"task_id": i, "config": dict(row), "command": train_command(dict(row))}
             for i, row in enumerate(rows)]
    manifest: dict[str, Any] = {
        "schema_version": 1, "suite": suite, "group": group,
        "created_at": datetime.now(timezone.utc).isoformat(), "tasks": tasks,
    }
    if metadata:
        manifest["metadata"] = metadata
    digest_input = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(digest_input).hexdigest()
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("tasks"), list):
        raise ValueError(f"Unsupported or invalid manifest: {path}")
    for i, task in enumerate(manifest["tasks"]):
        if task.get("task_id") != i or not isinstance(task.get("command"), list):
            raise ValueError(f"Invalid task index or command at task {i}")
    return manifest


def run_task(manifest_path: str | Path, task_index: int, cwd: str | Path,
             log_dir: str | Path | None = None) -> int:
    manifest = read_manifest(manifest_path)
    if not 0 <= task_index < len(manifest["tasks"]):
        raise IndexError(f"task index {task_index} outside 0..{len(manifest['tasks']) - 1}")
    task = manifest["tasks"][task_index]
    command = task["command"]
    print(f"[{task_index + 1}/{len(manifest['tasks'])}] {' '.join(command)}", flush=True)
    if log_dir is None:
        return subprocess.run(command, cwd=cwd, check=False).returncode
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{manifest['suite']}-{manifest['group']}-task{task_index:05d}.log"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND: {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nEXIT_CODE: {result.returncode}\n")
    print(f"Task log: {path}", flush=True)
    return result.returncode


def _metric(run: dict[str, Any], metric: str) -> float | None:
    summary = run.get("summary", {})
    value = summary.get(metric, summary.get(f"{metric}.max"))
    return float(value) if value is not None else None


def select_top_k(runs: list[dict[str, Any]], metric: str = "val_ap", top_k: int = 3,
                 required_seeds: Iterable[int] = (0, 1, 2)) -> list[dict[str, Any]]:
    """Select only variants with exactly one completed result for each requested seed."""
    seeds = tuple(int(s) for s in required_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or top_k < 1:
        raise ValueError("required_seeds must be unique/non-empty and top_k positive")
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for run in runs:
        cfg = run.get("config", {})
        variant, seed = cfg.get("variant"), cfg.get("seed")
        if not variant or seed is None or run.get("state", "finished") != "finished":
            continue
        value = _metric(run, metric)
        if value is None:
            continue
        per_seed = grouped.setdefault(str(variant), {})
        # Duplicate seed runs make a variant ambiguous; exclude it instead of silently choosing one.
        seed = int(seed)
        if seed in per_seed:
            per_seed[seed] = {"duplicate": True}
        else:
            per_seed[seed] = {"run": run, "value": value}
    ranked = []
    for variant, values in grouped.items():
        if not set(seeds).issubset(values) or any(values[s].get("duplicate") for s in seeds):
            continue
        metrics = {seed: values[seed]["value"] for seed in seeds}
        ranked.append({"variant": variant, "mean": sum(metrics.values()) / len(seeds),
                       "per_seed": metrics, "runs": [values[s]["run"] for s in seeds]})
    ranked.sort(key=lambda row: (-row["mean"], row["variant"]))
    return ranked[:top_k]


def wandb_runs(project: str, group: str) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B operations require the wandb package") from exc
    api = wandb.Api()
    entity = api.default_entity
    return [{"id": r.id, "name": r.name, "state": r.state, "config": dict(r.config),
             "summary": dict(r.summary), "url": r.url}
            for r in api.runs(f"{entity}/{project}", filters={"group": group})]


def resolve_checkpoint(run: dict[str, Any], modeldev_dir: str | Path) -> Path:
    """Resolve this run's best checkpoint by W&B run ID recorded in checkpoints/*/args.json."""
    root = Path(modeldev_dir).resolve()
    run_id = str(run.get("id", ""))
    configured = run.get("config", {}).get("checkpoint_dir")
    dirs = []
    if configured:
        p = Path(configured)
        dirs.append(p if p.is_absolute() else root / p)
    for args_file in (root / "checkpoints").glob("**/args.json"):
        try:
            stored = json.loads(args_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(stored.get("wandb_run_id", "")) == run_id:
            dirs.append(args_file.parent)
    for directory in dirs:
        checkpoints = [p for p in directory.glob("*.ckpt") if p.name != "last.ckpt"] if directory.is_dir() else []
        if checkpoints:
            def key(path: Path):
                match = re.search(r"val_ap[=:_-]([0-9]+(?:\.[0-9]+)?)", path.name)
                return (float(match.group(1)) if match else -1.0, path.stat().st_mtime)
            return max(checkpoints, key=key)
    raise FileNotFoundError(f"No checkpoint found for W&B run {run_id} below {root / 'checkpoints'}")
