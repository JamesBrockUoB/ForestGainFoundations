"""Shared manifest, execution, selection, and checkpoint utilities."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def train_command(config: dict[str, Any]) -> list[str]:
    """Translate a train.py config into argv without invoking a shell."""
    command = [sys.executable, "train.py"]
    for key, value in config.items():
        flag = "--loss" if key == "loss_name" else "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            command.append(flag)
        elif isinstance(value, (tuple, list)):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    return command


def build_manifest(rows: Iterable[dict[str, Any]], suite: str, group: str,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    tasks = []
    for index, row in enumerate(rows):
        config = dict(row)
        tasks.append({"task_id": index, "config": config, "command": train_command(config)})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite": suite,
        "group": group,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
    }
    if metadata:
        manifest["metadata"] = metadata
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
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
    log_path = logs / f"{manifest['suite']}-{manifest['group']}-task{task_index:05d}.log"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND: {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nEXIT_CODE: {result.returncode}\n")
    print(f"Task log: {log_path}", flush=True)
    return result.returncode


def _metric(run: dict[str, Any], metric: str) -> float | None:
    summary = run.get("summary", {})
    value = summary.get(metric)
    if value is None:
        value = summary.get(f"{metric}.max")
    return float(value) if value is not None else None


def select_top_k(runs: list[dict[str, Any]], metric: str = "val_ap", top_k: int = 3,
                 required_seeds: Iterable[int] = (0, 1, 2)) -> list[dict[str, Any]]:
    """Select variants only when every required seed has a completed metric."""
    seeds = tuple(int(seed) for seed in required_seeds)
    if not seeds or top_k < 1:
        raise ValueError("required_seeds and top_k must be non-empty/positive")
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for run in runs:
        cfg = run.get("config", {})
        variant, seed = cfg.get("variant"), cfg.get("seed")
        if not variant or seed is None or run.get("state", "finished") != "finished":
            continue
        value = _metric(run, metric)
        if value is None:
            continue
        grouped.setdefault(str(variant), {})[int(seed)] = {"run": run, "value": value}

    ranked = []
    for variant, per_seed in grouped.items():
        if not set(seeds).issubset(per_seed):
            continue
        values = {seed: per_seed[seed]["value"] for seed in seeds}
        ranked.append({
            "variant": variant,
            "mean": sum(values.values()) / len(seeds),
            "per_seed": values,
            "runs": [per_seed[seed]["run"] for seed in seeds],
        })
    ranked.sort(key=lambda item: (-item["mean"], item["variant"]))
    return ranked[:top_k]


def wandb_runs(project: str, group: str) -> list[dict[str, Any]]:
    """Fetch run metadata needed for validation-only selection."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B selection requires the wandb package") from exc
    api = wandb.Api()
    entity = api.default_entity
    result = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        result.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "url": run.url,
        })
    return result


def resolve_checkpoint(run: dict[str, Any], modeldev_dir: str | Path) -> Path:
    """Find the checkpoint belonging to a W&B run using args.json run IDs."""
    root = Path(modeldev_dir).resolve()
    run_id = str(run.get("id", ""))
    config = run.get("config", {})
    configured = config.get("checkpoint_dir")
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        candidates.append(p if p.is_absolute() else root / p)
    for args_file in (root / "checkpoints").glob("**/args.json"):
        try:
            stored = json.loads(args_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(stored.get("wandb_run_id", "")) == run_id:
            candidates.append(args_file.parent)
    for directory in candidates:
        if not directory.is_dir():
            continue
        checkpoints = [p for p in directory.glob("*.ckpt") if p.name != "last.ckpt"]
        if checkpoints:
            def score(path: Path) -> tuple[float, float]:
                match = re.search(r"val_ap[=:_-]([0-9]+(?:\.[0-9]+)?)", path.name)
                return (float(match.group(1)) if match else -1.0, path.stat().st_mtime)
            return max(checkpoints, key=score)
    raise FileNotFoundError(f"No checkpoint found for W&B run {run_id} under {root / 'checkpoints'}")
