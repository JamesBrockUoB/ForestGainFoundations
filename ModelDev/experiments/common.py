from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def train_command(cfg: dict[str, Any]) -> list[str]:
    """Build a train.py invocation from a config dictionary."""
    cmd = [sys.executable, "train.py"]
    for key, value in cfg.items():
        flag = "--loss" if key == "loss_name" else "--" + key
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        elif isinstance(value, (list, tuple)):
            cmd += [flag, *map(str, value)]
        else:
            cmd += [flag, str(value)]
    return cmd


def build_manifest(rows: Iterable[dict[str, Any]], suite: str, group: str | None = None) -> dict[str, Any]:
    """Create a manifest for a list of task configs."""
    return {
        "suite": suite,
        "group": group,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"config": row, "command": train_command(row)}
            for row in rows
        ],
    }


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def run_task(task: dict[str, Any], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        task["command"],
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def _val_ap_from_ckpt_name(name: str) -> float:
    m = re.search(r"val_ap[:=]([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        return -1e9
    return float(m.group(1))


def resolve_best_checkpoint(checkpoint_dir: str | Path, run_id: str | None = None) -> Path:
    """Resolve the best *.ckpt in a checkpoint directory."""
    candidate_dir = Path(checkpoint_dir)
    if not candidate_dir.is_absolute():
        candidate_dir = Path(os.getcwd()) / candidate_dir

    candidates = list(candidate_dir.glob("*.ckpt"))
    if not candidates:
        if run_id:
            for args_file in Path(os.getcwd()).glob("checkpoints/**/args.json"):
                try:
                    args = json.loads(args_file.read_text())
                except Exception:
                    continue
                if str(args.get("wandb_run_id")) == str(run_id):
                    return resolve_best_checkpoint(args_file.parent, run_id)
        raise FileNotFoundError(f"No checkpoint files in {candidate_dir}")

    ckpts = [p for p in candidates if p.name != "last.ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"No best non-last checkpoint found in {candidate_dir}")
    return max(ckpts, key=_val_ap_from_ckpt_name)


def select_top_k(
    runs: list[dict[str, Any]],
    metric: str = "val_ap",
    top_k: int = 3,
    required_seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, Any]]:
    """Rank completed runs by mean validation AP over required seeds."""
    by_variant: dict[str, list[dict[str, Any]]] = {}

    for run in runs:
        cfg = run.get("config", {})
        variant = cfg.get("variant")
        seed = cfg.get("seed")
        if not variant or seed is None:
            continue
        by_variant.setdefault(variant, []).append(run)

    ranked = []
    for variant, rows in by_variant.items():
        seed_map = {}
        for row in rows:
            cfg = row.get("config", {})
            seed = cfg.get("seed")
            if seed is None:
                continue
            summary = row.get("summary", {})
            value = summary.get(metric)
            if value is None:
                value = summary.get(f"{metric}.max")
            if value is None:
                value = row.get(metric)
            if value is None:
                continue
            seed_map[int(seed)] = float(value)

        if not set(required_seeds).issubset(seed_map.keys()):
            continue

        mean = sum(seed_map[s] for s in required_seeds) / len(required_seeds)
        ranked.append(
            {
                "variant": variant,
                "mean": mean,
                "seeds": {s: seed_map[s] for s in required_seeds},
                "runs": rows,
            }
        )

    ranked.sort(key=lambda x: (-x["mean"], x["variant"]))
    return ranked[:top_k]
