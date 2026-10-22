"""Build and run reproducible Forest Gain experiment suites.

Examples
--------
python experiments.py benchmark --dry-run
python experiments.py data-sources --model tsvit --execute
python experiments.py losses --model tsvit --execute --max-runs 12

Each invocation writes a JSON manifest containing exact ``train.py`` commands.
Review the manifest first; use ``--execute`` only when ready to train.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SETS = (
    ("s1",), ("s2",), ("s1", "s2"), ("alphaearth",), ("tessera",),
    ("s1", "alphaearth"), ("s2", "alphaearth"),
    ("s1", "s2", "alphaearth"), ("s1", "tessera"),
    ("s2", "tessera"), ("s1", "s2", "tessera"),
    ("s1", "s2", "alphaearth", "tessera"),
)


def _base(args: argparse.Namespace, model: str, sources: tuple[str, ...]) -> dict:
    return {
        "model_type": model, "sources": list(sources), "epochs": args.epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "data_dir": args.data_dir, "holdout_dir": args.holdout_dir,
        "split_seed": args.split_seed, "seed": args.seed,
    }


def benchmark(args: argparse.Namespace) -> list[dict]:
    """Small structured search; S1/S2 only, as requested."""
    rows = []
    for model in ("tsvit", "sits_scd", "unet_lstm"):
        for lr, wd in itertools.product((1e-4, 3e-4), (1e-4, 1e-2)):
            if model == "tsvit":
                structures = (
                    {"embed_dim": 96, "num_heads": 4, "depth_temporal": 2,
                     "depth_spatial": 3, "patch_size": 16, "dropout": 0.1, "skip_channels": 24},
                    {"embed_dim": 128, "num_heads": 4, "depth_temporal": 2,
                     "depth_spatial": 4, "patch_size": 16, "dropout": 0.1, "skip_channels": 32},
                )
            elif model == "sits_scd":
                structures = ({"base_features": 48, "num_heads": 4}, {"base_features": 64, "num_heads": 8})
            else:
                structures = ({"base_features": 48}, {"base_features": 64})
            for structure in structures:
                rows.append(_base(args, model, ("s1", "s2")) | {
                    "learning_rate": lr, "weight_decay": wd, **structure
                })
    return rows


def data_sources(args: argparse.Namespace) -> list[dict]:
    return [_base(args, args.model, sources) | {"learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay} for sources in SOURCE_SETS]


def losses(args: argparse.Namespace) -> list[dict]:
    rows = []
    for name in ("bce", "dice", "focal_tversky", "bce_focal_tversky"):
        if name == "focal_tversky":
            params = itertools.product((0.3, 0.5, 0.7), (1.0, 1.33, 2.0))
        else:
            params = ((0.5, 1.0),)
        for alpha, gamma in params:
            rows.append(_base(args, args.model, tuple(args.sources)) | {
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "loss_name": name, "tversky_alpha": alpha,
                "tversky_beta": 1 - alpha, "tversky_gamma": gamma,
            })
    return rows


def command(row: dict) -> list[str]:
    cmd = [sys.executable, "train.py"]
    for key, value in row.items():
        flag = "--loss" if key == "loss_name" else "--" + key
        if key == "sources":
            cmd += [flag, *value]
        else:
            cmd += [flag, str(value)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=("benchmark", "data-sources", "losses"))
    p.add_argument("--execute", action="store_true", help="Run commands (default only writes manifest).")
    p.add_argument("--dry-run", action="store_true", help="Print commands as well as writing manifest.")
    p.add_argument("--max-runs", type=int, default=0, help="Optional cap for smoke tests / staged sweeps.")
    p.add_argument("--output-dir", default="experiment_manifests")
    p.add_argument("--model", default="tsvit", choices=("tsvit", "sits_scd", "unet_lstm"))
    p.add_argument("--sources", nargs="+", default=["s1", "s2"], choices=("s1", "s2", "alphaearth", "tessera"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--data-dir", default="../DataCollection/data/test_tiles")
    p.add_argument("--holdout-dir", default="../DataCollection/data/holdout_tiles")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = {"benchmark": benchmark, "data-sources": data_sources, "losses": losses}[args.suite](args)
    if args.max_runs:
        rows = rows[:args.max_runs]
    manifest = {"suite": args.suite, "created_at": datetime.now(timezone.utc).isoformat(),
                "runs": [{"config": r, "command": command(r)} for r in rows]}
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
