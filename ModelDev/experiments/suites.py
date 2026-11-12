"""Adapters that reuse the established experiment definitions while migration proceeds."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
from typing import Any
from .common import build_manifest

SUITES = ("benchmark", "data-sources", "losses", "hardness", "temporal", "augmentation", "maskclean")


def _legacy_module():
    path = Path(__file__).resolve().parents[1] / "experiments.py"
    spec = importlib.util.spec_from_file_location("_legacy_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy experiment definitions at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_plan(suite: str, *, model="tsvit", sources=("s1", "s2"), seeds=(0, 1, 2),
                group=None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build static plans through the existing suite generators for migration parity.

    Benchmark's adaptive Optuna search is not static and should continue to use its
    existing benchmark command until a dedicated study adapter is used.
    """
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite}; choose from {SUITES}")
    if suite == "benchmark":
        raise ValueError("Benchmark uses adaptive Optuna; use experiments.py benchmark until its study adapter is migrated")
    old = _legacy_module()
    overrides = dict(overrides or {})
    group = group or f"{suite}_manifest"
    ns = argparse.Namespace(
        suite=suite, model=model, models=[model], sources=list(sources), group=group,
        epochs=overrides.pop("epochs", 100), patience=overrides.pop("patience", 20),
        batch_size=overrides.pop("batch_size", 8), data_dir=overrides.pop("data_dir", "../DataCollection/data/forest_gain_tiles"),
        holdout_dir=overrides.pop("holdout_dir", "../DataCollection/data/holdout_tiles"),
        split_seed=overrides.pop("split_seed", 0), eval_holdout=False,
        learning_rate=overrides.pop("learning_rate", 3e-4), weight_decay=overrides.pop("weight_decay", 1e-2),
        model_opt=overrides.pop("model_opt", None), loss=overrides.pop("loss", "bce_focal_tversky"),
        loss_type=overrides.pop("loss_type", None), loss_stage=overrides.pop("loss_stage", "base"),
        carry_from=overrides.pop("carry_from", None), carry_loss=overrides.pop("carry_loss", None),
        carry_opt=overrides.pop("carry_opt", None), hl_layer=overrides.pop("hl_layer", None),
        year_sets=overrides.pop("year_sets", None), match_steps=overrides.pop("match_steps", True),
        mask_stage=overrides.pop("mask_stage", "all"), mask_sizes=overrides.pop("mask_sizes", [2, 5, 10, 25]),
        mask_confs=overrides.pop("mask_confs", [round(.5 + .05*i, 2) for i in range(9)]),
        mask_keep_noop=overrides.pop("mask_keep_noop", False), mask_clean_val=False,
        mask_label_sigma=overrides.pop("mask_label_sigma", 0.0),
        combo_confs=overrides.pop("combo_confs", [.6, .7, .8]),
        combo_morph=overrides.pop("combo_morph", [(5, 5), (10, 10)]),
        source_index=None, top_sources_from=None, top_k=3,
    )
    if overrides:
        raise ValueError(f"Unsupported planning options: {sorted(overrides)}")
    rows = getattr(old, suite)(ns)
    expanded = []
    for seed in seeds:
        for row in rows:
            expanded.append(dict(row) | {"seed": seed})
    return build_manifest(expanded, suite, group, {"source": "legacy-adapter", "seed_list": list(seeds)})
