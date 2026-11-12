"""Manifest-driven experiment framework for Forest Gain.

The framework is intentionally generic: each experiment emits a manifest that
contains all the task configs and commands, and the same CLI utilities can then
run, select, confirm, and evaluate those tasks.

The first implemented experiment is the focused loss screen, which compares the
six relevant loss recipes at their defaults and then lightly tunes the winner.
"""

from .common import (
    build_manifest,
    read_manifest,
    resolve_best_checkpoint,
    run_task,
    select_top_k,
    train_command,
    write_manifest,
)
from .losses import (
    DEFAULT_LOSSES,
    build_default_loss_manifest,
    build_tune_manifest,
    select_default_loss_winner,
)

__all__ = [
    "DEFAULT_LOSSES",
    "build_default_loss_manifest",
    "build_manifest",
    "build_tune_manifest",
    "read_manifest",
    "resolve_best_checkpoint",
    "run_task",
    "select_default_loss_winner",
    "select_top_k",
    "train_command",
    "write_manifest",
]
