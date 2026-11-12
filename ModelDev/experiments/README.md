# Manifest-driven experiment workflow (migration in progress)

Run commands from `ModelDev/` so `train.py` and the experiment package resolve consistently.
The existing `experiments.py` remains the source of truth for suite configuration generation during migration.

## Plan and run a suite

```bash
python -m experiments plan losses --stage defaults --model tsvit --seeds 0 1 2 \
  --group losses_tsvit_defaults --output experiment_manifests/losses-defaults.json
python -m experiments run experiment_manifests/losses-defaults.json
```

The initial loss screen includes BCE, Dice, Focal, BCE + Focal, Focal Tversky,
and BCE + Focal Tversky. Lovasz and OHEM are intentionally absent from this screen.

Other static legacy suites can be planned through the adapter (`data-sources`,
`hardness`, `temporal`, `augmentation`, and `maskclean`). Benchmark remains on
`experiments.py benchmark` because its Optuna search is adaptive rather than a
fixed list of commands.

## Slurm

```bash
experiments/slurm_submit.sh experiment_manifests/losses-defaults.json "$PWD" 18
```

The generic array worker uses each array index as a manifest task ID. Slurm output
and per-task training logs are written under `ModelDev/logs/`.

## Validation selection, retraining confirmation, and evaluation-only

Selection reads W&B runs and requires every requested sweep seed for a variant.
It ranks using validation metrics only:

```bash
python -m experiments select --project Forest-Gain-CD --group losses_tsvit_defaults \
  --metric val_ap --top-k 3 --seeds 0 1 2 --output experiment_manifests/losses-selection.json
```

Confirmation creates a separate manifest that retrains each selected configuration
for every requested confirmation seed and enables holdout evaluation during training:

```bash
python -m experiments confirm experiment_manifests/losses-selection.json \
  --seeds 10 11 12 --group losses_tsvit_confirm --output experiment_manifests/losses-confirm.json
python -m experiments run experiment_manifests/losses-confirm.json
```

For checkpoint-only evaluation (no retraining):

```bash
python -m experiments evaluate --checkpoint checkpoints/RUN/best.ckpt \
  --test-dir ../DataCollection/data/holdout_tiles
```

Reports export the W&B run configs and metrics to CSV:

```bash
python -m experiments report --project Forest-Gain-CD --group losses_tsvit_confirm \
  --output reports/losses-confirm.csv
```

This is an incremental migration. Review manifest diffs before submitting jobs;
keep holdout results out of selection decisions.
