# Manifest-driven experiment runner

Run from `ModelDev/`. Every experiment suite has a native builder under `experiments/`; no adapter to `experiments.py` is used. Each plan is a frozen JSON manifest with an exact argument vector per task and a SHA-256 digest.

## Plan and run

```bash
python -m experiments plan losses --stage defaults --model tsvit --seeds 0 1 2 --group losses-tsvit --output experiment_manifests/losses-tsvit.json
python -m experiments run experiment_manifests/losses-tsvit.json

# generic Slurm array; output and task logs go under ModelDev/logs
experiments/slurm_submit.sh experiment_manifests/losses-tsvit.json "$PWD" 18
```

Supported suites: `benchmark`, `data-sources`, `losses`, `hardness`, `temporal`, `augmentation`, and `maskclean`. `benchmark` is a deterministic seeded-random search plan (search seed is recorded), not adaptive TPE. Losses has `defaults` and `tune` stages; maskclean has `singles`, `combos`, and `all` stages. Inspect the printed task count and manifest before submission.

## Select and confirm

Selection reads W&B validation summaries only and requires exactly one finished result per requested seed; incomplete or duplicate-seed variants are excluded.

```bash
python -m experiments select --project Forest-Gain-CD --group losses-tsvit --metric val_ap --top-k 3 --seeds 0 1 2 --output experiment_manifests/selection.json
python -m experiments confirm experiment_manifests/selection.json --seeds 10 11 12 --group losses-confirm --output experiment_manifests/confirm.json
python -m experiments run experiment_manifests/confirm.json
```

Confirmation retrains the selected configurations and turns holdout evaluation on. Never select using holdout metrics.

## Evaluation and report

```bash
python -m experiments evaluate --checkpoint checkpoints/RUN/best.ckpt --test-dir ../DataCollection/data/holdout_tiles
python -m experiments report --project Forest-Gain-CD --group losses-confirm --output reports/losses-confirm.csv
```

`evaluate` scores an existing checkpoint without retraining. `confirm` is the separate top-k retrain-and-holdout procedure.
