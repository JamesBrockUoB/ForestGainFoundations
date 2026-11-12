# Manifest-driven experiment runner

Run commands from `ModelDev/`. The experiment package has native suite builders; it does not call or wrap `experiments.py`. Static sweeps produce frozen JSON manifests with exact `train.py` argument vectors, stable task indices, seed/split metadata, and a digest. Adaptive benchmark search is run through Optuna TPE directly and is not incorrectly flattened into a precomputed random grid.

## Experiment suites

- **Benchmark (Optuna TPE):** optimizer and architecture search for TSViT, SITS-SCD, or U-Net-LSTM using S1+S2 and BCE + focal Tversky. The objective is mean validation AP over the requested search seeds. TPE is multivariate, uses a journal storage backend for shared parallel workers, and records trial configs and per-seed outcomes. Each model uses a separate study. Holdout is not used in search.
- **Data sources:** compares the 15 configured S1, S2, AlphaEarth, and Tessera source combinations under a fixed model/loss/optimizer recipe.
- **Losses:** six default loss recipes (BCE, Dice, Focal, BCE + Focal, Focal Tversky, BCE + Focal Tversky) followed by a small one-factor-at-a-time tune plan for the selected recipe. The broader loss/reweight exploration is represented by the hardness suite and can be extended with explicit stages.
- **Hardness:** matched BCE-based controls for no reweighting, OHEM thresholds, and learned pixel-hardness settings/layers.
- **Temporal:** compares the two-frame, three-frame, four-frame, and full-sequence inputs on the same split and scored pixels.
- **Augmentation:** compares augmentation families/strengths, frame dropout, repeats, crops, and soft-label jitter where applicable. Repeats can be compute-matched by scaling epochs and patience.
- **Mask cleaning:** baseline, individual component removal/hole filling/confidence filter/hardening, and configured morphology/confidence combinations. Raw validation labels are used by default; `--clean-val` is an explicit alternate target. Holdout is skipped during sweeps.

Review generated plans and task counts before submitting jobs. `--split-seed` fixes the train/validation tile split independently of model seed. The holdout directory is carried in configs but is never evaluated for sweeps (`skip_test_eval=True`).

## Plan and run static suites

```bash
python -m experiments plan losses --stage defaults --model tsvit --seeds 0 1 2 \
  --group losses-tsvit --output experiment_manifests/losses-tsvit.json
python -m experiments run experiment_manifests/losses-tsvit.json

# Generic Slurm array; output and per-task logs go under ModelDev/logs.
experiments/slurm_submit.sh experiment_manifests/losses-tsvit.json "$PWD" 18
```

Other static suites use the same `plan` and `run` commands, e.g. `plan data-sources`, `plan hardness`, `plan temporal`, `plan augmentation`, or `plan maskclean --stage singles|combos|all`. Their shared options include model, sources, seeds, training limits, optimizer settings, data directories, and split seed. Suite-specific values are recorded in the manifest.

## Run adaptive benchmark TPE

```bash
python -m experiments benchmark --models tsvit --trials 25 \
  --search-seeds 0 1 2 --group fg-tsvit --study-name fg-tsvit \
  --storage-path experiment_manifests/fg-tsvit.journal
```

For shared-filesystem parallel workers, launch workers with the same study name, journal path, model, and trial budget, but distinct `--worker-id` values and preferably distinct `--sampler-seed` values. TPE's proposals depend on results arriving, so the exact future config list is inherently adaptive rather than a fixed task manifest; the Optuna journal, recorded trial configs, and worker JSON files are the reproducibility record. Use one study per model.

## Select and confirm

Selection queries W&B and uses validation metrics only. A variant is eligible only when it has exactly one finished result with a metric for every requested seed; incomplete or duplicate-seed variants are excluded.

```bash
python -m experiments select --project Forest-Gain-CD --group losses-tsvit \
  --metric val_ap --top-k 3 --seeds 0 1 2 \
  --output experiment_manifests/losses-selection.json
python -m experiments confirm experiment_manifests/losses-selection.json \
  --seeds 10 11 12 --group losses-confirm --output experiment_manifests/losses-confirm.json
python -m experiments run experiment_manifests/losses-confirm.json
```

Confirmation reconstructs the selected training configs, retrains each top-K choice for each confirmation seed, and enables holdout evaluation only for those confirmation runs. Never select using holdout metrics.

## Evaluation and reporting

```bash
# Score a checkpoint already on disk; no training occurs.
python -m experiments evaluate --checkpoint checkpoints/RUN/best.ckpt \
  --test-dir ../DataCollection/data/holdout_tiles
python -m experiments report --project Forest-Gain-CD --group losses-confirm \
  --output reports/losses-confirm.csv
```

`evaluate` supports raw or explicitly cleaned labels. Reports export run identity, variant/seed, and validation/holdout metrics. Keep logs in `ModelDev/logs/`; Slurm stdout/stderr and the task runner both record task-specific output there.
