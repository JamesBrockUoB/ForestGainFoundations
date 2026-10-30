# Forest Gain experiment scripts

Run commands from `ModelDev`. `experiments.py` writes a JSON manifest and prints the generated `train.py` commands. Except for the adaptive benchmark, it does not train unless `--execute` is supplied. Review the manifest before launching experiments.

Files: `experiments.py` (suites), `train.py` (training), `datasets.py` (data + augmentation), `mask_sweep.py` (post-processing), `report.py` (seed-averaged results from Weights & Biases), `run_fg_benchmark.sh` (SLURM array for the parallel benchmark).

## Experiment tracking (Weights & Biases)

Every generated command carries labels so runs are named, grouped and reportable:

| Label | Set by | Meaning |
|---|---|---|
| `--wandb_group` | `experiments.py` (`--group` overrides) | one launch, default `<suite>_<timestamp>`; benchmark trials use the Optuna study name |
| `--variant` | the suite (or auto-derived) | seed-independent label of a configuration, e.g. `repeats4`, `geo-only`, `trial07` |
| `--suite` | `experiments.py` | W&B job type |
| `--wandb_tags` | `experiments.py` | suite and model |
| run name | `train.py` | `<variant>-s<seed>` (or the checkpoint run name if no variant) |

`train.py` logs the full configuration to W&B, defines `val_ap` with a **max summary** (the run summary holds the best epoch, not the last), and writes the W&B run id and checkpoint directory to `checkpoints/<run>/args.json`. `--wandb_project` overrides the project (default `Forest-Gain-CD`). In the W&B UI, group by `variant` to collapse seeds into mean and min/max curves.

Aggregate a launch over seeds:

```bash
python report.py --group augmentation_20261030_004812 --baseline baseline
python report.py --suite losses --out reports/losses
```

This writes `<out>.runs.csv` (one row per run), `<out>.csv` and `<out>.md` (per variant: n, mean, std, min, max, difference from the baseline, best run) and `<out>.png`. It reports validation AP only. Keep `--split-seed` fixed so seeds are the only variation within a variant.

## 1. Model and optimiser benchmark

The benchmark uses **Optuna TPE Bayesian optimisation**, not a fixed Cartesian grid. It searches each architecture separately using S1 + S2 and maximises validation average precision (`val_ap`). `--trials` is the **total budget per model** (default 15, so 45 training runs across three models); choose 10–25 depending on compute budget and how much redundancy the early trials show.

```bash
# Write the search plan only (default: 15 trials per model, 45 total)
python experiments.py benchmark --dry-run

# Serial: one process, one in-memory study per model
python experiments.py benchmark --execute
python experiments.py benchmark --execute --trials 25
```

### Running trials in parallel (SLURM)

Parallel workers share one Optuna study stored in a journal file on a shared filesystem. Each worker asks the study for a trial, trains it, reports `val_ap`, and repeats until the total `--trials` budget has been claimed. Submit one array per model:

```bash
sbatch --export=ALL,MODEL=tsvit      run_fg_benchmark.sh
sbatch --export=ALL,MODEL=sits_scd   run_fg_benchmark.sh
sbatch --export=ALL,MODEL=unet_lstm  run_fg_benchmark.sh
sbatch --export=ALL,MODEL=tsvit,TRIALS=20 run_fg_benchmark.sh   # larger budget
```

`run_fg_benchmark.sh` is a SLURM array (`--array=0-5`: 6 workers per model, 18 concurrent jobs for all three models). All tasks of one submission share `SLURM_ARRAY_JOB_ID`, which names the study and the output directory (`experiment_manifests/benchmark_<model>_<jobid>/`). Resubmit with `STUDY_TAG=<old job id>` to resume the same study.

The matching flags in `experiments.py benchmark`:

| Flag | Meaning |
|---|---|
| `--study-name` | Shared Optuna study name; every worker must use the same one |
| `--storage-path` | Journal file on a shared filesystem; enables parallel workers. Needs exactly one model in `--models` |
| `--worker-id` | Worker index; offsets the sampler seed and names the output files |

Without `--storage-path` the benchmark runs serially with an in-memory study.

Notes:

- The first five trials per model are random (TPE start-up). With 6 workers they run at the same time, and TPE only adapts after they finish. In-flight trials are treated as poor results (`constant_liar`) so workers don't all explore the same region.
- Past about 6 workers per study TPE gains little, because later trials start before earlier results come back. Raise `--trials` instead.
- Workers can overshoot the budget by a trial or two if they ask at the same instant.
- The journal file needs a filesystem with working file locks. If you see lock errors, move `--storage-path` to another shared filesystem.
- Set the SLURM `--time` to roughly `ceil(trials / workers) × per-trial time` plus a buffer. If a task hits the time limit mid-trial, that trial stays "running" in the study and the other workers continue.
- Each worker writes `benchmark_<model>_worker<NN>.json` and `benchmark_summary_worker<NN>_*.json` to its output directory. The authoritative best trial is in the journal:

```python
import optuna
from experiments import _make_storage

study = optuna.load_study(
    study_name="forestgain_tsvit_<jobid>",
    storage=_make_storage("experiment_manifests/benchmark_tsvit_<jobid>/optuna_journal.log"),
)
print(study.best_trial.number + 1, study.best_value, study.best_params)
```

Each trial's W&B group is the study name and its variant is `trialNN`, where NN is the global trial number across workers.

### Search spaces

- **All models:** learning rate `3e-5`–`1e-3` (log scale), weight decay `1e-5`–`5e-2` (log scale), dropout `0`–`0.35`.
- **TSViT:** embedding dimension `{96,128,160}`, attention heads `{4,8}`, temporal depth `1`–`4`, spatial depth `2`–`6`, patch size `{8,16,32}`, skip channels `{16,24,32,48}`.
- **SITS_SCD:** base features `{32,48,64,96}`, attention heads `{2,4,8}`, temporal depth `1`–`4`, spatial depth `2`–`6`.
- **U-Net_LSTM:** base features `{24,32,48,64,96}`, skip channels `{16,24,32,48}`, temporal depth `1`–`3`.

This is a practical budgeted search, not a guarantee of finding the global optimum. Use the same fixed `--split-seed` across trials and compare the best candidates on additional seeds before declaring a winner.

**Holdout protection:** the updated `experiments.py` adds `--skip_test_eval` to every generated training command, and the updated `train.py` supports that flag. Replace your project's `train.py` with the included updated file before running suites. This keeps the holdout untouched during tuning and ablations. After choosing final checkpoints, run evaluation explicitly once on those checkpoints. The Bayesian objective is validation AP, never holdout performance.

## 2. Data-source ablation

Use one selected/frozen model and training configuration for all 12 source combinations:

```bash
python experiments.py data-sources --model tsvit --execute
```

The combinations are S1, S2, S1+S2, AlphaEarth, Tessera, S1+AlphaEarth, S2+AlphaEarth, S1+S2+AlphaEarth, S1+Tessera, S2+Tessera, S1+S2+Tessera, and S1+S2+AlphaEarth+Tessera. Only use this suite after selecting the model and optimiser settings from the benchmark. Confirm each product exists in the dataset pipeline.

## 3. Loss exploration

```bash
python experiments.py losses --model tsvit --sources s1 s2 --execute
```

The suite covers all nine requested families:

1. BCE
2. Dice
3. Focal Tversky
4. BCE + Focal Tversky
5. Focal
6. Lovasz hinge
7. BCE + Lovasz hinge
8. BCE + OHEM (and BCE + Focal Tversky + OHEM)
9. Focal + OHEM

Parameter exploration includes:

- Focal Tversky and BCE + Focal Tversky: `tversky.alpha` `{0.3,0.5,0.7}`, `beta = 1-alpha`, and `gamma` `{1,1.33,2}`.
- Focal: `gamma` `{1,2,3}` and `alpha` unset or `0.25`.
- Lovasz and BCE + Lovasz: pooled-batch and per-image reduction.
- OHEM: thresholds `0.7` and `0.9` for BCE-based recipes.
- Focal + OHEM: all combinations of focal gamma `{1,2,3}`, focal alpha unset or `0.25`, and OHEM threshold `{0.7,0.9}`.

**Focal + OHEM is implemented, not just listed as a command.** The Focal term calculates unreduced per-pixel focal losses and passes them to the configured reweighter. OHEM selects hard pixels using the model's confidence in the true class (threshold-based mmsegmentation-style selection), then the selected pixels' focal losses are reduced. `SegmentationLoss` handles the focal term's `(loss, auxiliary_loss, stats)` result just as it does BCE. Make sure the updated `terms.py` and loss package `__init__.py` are installed in the actual `losses/` package before running this suite.

The suite uses one fixed model, source set, data split, and unspecified training defaults. It is a structured comparison, not a full Bayesian optimisation of every loss parameter. Use multiple seeds for shortlisted loss configurations.

## 4. Pixel hardness

```bash
python experiments.py hardness --model tsvit --execute
```

The suite compares matched versions of BCE and BCE + Focal Tversky under:

- no reweighting;
- OHEM at thresholds `0.7` and `0.9`;
- learned hardness, sweeping `hardness.c` `{0.01,0.1}` and auxiliary loss scale `hardness.alpha` `{0.001,0.01,0.05}`.

The hardness branch mirrors the model's segmentation head on detached inputs, so the layer is the head module: TSViT `decoder` (a copy of the multi-conv decoder) and `decoder.final_head` (small default head on the fused features), SITS-SCD `final_head` and U-Net-LSTM `out_head` (small default head). `experiments.py` picks these per model (`HARDNESS_LAYERS`); override with `--hl-layer`, or pass `--hl_layer` to `train.py`. Run the suite once per model. That gives 12 (TSViT) or 6 (others) hardness configurations per loss family, plus 3 controls (no reweighting, OHEM 0.7, OHEM 0.9), i.e. 15 or 9 runs per loss family. The hardness branch consumes detached features, and the learned hardness weighting applies to the per-pixel BCE term; in BCE + Focal Tversky the Tversky term itself remains un-reweighted.

This is a parameter sweep rather than a full Bayesian search. After the initial sweep, compare the strongest hardness settings against the matched no-reweighting and OHEM controls over multiple seeds.

## 5. Temporal ablation

```bash
python experiments.py temporal --models tsvit sits_scd unet_lstm --seeds 0 1 2 --execute
```

Compares first+last year (2017, 2024), 2017/2021/2024, 2017/2020/2022/2024, and the whole sequence (`--year-sets` overrides). The valid-pixel mask is taken over all years so the scored pixels stay consistent across frame subsets.

## 6. Augmentation and sampling

```bash
python experiments.py augmentation --model tsvit --seeds 0 1 2 --execute
python experiments.py augmentation --loss-type soft --execute   # also sweeps soft-label jitter
```

Model, sources, loss, learning rate and weight decay are fixed. The baseline is `train.py`'s defaults: random 90-degree rotation and flip, S2 brightness jitter (`gain_sigma` 0.03, `frame_sigma` 0.01), no frame dropout, full tile, `repeats` 1. The suite (14 runs per seed, 18 with `--loss-type soft`) covers:

- **Augmentation family:** all off (`--no_augment`), geometric only (`--gain_sigma 0 --frame_sigma 0`), spectral only (`--no_geo_aug`).
- **Spectral strength:** `gain_sigma`/`frame_sigma` of 0.1/0.03 and 0.2/0.05.
- **Interior-frame dropout:** `frame_drop_p` 0.1 and 0.25.
- **Repeats:** 2, 4, 8, plus 4 with no augmentation as a control (identical duplicate samples).
- **Random training crop** (validation stays full tile): 128 and 192 px, each with 4 repeats. The crop must be a multiple of the model's patch size, and `train.py` builds the model with `img_size=crop_size` while validation uses full tiles, so check that your model tolerates that.
- **Soft-label jitter** `label_sigma` 0, 0.05, 0.1, 0.2 (soft loss only).

**Compute matching:** `repeats` multiplies optimiser steps per epoch. By default epochs and patience are divided by `repeats` (`--match-steps`; use `--no-match-steps` to disable) so every setting gets the same number of steps and differs only in sample diversity.

New `train.py` flags: `--gain_sigma`, `--frame_sigma`, `--frame_drop_p`, `--label_sigma` (default 0.1 with the soft loss, else 0), `--no_geo_aug`.

## Common controls

All non-benchmark suites accept:

- `--dry-run`: write and print a manifest without training (default if `--execute` is omitted).
- `--execute`: execute generated commands.
- `--max-runs N`: cap runs for a smoke test.
- `--output-dir DIR`: manifest directory (default `experiment_manifests`).
- `--seeds 0 1 2`: repeat configurations for each seed.
- `--loss-type hard|soft`: pass the selected loss type to each run.
- `--group NAME`: W&B group for the launch (default `<suite>_<timestamp>`); `--wandb-project NAME`: W&B project.
- `--match-steps` / `--no-match-steps`: augmentation suite only, see section 6.
- `--epochs`, `--patience`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--data-dir`, `--holdout-dir`, `--split-seed`.

The benchmark accepts `--trials`, `--models`, `--seed`, `--study-name`, `--storage-path`, `--worker-id`, `--output-dir`, `--epochs`, `--patience`, `--batch-size`, `--data-dir`, `--holdout-dir`, `--split-seed`, `--loss-type` and the W&B flags. It ignores `--learning-rate` and `--weight-decay` because those are searched, and it uses its own per-trial W&B group (the study name) instead of `--group`.

The updated `train.py` accepts `--skip_test_eval` to defer holdout evaluation, `--model_opt KEY=VAL ...` for architecture options, and `--loss_opt KEY=VAL ...` for loss parameters. Defaults for model options live in `train.py`; loss defaults live in `losses/config.py`. Models may ignore options that do not apply to them.

## 7. Mask optimisation

Run after selecting a checkpoint:

```bash
python mask_sweep.py checkpoints/<run>/<best>.ckpt --output results/mask_sweep.csv
```

The sweep needs no retraining. The CSV covers thresholding at `0.50`–`0.90`, removal and filling of connected components sized `1`, `2`, `5`, `10`, and `25` pixels, and maximum predicted-gain coverage at minimum positive-mask precision of `55`–`80%`. In this rare-event setting, precision is the relevant interpretation of positive-mask "accuracy"; `pixel_accuracy` is reported separately for completeness.