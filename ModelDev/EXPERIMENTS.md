# Forest Gain experiment scripts

Run commands from `ModelDev`. `experiments.py` writes a JSON manifest and prints the generated `train.py` commands. Except for the adaptive benchmark, it does not train unless `--execute` is supplied. Review the manifest before launching experiments.

Files: `experiments.py` (suites), `train.py` (training), `datasets.py` (data + augmentation), `mask_sweep.py` (post-processing), `report.py` (seed-averaged results from Weights & Biases), `run_fg_benchmark.sh` (SLURM launcher and array for the parallel benchmarks), `confirm_top_k.py` + `run_fg_confirm.sh` (multi-seed confirmation of the global top benchmark trials, with holdout evaluation).

## Experiment tracking (Weights & Biases)

Every generated command carries labels so runs are named, grouped and reportable:

| Label | Set by | Meaning |
|---|---|---|
| `--wandb_group` | `experiments.py` (`--group` overrides) | one launch, default `<suite>_<timestamp>`; benchmark trials use the Optuna study name; confirmation runs use `confirm_<model><jobid>-...` |
| `--variant` | the suite (or auto-derived) | seed-independent label of a configuration, e.g. `repeats4`, `geo-only`, `trial07`, `confirm-tsvit-trial07` |
| `--suite` | `experiments.py` | W&B job type (`benchmark`, `losses`, ..., `confirm`) |
| `--wandb_tags` | `experiments.py` | suite and model |
| run name | `train.py` | `<variant>-s<seed>` (or the checkpoint run name if no variant) |

`train.py` logs the full configuration to W&B, defines `val_ap` with a **max summary** (the run summary holds the best epoch, not the last), and writes the W&B run id and checkpoint directory to `checkpoints/<run>/args.json`. `--wandb_project` overrides the project (default `Forest-Gain-CD`). In the W&B UI, group by `variant` to collapse seeds into mean and min/max curves.

Aggregate a launch over seeds:

```bash
python report.py --group augmentation_20261030_004812 --baseline baseline
python report.py --suite losses --out reports/losses
```

This writes `<out>.runs.csv` (one row per run), `<out>.csv` and `<out>.md` (per variant: n, mean, std, min, max, difference from the baseline, best run) and `<out>.png`. It reports validation AP only. Keep `--split-seed` fixed so seeds are the only variation within a variant. Confirmation runs (which also have holdout metrics) are aggregated by `confirm_top_k.py aggregate`, see section 1.

## 1. Model and optimiser benchmark

The benchmark uses **Optuna TPE Bayesian optimisation**, not a fixed Cartesian grid. It searches each architecture separately using S1 + S2 and the `bce_focal_tversky` loss, and maximises validation average precision (`val_ap`). `--trials` is the **total budget per model** and is shared across parallel workers (default 50; there is no upper cap). The budget is exact: a worker that claims a slot beyond `--trials` hands it back without training.

```bash
# Write the search plan only (default: 50 trials per model)
python experiments.py benchmark --dry-run

# Serial: one process, one in-memory study per model
python experiments.py benchmark --execute --trials 50
```

### Sampler

- **Random start-up:** the first `--startup-trials` trials are random (default: a third of `--trials`, between 5 and 24, so 16 for 50 trials). With parallel workers, trials that start before any result exists are effectively random too.
- **Multivariate TPE:** parameters are modelled jointly, so interactions such as "the best learning rate depends on model size" are captured.
- **Constant liar:** in-flight trials count as poor results, so concurrent workers don't pile onto the same region.
- Failed trials (`FAIL`) are ignored by TPE but still use up budget. Check the failure rate, especially out-of-memory failures on large TSViT configurations (`patch_size=8` with a large `embed_dim` and deep spatial depth): TPE cannot learn to avoid configs it never sees a score for.

### Running trials in parallel (SLURM): benchmark, then confirmation

Run `run_fg_benchmark.sh` **with bash** (not `sbatch`) from a login node, in the directory where you keep your other job scripts and `logs/`. One command launches every model and the confirmation:

```bash
bash run_fg_benchmark.sh
TRIALS=60 TOP_K=3 CONFIRM_SEEDS="0 1 2" bash run_fg_benchmark.sh
MODELS="tsvit sits_scd" bash run_fg_benchmark.sh              # a subset of models
```

| Step | Job | What it does |
|---|---|---|
| 1 | one `run_fg_benchmark.sh` array per model (`--array=0-5`, 6 workers each) | parallel TPE workers on one shared study per model (`TRIALS`, default 50, per model) |
| 2 | one `run_fg_confirm.sh` array, `--dependency=afterany` on all benchmark arrays | pools the trials of every model, takes the **global** top `TOP_K` (default 3) by validation AP, and retrains each for every seed in `CONFIRM_SEEDS` (default `0 1 2`): 9 runs in total, one GPU each, each scored on the holdout tiles |
| 3 | small CPU job, `--dependency=afterany` on step 2 | `confirm_top_k.py aggregate`: mean ± std over seeds into `reports/confirm_<...>.{md,json,runs.csv}` in `ModelDev` |

The global top 3 can be any mix of architectures (for example two SITS-SCD configurations and one TSViT). Comparing architectures this way is fair because every benchmark trial uses the same sources, loss, split and seed.

`afterany` means steps 2 and 3 still run if a benchmark worker hits the time limit. All tasks of one benchmark array share `SLURM_ARRAY_JOB_ID`, which names that model's study (`forestgain_<model>_<jobid>`) and output directory (`experiment_manifests/benchmark_<model>_<jobid>/`). Resume a model's study with `TAG_<model>=<old job id>`, for example `TAG_tsvit=123456 bash run_fg_benchmark.sh`: it adds trials until the study holds `TRIALS`.

`sbatch --export=ALL,MODEL=tsvit run_fg_benchmark.sh` still runs one model's benchmark array only, with no confirmation.

The matching flags in `experiments.py benchmark`:

| Flag | Meaning |
|---|---|
| `--trials` | Total trials per model, shared by all workers (default 50) |
| `--startup-trials` | Random trials before TPE models the objective (default `min(24, max(5, trials // 3))`) |
| `--study-name` | Shared Optuna study name; every worker must use the same one |
| `--storage-path` | Journal file on a shared filesystem; enables parallel workers. Needs exactly one model in `--models` |
| `--worker-id` | Worker index; offsets the sampler seed and names the output files |

Without `--storage-path` the benchmark runs serially with an in-memory study.

Notes:

- Past about 6 workers per study TPE gains little, because later trials start before earlier results come back. Raise `--trials` instead.
- The journal file needs a filesystem with working file locks. If you see lock errors, move `--storage-path` to another shared filesystem.
- Set the SLURM `--time` to roughly `ceil(trials / workers) × per-trial time` plus a buffer (50 trials / 6 workers = 9 trials each, about 18 h at 2 h per trial). If a task hits the time limit mid-trial, that trial stays "running" in the study forever; it counts toward the budget but is never selected.
- Each trial's full training configuration is stored in the study (`trial.user_attrs["config"]`), which is how the confirmation step reproduces it exactly.
- Each worker writes `benchmark_<model>_worker<NN>.json` and `benchmark_summary_worker<NN>_*.json` to its output directory. The authoritative record is the journal:

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

After a first batch, look at parameter importance (`optuna.importance.get_param_importances(study)`) and at whether the best trials sit on a range boundary. If so, widen the range and continue the same study.

### Confirmation and holdout (`confirm_top_k.py`)

Benchmark trials are single-seed and scored on the same validation tiles that drive early stopping, so the best of many trials is partly the luckiest. The confirmation step pools the completed trials of all benchmark studies, takes the global top `TOP_K` by validation AP, and retrains each for several seeds. With the defaults that is 3 configurations × 3 seeds = **9 training runs in total**, not per model.

- **The holdout is evaluated here, and only here.** Benchmark runs and every other suite pass `--skip_test_eval`. Confirmation runs do not, so `train.py` scores each run's best checkpoint on `--holdout_dir` and logs `test_*` metrics to W&B. Benchmark runs never saw the holdout, so seed 0 is retrained rather than reused.
- **Selection rule:** the final model is the configuration with the best **mean validation AP** over its seeds. Holdout numbers are reported next to it but should not be used to choose between the three, otherwise the holdout stops being held out. Once chosen, do not change the configuration after looking at holdout results.
- **Report:** `reports/confirm_<...>.md` ranks the configurations (model, n, validation AP and every `test_*` metric as mean ± std), names the selected one, and gives its learning rate, weight decay and architecture. `.runs.csv` has one row per run with the W&B URL and checkpoint directory; `.json` has the same data for scripts. Aggregation reads W&B, so it needs the runs to have finished and W&B access from the node.
- **Labels:** confirmation runs use W&B job type `confirm`, group `confirm_<model><jobid>-...` and variant `confirm-<model>-trialNN`.

Studies are passed as `MODEL,STUDY_NAME,JOURNAL_PATH`, one per architecture. To run it by hand, or to confirm benchmarks that are already finished:

```bash
S="tsvit,forestgain_tsvit_<id1>,experiment_manifests/benchmark_tsvit_<id1>/optuna_journal.log \
   sits_scd,forestgain_sits_scd_<id2>,experiment_manifests/benchmark_sits_scd_<id2>/optuna_journal.log \
   unet_lstm,forestgain_unet_lstm_<id3>,experiment_manifests/benchmark_unet_lstm_<id3>/optuna_journal.log"
python confirm_top_k.py select --studies $S --group confirm_manual
python confirm_top_k.py select --studies $S --group confirm_manual --execute      # train all 9 serially
export CONFIRM_STUDIES="$S" CONFIRM_GROUP=confirm_manual TOP_K=3 CONFIRM_SEEDS="0 1 2"
sbatch --export=ALL --array=0-8 run_fg_confirm.sh                                # or on SLURM
python confirm_top_k.py aggregate --group confirm_manual --out reports/confirm_manual
```

`select` prints the global top-k trials and the exact 9 `train.py` commands and writes `confirm_plan.json` to `experiment_manifests/<group>/`. `run --index N` trains the N-th planned run (ranked trial-major: index = rank × n_seeds + seed position); every SLURM task derives the same top-k from the finished studies, so the tasks agree. Options: `--top-k`, `--seeds`, `--group`, `--wandb-project`, `--output-dir`. For a study created before trial configs were stored, the config is rebuilt from the trial's parameters using `--epochs/--patience/--batch-size/--data-dir/--holdout-dir/--split-seed`, which must match the benchmark's values.

After choosing a winner, the checkpoints of its three runs are listed in the `.runs.csv` for later work. Run `mask_sweep.py` (section 7) on a selected checkpoint.

**Holdout protection:** `experiments.py` adds `--skip_test_eval` to every generated training command, and `train.py` supports that flag. This keeps the holdout untouched during tuning and ablations. The same applies to the data-source, loss, hardness, temporal and augmentation suites: if you want holdout numbers for their final choices, run the chosen configuration once per seed with `--skip_test_eval` off, as the confirmation step does. The Bayesian objective is validation AP, never holdout performance.

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

The benchmark accepts `--trials`, `--startup-trials`, `--models`, `--seed`, `--study-name`, `--storage-path`, `--worker-id`, `--output-dir`, `--epochs`, `--patience`, `--batch-size`, `--data-dir`, `--holdout-dir`, `--split-seed`, `--loss-type` and the W&B flags. It ignores `--learning-rate` and `--weight-decay` because those are searched, and it uses its own per-trial W&B group (the study name) instead of `--group`.

`train.py` accepts `--skip_test_eval` to defer holdout evaluation, `--model_opt KEY=VAL ...` for architecture options, and `--loss_opt KEY=VAL ...` for loss parameters. Defaults for model options live in `train.py`; loss defaults live in `losses/config.py`. Models may ignore options that do not apply to them.

## 7. Mask optimisation

Run after selecting a checkpoint:

```bash
python mask_sweep.py checkpoints/<run>/<best>.ckpt --output results/mask_sweep.csv
```

The sweep needs no retraining. The CSV covers thresholding at `0.50`–`0.90`, removal and filling of connected components sized `1`, `2`, `5`, `10`, and `25` pixels, and maximum predicted-gain coverage at minimum positive-mask precision of `55`–`80%`. In this rare-event setting, precision is the relevant interpretation of positive-mask "accuracy"; `pixel_accuracy` is reported separately for completeness.