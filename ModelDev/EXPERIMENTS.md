# Forest Gain experiment scripts

Run commands from `ModelDev`. `experiments.py` writes a JSON manifest and prints the generated `train.py` commands. Except for the adaptive benchmark, it does not train unless `--execute` is supplied. Review the manifest before launching experiments.

Files: `experiments.py` (suites), `train.py` (training), `datasets.py` (data + augmentation), `mask_cleaning.py` (label-mask cleaning used in training), `report.py` (seed-averaged results from Weights & Biases), `run_fg_benchmark.sh` (SLURM launcher and arrays for the parallel benchmarks and the confirmation), `confirm_top_k.py` + `run_fg_confirm.sh` (multi-seed confirmation of the best benchmark trials, with holdout evaluation), `run_fg_data_sources.sh` (data-source sweep and confirmation of the top sources, with the best benchmark configuration of a model).

## Experiment tracking (Weights & Biases)

Every generated command carries labels so runs are named, grouped and reportable:

| Label | Set by | Meaning |
|---|---|---|
| `--wandb_group` | `experiments.py` (`--group` overrides) | one launch, default `<suite>_<timestamp>`; benchmark trials use the Optuna study name; confirmation runs use the `CONFIRM_GROUP` name |
| `--variant` | the suite (or auto-derived) | seed-independent label of a configuration, e.g. `repeats4`, `geo-only`, `trial07`, `confirm-tsvit-trial07` |
| `--suite` | `experiments.py` | W&B job type (`benchmark`, `losses`, ..., `confirm`) |
| `--wandb_tags` | `experiments.py` | suite and model |
| run name | `train.py` | `<variant>-s<seed>` (or the checkpoint run name if no variant) |

`train.py` logs the full configuration to W&B, defines `val_ap` with a **max summary** (the run summary holds the best epoch, not the last), and writes the W&B run id and checkpoint directory to `checkpoints/<run>/args.json`. `--wandb_project` overrides the project (default `Forest-Gain-CD`). In the W&B UI, group by `variant` to collapse seeds into mean and min/max curves.

**The summary key is `val_ap.max`.** Because of the max summary, the W&B API exposes the best-epoch validation AP as `val_ap.max`, not `val_ap`. `report.py`, `confirm_top_k.py aggregate` and the data-source confirmation read `val_ap.max` (falling back to `val_ap`). If a report shows empty validation AP columns, this is the first thing to check.

Aggregate a launch over seeds:

```bash
python report.py --group augmentation_20261030_004812 --baseline baseline
python report.py --suite losses --out reports/losses
```

This writes `<out>.runs.csv` (one row per run), `<out>.csv` and `<out>.md` (per variant: n, mean, std, min, max, difference from the baseline, best run) and `<out>.png`. It reports validation AP only. Keep `--split-seed` fixed so seeds are the only variation within a variant. Confirmation runs (which also have holdout metrics) are aggregated by `confirm_top_k.py aggregate`, see sections 1 and 2.

## Embedding data handling (`datasets.py`)

Tessera tiles can contain corrupt values that the export checks did not catch: finite but absurd values (up to 1.7e35, seen in `tile_1102_873` 2019 and `tile_1131_908` 2024; 159 pixels in total), and pixels that are bad in only some of their 128 channels. A single absurd value turns the loss into NaN and collapses training.

`read_physical` therefore treats every embedding value that is NaN, infinite or has an absolute value above `EMBEDDING_ABS_MAX` (`config.py`, 50.0; the largest legitimate Tessera value measured over all 4000 tile-years is about 17.3, AlphaEarth values stay within about ±0.55) as missing, and a pixel that is missing in **any** channel is made missing in **all** channels. `normalize()` then zero-fills it, exactly like the sea / uncovered pixels Tessera marks as NaN. The bound is absolute, so it needs no statistics and no re-run when new data is collected. The valid-pixel mask is deliberately not changed, so every source combination is still scored on identical pixels.

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

Submit `run_fg_benchmark.sh` **with sbatch** from the directory where you keep your job scripts (`/projects/b6be/jamesbrock.b6be`), after `mkdir -p logs`. It runs as a small launcher job (2 GB, 10 minutes, no GPU, set by its `#SBATCH` header) that submits everything else:

```bash
sbatch run_fg_benchmark.sh
TRIALS=60 TOP_K=3 CONFIRM_SEEDS="0 1 2" sbatch run_fg_benchmark.sh
MODELS="tsvit sits_scd" sbatch run_fg_benchmark.sh              # a subset of models
```

The launcher finds the real scripts through `SLURM_SUBMIT_DIR` (the directory you ran `sbatch` from), because Slurm runs a copy of the script from its spool directory. Worker arrays get their own resources (1 GPU, 32 GB, 24 h) explicitly. If your cluster refuses to submit jobs from compute nodes, run the launcher on the login node with `bash run_fg_benchmark.sh` instead; the script works either way.

| Step | Job | What it does |
|---|---|---|
| 1 | one array per model (`--array=0-5`, 6 workers each) | parallel TPE workers on one shared study per model (`TRIALS`, default 50, per model) |
| 2 | one `run_fg_confirm.sh` array, `--dependency=afterany` on all benchmark arrays | pools the trials of every model, takes the **global** top `TOP_K` (default 3) by validation AP, and retrains each for every seed in `CONFIRM_SEEDS` (default `0 1 2`): 9 runs in total, one GPU each, each scored on the holdout tiles |
| 3 | small CPU job, `--dependency=afterany` on step 2 | runs `run_fg_report.sh` (or the script in `REPORT_SCRIPT`) if it exists; otherwise the launcher prints a warning and skips it. Aggregate by hand with `confirm_top_k.py aggregate` (see below) |

The global top 3 can be any mix of architectures. To compare architectures instead (the best trial of each, several seeds each), run the confirmation by hand with `--stratify-by model`, see "Confirmation and holdout" below. Comparing architectures this way is fair because every benchmark trial uses the same sources, loss, split and seed.

`afterany` means steps 2 and 3 still run if a benchmark worker hits the time limit. All tasks of one benchmark array share `SLURM_ARRAY_JOB_ID`, which names that model's study (`forestgain_<model>_<jobid>`) and output directory (`experiment_manifests/benchmark_<model>_<jobid>/`). Resume a model's study with `TAG_<model>=<old job id>`, for example `TAG_tsvit=123456 sbatch run_fg_benchmark.sh`: it adds trials until the study holds `TRIALS`. Do not resume finished benchmarks just to confirm them; run the confirmation by hand instead.

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
df = study.trials_dataframe()            # every trial: params, value, state
```

Each trial's W&B group is the study name and its variant is `trialNN`, where NN is the global trial number across workers.

### Search spaces

- **All models:** learning rate `3e-5`–`1e-3` (log scale), weight decay `1e-5`–`5e-2` (log scale), dropout `0`–`0.35`.
- **TSViT:** embedding dimension `{96,128,160}`, attention heads `{4,8}`, temporal depth `1`–`4`, spatial depth `2`–`6`, patch size `{8,16,32}`, skip channels `{16,24,32,48}`. `base_features` is not searched for TSViT (it stays at the model default, 64).
- **SITS_SCD:** base features `{32,48,64,96}`, attention heads `{2,4,8}`, temporal depth `1`–`4`, spatial depth `2`–`6`.
- **U-Net_LSTM:** base features `{24,32,48,64,96}`, skip channels `{16,24,32,48}`, temporal depth `1`–`3`.

After a first batch, look at parameter importance (`optuna.importance.get_param_importances(study)`) and at whether the best trials sit on a range boundary. If so, widen the range and continue the same study.

### Confirmation and holdout (`confirm_top_k.py`)

Benchmark trials are single-seed and scored on the same validation tiles that drive early stopping, so the best of many trials is partly the luckiest. The confirmation step pools the completed trials of all benchmark studies, selects the top `--top-k` trials per stratum by validation AP (`--stratify-by`, see below), and retrains each for several seeds. The number of runs is **strata × top-k × seeds**.

- **The holdout is evaluated here, and only here.** Benchmark runs and every other suite pass `--skip_test_eval`. Confirmation runs do not, so `train.py` scores each run's best checkpoint on `--holdout_dir` and logs `test_*` metrics to W&B. Benchmark runs never saw the holdout, so seed 0 is retrained rather than reused.
- **Selection rule:** the final model is the configuration with the best **mean validation AP** over its seeds. Holdout numbers are reported next to it but should not be used to choose between configurations, otherwise the holdout stops being held out. Once chosen, do not change the configuration after looking at holdout results.
- **Report:** `reports/<group>.md` ranks the configurations (model, n, validation AP and every `test_*` metric as mean ± std), names the selected one, gives its learning rate, weight decay and architecture, and has a second table with the **best configuration of every architecture**. `.runs.csv` has one row per run with the W&B URL and checkpoint directory; `.json` has the same data for scripts. Aggregation reads W&B, so it needs the runs to have finished and W&B access from the node.
- **Labels:** confirmation runs use W&B job type `confirm`, the group you pass as `--group` / `CONFIRM_GROUP`, and variant `confirm-<model>-trialNN` (the study name replaces the model if one model has several studies).
- **Group names:** use a fresh group for every confirmation. Re-using one mixes old and new runs, and `aggregate` would count them as extra seeds.

`--stratify-by` options (`--top-k` is always per stratum):

| `--stratify-by` | picks |
|---|---|
| `none` (default) | the global top-k over all studies |
| `model` | the top-k within each architecture, e.g. `--top-k 1` = the best trial of every architecture |
| a config key or trial parameter | the top-k within each distinct value, e.g. `loss_name` (`loss` is an alias); trials without the key go to the stratum `unknown` |

Studies are passed as `MODEL,STUDY_NAME,JOURNAL_PATH`, one per study. To run it by hand, or to confirm benchmarks that are already finished:

```bash
P=/projects/b6be/jamesbrock.b6be/ForestGainFoundations/ModelDev/experiment_manifests
export CONFIRM_STUDIES="tsvit,forestgain_tsvit_<id1>,$P/benchmark_tsvit_<id1>/optuna_journal.log sits_scd,forestgain_sits_scd_<id2>,$P/benchmark_sits_scd_<id2>/optuna_journal.log unet_lstm,forestgain_unet_lstm_<id3>,$P/benchmark_unet_lstm_<id3>/optuna_journal.log"
export CONFIRM_GROUP=confirm_manual TOP_K=1 CONFIRM_SEEDS="0 1 2"
# run_fg_confirm.sh passes --stratify-by "$CONFIRM_STRATIFY" if your copy has that line; export CONFIRM_STRATIFY=model

cd ForestGainFoundations/ModelDev
python confirm_top_k.py select --studies $CONFIRM_STUDIES --top-k 1 --stratify-by model --group confirm_manual   # plan only
cd ../..
sbatch --export=ALL --array=0-8 run_fg_confirm.sh                         # train on SLURM (array size = strata x top-k x seeds)
cd ForestGainFoundations/ModelDev
python confirm_top_k.py aggregate --group confirm_manual --out reports/confirm_manual
```

The studies in `CONFIRM_STUDIES` are separated by spaces and must not be quoted separately, because `run_fg_confirm.sh` expands the variable unquoted. Always run `select` first: it prints the selected trials and the exact `train.py` commands and writes `confirm_plan.json` to `experiment_manifests/<group>/`, without training. `run --index N` trains the N-th planned run (ranked trial-major: index = selection position × n_seeds + seed position); every SLURM task derives the same selection from the finished studies, so the tasks agree. An array larger than the plan is harmless: the extra tasks print "nothing to do" and exit. Options: `--top-k`, `--stratify-by`, `--seeds`, `--group`, `--wandb-project`, `--output-dir`. For a study created before trial configs were stored, the config is rebuilt from the trial's parameters using `--epochs/--patience/--batch-size/--data-dir/--holdout-dir/--split-seed`, which must match the benchmark's values.

After choosing a winner, the checkpoints of its runs are listed in the `.runs.csv` for later work.

**Holdout protection:** `experiments.py` adds `--skip_test_eval` to every generated training command (except the confirmation of the top data sources, section 2), and `train.py` supports that flag. This keeps the holdout untouched during tuning and ablations. The same applies to the loss, hardness, temporal and augmentation suites: if you want holdout numbers for their final choices, run the chosen configuration once per seed with `--skip_test_eval` off, as the confirmation step does. The Bayesian objective is validation AP, never holdout performance.

## 2. Data-source ablation

Use one selected/frozen model and training configuration for all 15 source combinations (every non-empty subset of S1, S2, AlphaEarth and Tessera). The configuration is the best benchmark trial of the chosen model, so that only the data source changes. There are two steps: a **sweep** over all sources (validation AP only), then a **confirmation** of the top sources on the holdout.

```bash
# 1. sweep: 15 source sets x 3 seeds = 45 runs, one per array task, at most 20 at once
sbatch run_fg_data_sources.sh                                  # tsvit, benchmark trial 28
sbatch --export=ALL,MODEL=unet_lstm run_fg_data_sources.sh     # unet_lstm, benchmark trial 33
sbatch --export=ALL,MODEL=sits_scd run_fg_data_sources.sh      # sits_scd, benchmark trial 45

# when the sweep has finished: rank the sources (validation AP, mean over seeds)
python report.py --group data-sources_tsvit_<sweep job id> --metric val_ap \
    --baseline tsvit_s1+s2_bce_focal_tversky --out reports/data_sources_tsvit

# 2. confirmation: the top 3 sources x 3 seeds = 9 runs WITH holdout evaluation
sbatch --array=0-8 --export=ALL,MODEL=tsvit,CONFIRM_FROM=data-sources_tsvit_<sweep job id> run_fg_data_sources.sh
python confirm_top_k.py aggregate --group data-sources-confirm_tsvit_<confirm job id> --out reports/data_sources_confirm_tsvit
```

**Sweep.** `run_fg_data_sources.sh` is an array of 45 tasks (`--array=0-44%20`): one training run per (source set, seed). Task id = source index × number of seeds + seed position, so the first three tasks are source set 0 with seeds 0, 1, 2. Each task passes `--source-index N` to `experiments.py` so that it trains only that combination. The script sets the learning rate, weight decay, `--loss` and `--model-opt` from the model's best benchmark trial (override with `LR`, `WD`, `MODEL_OPT`, `LOSS`, `GROUP`, `MAX_RUNS`, `SEEDS` via `--export=ALL,...`). All tasks share one W&B group (`data-sources_<model>_<array job id>`), and each writes its manifest to its own directory (`experiment_manifests/<group>/seed<S>_src<N>/`). Each task requests **96 GB RAM and 4 hours** (`#SBATCH --mem=96G --time=04:00:00`): the DataLoader workers were OOM-killed at 32 GB when embedding sources are loaded, and `--time` is per run. Override with `sbatch --mem=... --time=...` for a slower model. If your limit is on submitted jobs rather than running jobs, submit the array in two halves (`--array=0-22` and `--array=23-44`) with the same `GROUP`. If you change the number of seeds, change `--array` to 15 × seeds, and `N_SETS` in the script if the number of source sets changes.

**Resubmitting cancelled or failed tasks.** Pin `GROUP` to the original sweep's group, otherwise a new array job id creates a new W&B group and splits the sweep, and give only the missing task ids:

```bash
grep -h "group:" logs/fg-data-sources-<job id>_<task>.out        # the group the original tasks used
sbatch --export=ALL,GROUP=<that group> --array=12-14,24-35%20 run_fg_data_sources.sh
```

Only resubmit tasks that have no finished run in W&B: the confirmation ranking averages every finished run of a source set in the group, so a duplicate (source set, seed) would be counted twice. Crashed and cancelled runs are ignored by the ranking (it only reads runs in state `finished` with a validation AP). The manifest folders under `experiment_manifests/<group>/` are written before training starts, so their existence does not mean a run completed; W&B is the source of truth. The task for a given source set and seed is `source index × 3 + seed position`.

**Confirmation.** With `CONFIRM_FROM=<sweep group>`, each task reads the finished sweep from W&B, ranks the source combinations by mean validation AP over the sweep's seeds, takes the top `TOP_K` (default 3), and trains its own (source set, seed) with holdout evaluation switched on (`--top-sources-from` in `experiments.py` implies it). The ranking is deterministic, so all tasks agree, and each task prints the full ranking at the top of its log. The sweep must have **finished** first, otherwise the ranking uses partial results. Runs use the group `data-sources-confirm_<model>_<job id>` and variant `confirm-<model>-<sources>`. `confirm_top_k.py aggregate` then gives validation AP and every `test_*` metric as mean ± std per source set, ranked by validation AP. As in section 1, choose by validation AP and report the holdout numbers next to it without using them to choose. The seeds are retrained (not reused from the sweep) so the holdout is scored on a fresh run of each.

To run it directly, pass the configuration explicitly; without `--model-opt` the architecture falls back to `train.py`'s defaults:

```bash
python experiments.py data-sources --model tsvit --seeds 0 1 2 --group data_sources_tsvit \
  --learning-rate 0.00034779544459228077 --weight-decay 0.005786816724872467 --loss bce_focal_tversky \
  --model-opt embed_dim=160 num_heads=4 depth_temporal=3 depth_spatial=2 patch_size=8 dropout=0.22185365097777363 skip_channels=24 \
  --dry-run
python experiments.py data-sources --model tsvit --seeds 0 1 2 --top-sources-from data_sources_tsvit --top-k 3 \
  --group data_sources_confirm_tsvit --learning-rate ... --weight-decay ... --model-opt ... --dry-run
```

The 15 combinations are `SOURCE_SETS` in `experiments.py`, indexed 0-14 in this order: S1, S2, S1+S2, AlphaEarth, Tessera, S1+AlphaEarth, S2+AlphaEarth, S1+S2+AlphaEarth, S1+Tessera, S2+Tessera, S1+S2+Tessera, S1+S2+AlphaEarth+Tessera, AlphaEarth+Tessera, S1+AlphaEarth+Tessera, S2+AlphaEarth+Tessera. The last three (indices 12-14, tasks 36-44) complete the full factorial; they were appended so that the indices, W&B groups and results of the first 12 are unchanged. Confirm each product exists in the dataset pipeline. Variants are named `<model>_<sources>_<loss>`. Tessera-containing sets rely on the embedding handling described above.

**Source bias.** The benchmark was run on S1 + S2, so the configuration is tuned for that input and may under-serve the others (especially the embedding sources, which have different channel statistics). To limit this:

- Use the same seeds for every source set and compare differences against the S1+S2 baseline, not only means. Treat differences smaller than the seed spread (about 0.005 validation AP for TSViT) as ties.
- Repeat the sweep with a second architecture (`MODEL=unet_lstm`). If the source ranking agrees across architectures, it is not an artefact of one model's hyperparameters.
- For the leading source sets and S1+S2, run a short tuning search (learning rate, weight decay, dropout) and confirm with 3 seeds. If the ranking holds, it is robust to tuning; if it changes, report both tables. This needs the benchmark's `("s1", "s2")` source set replaced by `args.sources` in `run_bayesian_benchmark` and a separate `--study-name` per source set.
- State the limitation. If a source beats S1+S2 even though the configuration favours S1+S2, the result is conservative.

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

The suite uses one fixed model, source set, data split, and unspecified training defaults. Pass the best benchmark configuration with `--learning-rate`, `--weight-decay` and `--model-opt` (as for the data-source ablation) so that the architecture is not left at `train.py`'s defaults. It is a structured comparison, not a full Bayesian optimisation of every loss parameter. Use multiple seeds for shortlisted loss configurations.

### Investigating the sweep and confirming the best of each loss

```bash
# 1. aggregate the sweep over seeds (validation AP only), baseline = plain BCE
python report.py --group losses_tsvit_<sweep job id> --baseline tsvit_s1+s2_bce --out reports/losses_tsvit

# 2. preview the best configuration per loss family, then the exact confirmation commands
python experiments.py losses --model tsvit --confirm-from losses_tsvit_<sweep job id> --seeds 0 1 2 --dry-run

# 3. confirm on SLURM: one task per (family, seed), holdout evaluated
SUITE=losses SWEEP_GROUP=losses_tsvit_<sweep job id> sbatch run_fg_confirm_suite.sh
python confirm_top_k.py aggregate --group losses-confirm_tsvit_<confirm job id> --out reports/losses_confirm_tsvit --no-confirmed-config
```

See "Confirmation and holdout" below for how this generalises to the other suites. A loss family is (loss, reweight): bce, dice, focal_tversky, bce_focal_tversky, focal, lovasz, bce_lovasz, and bce / bce_focal_tversky / focal with OHEM (10 families, so 30 tasks with 3 seeds). `--confirm-from` ranks every configuration in the sweep group by mean validation AP over its seeds (finished runs only), prints the full table per family with the selected one marked `*`, and retrains the winner with `--skip_test_eval` off, copying learning rate, weight decay, architecture and every loss option from the sweep run's own W&B config. Use `--confirm-top-k 2` to confirm the two best per family.

Reading the sweep: treat differences smaller than the seed spread (about 0.005 validation AP for TSViT, more for several losses) as ties; check whether a family's best setting sits on the edge of its grid (e.g. Tversky alpha 0.7, gamma 2.0, OHEM threshold 0.9), in which case widen the grid and re-run that family; and compare families by their confirmed holdout numbers, not the sweep's best-of-many validation AP. AP is threshold-free, while precision, recall, F1 and IoU depend on the 0.5 threshold, which Tversky/Lovasz-type losses shift, so read the `test_*` threshold metrics alongside AP. Confirming with the sweep's seeds mostly replicates the sweep (same initialisation and sample order); pass other seeds (`SEEDS="3 4 5"`) for an independent estimate. As in section 1, choose on validation AP and report the holdout numbers without using them to choose.

`aggregate` also writes `confirmed_configs.json` (best variant per model) unless `--no-confirmed-config` is given; use the flag for every group except the benchmark confirmation, otherwise it overwrites the confirmed TSViT entry with the best loss / data-source variant and drops `loss_opt` / `reweight`.

### Confirmation and holdout (all suites)

The holdout is only evaluated by confirmation runs, or by runs started with `--eval-holdout`:

| Suite | Holdout | How |
|---|---|---|
| benchmark (Optuna search) | confirm best trials | `experiments.py benchmark --confirm --studies MODEL,STUDY,JOURNAL ... --top-k K --stratify-by model --seeds 0 1 2` (`--confirm-index N` for one array task) |
| losses, hardness, maskclean | confirm best of each family | `experiments.py <suite> --confirm-from SWEEP_GROUP [--confirm-top-k K] [--confirm-index N]`; SLURM: `SUITE=<suite> SWEEP_GROUP=... sbatch run_fg_confirm_suite.sh` |
| data-sources | confirm the top sources, or log for all | `--top-sources-from SWEEP_GROUP --top-k K`, or `--eval-holdout` on the sweep |
| temporal, augmentation | log for every run (pure comparison) | `--eval-holdout` |

Families: losses and hardness group by (loss, reweight), maskclean by cleaning method (baseline, rm, fill, conf, harden and their combinations). Whatever the route, choose configurations on validation AP and report holdout numbers next to it. `confirm_top_k.py` keeps its `select` / `run` / `aggregate` commands (the selection logic now lives in `experiments.py`), so `run_fg_confirm.sh` and the benchmark launcher keep working unchanged.

## 4. Pixel hardness

```bash
python experiments.py hardness --model tsvit --execute
```

The suite compares matched versions of BCE and BCE + Focal Tversky under:

- no reweighting;
- OHEM at thresholds `0.7` and `0.9`;
- learned hardness, sweeping `hardness.c` `{0.01,0.1}` and auxiliary loss scale `hardness.alpha` `{0.001,0.01,0.05}`.

The hardness branch mirrors the model's segmentation head on detached inputs, so the layer is the head module: TSViT `decoder` (a copy of the multi-conv decoder) and `decoder.final_head` (small default head on the fused features), SITS-SCD `final_head` and U-Net-LSTM `out_head` (small default head). `experiments.py` picks these per model (`HARDNESS_LAYERS`); override with `--hl-layer`, or pass `--hl_layer` to `train.py`. Run the suite once per model. That gives 12 (TSViT) or 6 (others) hardness configurations per loss family, plus 3 controls (no reweighting, OHEM 0.7, OHEM 0.9), i.e. 15 or 9 runs per loss family. The hardness branch consumes detached features, and the learned hardness weighting applies to the per-pixel BCE term; in BCE + Focal Tversky the Tversky term itself remains un-reweighted. `--model-opt`, `--learning-rate` and `--weight-decay` apply as in section 3.

This is a parameter sweep rather than a full Bayesian search. After the initial sweep, compare the strongest hardness settings against the matched no-reweighting and OHEM controls over multiple seeds.

## 5. Temporal ablation

```bash
python experiments.py temporal --models tsvit sits_scd unet_lstm --seeds 0 1 2 --execute
```

Compares first+last year (2017, 2024), 2017/2021/2024, 2017/2020/2022/2024, and the whole sequence (`--year-sets` overrides). The valid-pixel mask is taken over all years so the scored pixels stay consistent across frame subsets. This suite compares several models, taking the optimal configurations for these models from the benchmarking experiments. If no configuration is found it uses the defaults of `train.py`.

## 6. Augmentation and sampling

```bash
python experiments.py augmentation --model tsvit --seeds 0 1 2 --execute
python experiments.py augmentation --loss-type soft --execute   # also sweeps soft-label jitter
```

Model, sources, loss, learning rate and weight decay are fixed (pass the best benchmark configuration with `--learning-rate`, `--weight-decay` and `--model-opt`). The baseline is `train.py`'s defaults: random 90-degree rotation and flip, S2 brightness jitter (`gain_sigma` 0.03, `frame_sigma` 0.01), no frame dropout, full tile, `repeats` 1. The suite (14 runs per seed, 18 with `--loss-type soft`) covers:

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
- `--model-opt KEY=VAL ...`: architecture options passed to every run of the data-sources, losses, hardness and augmentation suites (use the best benchmark configuration).
- `--loss NAME`: loss for the data-sources and temporal suites (default `bce_focal_tversky`, the benchmark loss).
- `--source-index N`: data-sources suite, run only the N-th source combination, 0-14 (for parallel SLURM tasks).
- `--top-sources-from GROUP` and `--top-k K`: data-sources suite, confirmation of the K best combinations of a finished sweep, with holdout evaluation (section 2).
- `--epochs`, `--patience`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--data-dir`, `--holdout-dir`, `--split-seed`.

The benchmark accepts `--trials`, `--startup-trials`, `--models`, `--seed`, `--study-name`, `--storage-path`, `--worker-id`, `--output-dir`, `--epochs`, `--patience`, `--batch-size`, `--data-dir`, `--holdout-dir`, `--split-seed`, `--loss-type` and the W&B flags. It ignores `--learning-rate`, `--weight-decay` and `--model-opt` because those are searched, and it uses its own per-trial W&B group (the study name) instead of `--group`.

`train.py` accepts `--skip_test_eval` to defer holdout evaluation, `--model_opt KEY=VAL ...` for architecture options, and `--loss_opt KEY=VAL ...` for loss parameters. Defaults for model options live in `train.py`; loss defaults live in `losses/config.py`. Models may ignore options that do not apply to them.

## 7. Mask cleaning during training (`maskclean` suite)

Trains on **cleaned training labels** with one fixed configuration (pass the best TSViT `--learning-rate`, `--weight-decay`, `--model-opt`, `--loss`), 3 seeds per run, to see how much label noise removal helps and how the model behaves on masks closer to a human annotation. Cleaning lives in `mask_cleaning.py` and is applied in `datasets.py` on the full tile before crop and augmentation.

```bash
python experiments.py maskclean --model tsvit --seeds 0 1 2 --loss-type soft --group maskclean_tsvit \
  --learning-rate ... --weight-decay ... --model-opt ... --dry-run
python experiments.py maskclean --mask-stage singles ...    # first the single methods; later --mask-stage combos
```

| Experiment | `train.py` flag | Values |
|---|---|---|
| Baseline | none | original masks |
| Small-component removal | `--min_gain_px N` | 2, 5, 10, 25 px |
| Hole filling | `--fill_holes_px N` | 2, 5, 10, 25 px |
| Confidence filtering | `--conf_filter T` | 0.50-0.90, step 0.05 |
| Confidence hardening | `--conf_harden T` | 0.50-0.90, step 0.05 |
| Combinations (`--mask-stage combos`) | several of the above | morphology pairs (`--combo-morph`, default 5,5 and 10,10) alone and with filter / harden at `--combo-confs` (default 0.6, 0.7, 0.8) |

Size 1 is a no-op (equal to the baseline) and is skipped (`--mask-keep-noop` runs it). With `--loss-type soft` the defaults give 41 configurations (123 runs with 3 seeds); with the hard loss the 9 hardening configurations and the harden combinations are skipped, because the hard loss only uses the binary mask and hardening would equal filtering.

**Order and semantics.** Confidence step, then removal, then filling. Filtering/hardening ignore (valid=0) gain pixels whose confidence is below T; true zeros stay valid negatives. Hardening additionally sets retained gain weights to 1. Removal drops 8-connected patches smaller than N px; filling fills 4-connected holes smaller than N px that do not touch the tile border and contain only valid pixels. Hardening needs binary weights, so `--mask-label-sigma` (default 0) fixes label jitter for every maskclean run; this keeps variants matched.

**Configuration.** Like the temporal and losses suites, the suite reads the model's best confirmed configuration (learning rate, weight decay, loss, `model_opt`) from `experiment_manifests/confirmed_configs.json` (`--model`, default `tsvit`), and falls back to `train.py` defaults with a warning if the file is missing. `--learning-rate`, `--weight-decay`, `--loss` and `--model-opt` are ignored by this suite.

**Comparability.** Cleaning is applied to training labels only. Early stopping and `val_ap` use the raw validation masks for every variant, so all variants are selected on the same target; `--mask-clean-val` (`--clean_val`) cleans them too. This is a change from the earlier `--min_gain_px` behaviour, which cleaned validation as well. The holdout stays raw and untouched during the sweep (`--skip_test_eval`).

**Raw and cleaned metrics side by side.** After training, `train.py` scores the best checkpoint on the validation tiles twice and logs both to W&B: `val_eval_*` (raw labels, every maskclean run including the baseline) and `val_eval_clean_*` (the cleaned labels of that variant). When the holdout is evaluated (confirmation runs) the same pair is logged as `test_*` and `test_clean_*`. Raw metrics are the comparable ones across variants. Cleaned metrics are scored on a different target per variant, so they show how well each model fits its own cleaned labels, not which variant is better. `eval.py` (`evaluate_checkpoint`, also its CLI) accepts `--min_gain_px --fill_holes_px --conf_filter --conf_harden` and a `tile_dirs` list for this.