# Experiment scripts

Run all commands from `ModelDev`. By default the runner only writes a manifest;
add `--execute` after checking the printed commands. Results are comparable because
each run fixes the data split and all unspecified `train.py` defaults.

```bash
# Tune S1+S2 architectures and optimiser settings (24 staged runs)
python experiments.py benchmark --dry-run
python experiments.py benchmark --execute

# Source ablations using the selected default model
python experiments.py data-sources --model tsvit --execute

# Compare BCE, Dice, Focal Tversky and BCE+Focal Tversky
python experiments.py losses --model tsvit --sources s1 s2 --execute
```

For the larger source ablation, use the best configuration selected from the
benchmark by supplying its values to `train.py` directly. Keep the same split
seed and holdout directory across all comparisons. The manifests in
`experiment_manifests/` are an auditable record of every exact command.

`train.py` now accepts `--base_features`, `--num_heads`, `--patch_size`,
`--embed_dim`, `--depth_temporal`, `--depth_spatial`, `--dropout`, and
`--skip_channels`, plus `--loss` (`bce`, `dice`, `focal_tversky`, or
`bce_focal_tversky`).

## Mask optimisation

Run mask evaluation only after choosing a checkpoint; it scores the untouched
holdout labels and writes all results to one CSV.

```bash
python mask_sweep.py checkpoints/<run>/<best>.ckpt --output results/mask_sweep.csv
```

The CSV contains thresholding at 0.50--0.90, removal and filling of connected
components sized 1, 2, 5, 10 and 25 pixels, and the maximum mask coverage at
minimum positive-mask precision of 55--80%. `coverage_pct` is the percentage
of valid pixels retained as predicted gain. In this rare-event setting,
positive-mask precision is the meaningful interpretation of mask “accuracy”; a
separate `pixel_accuracy` column is retained for completeness.
