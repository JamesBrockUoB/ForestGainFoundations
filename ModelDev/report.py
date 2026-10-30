#!/usr/bin/env python3
"""Aggregate Weights & Biases runs of one experiment launch into a seed-averaged table + chart.

    python report.py --project Forest-Gain-CD --group augmentation_20261030_004812 --baseline baseline
    python report.py --project Forest-Gain-CD --suite losses --metric val_ap --out reports/losses

Runs are grouped by their `variant` config value (seed-independent label set by experiments.py), so every row is
mean +/- std over seeds. Needs `pip install wandb pandas matplotlib` and a logged-in wandb. Writes <out>.csv, <out>.md, <out>.png.
`val_ap` is the BEST epoch (train.py defines the summary as max); validation only - holdout metrics (test_*) appear
only for runs where the holdout was evaluated on purpose.
"""

import argparse
from pathlib import Path

import pandas as pd


def fetch_runs(
    project, entity=None, group=None, suite=None, tags=None, metrics=("val_ap",)
):
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {"state": "finished"}
    if group:
        filters["group"] = group
    if suite:
        filters["jobType"] = suite
    if tags:
        filters["tags"] = {"$in": list(tags)}
    rows = []
    for run in api.runs(path, filters=filters):
        cfg = run.config
        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "group": run.group,
                "variant": cfg.get("variant") or run.name,
                "seed": cfg.get("seed"),
                "model": cfg.get("model_type"),
                "checkpoint_dir": cfg.get("checkpoint_dir"),
                "url": run.url,
                **{m: run.summary.get(m) for m in metrics},
            }
        )
    return pd.DataFrame(rows)


def aggregate(df, metric="val_ap", baseline=None):
    """One row per variant: n seeds, mean, std, min, max, best run; sorted best first; optional delta vs baseline."""
    d = df.dropna(subset=[metric])
    g = d.groupby("variant")[metric]
    out = g.agg(n="count", mean="mean", std="std", min="min", max="max")
    out["std"] = out["std"].fillna(0.0)  # a single seed has no spread
    out["best_run"] = d.loc[g.idxmax(), ["variant", "run_id"]].set_index("variant")[
        "run_id"
    ]
    out = out.sort_values("mean", ascending=False)
    if baseline is not None:
        if baseline not in out.index:
            raise SystemExit(
                f"baseline variant '{baseline}' not found; have: {list(out.index)}"
            )
        out["delta_vs_baseline"] = out["mean"] - out.loc[baseline, "mean"]
    return out


def to_markdown(table, metric):
    cols = ["n", "mean", "std", "min", "max"] + (
        ["delta_vs_baseline"] if "delta_vs_baseline" in table else []
    )
    lines = [f"| variant | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for variant, r in table.iterrows():
        lines.append(
            f"| {variant} | "
            + " | ".join(
                (
                    f"{r[c]:.0f}"
                    if c == "n"
                    else f"{r[c]:+.4f}" if c.startswith("delta") else f"{r[c]:.4f}"
                )
                for c in cols
            )
            + " |"
        )
    return (
        f"**{metric}** (best validation epoch; mean over seeds)\n\n"
        + "\n".join(lines)
        + "\n"
    )


def plot(table, metric, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    ax.barh(t.index, t["mean"], xerr=t["std"], color="#4c72b0", capsize=3)
    ax.set_xlabel(f"{metric} (mean ± std over seeds)")
    ax.set_xlim(
        max(0, t["mean"].min() - 3 * t["std"].max() - 0.02),
        t["mean"].max() + 2 * t["std"].max() + 0.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--project", default="Forest-Gain-CD")
    p.add_argument("--entity")
    p.add_argument(
        "--group", help="one experiment launch (see the manifest's wandb_group)"
    )
    p.add_argument("--suite", help="all runs of a suite (W&B job type)")
    p.add_argument("--tags", nargs="+")
    p.add_argument("--metric", default="val_ap")
    p.add_argument("--baseline", help="variant name to report deltas against")
    p.add_argument("--out", default="reports/report")
    a = p.parse_args()
    if not (a.group or a.suite or a.tags):
        p.error("give --group, --suite or --tags so unrelated runs are not mixed")
    df = fetch_runs(a.project, a.entity, a.group, a.suite, a.tags, metrics=(a.metric,))
    if df.empty:
        raise SystemExit("no finished runs matched")
    table = aggregate(df, a.metric, a.baseline)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".runs.csv"), index=False)
    table.to_csv(out.with_suffix(".csv"))
    out.with_suffix(".md").write_text(to_markdown(table, a.metric))
    plot(table, a.metric, out.with_suffix(".png"))
    print(to_markdown(table, a.metric))
    print(f"wrote {out}.runs.csv / .csv / .md / .png")


if __name__ == "__main__":
    main()
