import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import EMBEDDING_SOURCES, SOURCES

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT = Path("shap_out/shap_importance.npz")
OUT = Path("shap_analysis")
OUT.mkdir(exist_ok=True)

TOP_N = 20

data = np.load(INPUT, allow_pickle=True)

group_abs = data["grouped_abs"]  # (T, G)
group_sum = data["grouped_sum"]  # (T, G)
group_names = data["group_names"].astype(str)
years = data["years"]

n_tiles = int(data["n_tiles"])

print(f"Loaded SHAP results: {n_tiles} tiles")
print(f"Timesteps: {years}")
print(f"Features: {len(group_names)}")


# ---------------------------------------------------------------------
# Assign each feature to its original data source
# ---------------------------------------------------------------------

band_to_source = {}

for source, config in SOURCES.items():
    bands = config[1]
    for band in bands:
        band_to_source[band] = source


def get_source(name):
    # Individual embedding dimensions, e.g. alphaearth_000
    for source in EMBEDDING_SOURCES:
        if name.startswith(source + "_"):
            return source

    # Original Sentinel bands
    if name in band_to_source:
        return band_to_source[name]

    # Derived indices
    if name.startswith("idx_"):
        return "indices"

    return "other"


sources = np.array([get_source(name) for name in group_names])
unique_sources = list(dict.fromkeys(sources))


# ---------------------------------------------------------------------
# 01. SOURCE-LEVEL IMPORTANCE
# ---------------------------------------------------------------------

source_abs = {}
source_signed = {}

for source in unique_sources:
    mask = sources == source

    source_abs[source] = group_abs[:, mask].sum()
    source_signed[source] = group_sum[:, mask].sum()

total = sum(source_abs.values())

print("\n01. SOURCE-LEVEL IMPORTANCE")

source_rows = []

for source in sorted(source_abs, key=source_abs.get, reverse=True):
    absolute = source_abs[source]
    percentage = 100 * absolute / total if total else 0

    source_rows.append(
        [
            source,
            absolute,
            percentage,
            source_signed[source],
        ]
    )

    print(
        f"{source:<15} "
        f"absolute={absolute:.6e} "
        f"share={percentage:.2f}% "
        f"signed={source_signed[source]:+.6e}"
    )

with open(OUT / "shap01_source_importance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["source", "absolute_importance", "percentage", "signed_importance"]
    )
    writer.writerows(source_rows)

fig, ax = plt.subplots(figsize=(9, 5))

labels = [r[0] for r in source_rows]
values = [r[2] for r in source_rows]

ax.bar(labels, values)
ax.set_ylabel("Share of total absolute attribution (%)")
ax.set_title("SHAP source-level importance")
ax.tick_params(axis="x", rotation=30)

fig.tight_layout()
fig.savefig(OUT / "shap01_source_importance.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 02. TEMPORAL IMPORTANCE
# ---------------------------------------------------------------------

temporal_abs = group_abs.sum(axis=1)
temporal_signed = group_sum.sum(axis=1)

temporal_pct = (
    100 * temporal_abs / temporal_abs.sum()
    if temporal_abs.sum()
    else np.zeros_like(temporal_abs)
)

print("\n02. TEMPORAL IMPORTANCE")

for year, absolute, percentage, signed in zip(
    years, temporal_abs, temporal_pct, temporal_signed
):
    print(
        f"{year}: absolute={absolute:.6e} "
        f"share={percentage:.2f}% "
        f"signed={signed:+.6e}"
    )

with open(OUT / "shap02_temporal_importance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "year",
            "absolute_importance",
            "percentage",
            "signed_importance",
        ]
    )

    for row in zip(years, temporal_abs, temporal_pct, temporal_signed):
        writer.writerow(row)

# Total temporal importance
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(years, temporal_pct, marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Share of total absolute attribution (%)")
ax.set_title("SHAP temporal importance")
ax.set_xticks(years)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "shap02_temporal_importance.png", dpi=200)
plt.close(fig)

# Temporal importance separated by source
fig, ax = plt.subplots(figsize=(10, 6))

for source in unique_sources:
    mask = sources == source
    profile = group_abs[:, mask].sum(axis=1)

    ax.plot(years, profile, marker="o", label=source)

ax.set_xlabel("Year")
ax.set_ylabel("Absolute attribution")
ax.set_title("Temporal importance by data source")
ax.set_xticks(years)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "shap02_temporal_by_source.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 03. INDIVIDUAL EMBEDDING DIMENSION IMPORTANCE
# ---------------------------------------------------------------------

print("\n03. EMBEDDING DIMENSION IMPORTANCE")

embedding_rows = []

for source in EMBEDDING_SOURCES:
    indices = np.where(sources == source)[0]

    if len(indices) == 0:
        continue

    scores = group_abs[:, indices].sum(axis=0)
    signed_scores = group_sum[:, indices].sum(axis=0)

    order = np.argsort(-scores)

    print(f"\n{source}: top {min(TOP_N, len(order))} dimensions")

    for rank, position in enumerate(order[:TOP_N], start=1):
        idx = indices[position]
        name = group_names[idx]

        print(
            f"{rank:2d}. {name:<25} "
            f"absolute={scores[position]:.6e} "
            f"signed={signed_scores[position]:+.6e}"
        )

    for position in order:
        idx = indices[position]

        embedding_rows.append(
            [
                source,
                group_names[idx],
                scores[position],
                signed_scores[position],
                100 * scores[position] / scores.sum() if scores.sum() else 0,
            ]
        )

    # Plot top dimensions
    selected = order[:TOP_N]

    names = group_names[indices[selected]]
    values = scores[selected]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Total absolute attribution")
    ax.set_title(f"Top {TOP_N} {source} embedding dimensions")

    fig.tight_layout()
    fig.savefig(
        OUT / f"shap03_{source}_dimension_ranking.png",
        dpi=200,
    )
    plt.close(fig)

with open(OUT / "shap03_embedding_dimension_ranking.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "source",
            "dimension",
            "absolute_importance",
            "signed_importance",
            "percentage_within_source",
        ]
    )
    writer.writerows(embedding_rows)


print("\nAnalysis complete.")
print(f"Results saved in: {OUT.resolve()}")
