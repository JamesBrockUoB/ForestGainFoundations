#!/usr/bin/env python3
"""
Compare the 'threshold' and 'delta' gain rules on the SAME random pending tiles:
of the gain pixels each rule keeps in a tile, what is the mean tree cover %?

For each batch of pending tiles it runs the real cheap-stage statistics
(filtering.raster_stats.fetch_cheap_stats) once per rule and applies the cheap-stage
pass/fail logic. For tiles that pass, it also reduces the label layer's
forest_confidence band (end-year cover on the kept gain pixels) to one mean per tile.
Results are appended to a CSV; the registry is only read, never written. It stops
once BOTH rules have --quota cheap-valid tiles, then runs the statistical comparison
(unpaired + paired) and prints it.

    python compare_gain_rules.py
    python compare_gain_rules.py --quota 500
    python compare_gain_rules.py --analyse-only      # re-run the stats on the existing CSV

Re-running with the same --out resumes: tiles already in the CSV are skipped.
The delta size comes from GAIN_MIN_DELTA / settings, exactly as in the pipeline.
Put this file in the same folder as main.py. Extra deps: numpy, pandas, scipy.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from config import settings
from enums import TileStatus
from filtering import gain_rule as gr
from filtering.raster_stats import (
    _reduce_tiles,
    _with_retry,
    fetch_cheap_stats,
    tiles_to_feature_collection,
)
from gee.auth import get_ee_credentials
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer
from registry.store import _get_db
from scipy import stats as sps

RULES = ("threshold", "delta")
TILE_COLS = [
    "tile_id",
    "xi",
    "yi",
    "x_min_m",
    "y_min_m",
    "x_max_m",
    "y_max_m",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "biome",
]
POOL_FACTOR = 40  # random pending tiles drawn per quota slot (upper bound; usually stops far earlier)


@contextmanager
def rule_env(rule: str):
    """Select the gain rule the same way the pipeline does (GAIN_RULE env var)."""
    old = os.environ.get("GAIN_RULE")
    os.environ["GAIN_RULE"] = rule
    try:
        assert gr.gain_rule() == rule, "GAIN_RULE override did not take effect"
        yield
    finally:
        if old is None:
            os.environ.pop("GAIN_RULE", None)
        else:
            os.environ["GAIN_RULE"] = old


def cheap_valid(stats: dict) -> bool:
    """Mirror of evaluate_cheap_stats(): True when the tile would become CHEAP_VALID."""
    gain_frac, ndvi = stats.get("gain_frac"), stats.get("ndvi_trend")
    if not gain_frac:
        return False
    if gain_frac * 100.0 < settings.gain_pct_min:
        return False
    return ndvi is not None and ndvi > settings.ndvi_trend_min


def fetch_cover_means(tiles, ds, chunk_size=20, max_workers=5) -> dict:
    """Per tile: mean end-year tree cover (%) over the KEPT gain pixels, under the
    currently selected rule (forest_confidence = cover_end masked to validated gain)."""

    def run(chunk):
        def go():
            geom = tiles_to_feature_collection(chunk).geometry()
            _, _, conf = build_gain_layer(geom, ds)
            reduced = _reduce_tiles(
                conf.rename("cover_mean"), chunk, ["cover_mean"], tile_scale=2
            )
            return {tid: vals.get("cover_mean") for tid, vals in reduced.items()}

        return _with_retry(go)

    out = {}
    chunks = [tiles[i : i + chunk_size] for i in range(0, len(tiles), chunk_size)]
    if not chunks:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:  # finishes inside rule_env
        for r in ex.map(run, chunks):
            out.update(r)
    return out


def sample_pending(n: int, exclude: set) -> list[dict]:
    with _get_db()._conn() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(TILE_COLS)} FROM tiles WHERE status = ? "
            "ORDER BY RANDOM() LIMIT ?",
            (str(TileStatus.PENDING), n + len(exclude)),
        ).fetchall()
    tiles = [{c: r[c] for c in TILE_COLS} for r in rows if r["tile_id"] not in exclude]
    return tiles[:n]


A, B = "delta", "threshold"  # differences are reported as A - B


def boot_ci(x, y=None, n_boot=5000, seed=0):
    """95% percentile bootstrap CI for mean(x) (paired diffs) or mean(x) - mean(y)."""
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[ix].mean(axis=1)
    if y is not None:
        iy = rng.integers(0, len(y), (n_boot, len(y)))
        m = m - y[iy].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def _describe(name, x):
    print(
        f"  {name:<10} n={len(x):<5} mean={x.mean():.2f}  median={np.median(x):.2f}  "
        f"sd={x.std(ddof=1):.2f}"
    )


def unpaired(a, b):
    print(
        f"\n=== UNPAIRED: first cheap-valid tiles per rule ({A} vs {B}), mean cover % ==="
    )
    _describe(A, a)
    _describe(B, b)
    u, p_u = sps.mannwhitneyu(a, b, alternative="two-sided")
    cliff = 2 * u / (len(a) * len(b)) - 1
    _, p_t = sps.ttest_ind(a, b, equal_var=False)
    lo, hi = boot_ci(a, b)
    print(
        f"  mean diff ({A} - {B}) = {a.mean() - b.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
    )
    print(f"  Mann-Whitney U  p = {p_u:.4g}   Cliff's delta = {cliff:+.3f}")
    print(f"  Welch t-test    p = {p_t:.4g}")


def paired(wide):
    d = (wide[A] - wide[B]).to_numpy()
    print(
        f"\n=== PAIRED: {len(d)} tiles cheap-valid under both rules, mean cover % ==="
    )
    if len(d) < 2:
        print("  not enough tiles")
        return
    print(f"  mean {A} = {wide[A].mean():.2f}   mean {B} = {wide[B].mean():.2f}")
    print(
        f"  tiles where {A} is higher / equal / lower: {(d > 0).sum()} / {(d == 0).sum()} / {(d < 0).sum()}"
    )
    lo, hi = boot_ci(d)
    print(f"  mean diff ({A} - {B}) = {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    nz = d[d != 0]
    if len(nz):
        print(
            f"  Wilcoxon signed-rank (non-zero diffs, n={len(nz)}) p = {sps.wilcoxon(nz).pvalue:.4g}"
        )
    else:
        print("  identical under both rules on every tile: no test possible")


def analyse(csv, quota=None, metric="cover_mean"):
    df = pd.read_csv(csv)
    missing = {"tile_id", "rule", metric, "cheap_valid"} - set(df.columns)
    if missing:
        sys.exit(f"missing columns: {sorted(missing)}")
    df["cheap_valid"] = (
        df["cheap_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    bad = set(df["rule"]) - {A, B}
    if bad:
        sys.exit(f"unknown rule values: {bad}")
    ok = df[df["cheap_valid"] & df[metric].notna()]

    samples = {}
    for r in (A, B):
        v = ok[ok["rule"] == r][metric].to_numpy(dtype=float)
        if quota:
            if len(v) < quota:
                print(
                    f"WARNING: only {len(v)} cheap-valid tiles for {r} (< quota {quota})"
                )
            v = v[:quota]
        if len(v) < 2:
            sys.exit(f"need at least 2 cheap-valid tiles for {r}, have {len(v)}")
        samples[r] = v
    unpaired(samples[A], samples[B])

    wide = ok.pivot_table(
        index="tile_id", columns="rule", values=metric, aggfunc="first"
    )
    if set(wide.columns) >= {A, B}:
        paired(wide.dropna())

    n_tiles = df["tile_id"].nunique()
    valid = df[df["cheap_valid"]].groupby("rule")["tile_id"].nunique()
    print(
        f"\n  cheap-valid pass rate over {n_tiles} tiles: "
        f"{A} {valid.get(A, 0) / n_tiles:.1%}, {B} {valid.get(B, 0) / n_tiles:.1%}"
    )
    print(
        "\nNotes: the two rules keep different tiles (each must pass the cheap filter), so the unpaired\n"
        "comparison mixes the rule's effect with which tiles survive; the paired block avoids that.\n"
        "Tiles from the same area are not fully independent, so treat p-values as optimistic."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--quota",
        type=int,
        default=500,
        help="cheap-valid tiles needed per rule (default 500)",
    )
    ap.add_argument("--out", default="gain_rule_comparison.csv")
    ap.add_argument(
        "--analyse-only",
        action="store_true",
        help="skip collection; analyse the existing CSV",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if args.analyse_only:
        analyse(out, quota=args.quota)
        return

    ee.Initialize(get_ee_credentials(), project=settings.gee_project)
    ds = Datasets()
    for r in RULES:
        with rule_env(r):
            print(gr.describe())

    done, passes = set(), {r: 0 for r in RULES}
    if out.exists():
        prev = pd.read_csv(out)
        prev["cheap_valid"] = prev["cheap_valid"].astype(str).str.lower().eq("true")
        done = set(prev["tile_id"])
        for r in RULES:
            passes[r] = int(((prev["rule"] == r) & prev["cheap_valid"]).sum())
        print(
            f"Resuming: {len(done)} tiles already in {out} | cheap-valid so far {passes}"
        )

    pool = sample_pending(args.quota * POOL_FACTOR, done)
    batch_size = settings.filter_batch_size_cheap
    evaluated = 0

    for i in range(0, len(pool), batch_size):
        if all(passes[r] >= args.quota for r in RULES):
            break
        chunk = pool[i : i + batch_size]

        try:
            stats, cover = {}, {}
            for r in RULES:  # sequential: the rule is a process-wide env var
                with rule_env(r):
                    stats[r] = fetch_cheap_stats(chunk, ds)
                    passing = [
                        t
                        for t in chunk
                        if t["tile_id"] in stats[r]
                        and cheap_valid(stats[r][t["tile_id"]])
                    ]
                    cover[r] = fetch_cover_means(passing, ds)
        except Exception as exc:
            print(f"  batch {i // batch_size + 1} failed, skipping: {exc}")
            continue

        rows = []
        for t in chunk:
            per_rule = {r: stats[r].get(t["tile_id"]) for r in RULES}
            if any(v is None for v in per_rule.values()):
                continue  # keep the data paired: drop tiles missing under either rule
            oks = {r: cheap_valid(per_rule[r]) for r in RULES}
            if any(oks[r] and cover[r].get(t["tile_id"]) is None for r in RULES):
                continue  # passed the filter but no cover value came back: drop, keep pairs clean
            for r in RULES:
                rows.append(
                    {
                        "tile_id": t["tile_id"],
                        "rule": r,
                        "cover_mean": cover[r].get(t["tile_id"]) if oks[r] else None,
                        "cheap_valid": oks[r],
                        "gain_pct": (per_rule[r].get("gain_frac") or 0.0) * 100.0,
                        "biome": t["biome"],
                    }
                )
                passes[r] += oks[r]
        evaluated += len(rows) // len(RULES)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(
            f"  evaluated {evaluated} tiles | cheap-valid: "
            + ", ".join(f"{r} {passes[r]}/{args.quota}" for r in RULES)
        )

    if not all(passes[r] >= args.quota for r in RULES):
        print(
            f"WARNING: pool exhausted before every rule reached {args.quota}; analysing what we have."
        )
    print()
    analyse(out, quota=args.quota)


if __name__ == "__main__":
    main()
