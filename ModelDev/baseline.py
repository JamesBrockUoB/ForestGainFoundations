"""Per-pixel baselines (logistic regression / MLP / LightGBM) + zero-shot cosine
change scores, on exactly the same tiles, valid mask and labels as the deep models.

Purpose: tell apart "the signal is not in the inputs" from "the deep model is not
extracting it". It needs config.py (the updated one with SOURCES) but not torch.

    pip install scikit-learn lightgbm rasterio
    python baseline.py --sources alphaearth
    python baseline.py --sources tessera
    python baseline.py --sources s1 s2          # reference: should clearly learn

Every run prints input sanity checks first (value ranges, int8 leftovers, embedding
norms, NaN and clip fractions, label prevalence), then a results table.
"""

import argparse
import random
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from config import (
    BAND_FILE_INDEX,
    EMBEDDING_SOURCES,
    NORM_CLIP,
    NORM_NAN_FILL,
    NORM_STATS,
    S2_BANDS,
    S2_SCALE,
    SOURCES,
    VALID_MASK_BAND_INDEX,
    YEARS,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------- data
def split_tile_dirs(data_dir, val_frac=0.2, seed=0):
    """Identical logic to datasets.split_tile_dirs (kept here so torch isn't needed)."""
    tile_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    random.Random(seed).shuffle(tile_dirs)
    n_val = max(1, round(val_frac * len(tile_dirs)))
    return tile_dirs[n_val:], tile_dirs[:n_val]


def read_bands(path, band_names, mask_nodata):
    idx = [BAND_FILE_INDEX[b] for b in band_names]
    with rasterio.open(path) as src:
        x = src.read(idx).astype(np.float32)
        nodata, transform, shape = src.nodata, src.transform, (src.height, src.width)
    if mask_nodata and nodata is not None and not np.isnan(nodata):
        x[x == np.float32(nodata)] = np.nan
    for i, b in enumerate(band_names):
        if b in S2_BANDS:
            x[i] /= S2_SCALE
    return x, transform, shape


def load_tile(tile_dir, sources):
    """-> phys (T,C,H,W) physical units (NaN kept), valid (H,W), gain (H,W) bool."""
    frames, valid_masks, warned = [], [], False
    with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
        conf = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
        label_tf = src.transform
    for year in YEARS:
        parts = []
        for s in sources:
            template, bands = SOURCES[s]
            x, tf, shape = read_bands(
                tile_dir / template.format(year=year), bands, s in EMBEDDING_SOURCES
            )
            if (
                s in EMBEDDING_SOURCES
                and not warned
                and (tf != label_tf or shape != conf.shape)
            ):
                print(f"  !! grid mismatch vs label in {tile_dir.name}/{s}/{year}")
                warned = True
            parts.append(x)
        frames.append(np.concatenate(parts, axis=0))
        with rasterio.open(tile_dir / "composites" / f"s1s2_{year}.tif") as src:
            vm = src.read(VALID_MASK_BAND_INDEX).astype(np.float32)
        valid_masks.append(np.nan_to_num(vm, nan=0.0) > 0)
    return (
        np.stack(frames),
        np.all(valid_masks, axis=0),
        conf > 0,
    )


def normalise(phys, band_names):
    out = np.empty_like(phys)
    for i, b in enumerate(band_names):
        lo, hi = NORM_STATS[b]
        out[:, i] = (phys[:, i] - lo) / (hi - lo)
    if NORM_CLIP is not None:
        np.clip(out, NORM_CLIP[0], NORM_CLIP[1], out=out)
    return np.nan_to_num(
        out, nan=NORM_NAN_FILL, posinf=NORM_NAN_FILL, neginf=NORM_NAN_FILL
    )


def center_crop(arrs, s):
    h, w = arrs[0].shape[-2:]
    if not s or (h <= s and w <= s):
        return arrs
    t, l = (h - s) // 2, (w - s) // 2
    return [a[..., t : t + s, l : l + s] for a in arrs]


# ------------------------------------------------------------------- features
def change_features(P, sources):
    """P: (T, C, n) physical values of the sampled pixels.
    Per embedding source: cosine distance and L2 distance between consecutive
    years -> 2*(T-1) features each."""
    feats, c0 = [], 0
    for s in sources:
        c = len(SOURCES[s][1])
        if s in EMBEDDING_SOURCES:
            E = np.nan_to_num(P[:, c0 : c0 + c])
            a, b = E[1:], E[:-1]
            num = (a * b).sum(1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
            feats += [1.0 - num / den, np.linalg.norm(a - b, axis=1)]
        c0 += c
    return np.concatenate(feats, 0).T if feats else None  # (n, 2*(T-1)*n_emb)


def sample_tile(tile_dir, sources, band_names, mode, rng, args, with_change):
    phys, valid, gain = load_tile(tile_dir, sources)
    if mode == "val":
        phys, valid, gain = center_crop([phys, valid, gain], args.val_crop)
    T, C, H, W = phys.shape
    v, g = valid.reshape(-1), gain.reshape(-1)
    pos, neg = np.flatnonzero(v & g), np.flatnonzero(v & ~g)
    if mode == "train":
        pos = rng.permutation(pos)[: args.pos_cap]
        n_neg = min(len(neg), max(args.neg_ratio * len(pos), args.min_neg))
        idx = np.concatenate([pos, rng.permutation(neg)[:n_neg]])
    else:
        allv = np.flatnonzero(v)
        idx = rng.permutation(allv)[: args.val_px_per_tile]
    if len(idx) == 0:
        return None
    flat_phys = phys.reshape(T, C, -1)[:, :, idx]  # (T, C, n)
    X = normalise(flat_phys, band_names)  # (T, C, n), same scaling as the dataset
    X = X.reshape(T * C, -1).T  # (n, T*C)
    y = g[idx].astype(np.int8)
    ch = change_features(flat_phys, sources) if with_change else None
    return X, y, ch, flat_phys


# -------------------------------------------------------------------- metrics
def evaluate(y, p):
    ap = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    iou = prec * rec / np.maximum(prec + rec - prec * rec, 1e-12)
    k = int(np.nanargmax(iou[:-1])) if len(thr) else 0
    return {
        "ap": ap,
        "best_iou": float(iou[k]),
        "thr": float(thr[k]) if len(thr) else float("nan"),
        "iou@0.5": float(
            ((p > 0.5) & (y == 1)).sum() / max(((p > 0.5) | (y == 1)).sum(), 1)
        ),
    }


def make_models(names, seed):
    out = {}
    if "logreg" in names:
        out["logreg"] = LogisticRegression(C=0.1, max_iter=300, n_jobs=-1)
    if "mlp" in names:
        out["mlp"] = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            alpha=1e-3,
            batch_size=1024,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            max_iter=60,
            random_state=seed,
        )
    if "lgbm" in names:
        import lightgbm as lgb

        out["lgbm"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            random_state=seed,
            verbose=-1,
        )
    return out


# ----------------------------------------------------------------- sanity info
def sanity(sources, band_names, phys_sample, X_train, y_train, y_val):
    print("\n=== input sanity checks ===")
    T = phys_sample.shape[0]
    c0 = 0
    for s in sources:
        c = len(SOURCES[s][1])
        v = phys_sample[:, c0 : c0 + c]
        fin = v[np.isfinite(v)]
        if fin.size:
            ints = np.mean(np.abs(fin - np.round(fin)) < 1e-6)
            print(
                f"[{s}] physical: min {fin.min():.4g} max {fin.max():.4g} "
                f"mean {fin.mean():.4g} std {fin.std():.4g} | NaN {1 - fin.size / v.size:.2%}"
                f" | integer-valued {ints:.1%}"
            )
            if s in EMBEDDING_SOURCES:
                nrm = np.linalg.norm(np.nan_to_num(v[T // 2]), axis=0)
                print(
                    f"      embedding L2 norm (mid year): median {np.median(nrm):.3f} "
                    f"p5 {np.percentile(nrm, 5):.3f} p95 {np.percentile(nrm, 95):.3f}"
                )
                if ints > 0.9 and np.abs(fin).max() > 2:
                    print(
                        "      !! values look like raw int8 -- TESSERA needs "
                        "value * per-pixel scale before use"
                    )
        c0 += c
    print(
        f"normalised: at clip limits {np.mean((X_train <= NORM_CLIP[0] + 1e-6) | (X_train >= NORM_CLIP[1] - 1e-6)):.2%}"
        f" | exactly NORM_NAN_FILL {np.mean(X_train == NORM_NAN_FILL):.2%}"
    )
    print(
        f"label prevalence: train sample {y_train.mean():.2%} (rebalanced) | "
        f"val {y_val.mean():.3%}  <- AP of a random scorer equals this"
    )


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../DataCollection/data/test_tiles")
    ap.add_argument(
        "--sources", nargs="+", default=["alphaearth"], choices=list(SOURCES)
    )
    ap.add_argument("--models", nargs="+", default=["logreg", "lgbm", "mlp"])
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="must match train.py")
    ap.add_argument("--max_tiles", type=int, default=0, help="debug: limit train tiles")
    ap.add_argument(
        "--pos_cap", type=int, default=1500, help="positives per train tile"
    )
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--min_neg", type=int, default=1000)
    ap.add_argument("--val_px_per_tile", type=int, default=20000)
    ap.add_argument(
        "--val_crop",
        type=int,
        default=128,
        help="center crop like train.py val (0=full)",
    )
    args = ap.parse_args()

    sources = tuple(args.sources)
    band_names = tuple(b for s in sources for b in SOURCES[s][1])
    missing = [b for b in band_names if b not in NORM_STATS]
    if missing:
        raise SystemExit(
            f"No NORM_STATS for {missing[:3]}... run compute_embedding_stats.py"
        )
    has_emb = any(s in EMBEDDING_SOURCES for s in sources)

    train_dirs, val_dirs = split_tile_dirs(args.data_dir, args.val_frac, args.seed)
    if args.max_tiles:
        train_dirs = train_dirs[: args.max_tiles]
    print(
        f"train tiles {len(train_dirs)} | val tiles {len(val_dirs)} | sources {sources}"
    )
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    data = {}
    for mode, dirs in (("train", train_dirs), ("val", val_dirs)):
        Xs, ys, chs, phs = [], [], [], []
        for d in dirs:
            r = sample_tile(d, sources, band_names, mode, rng, args, has_emb)
            if r is None:
                continue
            Xs.append(r[0])
            ys.append(r[1])
            if r[2] is not None:
                chs.append(r[2])
            if mode == "val":
                phs.append(r[3])
        data[mode] = (
            np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(chs) if chs else None,
            np.concatenate(phs, axis=2) if phs else None,
        )
        print(f"{mode}: {len(data[mode][1]):,} pixels loaded ({time.time() - t0:.0f}s)")

    Xtr, ytr, Ctr, _ = data["train"]
    Xva, yva, Cva, Pva = data["val"]
    if yva.sum() == 0 or ytr.sum() == 0:
        raise SystemExit("no positive pixels in train or val sample")
    sanity(sources, band_names, Pva[:, :, : min(50000, Pva.shape[2])], Xtr, ytr, yva)

    rows = []
    # zero-shot cosine/L2 scores (no training at all)
    if has_emb:
        T = Pva.shape[0]
        c0 = 0
        for s in sources:
            c = len(SOURCES[s][1])
            if s in EMBEDDING_SOURCES:
                E = np.nan_to_num(Pva[:, c0 : c0 + c])
                a, b = E[1:], E[:-1]
                cd = 1 - (a * b).sum(1) / (
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
                )
                fl = 1 - (E[-1] * E[0]).sum(0) / (
                    np.linalg.norm(E[-1], axis=0) * np.linalg.norm(E[0], axis=0) + 1e-8
                )
                for name, sc in (
                    ("zero-shot max consecutive cosine dist", cd.max(0)),
                    ("zero-shot first-vs-last cosine dist", fl),
                    (
                        "zero-shot max consecutive L2 dist",
                        np.linalg.norm(a - b, axis=1).max(0),
                    ),
                ):
                    sc = (sc - sc.min()) / (np.ptp(sc) + 1e-12)
                    rows.append((f"{s}: {name}", evaluate(yva, sc)))
            c0 += c

    feature_sets = {"raw": (Xtr, Xva)}
    if has_emb:
        feature_sets["raw+change"] = (np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva]))
    for fname, (A, B) in feature_sets.items():
        sc = StandardScaler().fit(A)
        A, B = sc.transform(A), sc.transform(B)
        for mname, model in make_models(args.models, args.seed).items():
            t1 = time.time()
            model.fit(A, ytr)
            p = model.predict_proba(B)[:, 1]
            rows.append((f"{fname:10s} {mname}", evaluate(yva, p)))
            print(f"  fitted {fname}/{mname} in {time.time() - t1:.0f}s")

    print(f"\n=== results on {len(yva):,} val pixels (prevalence {yva.mean():.3%}) ===")
    print(f"{'':58s} {'AP':>7s} {'bestIoU':>8s} {'thr':>6s} {'IoU@.5':>7s}")
    for name, m in rows:
        print(
            f"{name:58s} {m['ap']:7.3f} {m['best_iou']:8.3f} {m['thr']:6.2f} {m['iou@0.5']:7.3f}"
        )
    print(f"{'random scorer (AP = prevalence)':58s} {yva.mean():7.3f}")


if __name__ == "__main__":
    main()
