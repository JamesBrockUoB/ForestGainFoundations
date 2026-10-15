"""Per-pixel baselines (logistic regression / MLP / LightGBM) + zero-shot cosine
change scores, on exactly the same tiles, valid mask and labels as the deep models.

Purpose: tell apart "the signal is not in the inputs" from "the deep model is not
extracting it". It needs config.py (the updated one with SOURCES) but not torch.

    pip install scikit-learn lightgbm rasterio
    python baseline.py --sources alphaearth
    python baseline.py --sources tessera
    python baseline.py --sources s1 s2

Memory-lean version: tiles are never loaded in full. Labels and valid masks (single
band) are read first, the pixels to sample are chosen, then each (year, source) is
read once, only the sampled pixels are kept, and the full array is freed
immediately. Val tiles are read through a centre-crop window. Zero-shot scores are
computed per tile, so raw val pixels are not kept around.
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
from rasterio.windows import Window
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


def crop_window(h, w, s):
    """Centre-crop window (None = full tile)."""
    if not s or (h <= s and w <= s):
        return None
    t, l = max(0, (h - s) // 2), max(0, (w - s) // 2)
    return Window(l, t, min(s, w), min(s, h))


def read_bands(path, band_names, mask_nodata, window=None):
    idx = [BAND_FILE_INDEX[b] for b in band_names]
    with rasterio.open(path) as src:
        x = src.read(idx, window=window, out_dtype="float32")  # no extra copy
        nodata, transform, shape = src.nodata, src.transform, (src.height, src.width)
    if mask_nodata and nodata is not None and not np.isnan(nodata):
        x[x == np.float32(nodata)] = np.nan
    for i, b in enumerate(band_names):
        if b in S2_BANDS:
            x[i] /= S2_SCALE
    return x, transform, shape


def load_masks(tile_dir, window):
    """Single-band reads only: -> valid (H,W) bool, gain (H,W) bool, label transform, full shape."""
    with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
        conf = np.nan_to_num(src.read(1, window=window, out_dtype="float32"), nan=0.0)
        label_tf, full_shape = src.transform, (src.height, src.width)
    valid = np.ones(conf.shape, dtype=bool)
    for year in YEARS:
        with rasterio.open(tile_dir / "composites" / f"s1s2_{year}.tif") as src:
            vm = src.read(VALID_MASK_BAND_INDEX, window=window, out_dtype="float32")
        valid &= np.nan_to_num(vm, nan=0.0) > 0
    return valid, conf > 0, label_tf, full_shape


def normalise(phys, sources):
    out = np.empty_like(phys)
    c0 = 0

    for s in sources:
        bands = SOURCES[s][1]
        c = len(bands)
        if s in EMBEDDING_SOURCES:
            out[:, c0 : c0 + c] = np.nan_to_num(
                phys[:, c0 : c0 + c],
                nan=NORM_NAN_FILL,
                posinf=NORM_NAN_FILL,
                neginf=NORM_NAN_FILL,
            )
        else:
            for i, b in enumerate(bands, start=c0):
                lo, hi = NORM_STATS[b]
                out[:, i] = (phys[:, i] - lo) / (hi - lo)

        c0 += c

    if NORM_CLIP is not None:
        # Clip only non-embedding channels.
        c0 = 0
        for s in sources:
            c = len(SOURCES[s][1])
            if s not in EMBEDDING_SOURCES:
                np.clip(
                    out[:, c0 : c0 + c],
                    NORM_CLIP[0],
                    NORM_CLIP[1],
                    out=out[:, c0 : c0 + c],
                )
            c0 += c

    return out


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


def zero_shot_scores(P, sources):
    """Training-free change scores per embedding source. -> {name: (n,) array}"""
    out, c0 = {}, 0
    for s in sources:
        c = len(SOURCES[s][1])
        if s in EMBEDDING_SOURCES:
            E = np.nan_to_num(P[:, c0 : c0 + c])
            a, b = E[1:], E[:-1]
            cd = 1 - (a * b).sum(1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
            )
            fl = 1 - (E[-1] * E[0]).sum(0) / (
                np.linalg.norm(E[-1], axis=0) * np.linalg.norm(E[0], axis=0) + 1e-8
            )
            out[f"{s}: zero-shot max consecutive cosine dist"] = cd.max(0)
            out[f"{s}: zero-shot first-vs-last cosine dist"] = fl
            out[f"{s}: zero-shot max consecutive L2 dist"] = np.linalg.norm(
                a - b, axis=1
            ).max(0)
        c0 += c
    return out


def sample_tile(tile_dir, sources, mode, rng, args, with_change, want_phys):
    window = None
    if mode == "val" and args.val_crop:
        with rasterio.open(tile_dir / "labels" / "gain_confidence.tif") as src:
            window = crop_window(src.height, src.width, args.val_crop)

    valid, gain, label_tf, full_shape = load_masks(tile_dir, window)
    v, g = valid.reshape(-1), gain.reshape(-1)
    pos, neg = np.flatnonzero(v & g), np.flatnonzero(v & ~g)
    if mode == "train":
        pos = rng.permutation(pos)[: args.pos_cap]
        n_neg = min(len(neg), max(args.neg_ratio * len(pos), args.min_neg))
        idx = np.concatenate([pos, rng.permutation(neg)[:n_neg]])
    else:
        idx = rng.permutation(np.flatnonzero(v))[: args.val_px_per_tile]
    if len(idx) == 0:
        return None
    idx = np.sort(idx)

    C = sum(len(SOURCES[s][1]) for s in sources)
    flat_phys = np.empty((len(YEARS), C, len(idx)), dtype=np.float32)
    warned = False
    for ti, year in enumerate(YEARS):
        c0 = 0
        for s in sources:
            template, bands = SOURCES[s]
            x, tf, shape = read_bands(
                tile_dir / template.format(year=year),
                bands,
                s in EMBEDDING_SOURCES,
                window,
            )
            if (
                s in EMBEDDING_SOURCES
                and not warned
                and (tf != label_tf or shape != full_shape)
            ):
                print(f"  !! grid mismatch vs label in {tile_dir.name}/{s}/{year}")
                warned = True
            flat_phys[ti, c0 : c0 + len(bands)] = x.reshape(len(bands), -1)[:, idx]
            c0 += len(bands)
            del x  # free the full band stack right away

    T = flat_phys.shape[0]
    X = normalise(flat_phys, sources).reshape(T * C, -1).T.astype(np.float32)
    y = g[idx].astype(np.int8)
    ch = change_features(flat_phys, sources) if with_change else None
    zs = (
        zero_shot_scores(flat_phys, sources)
        if (mode == "val" and with_change)
        else None
    )
    return X, y, ch, zs, (flat_phys if want_phys else None)


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
def sanity(sources, phys_sample, X_train, y_train, y_val):
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
    ap.add_argument(
        "--sanity_px", type=int, default=50000, help="pixels kept for sanity stats"
    )
    args = ap.parse_args()

    sources = tuple(args.sources)

    missing = [
        b
        for s in sources
        if s not in EMBEDDING_SOURCES
        for b in SOURCES[s][1]
        if b not in NORM_STATS
    ]
    if missing:
        raise SystemExit(f"No NORM_STATS for {missing[:3]}...")
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
    zs_all, phys_keep, phys_n = {}, [], 0
    for mode, dirs in (("train", train_dirs), ("val", val_dirs)):
        Xs, ys, chs = [], [], []
        for d in dirs:
            want_phys = mode == "val" and phys_n < args.sanity_px
            r = sample_tile(d, sources, mode, rng, args, has_emb, want_phys)
            if r is None:
                continue
            Xs.append(r[0])
            ys.append(r[1])
            if r[2] is not None:
                chs.append(r[2].astype(np.float32))
            if r[3] is not None:
                for k, vals in r[3].items():
                    zs_all.setdefault(k, []).append(vals.astype(np.float32))
            if r[4] is not None:
                phys_keep.append(r[4])
                phys_n += r[4].shape[2]
        data[mode] = (
            np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(chs) if chs else None,
        )
        del Xs, ys, chs
        print(f"{mode}: {len(data[mode][1]):,} pixels loaded ({time.time() - t0:.0f}s)")

    Xtr, ytr, Ctr = data["train"]
    Xva, yva, Cva = data["val"]
    if yva.sum() == 0 or ytr.sum() == 0:
        raise SystemExit("no positive pixels in train or val sample")
    sanity(sources, np.concatenate(phys_keep, axis=2), Xtr, ytr, yva)
    del phys_keep

    rows = []
    # zero-shot cosine/L2 scores (no training at all)
    for name, parts in zs_all.items():
        sc = np.concatenate(parts)
        sc = (sc - sc.min()) / (np.ptp(sc) + 1e-12)
        rows.append((name, evaluate(yva, sc)))

    feature_sets = {"raw": (Xtr, Xva)}
    if has_emb:
        feature_sets["raw+change"] = (np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva]))
    del Ctr, Cva
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
