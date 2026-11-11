#!/usr/bin/env python3
"""
Extract per-tile feature summaries for all valid pixels and for gain pixels only.

This script processes a directory containing many tile_* folders, reads each tile's
metadata.json, and writes CSV feature tables for:
  - all valid pixels
  - gain pixels only

It follows the same band indexing / valid-mask logic as ModelDev/datasets.py:
  - S1/S2 from composites/s1s2_{year}.tif
  - AlphaEarth and Tessera from embeddings/*.tif
  - gain confidence from labels/gain_confidence.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ModelDev.config import (
    BAND_FILE_INDEX,
    EMBEDDING_ABS_MAX,
    S2_BANDS,
    S2_SCALE,
    SOURCES,
    VALID_MASK_BAND_INDEX,
    YEARS,
)


def iter_tile_dirs(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    tiles = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("tile_")
    )
    if not tiles:
        raise FileNotFoundError(f"No tile_* dirs found in {data_dir}")
    return tiles


def read_tile_metadata(tile_dir: Path) -> dict:
    """
    Read tile metadata.json if present.
    Keeps the useful metadata fields for interpretation later.
    """
    path = tile_dir / "metadata.json"
    if not path.exists():
        return {}

    try:
        with path.open() as f:
            metadata = json.load(f)
    except Exception:
        return {}

    bounds = metadata.get("bounds") or {}
    return {
        "tile_id": metadata.get("tile_id", tile_dir.name),
        "biome": metadata.get("biome"),
        "region": metadata.get("region"),
        "country": metadata.get("country"),
        "gain_pct_metadata": metadata.get("gain_pct"),
        "min_lon": bounds.get("min_lon"),
        "min_lat": bounds.get("min_lat"),
        "max_lon": bounds.get("max_lon"),
        "max_lat": bounds.get("max_lat"),
    }


def read_physical(
    path: Path, band_names: list[str], mask_nodata: bool = False
) -> np.ndarray:
    """
    Read the exact configured band indices for a raster, matching ModelDev.datasets.
    """
    indices = [BAND_FILE_INDEX[b] for b in band_names]

    with rasterio.open(path) as src:
        arr = src.read(indices).astype(np.float32)
        nodata = src.nodata

    if mask_nodata:
        if nodata is not None and not np.isnan(nodata):
            arr[arr == np.float32(nodata)] = np.nan

        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(arr) | (np.abs(arr) > EMBEDDING_ABS_MAX)
            arr[:, bad.any(axis=0)] = np.nan

    for i, band in enumerate(band_names):
        if band in S2_BANDS:
            arr[i] /= S2_SCALE

    return arr


def valid_mask_for_year(tile_dir: Path, year: int) -> np.ndarray:
    comp = tile_dir / "composites" / f"s1s2_{year}.tif"
    if not comp.exists():
        return np.array([], dtype=np.float32)

    with rasterio.open(comp) as src:
        vm = src.read(VALID_MASK_BAND_INDEX).astype(np.float32)
    vm = np.nan_to_num(vm, nan=0.0)
    return (vm > 0).astype(bool)


def read_gain_layers(tile_dir: Path):
    """Read gain confidence and derive the gain mask using confidence > 0."""
    path = tile_dir / "labels" / "gain_confidence.tif"
    if not path.exists():
        return None, None

    with rasterio.open(path) as src:
        confidence = src.read(1).astype(np.float32)

    # Treat NaN and infinities as background; any finite positive value is gain.
    confidence = np.where(np.isfinite(confidence), confidence, 0.0).astype(np.float32)
    gain_mask = (confidence > 0).astype(np.float32)

    return gain_mask, confidence


def frame_stats(
    frame: np.ndarray,
    valid_mask: np.ndarray,
    band_names: list[str],
    prefix: str,
) -> dict:
    out = {}
    if frame.shape[1:] != valid_mask.shape:
        return out

    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return out

    flat = frame[:, valid]

    for i, band in enumerate(band_names):
        vals = flat[i].astype(np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{prefix}_{band}_median"] = float(np.median(vals))
        out[f"{prefix}_{band}_p10"] = float(np.percentile(vals, 10))
        out[f"{prefix}_{band}_p90"] = float(np.percentile(vals, 90))
        out[f"{prefix}_{band}_std"] = float(np.std(vals))
        out[f"{prefix}_{band}_iqr"] = float(
            np.percentile(vals, 75) - np.percentile(vals, 25)
        )

    return out


def temporal_band_trends(
    year_frames: dict[int, dict[str, np.ndarray]],
    valid_mask: np.ndarray,
    source: str,
    band_names: list[str],
) -> dict:
    out = {}

    for band_idx, band in enumerate(band_names):
        annual_vals = []

        for year in sorted(year_frames):
            frame = year_frames[year].get(source)
            if frame is None:
                continue
            vals = frame[band_idx, valid_mask].astype(np.float32)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            annual_vals.append(float(np.median(vals)))

        if len(annual_vals) < 2:
            continue

        out[f"{source}_{band}_first"] = float(annual_vals[0])
        out[f"{source}_{band}_last"] = float(annual_vals[-1])
        out[f"{source}_{band}_delta"] = float(annual_vals[-1] - annual_vals[0])
        out[f"{source}_{band}_trend"] = float(
            np.polyfit(
                np.arange(len(annual_vals)),
                np.asarray(annual_vals, dtype=np.float32),
                1,
            )[0]
        )
        out[f"{source}_{band}_var"] = float(np.var(annual_vals))

    return out


def tile_all_valid_features(
    tile_dir: Path,
    years: Iterable[int],
    sources: Iterable[str],
) -> dict:
    years = tuple(sorted(int(y) for y in years))
    sources = tuple(sources)

    year_frames: dict[int, dict[str, np.ndarray]] = {}
    valid_masks = []

    for year in years:
        per_source = {}
        for source in sources:
            template, bands = SOURCES[source]
            path = tile_dir / template.format(year=year)
            if not path.exists():
                continue

            arr = read_physical(
                path,
                list(bands),
                mask_nodata=(source in ("alphaearth", "tessera")),
            )
            per_source[source] = arr

        if per_source:
            year_frames[year] = per_source

        valid = valid_mask_for_year(tile_dir, year)
        if valid.size:
            valid_masks.append(valid)

    if not year_frames:
        return {}

    if valid_masks:
        valid = np.logical_and.reduce(np.stack(valid_masks, axis=0), axis=0)
    else:
        return {}

    if valid.size == 0 or valid.sum() == 0:
        return {}

    metadata = read_tile_metadata(tile_dir)

    out = {
        **metadata,
        "tile_id": tile_dir.name,
        "n_valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }

    # Per-year summaries
    for year in years:
        if year not in year_frames:
            continue
        for source in sources:
            if source not in year_frames[year]:
                continue
            arr = year_frames[year][source]
            bands = list(SOURCES[source][1])
            out.update(frame_stats(arr, valid, bands, f"{year}_{source}"))

    # Temporal trajectories
    for source in sources:
        bands = list(SOURCES[source][1])
        subset = {
            y: {source: year_frames[y][source]}
            for y in year_frames
            if source in year_frames[y]
        }
        out.update(temporal_band_trends(subset, valid, source, bands))

    # Gain summaries
    gain_mask, gain_conf = read_gain_layers(tile_dir)
    if gain_mask is not None:
        gain_mask = np.asarray(gain_mask, dtype=np.float32)
        gain_mask = np.nan_to_num(gain_mask, nan=0.0)

        if gain_conf is not None:
            gain_conf = np.asarray(gain_conf, dtype=np.float32)
            gain_conf = np.nan_to_num(gain_conf, nan=0.0)

        gain_valid = valid & (gain_mask > 0.5)
        out["gain_n_pixels"] = int(gain_valid.sum())
        out["gain_fraction"] = float(gain_valid.mean()) if valid.size else 0.0

        if gain_valid.sum() > 0:
            for year in years:
                if year not in year_frames:
                    continue
                for source in sources:
                    if source not in year_frames[year]:
                        continue
                    arr = year_frames[year][source]
                    bands = list(SOURCES[source][1])
                    gain_arr = arr[:, gain_valid]
                    for i, band in enumerate(bands):
                        vals = gain_arr[i].astype(np.float32)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        out[f"gain_{year}_{source}_{band}_median"] = float(
                            np.median(vals)
                        )
                        out[f"gain_{year}_{source}_{band}_p10"] = float(
                            np.percentile(vals, 10)
                        )
                        out[f"gain_{year}_{source}_{band}_p90"] = float(
                            np.percentile(vals, 90)
                        )
                        out[f"gain_{year}_{source}_{band}_std"] = float(np.std(vals))

            if gain_conf is not None:
                conf_vals = gain_conf[gain_valid]
                conf_vals = conf_vals[np.isfinite(conf_vals)]
                if conf_vals.size > 0:
                    out["gain_conf_mean"] = float(np.mean(conf_vals))
                    out["gain_conf_p10"] = float(np.percentile(conf_vals, 10))
                    out["gain_conf_p90"] = float(np.percentile(conf_vals, 90))

    return out


def tile_gain_only_features(
    tile_dir: Path,
    years: Iterable[int],
    sources: Iterable[str],
) -> dict:
    years = tuple(sorted(int(y) for y in years))
    sources = tuple(sources)

    gain_mask, gain_conf = read_gain_layers(tile_dir)
    if gain_mask is None:
        return {}

    gain_mask = np.asarray(gain_mask, dtype=np.float32)
    gain_mask = np.nan_to_num(gain_mask, nan=0.0)

    valid_masks = []
    for year in years:
        vm = valid_mask_for_year(tile_dir, year)
        if vm.size:
            valid_masks.append(vm)

    if not valid_masks:
        return {}

    valid = (
        np.logical_and.reduce(np.stack(valid_masks, axis=0), axis=0)
        if len(valid_masks) > 1
        else valid_masks[0]
    )

    gain_valid = valid & (gain_mask > 0.5)
    if not gain_valid.any():
        return {}

    metadata = read_tile_metadata(tile_dir)

    out = {
        **metadata,
        "tile_id": tile_dir.name,
        "gain_n_pixels": int(gain_valid.sum()),
        "gain_fraction": float(gain_valid.mean()) if valid.size else 0.0,
    }

    for year in years:
        per_source = {}
        for source in sources:
            template, bands = SOURCES[source]
            path = tile_dir / template.format(year=year)
            if not path.exists():
                continue

            arr = read_physical(
                path,
                list(bands),
                mask_nodata=(source in ("alphaearth", "tessera")),
            )
            per_source[source] = arr

        for source in sources:
            arr = per_source.get(source)
            if arr is None:
                continue
            bands = list(SOURCES[source][1])
            gain_arr = arr[:, gain_valid]
            for i, band in enumerate(bands):
                vals = gain_arr[i].astype(np.float32)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                out[f"gain_{year}_{source}_{band}_median"] = float(np.median(vals))
                out[f"gain_{year}_{source}_{band}_p10"] = float(np.percentile(vals, 10))
                out[f"gain_{year}_{source}_{band}_p90"] = float(np.percentile(vals, 90))
                out[f"gain_{year}_{source}_{band}_std"] = float(np.std(vals))
                out[f"gain_{year}_{source}_{band}_iqr"] = float(
                    np.percentile(vals, 75) - np.percentile(vals, 25)
                )

    if gain_conf is not None:
        gain_conf = np.asarray(gain_conf, dtype=np.float32)
        gain_conf = np.nan_to_num(gain_conf, nan=0.0)
        conf_vals = gain_conf[gain_valid]
        conf_vals = conf_vals[np.isfinite(conf_vals)]
        if conf_vals.size:
            out["gain_conf_mean"] = float(np.mean(conf_vals))
            out["gain_conf_p10"] = float(np.percentile(conf_vals, 10))
            out["gain_conf_p90"] = float(np.percentile(conf_vals, 90))

    return out


def build_features(
    data_dir: str | Path,
    years: Iterable[int],
    sources: Iterable[str],
    mode: str = "all_valid",
) -> pd.DataFrame:
    tile_dirs = iter_tile_dirs(data_dir)
    rows = []

    if mode == "all_valid":
        for tile_dir in tile_dirs:
            row = tile_all_valid_features(tile_dir, years, sources)
            if row:
                rows.append(row)
    elif mode == "gain_only":
        for tile_dir in tile_dirs:
            row = tile_gain_only_features(tile_dir, years, sources)
            if row:
                rows.append(row)
    else:
        raise ValueError("mode must be 'all_valid' or 'gain_only'")

    if not rows:
        raise ValueError(f"No usable rows found for mode='{mode}' in {data_dir}")

    df = pd.DataFrame(rows)
    return df.sort_values("tile_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Tile feature extraction")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["s1", "s2", "alphaearth", "tessera"],
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument(
        "--mode",
        choices=["all_valid", "gain_only"],
        default="all_valid",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_features(args.data_dir, args.years, args.sources, mode=args.mode)
    out_path = out_dir / f"tile_{args.mode}_features.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} tiles to {out_path}")
    print(df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()
