from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import ee
import numpy as np
from config import settings
from export.composites import (
    combine_year_validity,
    hemisphere_from_tile,
    s1_availability,
    s2_availability,
    s2_ndvi_trend,
)
from gee_datasets.registry import Datasets
from labels.gain import build_gain_layer
from tiling.grid import footprint_bounds_lonlat

NO_GAIN_SENTINEL = -9999.0

CHEAP_BAND_NAMES = [
    "gain_frac",
    "ndvi_trend",
]

GAIN_VALID_BAND = "gain_valid_frac"

S2_BAND_NAMES = [f"s2_{y}" for y in settings.years]
S1_BAND_NAMES = [f"s1_{y}" for y in settings.years]

# Gain stays at native 10 m (1% threshold needs the accuracy); only the NDVI
# trend, a mean slope compared to 0, runs coarser.
NDVI_SCALE = 20


def split_by_hemisphere(tiles: list[dict]) -> tuple[list[dict], list[dict]]:
    north, south = [], []
    for t in tiles:
        if hemisphere_from_tile(t["min_lat"], t["max_lat"]):
            north.append(t)
        else:
            south.append(t)
    return north, south


# GeoTesseraZarr (>= 0.11) streams v1.1 / dclimate, whose tile registry (one row per 2048 px tile, zone and
# year) replaces the old NPY manifest. Load it once, keep usable rows as sorted arrays, cache on disk.
_VERSION = str(getattr(settings, "tessera_dataset_version", "1.1")).lstrip("vV")
_VARIANT = str(getattr(settings, "tessera_dataset_variant", "dclimate"))
_CACHE = Path(
    getattr(settings, "tessera_cache_dir", None) or "~/.cache/geotessera"
).expanduser()


@lru_cache(maxsize=1)
def _coverage_index() -> dict[int, tuple]:
    """{year: (south, north, west, east, max_height)} of usable tiles (embedded, <= 50% refused)."""
    from geotessera.icechunk import TileRegistry
    from geotessera.registry import find_dataset

    reg = TileRegistry(find_dataset(_VERSION, _VARIANT).tile_registry, cache_dir=_CACHE)
    key = hashlib.md5("|".join(p[3] for p in reg.parts()).encode()).hexdigest()[:12]
    path = _CACHE / f"coverage_{_VERSION}_{_VARIANT}_{key}.npz"
    if not path.exists():
        print(
            "TESSERA: building coverage index (first run only, downloads the tile registry)...",
            flush=True,
        )
        df = reg.tiles()
        refused = (df.refused_px / df.eligible_px.where(df.eligible_px > 0)).fillna(0)
        df = df[df.embedded.astype(bool) & (refused <= 0.5)]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            year=df.year,
            s=df.bbox_south,
            n=df.bbox_north,
            w=df.bbox_west,
            e=df.bbox_east,
        )
        tmp.replace(path)
    z = np.load(path)
    index = {}
    for y in np.unique(z["year"]):
        m = z["year"] == y
        o = np.argsort(z["s"][m])
        s, n, w, e = (z[k][m][o] for k in "snwe")
        index[int(y)] = (s, n, w, e, float((n - s).max()))
    return index


def _has_data(index, year, bbox) -> bool:
    s, n, w, e, h = index[year]
    west, south, east, north = bbox
    lo, hi = np.searchsorted(s, south - h), np.searchsorted(s, north, "right")
    s, n, w, e = s[lo:hi], n[lo:hi], w[lo:hi], e[lo:hi]
    lon = np.where(
        w > e, (w <= east) | (e >= west), (w <= east) & (e >= west)
    )  # w > e: antimeridian
    return bool((lon & (n >= south)).any())


def warm_tessera_cache() -> None:
    _coverage_index()


def check_tessera_coverage(tiles, logger=None):
    index = _coverage_index()
    missing_by_tile: dict[str, list[int]] = {}

    for tile in tiles:
        bbox = footprint_bounds_lonlat(
            tile["x_min_m"], tile["y_min_m"], tile["x_max_m"], tile["y_max_m"]
        )
        missing = [
            y for y in settings.years if y not in index or not _has_data(index, y, bbox)
        ]
        if missing:
            missing_by_tile[tile["tile_id"]] = missing
            if logger:
                logger.debug(
                    f"TESSERA coverage missing | tile={tile['tile_id']} | years={missing}"
                )

    covered = [tile for tile in tiles if tile["tile_id"] not in missing_by_tile]
    return covered, missing_by_tile


def build_ndvi_stats_image(
    geom: ee.Geometry,
    ds: Datasets,
    *,
    north: bool,
) -> ee.Image:
    gain_validated, _, _ = build_gain_layer(geom, ds)
    return (
        s2_ndvi_trend(geom, settings.years, north=north)
        .updateMask(gain_validated.selfMask())
        .rename("ndvi_trend")
    )


def build_imagery_stats_image(geom: ee.Geometry, ds: Datasets) -> ee.Image:
    """
    Bands: s2_<year> per-year availability (cloud+snow masked), and
    gain_valid_frac = gain pixels that are valid in every year.
    """
    per_year = [s2_availability(geom, y) for y in settings.years]
    all_valid = combine_year_validity(per_year)

    gain_validated, _, _ = build_gain_layer(geom, ds)
    gain_valid = gain_validated.unmask(0).And(all_valid).rename(GAIN_VALID_BAND)

    bands = [img.rename(f"s2_{y}") for img, y in zip(per_year, settings.years)]
    bands.append(gain_valid)
    return ee.Image.cat(bands).clip(geom)


def tiles_to_feature_collection(tiles: list[dict]) -> ee.FeatureCollection:
    features = []
    for t in tiles:
        geom = ee.Geometry.Rectangle(
            [t["x_min_m"], t["y_min_m"], t["x_max_m"], t["y_max_m"]],
            proj=ee.Projection(settings.crs_wkt),
            geodesic=False,
        )
        features.append(ee.Feature(geom, {"tile_id": t["tile_id"]}))
    return ee.FeatureCollection(features)


def _reduce_tiles(
    stats: ee.Image,
    tiles: list[dict],
    band_names: list[str],
    *,
    tile_scale: int = 4,
    scale: int | None = None,
) -> dict[str, dict[str, float]]:
    fc = tiles_to_feature_collection(tiles)

    reducer = ee.Reducer.mean()
    if len(band_names) == 1:
        # Single-band reduceRegions names the output property "mean", not the
        # band name. Force it, otherwise every lookup below returns None.
        reducer = reducer.setOutputs(band_names)

    reduced = stats.reduceRegions(
        collection=fc,
        reducer=reducer,
        scale=scale or settings.scale,
        tileScale=tile_scale,
    )
    result = reduced.getInfo()
    out = {}
    for feature in result["features"]:
        props = feature["properties"]
        out[props["tile_id"]] = {band: props.get(band) for band in band_names}
    return out


def _with_retry(fn, *, attempts: int = 4, base_delay: float = 5.0):
    """Retry transient transport/quota errors only; real EE errors raise at once."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            transient = (
                "Connection aborted" in msg
                or "RemoteDisconnected" in msg
                or "timed out" in msg.lower()
                or "429" in msg
                or "Too many concurrent" in msg
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)


def fetch_cheap_stats(
    tiles: list[dict],
    ds: Datasets,
    *,
    chunk_size: int = 20,
    max_workers: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Per chunk: gain at 10 m first (no Sentinel-2), then NDVI trend at
    NDVI_SCALE only for tiles whose gain clears the threshold. Chunked per
    hemisphere (NDVI trend is leaf-on), fetched concurrently, transient
    errors retried.
    """
    north_tiles, south_tiles = split_by_hemisphere(tiles)

    jobs = []
    for group, north in ((north_tiles, True), (south_tiles, False)):
        for i in range(0, len(group), chunk_size):
            jobs.append((group[i : i + chunk_size], north))

    def run(job):
        chunk, north = job

        def go():
            geom = tiles_to_feature_collection(chunk).geometry()
            # The label layer (labels/gain.py), which follows GAIN_RULE.
            _, gain_binary, _ = build_gain_layer(geom, ds)
            gain = _reduce_tiles(
                gain_binary.rename("gain_frac"), chunk, ["gain_frac"], tile_scale=2
            )
            result = {
                t["tile_id"]: {
                    "gain_frac": gain.get(t["tile_id"], {}).get("gain_frac"),
                    "ndvi_trend": None,
                }
                for t in chunk
            }

            keep = [
                t
                for t in chunk
                if (result[t["tile_id"]]["gain_frac"] or 0.0) * 100.0
                >= settings.gain_pct_min
            ]
            if keep:
                geom_k = tiles_to_feature_collection(keep).geometry()
                ndvi = _reduce_tiles(
                    build_ndvi_stats_image(geom_k, ds, north=north),
                    keep,
                    ["ndvi_trend"],
                    tile_scale=2,
                    scale=NDVI_SCALE,
                )
                for tid, vals in ndvi.items():
                    result[tid]["ndvi_trend"] = vals.get("ndvi_trend")
            return result

        return _with_retry(go)

    out: dict[str, dict[str, float]] = {}
    if not jobs:
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(run, jobs):
            out.update(result)

    return out


def fetch_imagery_stats(
    tiles: list[dict],
    ds: Datasets | None = None,
    *,
    chunk_size: int = 10,
    max_workers: int = 4,
) -> dict[str, dict[str, float | None]]:
    ds = ds or Datasets()
    years = settings.years
    s2_names = S2_BAND_NAMES + [GAIN_VALID_BAND]

    def add_s1(feature):
        for y in years:
            feature = feature.set(f"s1_{y}", s1_availability(feature, y))
        return feature

    def one_chunk(chunk):
        fc = tiles_to_feature_collection(chunk)
        geom = fc.geometry()

        per_year = [s2_availability(geom, y) for y in years]
        all_valid = combine_year_validity(per_year)

        gain_validated, _, _ = build_gain_layer(geom, ds)
        gain_valid = gain_validated.unmask(0).And(all_valid).rename(GAIN_VALID_BAND)

        bands = [img.rename(f"s2_{y}") for img, y in zip(per_year, years)]
        bands.append(gain_valid)

        reduced = ee.Image.cat(bands).reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=settings.scale,
            tileScale=4,
        )

        info = reduced.map(add_s1).getInfo()

        out = {}
        for f in info["features"]:
            p = f["properties"]
            out[p["tile_id"]] = {
                **{b: p.get(b) for b in s2_names},
                **{f"s1_{y}": p.get(f"s1_{y}") for y in years},
            }

        return out

    chunks = [tiles[i : i + chunk_size] for i in range(0, len(tiles), chunk_size)]

    results: dict[str, dict[str, float | None]] = {}
    if not chunks:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(
            lambda c: _with_retry(lambda: one_chunk(c)),
            chunks,
        ):
            results.update(r)

    return results
