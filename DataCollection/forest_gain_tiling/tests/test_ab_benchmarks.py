import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import ee
import pytest
from config import settings
from export import composites
from inspector.service import point_centred_tile

TILE_POINTS: List[Tuple[str, float, float]] = [
    ("ireland_west_connemara", -9.9000, 53.4000),
    ("ireland_wicklow_mountains", -6.2000, 52.9000),
    ("spain_galicia_inland", -8.5000, 43.0000),
    ("spain_asturias_picos", -5.0000, 43.1000),
    ("germany_eifel", 6.5000, 50.4000),
    ("italy_cilento", 15.3000, 40.2000),
    ("england_fens", 0.2000, 52.5000),
    ("england_norfolk_broads", 1.4000, 52.6000),
    ("poland_bialowieza", 23.8500, 52.7000),
    ("france_luberon", 5.4000, 43.9000),
]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = composites.CLOUD_SCORE_PLUS_COLLECTION
CLOUD_SCORE_BAND = composites.CLOUD_SCORE_PLUS_BAND
DEFAULT_PROXY_BAND = getattr(settings, "s2_check_band", "B2")
DEFAULT_SCALE = int(getattr(settings, "scale", 10))

TEST_TILES = TILE_POINTS[:3]
TEST_YEARS = list(settings.period_years)[:2]

# Number of timed A/B trials per comparison. Order is shuffled on every
# trial (see _run_ab_trials) so no variant systematically runs first
# (paying a one-off cold-start cost) or last (benefiting from every
# other variant having just warmed the cache).
N_TRIALS = 5

# The batching benchmark's "individual" variants make O(tiles * years)
# separate getInfo() calls per trial, so a lower trial count keeps the
# test's runtime reasonable.
BATCHING_N_TRIALS = 3


def _assert_equal(old, new, label: str):
    assert old == new, f"{label}: old={old}, new={new}"


def _time_getinfo(expr: Any) -> Dict[str, Any]:
    try:
        t0 = time.time()
        info = expr.getInfo()
        return {"duration_s": time.time() - t0, "result": info}
    except Exception as e:
        return {"duration_s": None, "error": str(e)}


def _nonce() -> float:
    return time.time() + random.random()


def _bust(value):
    zero = ee.Number(0).multiply(_nonce())
    if isinstance(value, ee.Image):
        return value.add(ee.Image.constant(zero))
    return ee.Number(value).add(zero)


def _mean(values: List[float]):
    return sum(values) / len(values) if values else None


def _median(values: List[float]):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _stdev(values: List[float]):
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _trial_summary(values: List[float]) -> Dict[str, Any]:
    return {
        "durations_s": values,
        "mean_s": _mean(values),
        "median_s": _median(values),
        "stdev_s": _stdev(values),
        "n": len(values),
    }


def _aggregate_mean_s(per_year: Dict[str, Any]):
    """Sum each year's mean duration -- an estimate of total pipeline
    time across the period, comparable fairly between variants because
    every year's mean already comes from the same shuffled-order,
    cache-free trial set."""
    vals = [
        v["mean_s"]
        for v in per_year.values()
        if isinstance(v, dict) and v.get("mean_s") is not None
    ]
    return sum(vals) if vals else None


def _run_ab_trials(
    variant_fns: Dict[str, Callable[[], "float | None"]],
    n_trials: int = N_TRIALS,
) -> Dict[str, List[float]]:
    """Time each zero-arg callable in `variant_fns` `n_trials` times,
    shuffling the run order every trial. Each callable must apply its
    own cache-busting (see `_bust`) so repeated trials of the *same*
    variant don't just replay an earlier trial's cached result.

    Runs one untimed warm-up pass first (fixed order) so raw-tile
    caches are equally warm before any timed trial begins.
    """
    names = list(variant_fns.keys())
    durations: Dict[str, List[float]] = {name: [] for name in names}

    for name in names:
        variant_fns[name]()

    for _ in range(n_trials):
        order = names[:]
        random.shuffle(order)
        for name in order:
            duration = variant_fns[name]()
            if duration is not None:
                durations[name].append(duration)

    return durations


def _summary_path(cache_path: Path, name: str) -> Path:
    """Single source of truth for benchmark summary file naming, so
    every perf test writes to `{cache_stem}_{name}_summary.txt`."""
    return cache_path.with_name(f"{cache_path.stem}_{name}_summary.txt")


def _build_base_ic(geom: ee.Geometry, year: int, apply_scene_prefilter: bool):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    ic = ee.ImageCollection(S2_COLLECTION).filterDate(start, end).filterBounds(geom)
    if apply_scene_prefilter:
        ic = ic.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    return ic


def _old_join_savefirst(ic):
    cs_col = ee.ImageCollection(CLOUD_SCORE_COLLECTION)
    return ee.ImageCollection(
        ee.Join.saveFirst("cloud_score_plus").apply(
            primary=ic,
            secondary=cs_col,
            condition=ee.Filter.equals(
                leftField="system:index",
                rightField="system:index",
            ),
        )
    )


def _apply_old_mask_unwrap(joined_ic, cs_thresh):
    def _unwrap(img):
        cs = ee.Image(img.get("cloud_score_plus"))
        return img.updateMask(cs.select(CLOUD_SCORE_BAND).gte(cs_thresh))

    return joined_ic.map(_unwrap)


def _apply_new_link_mask(base_ic, geom, year, cs_thresh):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    linked = composites._join_cloud_score_plus(
        base_ic,
        geom,
        start,
        end,
    )
    return linked.map(lambda i: composites._mask_cloud_score_plus(i, cs_thresh))


def mask_count_gt0_from_masked_ic(masked_ic, proxy_band):
    return masked_ic.select(proxy_band).count().gt(0).rename("valid").toByte()


def mask_mapmax_from_masked_ic(masked_ic):
    def _per(img):
        return img.mask().reduce(ee.Reducer.max()).rename("valid")

    return masked_ic.map(_per).max().rename("valid").toByte()


def point_tiles_from_list(points, period):
    return [point_centred_tile(lon, lat, period) for (_id, lon, lat) in points]


def _load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _write_summary(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _lonlat_geom(tile):
    return ee.Geometry.Rectangle(
        [
            tile["min_lon"],
            tile["min_lat"],
            tile["max_lon"],
            tile["max_lat"],
        ],
        proj="EPSG:4326",
        geodesic=False,
    )


def _production_tile_geom(tile):
    return ee.Geometry.Rectangle(
        [
            tile["x_min_m"],
            tile["y_min_m"],
            tile["x_max_m"],
            tile["y_max_m"],
        ],
        proj=ee.Projection(settings.crs_wkt),
        geodesic=False,
    )


def _tiles_feature_collection(tiles):
    return ee.FeatureCollection(
        [
            ee.Feature(
                _production_tile_geom(tile),
                {"tile_id": tile["tile_id"]},
            )
            for tile in tiles
        ]
    )


def _build_batched_s2_image(tiles, years):
    tile_images = []

    for tile in tiles:
        geom = _production_tile_geom(tile)

        tile_image = ee.Image.cat(
            [
                composites.s2_availability(geom, year).rename(f"s2_{year}")
                for year in years
            ]
        ).clip(geom)

        tile_images.append(tile_image)

    return ee.ImageCollection(tile_images).mosaic()


def run_join_measurements(
    tile,
    years,
    *,
    cs_thresh,
    scale=DEFAULT_SCALE,
    proxy_band=DEFAULT_PROXY_BAND,
    n_trials=N_TRIALS,
):
    geom = _lonlat_geom(tile)
    result = {"old_join": {"per_year": {}}, "new_link": {"per_year": {}}}

    for year in years:

        def _time_old_join(year=year) -> "float | None":
            try:
                base_ic = _build_base_ic(geom, year, apply_scene_prefilter=True)
                masked_ic = _apply_old_mask_unwrap(
                    _old_join_savefirst(base_ic), cs_thresh
                )
                mask_img = _bust(mask_count_gt0_from_masked_ic(masked_ic, proxy_band))
                timed = _time_getinfo(
                    mask_img.reduceRegion(
                        ee.Reducer.sum(), geometry=geom, scale=scale, maxPixels=1e13
                    )
                )
                return timed.get("duration_s")
            except Exception:
                return None

        def _time_new_link(year=year) -> "float | None":
            try:
                base_ic = _build_base_ic(geom, year, apply_scene_prefilter=True)
                masked_ic = _apply_new_link_mask(base_ic, geom, year, cs_thresh)
                mask_img = _bust(mask_count_gt0_from_masked_ic(masked_ic, proxy_band))
                timed = _time_getinfo(
                    mask_img.reduceRegion(
                        ee.Reducer.sum(), geometry=geom, scale=scale, maxPixels=1e13
                    )
                )
                return timed.get("duration_s")
            except Exception:
                return None

        durations = _run_ab_trials(
            {"old_join": _time_old_join, "new_link": _time_new_link},
            n_trials=n_trials,
        )

        for variant, values in durations.items():
            result[variant]["per_year"][str(year)] = _trial_summary(values)

    return result


@pytest.mark.slow
def test_join_performance(bench_refresh, bench_cache_path):
    try:
        ee.Initialize()
    except Exception as e:
        pytest.skip(f"Earth Engine not initialized: {e}")

    years = list(settings.period_years)
    cache_path = Path(bench_cache_path)
    cache = _load_cache(cache_path)
    cache.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    cache.setdefault("years", years)
    cache.setdefault("tiles", {})

    tiles = point_tiles_from_list(TILE_POINTS, settings.period)
    cs_thresh = float(getattr(settings, "cloud_score_thresh", 0.5))

    for tile in tiles:
        tid = tile["tile_id"]
        tile_cache = cache["tiles"].setdefault(tid, {})

        if not bench_refresh and tile_cache.get("join"):
            print(f"[join] Using cached results for {tid}")
            continue

        print(f"[join] Running join benchmark for {tid}")

        tile_cache["join"] = run_join_measurements(
            tile,
            years,
            cs_thresh=cs_thresh,
            scale=int(getattr(settings, "scale", DEFAULT_SCALE)),
            proxy_band=getattr(settings, "s2_check_band", DEFAULT_PROXY_BAND),
        )

        _save_cache(cache_path, cache)

    for tile in tiles:
        tid = tile["tile_id"]
        j = cache["tiles"].get(tid, {}).get("join", {})

        assert j, f"No join results for {tid}"

        for variant in ("old_join", "new_link"):
            per_year = j.get(variant, {}).get("per_year", {})
            assert per_year, f"No per_year data for {variant} on {tid}"
            assert any(
                isinstance(v, dict) and v.get("n", 0) > 0 for v in per_year.values()
            ), f"No successful trials for {variant} on {tid}"

    summary_lines = [
        "Join benchmark -- total mean duration_s per tile "
        "(summed across years, each year averaged over "
        f"{N_TRIALS} shuffled-order, no cache trials):",
    ]

    agg_totals = {"old_join": 0.0, "new_link": 0.0}
    agg_any = {"old_join": False, "new_link": False}

    for tile in tiles:
        tid = tile["tile_id"]
        j = cache["tiles"][tid]["join"]
        row = {}

        for variant in ("old_join", "new_link"):
            total = _aggregate_mean_s(j[variant]["per_year"])
            row[variant] = total
            if total is not None:
                agg_totals[variant] += total
                agg_any[variant] = True

        summary_lines.append(
            f"  {tid}: old_join={row['old_join']}  new_link={row['new_link']}"
        )

    summary_lines.append("")
    summary_lines.append(f"Aggregate across {len(tiles)} tiles:")

    for variant in ("old_join", "new_link"):
        total = agg_totals[variant] if agg_any[variant] else None
        summary_lines.append(f"  {variant} total_mean_duration_s={total}")

    for line in summary_lines:
        print(line)

    _write_summary(_summary_path(cache_path, "join"), summary_lines)


def run_mask_measurements(
    tile,
    years,
    *,
    cs_thresh,
    scale=DEFAULT_SCALE,
    proxy_band=DEFAULT_PROXY_BAND,
    n_trials=N_TRIALS,
):
    geom = _lonlat_geom(tile)
    result = {"count_gt0": {"per_year": {}}, "mapmax": {"per_year": {}}}

    for year in years:

        def _time_count_gt0(year=year) -> "float | None":
            try:
                base_ic = _build_base_ic(geom, year, apply_scene_prefilter=True)
                masked_ic = _apply_new_link_mask(base_ic, geom, year, cs_thresh)
                m = _bust(mask_count_gt0_from_masked_ic(masked_ic, proxy_band))
                timed = _time_getinfo(
                    m.reduceRegion(
                        ee.Reducer.sum(), geometry=geom, scale=scale, maxPixels=1e13
                    )
                )
                return timed.get("duration_s")
            except Exception:
                return None

        def _time_mapmax(year=year) -> "float | None":
            try:
                base_ic = _build_base_ic(geom, year, apply_scene_prefilter=True)
                masked_ic = _apply_new_link_mask(base_ic, geom, year, cs_thresh)
                m = _bust(mask_mapmax_from_masked_ic(masked_ic))
                timed = _time_getinfo(
                    m.reduceRegion(
                        ee.Reducer.sum(), geometry=geom, scale=scale, maxPixels=1e13
                    )
                )
                return timed.get("duration_s")
            except Exception:
                return None

        durations = _run_ab_trials(
            {"count_gt0": _time_count_gt0, "mapmax": _time_mapmax},
            n_trials=n_trials,
        )

        for variant, values in durations.items():
            result[variant]["per_year"][str(year)] = _trial_summary(values)

    return result


@pytest.mark.slow
def test_mask_performance(bench_refresh, bench_cache_path):
    try:
        ee.Initialize()
    except Exception as e:
        pytest.skip(f"Earth Engine not initialized: {e}")

    years = list(settings.period_years)
    cache_path = Path(bench_cache_path)
    cache = _load_cache(cache_path)
    cache.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    cache.setdefault("years", years)
    cache.setdefault("tiles", {})

    tiles = point_tiles_from_list(TILE_POINTS, settings.period)
    cs_thresh = float(getattr(settings, "cloud_score_thresh", 0.5))

    for tile in tiles:
        tid = tile["tile_id"]
        tile_cache = cache["tiles"].setdefault(tid, {})

        if not bench_refresh and tile_cache.get("mask"):
            print(f"[mask] Using cached results for {tid}")
            continue

        print(f"[mask] Running mask benchmark for {tid}")

        tile_cache["mask"] = run_mask_measurements(
            tile,
            years,
            cs_thresh=cs_thresh,
            scale=int(getattr(settings, "scale", DEFAULT_SCALE)),
            proxy_band=getattr(settings, "s2_check_band", DEFAULT_PROXY_BAND),
        )

        _save_cache(cache_path, cache)

    for tile in tiles:
        tid = tile["tile_id"]
        m = cache["tiles"].get(tid, {}).get("mask", {})

        assert m, f"No mask results for {tid}"

        for method in ("count_gt0", "mapmax"):
            per_year = m.get(method, {}).get("per_year", {})
            assert per_year, f"No per_year data for {method} on {tid}"
            assert any(
                isinstance(v, dict) and v.get("n", 0) > 0 for v in per_year.values()
            ), f"No successful trials for {method} on {tid}"

    summary_lines = [
        "Mask benchmark -- total mean duration_s per tile "
        "(summed across years, each year averaged over "
        f"{N_TRIALS} shuffled-order, no-cache trials):",
    ]

    agg_totals = {"count_gt0": 0.0, "mapmax": 0.0}
    agg_any = {"count_gt0": False, "mapmax": False}

    for tile in tiles:
        tid = tile["tile_id"]
        m = cache["tiles"][tid]["mask"]
        row = {}

        for method in ("count_gt0", "mapmax"):
            total = _aggregate_mean_s(m[method]["per_year"])
            row[method] = total
            if total is not None:
                agg_totals[method] += total
                agg_any[method] = True

        summary_lines.append(
            f"  {tid}: count_gt0={row['count_gt0']}  mapmax={row['mapmax']}"
        )

    summary_lines.append("")
    summary_lines.append(f"Aggregate across {len(tiles)} tiles:")

    for method in ("count_gt0", "mapmax"):
        total = agg_totals[method] if agg_any[method] else None
        summary_lines.append(f"  {method} total_mean_duration_s={total}")

    for line in summary_lines:
        print(line)

    _write_summary(_summary_path(cache_path, "mask"), summary_lines)


def run_combined_vs_separate(
    tiles,
    years,
    *,
    scale=DEFAULT_SCALE,
    n_trials=N_TRIALS,
):
    fc = _tiles_feature_collection(tiles)
    geom = fc.geometry()

    def _time_combined() -> "float | None":
        bands = [
            composites.s2_availability(geom, year).rename(f"s2_{year}")
            for year in years
        ]
        stats = _bust(ee.Image.cat(bands))
        timed = _time_getinfo(
            stats.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=scale,
                tileScale=8,
            )
        )
        return timed.get("duration_s")

    def _time_separate() -> "float | None":
        total = 0.0
        any_success = False
        for year in years:
            band = _bust(composites.s2_availability(geom, year).rename(f"s2_{year}"))
            timed = _time_getinfo(
                band.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                    tileScale=8,
                )
            )
            duration = timed.get("duration_s")
            if duration is not None:
                total += duration
                any_success = True
        return total if any_success else None

    durations = _run_ab_trials(
        {"combined": _time_combined, "separate": _time_separate},
        n_trials=n_trials,
    )

    return {
        "combined": _trial_summary(durations["combined"]),
        "separate": _trial_summary(durations["separate"]),
    }


@pytest.mark.slow
def test_combined_vs_separate(
    bench_refresh,
    bench_cache_path,
):
    try:
        ee.Initialize()
    except Exception as e:
        pytest.skip(f"Earth Engine not initialized: {e}")

    years = list(settings.period_years)
    cache_path = Path(bench_cache_path)
    cache = _load_cache(cache_path)

    cache.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    cache.setdefault("years", years)

    tiles = point_tiles_from_list(TILE_POINTS, settings.period)

    if bench_refresh or not cache.get("combined"):
        cache["combined"] = run_combined_vs_separate(
            tiles,
            years,
            scale=int(getattr(settings, "scale", DEFAULT_SCALE)),
        )
        _save_cache(cache_path, cache)

    result = cache["combined"]

    for variant in ("combined", "separate"):
        assert result[variant]["n"] > 0, f"No successful trials for {variant}"

    combined = result["combined"]
    separate = result["separate"]

    summary_lines = [
        "Combined vs separate S2 benchmark:",
        f"Tiles: {len(tiles)}",
        f"Years: {years}",
        f"Trials: {N_TRIALS}, run order shuffled and each call "
        "cache-free so neither variant benefits from the other's "
        "warm request cache.",
        "",
        f"combined: mean={combined['mean_s']:.3f}s "
        f"median={combined['median_s']:.3f}s "
        f"stdev={combined['stdev_s']:.3f}s n={combined['n']}",
        f"separate: mean={separate['mean_s']:.3f}s "
        f"median={separate['median_s']:.3f}s "
        f"stdev={separate['stdev_s']:.3f}s n={separate['n']}",
    ]

    if combined["mean_s"] and separate["mean_s"]:
        speedup = separate["mean_s"] / combined["mean_s"]
        summary_lines.append(f"combined is {speedup:.2f}x the speed of separate")

    for line in summary_lines:
        print(line)

    _write_summary(_summary_path(cache_path, "combined"), summary_lines)


def run_batching_measurements(
    tiles,
    years,
    *,
    scale=DEFAULT_SCALE,
    n_trials=BATCHING_N_TRIALS,
):
    def _s2_individual() -> "float | None":
        total = 0.0
        any_success = False

        for tile in tiles:
            geom = _production_tile_geom(tile)

            for year in years:
                try:
                    img = _bust(
                        composites.s2_availability(geom, year).rename(f"s2_{year}")
                    )
                    t0 = time.time()
                    img.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geom,
                        scale=scale,
                        maxPixels=1e13,
                    ).getInfo()
                    total += time.time() - t0
                    any_success = True
                except Exception:
                    pass

        return total if any_success else None

    def _s2_batched() -> "float | None":
        fc = _tiles_feature_collection(tiles)
        stats = _bust(_build_batched_s2_image(tiles, years))

        timed = _time_getinfo(
            stats.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=scale,
                tileScale=8,
            )
        )

        return timed.get("duration_s")

    def _s1_individual() -> "float | None":
        total = 0.0
        any_success = False

        for tile in tiles:
            geom = _production_tile_geom(tile)
            feature = ee.Feature(geom, {"tile_id": tile["tile_id"]})

            for year in years:
                try:
                    t0 = time.time()
                    _bust(composites.s1_availability(feature, year)).getInfo()
                    total += time.time() - t0
                    any_success = True
                except Exception:
                    pass

        return total if any_success else None

    def _s1_batched() -> "float | None":
        fc = _tiles_feature_collection(tiles)

        def add_s1_stats(feature):
            for year in years:
                feature = feature.set(
                    f"s1_{year}",
                    _bust(composites.s1_availability(feature, year)),
                )
            return feature

        timed = _time_getinfo(fc.map(add_s1_stats))

        return timed.get("duration_s")

    s2_durations = _run_ab_trials(
        {"s2_individual": _s2_individual, "s2_batched": _s2_batched},
        n_trials=n_trials,
    )
    s1_durations = _run_ab_trials(
        {"s1_individual": _s1_individual, "s1_batched": _s1_batched},
        n_trials=n_trials,
    )

    return {
        "s2_individual": _trial_summary(s2_durations["s2_individual"]),
        "s2_batched": _trial_summary(s2_durations["s2_batched"]),
        "s1_individual": _trial_summary(s1_durations["s1_individual"]),
        "s1_batched": _trial_summary(s1_durations["s1_batched"]),
    }


@pytest.mark.slow
def test_batching_performance(
    bench_refresh,
    bench_cache_path,
):
    try:
        ee.Initialize()
    except Exception as e:
        pytest.skip(f"Earth Engine not initialized: {e}")

    years = list(settings.period_years)
    cache_path = Path(bench_cache_path)
    cache = _load_cache(cache_path)
    cache.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    cache.setdefault("years", years)
    cache.setdefault("batching", {})

    tiles = point_tiles_from_list(TILE_POINTS, settings.period)

    if bench_refresh or not cache["batching"]:
        print("[batching] Running batching benchmark for full tile set")

        cache["batching"] = run_batching_measurements(
            tiles,
            years,
            scale=int(getattr(settings, "scale", DEFAULT_SCALE)),
        )

        _save_cache(cache_path, cache)

    b = cache["batching"]

    for variant in ("s2_individual", "s2_batched", "s1_individual", "s1_batched"):
        assert b.get(variant, {}).get("n", 0) > 0, f"No successful trials for {variant}"

    summary_lines = [
        f"Batching benchmark -- mean duration_s across {len(tiles)} tiles "
        f"({BATCHING_N_TRIALS} shuffled-order, cache free trials):",
    ]

    for sensor in ("s2", "s1"):
        indiv = b[f"{sensor}_individual"]
        batched = b[f"{sensor}_batched"]

        summary_lines.append(
            f"  {sensor}: individual mean={indiv['mean_s']} "
            f"(n={indiv['n']})  batched mean={batched['mean_s']} "
            f"(n={batched['n']})"
        )

    for line in summary_lines:
        print(line)

    _write_summary(_summary_path(cache_path, "batching"), summary_lines)


def _build_composite_variant(geom, year, method, cs_thresh):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    base_ic = _build_base_ic(geom, year, apply_scene_prefilter=True)
    masked_ic = _apply_new_link_mask(base_ic, geom, year, cs_thresh)

    masked_ic = masked_ic.select(
        composites.NATIVE_10M_BANDS + composites.NATIVE_20M_BANDS
    ).map(composites._upsample_20m_bands_to_10m)

    if method == "median":
        return masked_ic.median()

    if method == "mean":
        return masked_ic.mean()

    if method == "quality_mosaic":

        def _with_quality(img):
            linked = composites._join_cloud_score_plus(
                ee.ImageCollection([img]),
                geom,
                start,
                end,
            ).first()

            return img.addBands(linked.select(composites.CLOUD_SCORE_PLUS_BAND))

        return masked_ic.map(_with_quality).qualityMosaic(
            composites.CLOUD_SCORE_PLUS_BAND
        )

    raise ValueError(f"unknown method: {method}")


def run_composite_method_measurements(
    tile,
    years,
    *,
    cs_thresh,
    scale=DEFAULT_SCALE,
    n_trials=N_TRIALS,
):
    geom = _lonlat_geom(tile)
    methods = ["median", "mean", "quality_mosaic"]
    result = {method: {"per_year": {}} for method in methods}

    for year in years:

        def _make_timer(method, year=year):
            def _time_method() -> "float | None":
                try:
                    img = _bust(_build_composite_variant(geom, year, method, cs_thresh))
                    timed = _time_getinfo(
                        img.reduceRegion(
                            ee.Reducer.mean(),
                            geometry=geom,
                            scale=scale,
                            maxPixels=1e13,
                        )
                    )
                    return timed.get("duration_s")
                except Exception:
                    return None

            return _time_method

        durations = _run_ab_trials(
            {method: _make_timer(method) for method in methods},
            n_trials=n_trials,
        )

        for method, values in durations.items():
            result[method]["per_year"][str(year)] = _trial_summary(values)

    return result


@pytest.mark.slow
def test_composite_method_performance(
    bench_refresh,
    bench_cache_path,
):
    try:
        ee.Initialize()
    except Exception as e:
        pytest.skip(f"Earth Engine not initialized: {e}")

    years = list(settings.period_years)
    cache_path = Path(bench_cache_path)
    cache = _load_cache(cache_path)
    cache.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    cache.setdefault("years", years)
    cache.setdefault("tiles", {})

    tiles = point_tiles_from_list(TILE_POINTS, settings.period)
    cs_thresh = float(getattr(settings, "cloud_score_thresh", 0.5))

    for tile in tiles:
        tid = tile["tile_id"]
        tile_cache = cache["tiles"].setdefault(tid, {})

        if not bench_refresh and tile_cache.get("composite_method"):
            print(f"[composite_method] Using cached results for {tid}")
            continue

        print(f"[composite_method] Running composite method benchmark for {tid}")

        tile_cache["composite_method"] = run_composite_method_measurements(
            tile,
            years,
            cs_thresh=cs_thresh,
            scale=int(getattr(settings, "scale", DEFAULT_SCALE)),
        )

        _save_cache(cache_path, cache)

    for tile in tiles:
        tid = tile["tile_id"]
        c = cache["tiles"].get(tid, {}).get("composite_method", {})

        assert c, f"No composite method results for {tid}"

        for method in ("median", "mean", "quality_mosaic"):
            per_year = c.get(method, {}).get("per_year", {})
            assert per_year, f"No per_year data for {method} on {tid}"
            assert any(
                isinstance(v, dict) and v.get("n", 0) > 0 for v in per_year.values()
            ), f"No successful trials for {method} on {tid}"

    summary_lines = [
        "Composite method benchmark -- total mean duration_s per tile "
        "(summed across years, each year averaged over "
        f"{N_TRIALS} shuffled-order, cache free trials):",
    ]

    agg_totals = {"median": 0.0, "mean": 0.0, "quality_mosaic": 0.0}
    agg_any = {"median": False, "mean": False, "quality_mosaic": False}

    for tile in tiles:
        tid = tile["tile_id"]
        c = cache["tiles"][tid]["composite_method"]
        row = {}

        for method in ("median", "mean", "quality_mosaic"):
            total = _aggregate_mean_s(c[method]["per_year"])
            row[method] = total
            if total is not None:
                agg_totals[method] += total
                agg_any[method] = True

        summary_lines.append(
            f"  {tid}: median={row['median']}  mean={row['mean']}  "
            f"quality_mosaic={row['quality_mosaic']}"
        )

    summary_lines.append("")
    summary_lines.append(f"Aggregate across {len(tiles)} tiles:")

    for method in ("median", "mean", "quality_mosaic"):
        total = agg_totals[method] if agg_any[method] else None
        summary_lines.append(f"  {method} total_mean_duration_s={total}")

    for line in summary_lines:
        print(line)

    _write_summary(_summary_path(cache_path, "composite_method"), summary_lines)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "join",
        "mask",
        "combined",
        "batching",
    ],
)
def test_equivalence(name):
    ee.Initialize()

    tiles = point_tiles_from_list(
        TEST_TILES,
        settings.period,
    )

    if name == "join":
        old, new = _join_outputs(
            tiles,
            TEST_YEARS,
        )
    elif name == "mask":
        old, new = _mask_outputs(
            tiles,
            TEST_YEARS,
        )
    elif name == "combined":
        old, new = _combined_outputs(
            tiles,
            TEST_YEARS,
        )
    elif name == "batching":
        old, new = _batching_outputs(
            tiles,
            TEST_YEARS,
        )
    else:
        raise AssertionError(name)

    _assert_equal(old, new, name)


def _join_outputs(tiles, years):
    old = {}
    new = {}

    for tile in tiles:
        geom = _lonlat_geom(tile)

        for year in years:
            base_ic = _build_base_ic(
                geom,
                year,
                apply_scene_prefilter=True,
            )

            old_img = _apply_old_mask_unwrap(
                _old_join_savefirst(base_ic),
                float(
                    getattr(
                        settings,
                        "cloud_score_thresh",
                        0.5,
                    )
                ),
            )

            new_img = _apply_new_link_mask(
                base_ic,
                geom,
                year,
                float(
                    getattr(
                        settings,
                        "cloud_score_thresh",
                        0.5,
                    )
                ),
            )

            old[f"{tile['tile_id']}_{year}"] = (
                mask_count_gt0_from_masked_ic(
                    old_img,
                    getattr(
                        settings,
                        "s2_check_band",
                        DEFAULT_PROXY_BAND,
                    ),
                )
                .reduceRegion(
                    ee.Reducer.sum(),
                    geometry=geom,
                    scale=int(
                        getattr(
                            settings,
                            "scale",
                            DEFAULT_SCALE,
                        )
                    ),
                    maxPixels=1e13,
                )
                .get("valid")
            )

            new[f"{tile['tile_id']}_{year}"] = (
                mask_count_gt0_from_masked_ic(
                    new_img,
                    getattr(
                        settings,
                        "s2_check_band",
                        DEFAULT_PROXY_BAND,
                    ),
                )
                .reduceRegion(
                    ee.Reducer.sum(),
                    geometry=geom,
                    scale=int(
                        getattr(
                            settings,
                            "scale",
                            DEFAULT_SCALE,
                        )
                    ),
                    maxPixels=1e13,
                )
                .get("valid")
            )

    return (
        ee.Dictionary(old).getInfo(),
        ee.Dictionary(new).getInfo(),
    )


def _mask_outputs(tiles, years):
    old = {}
    new = {}

    for tile in tiles:
        geom = _lonlat_geom(tile)

        for year in years:
            ic = _build_base_ic(
                geom,
                year,
                apply_scene_prefilter=True,
            )

            masked = _apply_new_link_mask(
                ic,
                geom,
                year,
                float(
                    getattr(
                        settings,
                        "cloud_score_thresh",
                        0.5,
                    )
                ),
            )

            proxy = getattr(
                settings,
                "s2_check_band",
                DEFAULT_PROXY_BAND,
            )

            old_img = mask_count_gt0_from_masked_ic(
                masked,
                proxy,
            ).rename("old")

            new_img = mask_mapmax_from_masked_ic(
                masked,
            ).rename("new")

            result = old_img.addBands(new_img).reduceRegion(
                ee.Reducer.sum(),
                geometry=geom,
                scale=int(
                    getattr(
                        settings,
                        "scale",
                        DEFAULT_SCALE,
                    )
                ),
                maxPixels=1e13,
            )

            key = f"{tile['tile_id']}_{year}"
            old[key] = result.get("old")
            new[key] = result.get("new")

    result = ee.Dictionary(
        {
            "old": ee.Dictionary(old),
            "new": ee.Dictionary(new),
        }
    ).getInfo()

    return result["old"], result["new"]


def _combined_outputs(tiles, years):
    fc = _tiles_feature_collection(tiles)
    geom = fc.geometry()
    scale = int(getattr(settings, "scale", DEFAULT_SCALE))

    # combined: multi-band image, reduceRegions names each output column
    # after its band -- no ambiguity here since band count > 1.
    combined_bands = ee.Image.cat(
        [composites.s2_availability(geom, year).rename(f"s2_{year}") for year in years]
    )
    combined_result = combined_bands.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=8,
    ).getInfo()

    old = {}
    for feature in combined_result["features"]:
        props = feature["properties"]
        tile_id = props["tile_id"]
        for year in years:
            old[f"{tile_id}_{year}"] = props.get(f"s2_{year}")

    # separate: single-band image per call. reduceRegions collapses the
    # output column to just the reducer's name ("mean") when there's
    # exactly one band and one reducer output -- setOutputs pins the
    # column name explicitly so this doesn't depend on that band-count
    # coincidence.
    new = {}
    for year in years:
        band = composites.s2_availability(geom, year).rename(f"s2_{year}")
        reducer = ee.Reducer.mean().setOutputs([f"s2_{year}"])
        year_result = band.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=scale,
            tileScale=8,
        ).getInfo()
        for feature in year_result["features"]:
            props = feature["properties"]
            tile_id = props["tile_id"]
            new[f"{tile_id}_{year}"] = props.get(f"s2_{year}")

    return old, new


def _batching_outputs(tiles, years):
    old = {}
    new = {}

    scale = int(
        getattr(
            settings,
            "scale",
            DEFAULT_SCALE,
        )
    )

    for tile in tiles:
        geom = _production_tile_geom(tile)

        for year in years:
            key = f"{tile['tile_id']}_{year}"

            old[key] = (
                composites.s2_availability(
                    geom,
                    year,
                )
                .rename(f"s2_{year}")
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=scale,
                    maxPixels=1e13,
                )
                .get(f"s2_{year}")
            )

    fc = _tiles_feature_collection(tiles)
    stats = _build_batched_s2_image(
        tiles,
        years,
    )

    batched = stats.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=8,
    ).getInfo()

    for feature in batched["features"]:
        props = feature["properties"]
        tile_id = props["tile_id"]

        for year in years:
            new[f"{tile_id}_{year}"] = props.get(f"s2_{year}")

    return (
        ee.Dictionary(old).getInfo(),
        new,
    )
