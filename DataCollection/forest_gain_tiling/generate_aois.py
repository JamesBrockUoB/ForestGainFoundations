"""
generate_aois.py

AOI generation with basic sanity checks.

Runs in two modes controlled by the USE_HPC environment variable:
USE_HPC=0 (default) — single-process, sequential batches, suitable for local dev
USE_HPC=1           — multiprocess workers + dedicated writer thread for HPC/SLURM

Period is controlled by the PERIOD environment variable:
PERIOD=p1 (default) — 2017 → 2020
PERIOD=p2           — 2020 → 2024

Usage:
# Local
PERIOD=p1 python generate_aois.py

# HPC
PERIOD=p2 USE_HPC=1 NUM_WORKERS=32 sbatch submit_aoi_generation.sh

Validity checks
────────────────────────────────────────────────────────────────────────────────
1. Has land          USDOS/LSIB_SIMPLE/2017 — excludes open ocean
2. Has vegetation    Dynamic World (GOOGLE/DYNAMICWORLD/V1), median composite
                     for the active period's year_start — top class is trees,
                     grass, flooded_vegetation, crops, or shrub_and_scrub ≥ 1%.
3. Has imagery       S2 (COPERNICUS/S2_SR_HARMONIZED) and S1 (COPERNICUS/S1_GRD)
                    — every calendar year in the active period must clear 5%
                    valid-pixel coverage for both sensors
                    PERIOD=p1 → 2017,2018,2019,2020
                    PERIOD=p2 → 2020,2021,2022,2023,2024
4. Has forest gain   DT year_start→year_end for the active period — at least 0.1%
                    of cell must show tree cover gain with over 50% confidence
                    counting as tree cover

Output fields per AOI
────────────────────────────────────────────────────────────────────────────────
id, minLon, minLat, maxLon, maxLat
valid
rejection_reason
veg_fraction
forest_gain_frac
has_imagery
"""

import json
import logging
import multiprocessing as mp
import os
import random
import threading
import time
from pathlib import Path

import ee
from config import settings
from dotenv import load_dotenv
from gee.auth import get_ee_credentials

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))

PERIOD = os.getenv("PERIOD", "p1")  # "p1" = 2017→2020, "p2" = 2020→2024

PERIOD_YEARS = {
    "p1": (2017, 2020),
    "p2": (2020, 2024),
}

if PERIOD not in PERIOD_YEARS:
    raise ValueError(f"PERIOD must be one of {list(PERIOD_YEARS)}, got {PERIOD!r}")

YEAR_START, YEAR_END = PERIOD_YEARS[PERIOD]

OUTPUT_FILE = PROJECT_ROOT / OUTPUT_DIR / f"aois/valid_aois_clearer_{PERIOD}.json"
REJECTED_OUTPUT_FILE = (
    PROJECT_ROOT / OUTPUT_DIR / f"aois/rejected_aois_clearer_{PERIOD}.json"
)
CHECKPOINT = (
    PROJECT_ROOT / OUTPUT_DIR / f"aois/aoi_filter_checkpoint_clearer_{PERIOD}.json"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
AOI_STEP = float(os.getenv("AOI_STEP", 0.25))

USE_HPC = os.getenv("USE_HPC", "0") == "1"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

SEARCH_MODE = os.getenv("SEARCH_MODE", "asset")

MIN_VEG_FRACTION = 0.01
MIN_LAND_FRACTION = 0.01
MIN_GAIN_FRACTION = 0.001
MIN_IMAGERY_FRACTION = 0.05

# Coverage-check band subsets. NOT for spectral analysis — these are a
# deliberately small, high-resolution proxy for "is there usable imagery
# here at all", not the full band set consumed by any downstream analysis.
S2_BANDS = [
    "B2",
    "B4",
    "B8",
]  # blue + red + NIR, 10m — strong proxy for full-scene coverage
S1_BANDS = ["VV", "VH"]

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_VEGETATED_LABELS = [1, 2, 3, 4, 5]  # trees, grass, flooded_veg, crops, shrub/scrub

# AOI_LIST_CACHE is intentionally NOT period-namespaced: the raw 0.25° grid
# of candidate land cells is identical regardless of which period is active.
AOI_LIST_CACHE = PROJECT_ROOT / OUTPUT_DIR / "aois/all_aois_clearer.json"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "aoi_generation_clearer.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# OAuth user credentials (from `ee.Authenticate()`), not a service account.
# get_ee_credentials() reads the refresh token from
# settings.ee_credentials_path if set (env var EE_CREDENTIALS_PATH),
# otherwise the library default ~/.config/earthengine/credentials, and
# merges in ee's bundled OAuth client id/secret/token_uri.
ee.Initialize(get_ee_credentials(), project=settings.gee_project)

logger.info(
    f"GEE initialised | project={settings.gee_project} | HPC={USE_HPC} | PERIOD={PERIOD}"
)


def _dw_vegetation_mask(year):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    dw = ee.ImageCollection(DW_COLLECTION).filterDate(start, end).median()

    return (
        dw.select("label")
        .remap(
            [1, 2, 3, 4, 5],  # trees, grass, flooded vegetation, crops, shrub/scrub
            [1, 1, 1, 1, 1],
            0,
        )
        .rename("dw_veg")
        .unmask(0)
    )


def _build_gee_datasets():
    _land_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    _land_raster = ee.Image(0).paint(_land_fc, 1).unmask(0).rename("land")

    _dw_veg = _dw_vegetation_mask(YEAR_START).rename("dw_veg")

    def load_dt_mosaic(year):
        return (
            ee.Image(
                f"projects/symbolic-base-346316/assets/dt_tree_cover_{year}_mosaic"
            )
            .select([0])
            .divide(2.55)
            .rename("tree_cover_pct")
        )

    _cover_start = load_dt_mosaic(YEAR_START)
    _cover_end = load_dt_mosaic(YEAR_END)

    _forest_start = _cover_start.gt(settings.non_tree_threshold_frac).unmask(0)
    _forest_end = _cover_end.gt(settings.min_tree_threshold_frac).unmask(0)

    _gain_mask = _forest_start.Not().And(_forest_end).rename("gain").unmask(0)

    _ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    return _land_raster, _dw_veg, _gain_mask, _ecoregions


def get_asset_bounds(year_start=2017, year_end=2020, padding=0.5):
    """
    Derive the geographic bounding box of the deadtrees gain layer
    by unioning the tile geometries for the start and end years.
    Returns (min_lon, min_lat, max_lon, max_lat) with optional padding.
    """

    def load_mosaic(year):
        return ee.Image(
            f"projects/symbolic-base-346316/assets/dt_tree_cover_{year}_mosaic"
        )

    bounds = (
        load_mosaic(year_start)
        .geometry()
        .union(load_mosaic(year_end).geometry(), 1)
        .bounds(1, "EPSG:4326")
        .coordinates()
        .get(0)
    )

    coords = ee.List(bounds).getInfo()

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    min_lon = max(-180.0, min(lons) - padding)
    min_lat = max(-90.0, min(lats) - padding)
    max_lon = min(180.0, max(lons) + padding)
    max_lat = min(90.0, max(lats) + padding)

    logger.info(
        f"Asset bounds derived: "
        f"lon [{min_lon:.2f}, {max_lon:.2f}] "
        f"lat [{min_lat:.2f}, {max_lat:.2f}]"
    )

    return min_lon, min_lat, max_lon, max_lat


def safe_num(val, default=0):
    return ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(val, None),
            default,
            val,
        )
    )


def atomic_json_write(path, obj, indent=None):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    tmp.replace(path)


def land_fraction(_land_raster, geom, scale=1000):
    val = _land_raster.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e9,
    ).get("land")

    return safe_num(val, 0)


def mask_s2_scl(img):
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(0))
    return img.updateMask(mask)


def _year_valid_fraction(collection_id, geom, start, end, bands, mask_fn=None):
    """
    Fraction of geom covered by at least one fully-unmasked pixel, across all
    images in collection_id intersecting geom within [start, end), for the
    given bands.
    """
    col = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(geom)

    if "S1_GRD" in collection_id:
        col = col.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        )
        col = col.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
        )

    if mask_fn is not None:
        col = col.map(mask_fn)

    col = col.select(bands)

    def valid_mask(img):
        return img.mask().reduce(ee.Reducer.min()).rename("valid")

    valid = col.map(valid_mask).max()

    valid = ee.Image(
        ee.Algorithms.If(
            valid.bandNames().size().gt(0), valid, ee.Image(0).rename("valid")
        )
    )

    return valid.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=500, maxPixels=1e9
    ).get("valid")


def has_usable_imagery(geom, min_frac=MIN_IMAGERY_FRACTION):
    """
    Requires, for every calendar year in [YEAR_START, YEAR_END] inclusive,
    both S2 and S1 valid-pixel coverage >= min_frac over geom.
    """
    conditions = []

    for year in range(YEAR_START, YEAR_END + 1):
        start, end = f"{year}-01-01", f"{year + 1}-01-01"

        s2_frac = safe_num(
            _year_valid_fraction(
                "COPERNICUS/S2_SR_HARMONIZED", geom, start, end, S2_BANDS, mask_s2_scl
            ),
            0,
        )
        s1_frac = safe_num(
            _year_valid_fraction("COPERNICUS/S1_GRD", geom, start, end, S1_BANDS),
            0,
        )

        conditions.append(s2_frac.gte(min_frac).And(s1_frac.gte(min_frac)))

    result = conditions[0]
    for c in conditions[1:]:
        result = result.And(c)

    return result


def forest_gain_fraction_dt(_gain_mask, geom, scale=100):
    """
    Fraction of pixels in geom that transitioned from non-forest (YEAR_START)
    to forest (YEAR_END) using the deadtrees.earth product at 10m resolution.
    Reduced at 100m scale for speed — sufficient for AOI-level filtering.
    """
    val = _gain_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e9,
    ).get("gain")

    return safe_num(val, 0)


def rejection_reason_str(reason_code):
    if reason_code == 0:
        return "valid"

    reasons = []

    if reason_code & 0x1:
        reasons.append("insufficient_veg")
    if reason_code & 0x2:
        reasons.append("missing_imagery")
    if reason_code & 0x4:
        reasons.append("no_land")
    if reason_code & 0x8:
        reasons.append("no_forest_gain")

    return " + ".join(reasons) if reasons else "unknown"


def generate_aois(
    step=AOI_STEP,
    batch_size=2000,
    min_lon=-180.0,
    min_lat=-60.0,
    max_lon=180.0,
    max_lat=85.0,
):
    import math

    cells = []

    # Snap to grid boundaries
    lat_start = math.floor(min_lat / step) * step
    lon_start = math.floor(min_lon / step) * step
    lat_end = math.ceil(max_lat / step) * step
    lon_end = math.ceil(max_lon / step) * step

    lat = lat_start
    while lat < lat_end:
        lon = lon_start
        while lon < lon_end:
            cells.append(
                {
                    "minLon": round(lon, 4),
                    "minLat": round(lat, 4),
                    "maxLon": round(min(lon + step, 180.0), 4),
                    "maxLat": round(min(lat + step, 85.0), 4),
                    "id": f"aoi_{round(lon,4)}_{round(lat,4)}",
                }
            )
            lon += step
        lat += step

    total = len(cells)
    logger.info(f"[AOI] total cells in search area: {total}")

    land_mask = (
        ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
        .select("discrete_classification")
        .neq(200)
    )

    valid = []
    batches = math.ceil(total / batch_size)
    asset_geom = ee.Image(
        "projects/symbolic-base-346316/assets/dt_tree_cover_2020_mosaic"
    ).geometry()

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        batch = cells[start:end]

        logger.info(f"[AOI] processing batch {i+1}/{batches} ({start}-{end})")

        features = [
            ee.Feature(
                ee.Geometry.Rectangle(
                    [c["minLon"], c["minLat"], c["maxLon"], c["maxLat"]]
                ),
                c,
            )
            for c in batch
        ]

        fc = ee.FeatureCollection(features)
        fc = fc.filterBounds(asset_geom)

        def add_land(f):
            frac = land_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=f.geometry(),
                scale=1000,
                maxPixels=1e9,
            ).get("discrete_classification")
            return f.set("land_frac", frac)

        fc = fc.map(add_land)
        fc = fc.filter(ee.Filter.gt("land_frac", 0))

        batch_result = fc.getInfo()["features"]
        valid.extend(batch_result)

        logger.info(f"[AOI] batch {i+1}/{batches} → valid so far: {len(valid)}")

    return valid


def process_batch(_land_raster, _dw_veg, _gain_mask, _ecoregions, batch):
    batch = [a.get("properties", a) if isinstance(a, dict) else a for a in batch]
    features = [
        ee.Feature(
            ee.Geometry.Rectangle([a["minLon"], a["minLat"], a["maxLon"], a["maxLat"]]),
            a,
        )
        for a in batch
    ]

    fc = ee.FeatureCollection(features)

    def add_all_properties(f):
        geom = f.geometry()

        area_km2 = geom.area().divide(1e6)
        centroid = geom.centroid(1)
        coords = ee.List(centroid.coordinates())

        lf = land_fraction(_land_raster, geom)
        has_land = lf.gte(MIN_LAND_FRACTION)

        vf = safe_num(
            _dw_veg.reduceRegion(
                ee.Reducer.mean(),
                geometry=geom,
                scale=1000,
                maxPixels=1e9,
            ).get("dw_veg"),
            0,
        )

        has_veg = vf.gte(MIN_VEG_FRACTION)

        fg = forest_gain_fraction_dt(_gain_mask, geom)
        has_gain = fg.gte(MIN_GAIN_FRACTION)

        has_img = has_usable_imagery(geom)

        intersecting = _ecoregions.filterBounds(geom)
        eco_with_area = intersecting.map(
            lambda e: e.set("eco_area", e.geometry().area(1))
        )
        ranked = eco_with_area.sort("eco_area", True)
        eco = ee.Feature(
            ee.Algorithms.If(ranked.size().gt(0), ranked.first(), ee.Feature(None))
        )

        biome_name_raw = eco.get("BIOME_NAME")
        biome_num_raw = eco.get("BIOME_NUM")
        realm_raw = eco.get("REALM")

        biome_name = ee.String(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(biome_name_raw, None),
                "Unknown",
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(biome_name_raw, "N/A"),
                    "Unknown",
                    ee.Algorithms.If(
                        ee.Algorithms.IsEqual(biome_name_raw, ""),
                        "Unknown",
                        biome_name_raw,
                    ),
                ),
            )
        )

        biome_num = ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(biome_num_raw, None), -1, biome_num_raw
            )
        )
        realm = ee.String(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(realm_raw, None),
                "Unknown",
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(realm_raw, "N/A"),
                    "Unknown",
                    ee.Algorithms.If(
                        ee.Algorithms.IsEqual(realm_raw, ""), "Unknown", realm_raw
                    ),
                ),
            )
        )

        is_rock_ice = biome_num.eq(11)
        biome_name = ee.String(
            ee.Algorithms.If(is_rock_ice, "Rock and Ice", biome_name)
        )
        realm = ee.String(ee.Algorithms.If(is_rock_ice, "Global", realm))

        return f.set(
            "aoi_area_km2",
            area_km2,
            "centroid_lon",
            ee.Number(coords.get(0)),
            "centroid_lat",
            ee.Number(coords.get(1)),
            "land_frac",
            lf,
            "has_land",
            has_land,
            "veg_fraction",
            vf,
            "has_veg",
            has_veg,
            "forest_gain_frac",
            fg,
            "has_gain",
            has_gain,
            "has_imagery",
            has_img,
            "biome_name",
            biome_name,
            "biome_num",
            biome_num,
            "region",
            realm,
        )

    fc = fc.map(add_all_properties)

    def add_validity(f):
        has_land = ee.Number(f.get("has_land"))
        has_veg = ee.Number(f.get("has_veg"))
        has_gain = ee.Number(f.get("has_gain"))
        has_img = ee.Number(f.get("has_imagery"))

        valid = has_land.And(has_veg).And(has_gain).And(has_img)

        reason = ee.String(
            ee.Algorithms.If(
                valid.eq(0),
                ee.Algorithms.If(
                    has_land.eq(0),
                    "no_land",
                    ee.Algorithms.If(
                        has_veg.eq(0),
                        "insufficient_veg",
                        ee.Algorithms.If(
                            has_gain.eq(0), "no_forest_gain", "missing_imagery"
                        ),
                    ),
                ),
                "valid",
            )
        )

        return f.set("valid", valid, "rejection_reason", reason)

    fc = fc.map(add_validity)

    all_results = fc.getInfo()["features"]

    valid_out = [f for f in all_results if f["properties"].get("valid") == 1]
    rejected_out = [f for f in all_results if f["properties"].get("valid") == 0]

    return valid_out, rejected_out


def run_local(remaining, loaded_valid, loaded_rejected):
    _land_raster, _dw_veg, _gain_mask, _ecoregions = _build_gee_datasets()

    valid_aois = []
    rejected_aois = []

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]

        try:
            valid_batch, rejected_batch = process_batch(
                _land_raster, _dw_veg, _gain_mask, _ecoregions, batch
            )
            valid_aois.extend([f["properties"] for f in valid_batch])
            rejected_aois.extend([f["properties"] for f in rejected_batch])

        except Exception as e:
            logger.error(f"Batch error (offset {i}): {e} — will retry next run")

        atomic_json_write(
            CHECKPOINT,
            {
                "valid": loaded_valid + valid_aois,
                "rejected": loaded_rejected + rejected_aois,
            },
        )

        logger.info(
            f"  {i + len(batch)}/{len(remaining)} processed — "
            f"{len(valid_aois)} valid"
        )

        time.sleep(0.2)

    return valid_aois, rejected_aois


def _worker(batch_queue, result_queue, worker_id):
    time.sleep(worker_id * 5)
    # See the module-level ee.Initialize call above for why this uses
    # get_ee_credentials() instead of a service account: each worker
    # process needs its own ee.Initialize call since GEE state isn't
    # inherited across the multiprocessing fork/spawn boundary.
    ee.Initialize(get_ee_credentials(), project=settings.gee_project)
    _land_raster, _dw_veg, _gain_mask, _ecoregions = _build_gee_datasets()

    while True:
        item = batch_queue.get()

        if item is None:
            break

        batch_idx, batch = item

        for attempt in range(8):
            try:
                valid, rejected = process_batch(
                    _land_raster, _dw_veg, _gain_mask, _ecoregions, batch
                )
                result_queue.put(("batch_result", batch_idx, valid, rejected))
                time.sleep(random.uniform(1, 3))
                break
            except Exception as e:
                err = str(e)
                if (
                    "429" in err
                    or "concurrent" in err.lower()
                    or "quota" in err.lower()
                    or "memory" in err.lower()
                ):
                    wait = (2**attempt) + random.uniform(0, 2)
                    logger.warning(
                        f"Worker {worker_id} | Batch {batch_idx} | "
                        f"Rate limited, retry {attempt+1}/5 in {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Worker {worker_id} | Batch {batch_idx}: {e}")
                    result_queue.put(("error", batch_idx, err))
                    break
        else:
            logger.error(f"Worker {worker_id} | Batch {batch_idx}: exhausted retries")
            result_queue.put(("error", batch_idx, "exhausted retries"))


def _writer(
    result_queue,
    total_batches,
    loaded_valid,
    loaded_rejected,
    out,
):
    valid_aois = []
    rejected_aois = []

    done = 0
    t0 = time.time()

    while done < total_batches:
        try:
            msg = result_queue.get(timeout=600)

        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total_batches})")
            continue

        msg_type = msg[0]

        if msg_type == "batch_result":
            _, batch_idx, valid, rejected = msg

            valid_aois.extend([f["properties"] for f in valid])
            rejected_aois.extend([f["properties"] for f in rejected])

            done += 1

        elif msg_type == "error":
            _, batch_idx, err = msg

            logger.warning(f"Batch {batch_idx} error: {err}")

            done += 1

        if done % 10 == 0:
            atomic_json_write(
                CHECKPOINT,
                {
                    "valid": loaded_valid + valid_aois,
                    "rejected": loaded_rejected + rejected_aois,
                },
            )

            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0

            logger.info(
                f"Checkpoint {done}/{total_batches} | "
                f"{len(valid_aois)} valid | "
                f"{rate:.1f} batches/min | "
                f"{elapsed:.1f}min elapsed"
            )

    atomic_json_write(
        OUTPUT_FILE,
        loaded_valid + valid_aois,
        indent=2,
    )

    atomic_json_write(
        REJECTED_OUTPUT_FILE,
        loaded_rejected + rejected_aois,
        indent=2,
    )

    logger.info(
        f"✓ Final output: "
        f"{len(loaded_valid) + len(valid_aois)} valid AOIs → {OUTPUT_FILE}"
    )

    logger.info(
        f"✓ Rejected output: "
        f"{len(loaded_rejected) + len(rejected_aois)} rejected AOIs → "
        f"{REJECTED_OUTPUT_FILE}"
    )

    out["valid"] = valid_aois
    out["rejected"] = rejected_aois


def run_hpc(remaining, loaded_valid, loaded_rejected):
    batches = [
        remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)
    ]

    batch_queue = mp.Queue()
    result_queue = mp.Queue()

    manager = mp.Manager()
    out_dict = manager.dict()

    workers = [
        mp.Process(
            target=_worker,
            args=(batch_queue, result_queue, i),
        )
        for i in range(NUM_WORKERS)
    ]

    for w in workers:
        w.start()

    writer_thread = threading.Thread(
        target=_writer,
        args=(result_queue, len(batches), loaded_valid, loaded_rejected, out_dict),
        daemon=False,
    )

    writer_thread.start()

    logger.info(f"Started {NUM_WORKERS} workers, 1 writer thread")

    for i, batch in enumerate(batches):
        batch_queue.put((i, batch))

    for _ in range(NUM_WORKERS):
        batch_queue.put(None)

    logger.info(f"Queued {len(batches)} batches")

    for w in workers:
        w.join()

    writer_thread.join()

    return (
        list(out_dict.get("valid", [])),
        list(out_dict.get("rejected", [])),
    )


def print_summary(valid_aois, rejected_aois):
    rejection_counts = {}

    for a in rejected_aois:
        reason = a.get("rejection_reason", "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    total_processed = len(valid_aois) + len(rejected_aois)

    logger.info(f"\n{'='*60}")
    logger.info(f"Total processed : {total_processed}")
    logger.info(f"Valid AOIs      : {len(valid_aois)}  → {OUTPUT_FILE}")
    logger.info(f"Rejected AOIs   : {len(rejected_aois)}")

    if rejection_counts:
        logger.info("\nRejection breakdown:")

        for reason, n in sorted(
            rejection_counts.items(),
            key=lambda x: -x[1],
        ):
            pct = 100 * n / len(rejected_aois)

            logger.info(f"  {reason:30s}: {n:7d} ({pct:5.1f}%)")


if __name__ == "__main__":
    if AOI_LIST_CACHE.exists():
        with open(AOI_LIST_CACHE) as f:
            all_aois = json.load(f)
        logger.info(f"Loaded {len(all_aois)} AOIs from cache")
    else:
        logger.info(f"Generating AOI list — mode={SEARCH_MODE}")

        if SEARCH_MODE == "asset":
            min_lon, min_lat, max_lon, max_lat = get_asset_bounds(
                year_start=YEAR_START, year_end=YEAR_END
            )
        else:
            min_lon, min_lat, max_lon, max_lat = -180.0, -60.0, 180.0, 85.0

        all_aois = generate_aois(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )
        atomic_json_write(AOI_LIST_CACHE, all_aois)
        logger.info(f"Cached {len(all_aois)} land cells → {AOI_LIST_CACHE}")

    all_aois = [a.get("properties", a) for a in all_aois]

    logger.info(f"Total {AOI_STEP}° cells: {len(all_aois)}")

    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            data = json.load(f)

        if isinstance(data, list):
            loaded_valid = data
            loaded_rejected = []
        else:
            loaded_valid = data.get("valid", [])
            loaded_rejected = data.get("rejected", [])

        already_done = {a["id"] for a in loaded_valid} | {
            a["id"] for a in loaded_rejected
        }

        remaining = [a for a in all_aois if a["id"] not in already_done]

        logger.info(
            f"Resuming — "
            f"{len(loaded_valid)} valid, "
            f"{len(loaded_rejected)} rejected, "
            f"{len(remaining)} remaining"
        )
    elif OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            loaded_valid = json.load(f)
        loaded_rejected = []
        already_done = {a["id"] for a in loaded_valid}
        remaining = [a for a in all_aois if a["id"] not in already_done]
    else:
        remaining = all_aois
        loaded_valid = []
        loaded_rejected = []

        logger.info(f"Starting fresh — {len(remaining)} cells to process")

    if USE_HPC:
        logger.info(f"Mode: HPC | workers={NUM_WORKERS} | batch_size={BATCH_SIZE}")
        new_valid, new_rejected = run_hpc(remaining, loaded_valid, loaded_rejected)

    else:
        logger.info(f"Mode: local | batch_size={BATCH_SIZE}")
        new_valid, new_rejected = run_local(remaining, loaded_valid, loaded_rejected)

    valid_aois = loaded_valid + new_valid
    rejected_aois = loaded_rejected + new_rejected

    atomic_json_write(OUTPUT_FILE, valid_aois, indent=2)
    atomic_json_write(REJECTED_OUTPUT_FILE, rejected_aois, indent=2)

    logger.info(f"✓ Final output: {len(valid_aois)} valid AOIs → {OUTPUT_FILE}")
    logger.info(
        f"✓ Rejected output: {len(rejected_aois)} rejected AOIs → {REJECTED_OUTPUT_FILE}"
    )

    print_summary(valid_aois, rejected_aois)
