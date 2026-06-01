"""
generate_aois.py

AOI generation with basic sanity checks.

Runs in two modes controlled by the USE_HPC environment variable:
  USE_HPC=0 (default) — single-process, sequential batches, suitable for local dev
  USE_HPC=1           — multiprocess workers + dedicated writer thread for HPC/SLURM

Usage:
  # Local
  python generate_aois.py

  # HPC
  USE_HPC=1 NUM_WORKERS=32 sbatch submit_aoi_generation.sh

Validity checks
────────────────────────────────────────────────────────────────────────────────
1. Has land          USDOS/LSIB_SIMPLE/2017 — excludes open ocean
2. Has vegetation    ESA WorldCover trees (10) or mangrove (95) ≥ 1%
                     Threshold relaxed to 0.5% if UMD forest gain confirmed
                     (UMD 30m and ESA 1km aggregation can disagree)
3. Has S2 imagery    COPERNICUS/S2_SR_HARMONIZED — 2016, 2020 and 2025 required
4. Has forest gain   UMD GLCLUC 2015→2020 — at least 0.1% of cell must show
                     tree cover gain (class 25-96, 125-196 in 2020 but not 2015)

Output fields per AOI
────────────────────────────────────────────────────────────────────────────────
id, minLon, minLat, maxLon, maxLat
valid
rejection_reason
veg_fraction
forest_gain_frac
s2_count_2016/2020/2025
"""

import json
import logging
import multiprocessing as mp
import os
import threading
import time
import random
from pathlib import Path

import ee
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GEE_PROJECT = os.getenv("GEE_PROJECT")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))

OUTPUT_FILE = OUTPUT_DIR / os.getenv("OUTPUT_FILE", "aois/valid_aois.json")

REJECTED_OUTPUT_FILE = OUTPUT_FILE.parent / "rejected_aois.json"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTPUT_FILE.parent / "aoi_filter_checkpoint.json"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
AOI_STEP = float(os.getenv("AOI_STEP", 0.25))

USE_HPC = os.getenv("USE_HPC", "0") == "1"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

MIN_VEG_FRACTION = 0.01
MIN_LAND_FRACTION = 0.05
MIN_GAIN_FRACTION = 0.001

AOI_LIST_CACHE = OUTPUT_FILE.parent / "all_aois.json"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "aoi_generation.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

credentials = ee.ServiceAccountCredentials(None, creds_path)

ee.Initialize(credentials, project=GEE_PROJECT)

logger.info(f"GEE initialised | project={GEE_PROJECT} | HPC={USE_HPC}")


def _build_gee_datasets():
    _land_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    _land_raster = ee.Image(0).paint(_land_fc, 1).unmask(0).rename("land")

    _esa_wc = ee.Image("ESA/WorldCover/v100/2020")
    _esa_trees = _esa_wc.eq(10).unmask(0)
    _esa_mangrove = _esa_wc.eq(95).unmask(0)
    _esa_veg = _esa_trees.Or(_esa_mangrove).unmask(0).rename("esa_veg")

    _glulc_2015 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").select([0])
    _glulc_2020 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").select([0])

    _tree_classes = ee.List.sequence(25, 96).cat(ee.List.sequence(125, 196))
    _ones = ee.List.repeat(1, _tree_classes.length())

    _tree_2015 = _glulc_2015.remap(_tree_classes, _ones, 0)
    _tree_2020 = _glulc_2020.remap(_tree_classes, _ones, 0)
    _gain_mask = _tree_2020.And(_tree_2015.Not()).select([0]).rename("gain").unmask(0)

    _ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    return _land_raster, _esa_veg, _gain_mask, _ecoregions


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


def has_usable_s2(geom):
    def year_frac(start, end):
        bands = [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        ]

        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(geom)
            .map(mask_s2_scl)
            .select(bands)
        )

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

    fracs = ee.Dictionary(
        {
            "2016": safe_num(year_frac("2015-01-01", "2016-12-31"), 0),
            "2020": safe_num(year_frac("2020-01-01", "2020-12-31"), 0),
            "2025": safe_num(year_frac("2025-01-01", "2025-12-31"), 0),
        }
    )

    return (
        fracs.getNumber("2016")
        .gte(0.05)
        .And(fracs.getNumber("2020").gte(0.05))
        .And(fracs.getNumber("2025").gte(0.05))
    )


def forest_gain_fraction_umd(_gain_mask, geom, scale=30):
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
        reasons.append("missing_s2")
    if reason_code & 0x4:
        reasons.append("no_land")
    if reason_code & 0x8:
        reasons.append("no_forest_gain")

    return " + ".join(reasons) if reasons else "unknown"


def generate_global_aois(step=AOI_STEP, batch_size=2000):
    import ee
    import math

    cells = []

    lat = -60.0
    while lat < 85.0:
        lon = -180.0
        while lon < 180.0:
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
    print(f"[AOI] total cells: {total}")

    land_mask = (
        ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
        .select("discrete_classification")
        .neq(200)
    )

    valid = []
    batches = math.ceil(total / batch_size)

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        batch = cells[start:end]

        print(f"[AOI] processing batch {i+1}/{batches} ({start}-{end})")

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

        print(f"[AOI] batch {i+1}/{batches} → valid so far: {len(valid)}")

    return valid


def process_batch(_land_raster, _esa_veg, _gain_mask, _ecoregions, batch):
    batch = [a.get("properties", a) if isinstance(a, dict) else a for a in batch]
    features = [
        ee.Feature(
            ee.Geometry.Rectangle([a["minLon"], a["minLat"], a["maxLon"], a["maxLat"]]),
            a,
        )
        for a in batch
    ]

    fc = ee.FeatureCollection(features)

    def add_geometry_metadata(f):
        geom = f.geometry()

        area_km2 = geom.area().divide(1e6)

        centroid = geom.centroid(1)

        coords = ee.List(centroid.coordinates())

        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))

        return f.set(
            "aoi_area_km2",
            area_km2,
            "centroid_lon",
            lon,
            "centroid_lat",
            lat,
        )

    fc = fc.map(add_geometry_metadata)

    def add_ecoregion_metadata(f):
        centroid = f.geometry().centroid(1)

        eco = ee.Feature(_ecoregions.filterBounds(centroid).sort("ECO_ID").first())

        biome_name = ee.String(
            ee.Algorithms.If(
                eco,
                ee.Feature(eco).get("BIOME_NAME"),
                "Unknown",
            )
        )

        biome_num = ee.Number(
            ee.Algorithms.If(
                eco,
                ee.Feature(eco).get("BIOME_NUM"),
                -1,
            )
        )

        region = ee.String(
            ee.Algorithms.If(
                eco,
                ee.Feature(eco).get("REALM"),
                "Unknown",
            )
        )

        return f.set(
            "biome_name",
            biome_name,
            "biome_num",
            biome_num,
            "region",
            region,
        )

    fc = fc.map(add_ecoregion_metadata)

    def add_land(f):
        lf = land_fraction(_land_raster, f.geometry())
        return f.set("land_frac", lf, "has_land", lf.gte(MIN_LAND_FRACTION))

    fc = fc.map(add_land)
    has_land_fc = fc.filter(ee.Filter.eq("has_land", 1))
    no_land_fc = fc.filter(ee.Filter.eq("has_land", 0))

    def add_veg(f):
        vf = safe_num(
            _esa_veg.reduceRegion(
                ee.Reducer.mean(), geometry=f.geometry(), scale=1000, maxPixels=1e9
            ).get("esa_veg"),
            0,
        )
        return f.set("veg_fraction", vf, "has_veg", vf.gte(MIN_VEG_FRACTION))

    has_land_fc = has_land_fc.map(add_veg)
    has_veg_fc = has_land_fc.filter(ee.Filter.eq("has_veg", 1))
    no_veg_fc = has_land_fc.filter(ee.Filter.eq("has_veg", 0))

    def add_gain(f):
        fg = forest_gain_fraction_umd(_gain_mask, f.geometry())
        return f.set("forest_gain_frac", fg, "has_gain", fg.gte(MIN_GAIN_FRACTION))

    has_veg_fc = has_veg_fc.map(add_gain)
    has_gain_fc = has_veg_fc.filter(ee.Filter.eq("has_gain", 1))
    no_gain_fc = has_veg_fc.filter(ee.Filter.eq("has_gain", 0))

    def add_s2(f):
        return f.set("has_s2", has_usable_s2(f.geometry()))

    has_gain_fc = has_gain_fc.map(add_s2)
    valid_fc = has_gain_fc.filter(ee.Filter.eq("has_s2", 1))
    no_s2_fc = has_gain_fc.filter(ee.Filter.eq("has_s2", 0))

    no_land_fc = no_land_fc.map(
        lambda f: f.set(
            "rejection_reason",
            "no_land",
            "valid",
            0,
            "veg_fraction",
            0,
            "forest_gain_frac",
            0,
        )
    )
    no_veg_fc = no_veg_fc.map(
        lambda f: f.set(
            "rejection_reason",
            "insufficient_veg",
            "valid",
            0,
            "forest_gain_frac",
            0,
        )
    )
    no_gain_fc = no_gain_fc.map(
        lambda f: f.set("rejection_reason", "no_forest_gain", "valid", 0)
    )
    no_s2_fc = no_s2_fc.map(
        lambda f: f.set("rejection_reason", "missing_s2", "valid", 0)
    )
    valid_fc = valid_fc.map(lambda f: f.set("rejection_reason", "valid", "valid", 1))

    all_rejected = no_land_fc.merge(no_veg_fc).merge(no_gain_fc).merge(no_s2_fc)

    all_fc = valid_fc.merge(all_rejected)
    all_results = all_fc.getInfo()["features"]

    valid_out = [f for f in all_results if f["properties"].get("valid") == 1]
    rejected_out = [f for f in all_results if f["properties"].get("valid") == 0]

    return valid_out, rejected_out


def run_local(remaining, loaded_valid, loaded_rejected):
    _land_raster, _esa_veg, _gain_mask, _ecoregions = _build_gee_datasets()

    valid_aois = []
    rejected_aois = []

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]

        try:
            valid_batch, rejected_batch = process_batch(
                _land_raster, _esa_veg, _gain_mask, _ecoregions, batch
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
    _creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    _credentials = ee.ServiceAccountCredentials(None, _creds_path)
    time.sleep(worker_id * 5)
    ee.Initialize(_credentials, project=GEE_PROJECT)

    _land_raster, _esa_veg, _gain_mask, _ecoregions = _build_gee_datasets()

    while True:
        item = batch_queue.get()

        if item is None:
            break

        batch_idx, batch = item

        for attempt in range(8):
            try:
                valid, rejected = process_batch(
                    _land_raster, _esa_veg, _gain_mask, _ecoregions, batch
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
        logger.info("Generating AOI list — this may take a while...")
        all_aois = generate_global_aois()
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
