"""
generate_aois.py

Generate and validate 0.25° AOIs for the active forest-gain period.

Environment
-----------
  PERIOD=p1 (default) — 2017 → 2020
  PERIOD=p2           — 2020 → 2024
  USE_HPC=0 (default) — local sequential processing
  USE_HPC=1           — multiprocessing for HPC/SLURM
  NUM_WORKERS=4       — HPC worker count
  BATCH_SIZE=50       — AOIs per processing batch
  AOI_STEP=0.25       — AOI grid size in degrees
  SEARCH_MODE=asset    — derive search bounds from DT assets
  CLOUD_SCORE_THRESH=0.6 — Cloud Score+ cs_cdf threshold for S2 masking

Usage
-----
  PERIOD=p1 python generate_aois.py
  PERIOD=p2 USE_HPC=1 NUM_WORKERS=4 sbatch submit_aoi_generation.sh

The script resumes from its period-specific checkpoint and writes:
  data/aois/valid_aois_<period>.json
  data/aois/rejected_aois_<period>.json

Validity checks
---------------
  • Land coverage
  • ≥1% Dynamic World vegetation
  • S2: pixel coverage (Cloud Score+ masked, full year) ≥ MIN_IMAGERY_FRACTION
  • S1: ≥ settings.min_s1_observations qualifying acquisitions, every year
  • ≥0.1% forest gain between the period's start and end year

Output fields include
---------------------
  id, bounds, area/centroid, land/vegetation/gain fractions,
  imagery status, biome, region, country, validity and rejection reason.
"""

import json
import logging
import multiprocessing as mp
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

OUTPUT_FILE = PROJECT_ROOT / OUTPUT_DIR / f"aois/valid_aois_{PERIOD}.json"
REJECTED_OUTPUT_FILE = PROJECT_ROOT / OUTPUT_DIR / f"aois/rejected_aois_{PERIOD}.json"
CHECKPOINT = PROJECT_ROOT / OUTPUT_DIR / f"aois/aoi_filter_checkpoint_{PERIOD}.json"

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

S2_BANDS = [settings.s2_check_band]

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_VEGETATED_LABELS = [1, 2, 3, 4, 5]  # trees, grass, flooded_veg, crops, shrub/scrub

# Cloud Score+ — cs_cdf is the calibrated variant Google recommends thresholding against;
# 0.6 is a starting default
CLOUD_SCORE_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_PLUS_BAND = "cs_cdf"
CLOUD_SCORE_THRESH = float(os.getenv("CLOUD_SCORE_THRESH", "0.6"))

AOI_LIST_CACHE = PROJECT_ROOT / OUTPUT_DIR / f"aois/all_aois_{PERIOD}.json"

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

    _land_veg = _land_raster.addBands(_dw_veg)

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
    _countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

    _realm_names = _ecoregions.aggregate_array("REALM").distinct().getInfo()
    _realm_name_to_id = {name: i for i, name in enumerate(_realm_names)}
    _id_to_realm_name = {i: name for name, i in _realm_name_to_id.items()}

    _ecoregions_indexed = _ecoregions.map(
        lambda f: f.set(
            "realm_id", ee.Dictionary(_realm_name_to_id).get(f.get("REALM"))
        )
    )

    _eco_raster = _ecoregions_indexed.reduceToImage(
        ["BIOME_NUM"], ee.Reducer.first()
    ).rename("biome_num")
    _realm_raster = _ecoregions_indexed.reduceToImage(
        ["realm_id"], ee.Reducer.first()
    ).rename("realm_id")

    _biome_name_by_num = {}
    for f in (
        _ecoregions.distinct(["BIOME_NUM"])
        .select(["BIOME_NUM", "BIOME_NAME"])
        .getInfo()["features"]
    ):
        p = f["properties"]
        bn = p.get("BIOME_NUM")
        if bn is not None:
            _biome_name_by_num[int(bn)] = p.get("BIOME_NAME") or "Unknown"

    _country_names = _countries.aggregate_array("country_na").distinct().getInfo()
    _country_name_to_id = {name: i for i, name in enumerate(_country_names)}
    _id_to_country_name = {i: name for name, i in _country_name_to_id.items()}

    _countries_indexed = _countries.map(
        lambda f: f.set(
            "country_id",
            ee.Dictionary(_country_name_to_id).get(f.get("country_na")),
        )
    )
    _country_raster = _countries_indexed.reduceToImage(
        ["country_id"], ee.Reducer.first()
    ).rename("country_id")

    _eco_country_img = _eco_raster.addBands(_realm_raster).addBands(_country_raster)

    _lookups = {
        "biome_name": _biome_name_by_num,
        "realm": _id_to_realm_name,
        "country": _id_to_country_name,
    }

    return _land_veg, _gain_mask, _ecoregions, _countries, _eco_country_img, _lookups


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


def _join_cloud_score_plus(ic, geom, start, end):
    """Link Cloud Score+ QA band onto each S2 image via
    ImageCollection.linkCollection — Google's recommended pattern for
    this dataset. The linked band is attached directly as a band on
    each image, matched by system:index, rather than nested behind a
    property lookup (the older Join.saveFirst pattern this replaces)."""
    cs_col = (
        ee.ImageCollection(CLOUD_SCORE_PLUS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(geom)
    )
    return ic.linkCollection(cs_col, [CLOUD_SCORE_PLUS_BAND])


def mask_s2_cloud_score_plus(img):
    """Mask using the linked Cloud Score+ cs_cdf band — a plain band on
    img after linkCollection, no unwrapping needed."""
    cs = img.select(CLOUD_SCORE_PLUS_BAND)
    return img.updateMask(cs.gte(CLOUD_SCORE_THRESH))


def _s2_year_valid_mask(geom, start, end, bands):
    """
    Per-pixel indicator of whether geom is covered by at least one
    Cloud Score+ unmasked S2 image within [start, end), for the given
    bands. Uses ImageCollection.count() (a native collection-level
    reducer) rather than a per-image map+max reduction, and applies the
    CLOUDY_PIXEL_PERCENTAGE scene-level prefilter before the Cloud
    Score+ link so near-fully-cloudy scenes are dropped before the
    per-pixel work runs at all.
    """
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    )
    col = _join_cloud_score_plus(col, geom, start, end)
    col = col.map(mask_s2_cloud_score_plus).select(bands)

    counts = col.select(bands[0]).count()
    return counts.gt(0).unmask(0).rename("valid")


def build_year_sensor_masks_combined(batch_geom, year_start=YEAR_START, year_end=YEAR_END):
    """
    Combined multi-band S2 coverage image: one band per year
    (s2_<year>), each built once against the batch-wide geometry. Bands
    are reduced together in a single reduceRegions call downstream
    instead of one call per year — trades N separate EE round-trips
    (each paying its own queueing/compute overhead) for one larger call,
    which wins when per-call overhead dominates rather than per-call
    pixel volume (AOI-scale geometries at scale=500 are light enough
    that this is the right tradeoff; see raster_stats.py's tile-export
    pipeline for the opposite case, where pixel volume dominates and
    splitting by year is the correct call instead).
    """
    bands = []
    for year in range(year_start, year_end + 1):
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        mask = _s2_year_valid_mask(batch_geom, start, end, S2_BANDS)
        bands.append(mask.rename(f"s2_{year}"))
    return ee.Image.cat(bands)


def s1_scene_counts_for_batch(fc: ee.FeatureCollection, year: int) -> dict[str, int]:
    """
    Acquisition-level S1 scene count per AOI feature, for one year. Same
    qualifying filters as composites.s1_observation_count (IW mode, dual
    VV/VH polarisation), batched across the whole FeatureCollection in a
    single getInfo() call rather than one call per AOI.
    """
    start, end = f"{year}-01-01", f"{year + 1}-01-01"

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    def _count_for_feature(f):
        n = col.filterBounds(f.geometry()).size()
        return f.set("s1_count", n)

    fc_counted = fc.map(_count_for_feature)
    info = _reduce_regions_getinfo_with_retry(fc_counted)

    out = {}
    for f in info.get("features", []):
        props = f.get("properties", {})
        aid = props.get("id") or f.get("id")
        out[aid] = int(props.get("s1_count", 0) or 0)
    return out


def _reduce_regions_getinfo_with_retry(fc_obj, attempts=6, backoff_base=2.0):
    """
    Call getInfo() on an EE object (FeatureCollection from reduceRegions, etc.)
    with exponential backoff retries. Compatible with EE clients that don't accept
    a timeout kwarg.
    Returns the getInfo() result (dict).
    """
    last_err = None
    for attempt in range(attempts):
        try:
            info = fc_obj.getInfo()
            return info
        except Exception as e:
            last_err = e
            wait = (backoff_base**attempt) + random.uniform(0, 2)
            logger.warning(
                f"getInfo attempt {attempt + 1}/{attempts} failed: {e}; sleeping {wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"Exceeded retries for getInfo(): last error: {last_err}")


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


def _clean_str(raw):
    return ee.String(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(raw, None),
            "Unknown",
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(raw, "N/A"),
                "Unknown",
                ee.Algorithms.If(ee.Algorithms.IsEqual(raw, ""), "Unknown", raw),
            ),
        )
    )


def process_batch(
    _land_veg, _gain_mask, _ecoregions, _countries, _eco_country_img, _lookups, batch
):
    batch = [a.get("properties", a) if isinstance(a, dict) else a for a in batch]

    features = [
        ee.Feature(
            ee.Geometry.Rectangle([a["minLon"], a["minLat"], a["maxLon"], a["maxLat"]]),
            a,
        )
        for a in batch
    ]
    fc = ee.FeatureCollection(features)
    batch_geom = fc.geometry().bounds(maxError=1)

    t0 = time.time()

    year_names = [f"s2_{y}" for y in range(YEAR_START, YEAR_END + 1)]

    combined_s2 = build_year_sensor_masks_combined(batch_geom)
    s2_rr = combined_s2.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=500, tileScale=4
    )
    s2_info = _reduce_regions_getinfo_with_retry(s2_rr)

    mask_results = {("s2", y): {} for y in range(YEAR_START, YEAR_END + 1)}
    for f in s2_info.get("features", []):
        props = f.get("properties", {})
        aid = props.get("id") or f.get("id")
        for y, band in zip(range(YEAR_START, YEAR_END + 1), year_names):
            val = props.get(band)
            mask_results[("s2", y)][aid] = float(val or 0.0)

    logger.info(f"  [timing] s2 imagery masks: {time.time() - t0:.1f}s")

    t0 = time.time()
    s1_counts_by_year = {
        year: s1_scene_counts_for_batch(fc, year)
        for year in range(YEAR_START, YEAR_END + 1)
    }
    logger.info(f"  [timing] s1 scene counts: {time.time() - t0:.1f}s")

    t0 = time.time()
    lv_rr = _land_veg.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=1000
    )
    lv_info = _reduce_regions_getinfo_with_retry(lv_rr)
    landveg_map = {}
    for f in lv_info.get("features", []):
        props = f.get("properties", {})
        key = props.get("id") or f.get("id")
        land = props.get("land")
        if land is None:
            land = props.get("discrete_classification") or props.get("mean") or 0.0
        dw = props.get("dw_veg")
        if dw is None:
            dw = props.get("mean_dw_veg") or 0.0
        landveg_map[key] = {"land": float(land or 0.0), "dw_veg": float(dw or 0.0)}
    logger.info(f"  [timing] land/veg: {time.time() - t0:.1f}s")

    t0 = time.time()
    gain_rr = _gain_mask.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=100
    )
    gain_info = _reduce_regions_getinfo_with_retry(gain_rr)
    gain_map = {}
    for f in gain_info.get("features", []):
        props = f.get("properties", {})
        key = props.get("id") or f.get("id")
        gain_val = props.get("gain")
        if gain_val is None:
            gain_val = props.get("mean") or 0.0
        gain_map[key] = float(gain_val or 0.0)
    logger.info(f"  [timing] gain: {time.time() - t0:.1f}s")

    t0 = time.time()

    def add_centroid_lookup(f):
        centroid = f.geometry().centroid(1)

        eco_hit = _ecoregions.filterBounds(centroid)
        country_hit = _countries.filterBounds(centroid)
        has_eco = eco_hit.size().gt(0)
        has_country = country_hit.size().gt(0)

        eco = eco_hit.first()
        country = country_hit.first()

        biome_name = ee.Algorithms.If(has_eco, _clean_str(eco.get("BIOME_NAME")), None)
        biome_num = ee.Algorithms.If(has_eco, eco.get("BIOME_NUM"), None)
        realm = ee.Algorithms.If(has_eco, _clean_str(eco.get("REALM")), None)
        country_name = ee.Algorithms.If(
            has_country, _clean_str(country.get("country_na")), None
        )

        return f.set(
            {
                "has_eco": has_eco,
                "has_country": has_country,
                "biome_name": biome_name,
                "biome_num": biome_num,
                "realm": realm,
                "country_name": country_name,
            }
        )

    fc_centroid = fc.map(add_centroid_lookup)
    centroid_info = _reduce_regions_getinfo_with_retry(fc_centroid)

    eco_map = {}
    country_map = {}
    fallback_ids = set()

    for f in centroid_info.get("features", []):
        props = f.get("properties", {})
        aid = props.get("id")

        if props.get("has_eco"):
            biome_num_raw = props.get("biome_num")
            try:
                biome_num = int(biome_num_raw) if biome_num_raw is not None else -1
            except (TypeError, ValueError):
                biome_num = -1
            if biome_num == 11:
                eco_map[aid] = {
                    "biome_name": "Rock and Ice",
                    "biome_num": biome_num,
                    "realm": "Global",
                }
            else:
                eco_map[aid] = {
                    "biome_name": props.get("biome_name") or "Unknown",
                    "biome_num": biome_num,
                    "realm": props.get("realm") or "Unknown",
                }
        else:
            fallback_ids.add(aid)

        if props.get("has_country"):
            country_map[aid] = props.get("country_name") or "Unknown"
        else:
            fallback_ids.add(aid)

    # --- 4b) slow path, only for AOIs whose centroid missed. Majority-pixel
    # vote via the pre-rasterized image, targeted at just this subset
    # instead of running for every AOI in the batch.
    if fallback_ids:
        fc_fallback = fc.filter(ee.Filter.inList("id", list(fallback_ids)))
        fb_rr = _eco_country_img.reduceRegions(
            collection=fc_fallback, reducer=ee.Reducer.mode(), scale=1000
        )
        fb_info = _reduce_regions_getinfo_with_retry(fb_rr)

        for f in fb_info.get("features", []):
            props = f.get("properties", {})
            aid = props.get("id") or f.get("id")

            if aid not in eco_map:
                try:
                    biome_num = (
                        int(props.get("biome_num"))
                        if props.get("biome_num") is not None
                        else -1
                    )
                except (TypeError, ValueError):
                    biome_num = -1

                if biome_num == 11:
                    eco_map[aid] = {
                        "biome_name": "Rock and Ice",
                        "biome_num": biome_num,
                        "realm": "Global",
                    }
                else:
                    biome_name = _lookups["biome_name"].get(biome_num, "Unknown")
                    try:
                        realm_id = int(props.get("realm_id"))
                        realm = _lookups["realm"].get(realm_id, "Unknown")
                    except (TypeError, ValueError):
                        realm = "Unknown"
                    eco_map[aid] = {
                        "biome_name": biome_name,
                        "biome_num": biome_num,
                        "realm": realm,
                    }

            if aid not in country_map:
                try:
                    country_id = int(props.get("country_id"))
                    country_map[aid] = _lookups["country"].get(country_id, "Unknown")
                except (TypeError, ValueError):
                    country_map[aid] = "Unknown"

    # --- 5) assemble outputs ---
    valid_out = []
    rejected_out = []

    for a in batch:
        aid = a["id"]
        lv = landveg_map.get(aid, {"land": 0.0, "dw_veg": 0.0})
        land_frac = float(lv["land"])
        veg_frac = float(lv["dw_veg"])
        has_land = land_frac >= MIN_LAND_FRACTION
        has_veg = veg_frac >= MIN_VEG_FRACTION

        fg = gain_map.get(aid, 0.0)
        has_gain = fg >= MIN_GAIN_FRACTION

        worst_s2 = 1.0
        for (sensor, year), per in mask_results.items():
            v = per.get(aid, 0.0)
            if v < worst_s2:
                worst_s2 = v
        has_s2_img = worst_s2 >= MIN_IMAGERY_FRACTION

        has_s1_img = all(
            s1_counts_by_year[year].get(aid, 0) >= settings.min_s1_observations
            for year in range(YEAR_START, YEAR_END + 1)
        )

        has_img = has_s2_img and has_s1_img

        biome = eco_map.get(
            aid, {"biome_name": "Unknown", "biome_num": -1, "realm": "Unknown"}
        )
        country = country_map.get(aid, "Unknown")

        centroid_lon = (a["minLon"] + a["maxLon"]) / 2.0
        centroid_lat = (a["minLat"] + a["maxLat"]) / 2.0
        R = 6371.0088
        import math as _math

        lon1 = _math.radians(a["minLon"])
        lon2 = _math.radians(a["maxLon"])
        lat1 = _math.radians(a["minLat"])
        lat2 = _math.radians(a["maxLat"])
        area_km2 = abs(R * R * (lon2 - lon1) * (_math.sin(lat2) - _math.sin(lat1)))

        props = {
            "id": aid,
            "minLon": a["minLon"],
            "minLat": a["minLat"],
            "maxLon": a["maxLon"],
            "maxLat": a["maxLat"],
            "aoi_area_km2": float(area_km2),
            "centroid_lon": float(centroid_lon),
            "centroid_lat": float(centroid_lat),
            "land_frac": float(land_frac),
            "has_land": int(has_land),
            "veg_fraction": float(veg_frac),
            "has_veg": int(has_veg),
            "forest_gain_frac": float(fg),
            "has_gain": int(has_gain),
            "has_imagery": int(has_img),
            "biome_name": biome["biome_name"],
            "biome_num": int(biome["biome_num"]),
            "region": biome["realm"],
            "country": country,
        }

        valid_flag = bool(has_land and has_veg and has_gain and has_img)
        if valid_flag:
            props["valid"] = 1
            props["rejection_reason"] = "valid"
            valid_out.append({"type": "Feature", "properties": props})
        else:
            props["valid"] = 0
            if not has_land:
                props["rejection_reason"] = "no_land"
            elif not has_veg:
                props["rejection_reason"] = "insufficient_veg"
            elif not has_gain:
                props["rejection_reason"] = "no_forest_gain"
            else:
                props["rejection_reason"] = "missing_imagery"
            rejected_out.append({"type": "Feature", "properties": props})

    return valid_out, rejected_out


def run_local(remaining, loaded_valid, loaded_rejected):
    _land_veg, _gain_mask, _ecoregions, _countries, _eco_country_img, _lookups = (
        _build_gee_datasets()
    )

    valid_aois = []
    rejected_aois = []

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]

        try:
            valid_batch, rejected_batch = process_batch(
                _land_veg,
                _gain_mask,
                _ecoregions,
                _countries,
                _eco_country_img,
                _lookups,
                batch,
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
            f"{len(loaded_valid) + len(valid_aois)} valid"
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
    _land_veg, _gain_mask, _ecoregions, _countries, _eco_country_img, _lookups = (
        _build_gee_datasets()
    )

    while True:
        item = batch_queue.get()

        if item is None:
            break

        batch_idx, batch = item

        for attempt in range(8):
            try:
                valid_batch, rejected_batch = process_batch(
                    _land_veg,
                    _gain_mask,
                    _ecoregions,
                    _countries,
                    _eco_country_img,
                    _lookups,
                    batch,
                )
                result_queue.put(
                    ("batch_result", batch_idx, valid_batch, rejected_batch)
                )
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
                f"{len(loaded_valid) + len(valid_aois)} valid | "
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
