"""
gee.py

Export forest-gain tiles for training a vision foundation model.

Commands
--------
  python gee.py plan                              # build tile registry, print summary
  python gee.py run                               # process all pending tiles
  python gee.py run --limit 500                   # next N pending tiles
  python gee.py run --biome "Boreal Forests"      # filter by biome (substring match)
  python gee.py run --region Neotropic            # filter by region
  python gee.py run --aoi-id aoi_-73.25_-52.75   # single AOI (debug)
  python gee.py run --status failed               # retry failed tiles
  python gee.py status                            # print registry summary
  python gee.py audit                             # report AOIs with no tiles

Registry
--------
  data/tiles/tile_registry.json — one entry per tile, persists across runs.

  Status lifecycle:
    pending   → not yet processed
    submitted → GEE export task in flight
    complete  → exported successfully
    rejected  → failed viability / gain checks (will not be retried by default)
    failed    → GEE error or submission error (retry with --status failed)

  Each entry:
    tile_id, xi, yi, x/y bounds (metres), lon/lat bounds,
    biome, region, aoi_ids,
    status, gee_task_id, submitted_at, completed_at,
    rejection_reason, error

Tile geometry
-------------
  Global fixed grid in EPSG:3857, origin (0,0), cell size 1280 m × 1280 m.
  Tile retained if any single valid AOI covers ≥ MIN_AOI_OVERLAP_FRAC of it.
  CRS transform is pixel-aligned per tile — no seams between adjacent tiles.

Pseudo-label scores (bands in exported GeoTIFF)
-----------------------------------------------
  score_agrocrop, score_nat_regen, score_plantation, score_restoration,
  dominant_class, label_confidence

Parallelism
-----------
  USE_HPC=0 (default) — sequential, single-process. Suitable for local testing
                         on a small subset of tiles.
  USE_HPC=1           — multiprocess workers + dedicated writer thread for
                         HPC/SLURM environments.

Stratified sampling
-------------------
  --stratify biome|region   Draw a proportionally or equally balanced sample
                             across strata up to --limit total tiles, rather than
                             taking the first N from the filtered list.
  --stratify-mode prop|equal
    prop  (default) — each stratum gets floor(limit * stratum_share) tiles
    equal           — each stratum gets floor(limit / n_strata) tiles

Usage:
  # Local (sequential, small test run)
  python gee.py run --limit 20

  # HPC
  USE_HPC=1 NUM_WORKERS=32 sbatch submit_tile_export.sh

  # Stratified sample of 500 tiles balanced across biomes
  python gee.py run --limit 500 --stratify biome

  # Equal sample of 200 tiles across regions
  python gee.py run --limit 200 --stratify region --stratify-mode equal
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ee
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GEE_PROJECT = os.getenv("GEE_PROJECT")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))
AOI_FILE = Path(os.getenv("OUTPUT_FILE", "aois/valid_aois.json"))
DRIVE_FOLDER = os.getenv("DRIVE_FOLDER", "forest_gain_tiles")
DRIVE_REMOTE = os.getenv("DRIVE_REMOTE", "gdrive")
HPC_BASE = os.getenv("HPC_BASE")
POLL_INTERVAL = 30

TILE_PIXELS = 128
SCALE = 10
TILE_SIZE_M = TILE_PIXELS * SCALE  # 1280 m
CRS = "EPSG:3857"
MIN_AOI_OVERLAP_FRAC = 0.50
GAIN_PCT_MIN = 1.0
NDVI_DELTA_MIN = 0.0
GAIN_CANOPY_MIN = 3.0

VALID_AOIS_PATH = OUTPUT_DIR / AOI_FILE
REGISTRY_PATH = OUTPUT_DIR / "tiles" / "tile_registry.json"
AOI_AUDIT_PATH = OUTPUT_DIR / "tiles" / "aoi_tile_audit.json"
LOG_DIR = Path("logs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tiles").mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

USE_HPC = os.getenv("USE_HPC", "0") == "1"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

STATUS_PENDING = "pending"
STATUS_SUBMITTED = "submitted"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"  # failed viability / gain checks — do not retry
STATUS_FAILED = "failed"  # GEE / submission error — retry with --status failed

TERMINAL_STATUSES = {STATUS_COMPLETE, STATUS_REJECTED}


def setup_logging(command: str) -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"gee_{command}_{ts}.log"
    logger = logging.getLogger("gee")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Log: {logfile}")
    return logger


def load_registry() -> dict:
    """Returns {tile_id: entry}."""
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        entries = json.load(f)
    return {e["tile_id"]: e for e in entries}


def save_registry(registry: dict):
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(registry.values()), f, indent=2)
    tmp.replace(REGISTRY_PATH)


def update_tile(registry: dict, tile_id: str, **kwargs):
    """Patch fields on a registry entry and flush to disk immediately."""
    registry[tile_id].update(kwargs)
    save_registry(registry)


def registry_summary(registry: dict) -> str:
    status_counts = Counter(e["status"] for e in registry.values())
    biome_counts = Counter(
        e.get("biome", "Unknown")
        for e in registry.values()
        if e["status"] == STATUS_COMPLETE
    )
    region_counts = Counter(
        e.get("region", "Unknown")
        for e in registry.values()
        if e["status"] == STATUS_COMPLETE
    )
    rejection_counts = Counter(
        e.get("rejection_reason", "unknown")
        for e in registry.values()
        if e["status"] == STATUS_REJECTED
    )

    lines = [
        "",
        "═" * 60,
        "  TILE REGISTRY SUMMARY",
        "═" * 60,
        f"  Total tiles    : {len(registry):>10,}",
    ]
    for s in [
        STATUS_PENDING,
        STATUS_SUBMITTED,
        STATUS_COMPLETE,
        STATUS_REJECTED,
        STATUS_FAILED,
    ]:
        lines.append(f"  {s:<14} : {status_counts.get(s, 0):>10,}")

    if rejection_counts:
        lines += ["", "  Rejected by reason:"]
        for r, n in rejection_counts.most_common():
            lines.append(f"    {r:<35} {n:>8,}")

    if biome_counts:
        lines += ["", "  Complete by biome:"]
        for b, n in biome_counts.most_common():
            lines.append(f"    {b:<45} {n:>7,}")

    if region_counts:
        lines += ["", "  Complete by region:"]
        for r, n in region_counts.most_common():
            lines.append(f"    {r:<30} {n:>7,}")

    lines += ["═" * 60, ""]
    return "\n".join(lines)


def build_aoi_audit(registry: dict, valid_aois: list[dict]) -> dict:
    """
    Returns a dict keyed by aoi_id with tile counts per status.
    AOIs with zero complete tiles are flagged.
    """
    aoi_tile_counts: dict[str, Counter] = defaultdict(Counter)

    for entry in registry.values():
        for aoi_id in entry.get("aoi_ids", []):
            aoi_tile_counts[aoi_id][entry["status"]] += 1

    result = {}
    for aoi in valid_aois:
        aoi_id = aoi["id"]
        counts = aoi_tile_counts.get(aoi_id, Counter())
        result[aoi_id] = {
            "biome": aoi.get("biome_name", "Unknown"),
            "region": aoi.get("region", "Unknown"),
            "tile_counts": dict(counts),
            "total_tiles": sum(counts.values()),
            "complete_tiles": counts.get(STATUS_COMPLETE, 0),
            "has_coverage": counts.get(STATUS_COMPLETE, 0) > 0,
        }

    return result


def save_aoi_audit(audit: dict):
    tmp = AOI_AUDIT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2)
    tmp.replace(AOI_AUDIT_PATH)


def audit_summary(audit: dict) -> str:
    total = len(audit)
    covered = sum(1 for v in audit.values() if v["has_coverage"])
    uncovered = total - covered

    by_biome: dict[str, dict] = defaultdict(lambda: {"total": 0, "covered": 0})
    for v in audit.values():
        b = v["biome"]
        by_biome[b]["total"] += 1
        by_biome[b]["covered"] += int(v["has_coverage"])

    lines = [
        "",
        "═" * 60,
        "  AOI COVERAGE AUDIT",
        "═" * 60,
        f"  Total AOIs     : {total:>10,}",
        f"  With coverage  : {covered:>10,}  ({100*covered/max(total,1):.1f}%)",
        f"  No coverage    : {uncovered:>10,}  ({100*uncovered/max(total,1):.1f}%)",
        "",
        "  By biome (total | covered | gap%):",
    ]
    for b, c in sorted(by_biome.items(), key=lambda x: -x[1]["total"]):
        gap = 100 * (c["total"] - c["covered"]) / max(c["total"], 1)
        lines.append(
            f"    {b:<45} {c['total']:>6,}  {c['covered']:>6,}  {gap:5.1f}% gap"
        )

    lines += ["═" * 60, ""]
    return "\n".join(lines)


def _snap(coord_m: float, down: bool) -> float:
    fn = math.floor if down else math.ceil
    return fn(coord_m / TILE_SIZE_M) * TILE_SIZE_M


def _aoi_to_3857(aoi: dict) -> tuple[float, float, float, float]:
    R = 6_378_137.0

    def lon2x(lon):
        return R * math.radians(lon)

    def lat2y(lat):
        return R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    return (
        lon2x(aoi["minLon"]),
        lat2y(aoi["minLat"]),
        lon2x(aoi["maxLon"]),
        lat2y(aoi["maxLat"]),
    )


def _x2lon(x: float) -> float:
    return math.degrees(x / 6_378_137.0)


def _y2lat(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(y / 6_378_137.0)) - math.pi / 2)


def build_global_grid(valid_aois: list[dict], logger: logging.Logger) -> list[dict]:
    """
    Pure Python — no GEE.  Returns list of tile dicts ready for the registry.
    Each tile includes biome / region / aoi_ids from overlapping AOIs.
    """
    logger.info("Projecting AOI bounds to EPSG:3857…")
    aoi_bounds_m = [_aoi_to_3857(a) for a in valid_aois]

    global_xmin = _snap(min(b[0] for b in aoi_bounds_m), down=True)
    global_ymin = _snap(min(b[1] for b in aoi_bounds_m), down=True)
    global_xmax = _snap(max(b[2] for b in aoi_bounds_m), down=False)
    global_ymax = _snap(max(b[3] for b in aoi_bounds_m), down=False)

    n_cols = round((global_xmax - global_xmin) / TILE_SIZE_M)
    n_rows = round((global_ymax - global_ymin) / TILE_SIZE_M)
    logger.info(f"Grid: {n_cols} cols × {n_rows} rows = {n_cols*n_rows:,} candidates")

    tile_area = TILE_SIZE_M**2
    retained = []

    for ci in tqdm(range(n_cols), desc="Building grid", unit="col"):
        tx_min = global_xmin + ci * TILE_SIZE_M
        tx_max = tx_min + TILE_SIZE_M

        for ri in range(n_rows):
            ty_min = global_ymin + ri * TILE_SIZE_M
            ty_max = ty_min + TILE_SIZE_M

            overlapping_aois = []
            for aoi, (ax_min, ay_min, ax_max, ay_max) in zip(valid_aois, aoi_bounds_m):
                ix_w = min(tx_max, ax_max) - max(tx_min, ax_min)
                iy_h = min(ty_max, ay_max) - max(ty_min, ay_min)
                if ix_w <= 0 or iy_h <= 0:
                    continue
                if (ix_w * iy_h) / tile_area >= MIN_AOI_OVERLAP_FRAC:
                    overlapping_aois.append(aoi)

            if not overlapping_aois:
                continue

            xi = round(tx_min / TILE_SIZE_M)
            yi = round(ty_min / TILE_SIZE_M)
            primary = overlapping_aois[0]

            retained.append(
                {
                    "tile_id": f"tile_{xi}_{yi}",
                    "xi": xi,
                    "yi": yi,
                    "x_min_m": tx_min,
                    "y_min_m": ty_min,
                    "x_max_m": tx_max,
                    "y_max_m": ty_max,
                    "min_lon": _x2lon(tx_min),
                    "min_lat": _y2lat(ty_min),
                    "max_lon": _x2lon(tx_max),
                    "max_lat": _y2lat(ty_max),
                    "biome": primary.get("biome_name", "Unknown"),
                    "region": primary.get("region", "Unknown"),
                    "aoi_ids": [a["id"] for a in overlapping_aois],
                    # Registry fields
                    "status": STATUS_PENDING,
                    "gee_task_id": None,
                    "submitted_at": None,
                    "completed_at": None,
                    "rejection_reason": None,
                    "error": None,
                }
            )

    logger.info(f"Retained {len(retained):,} tiles")
    return retained


def plan_summary(tiles: list[dict]) -> str:
    biome_counts = Counter(t["biome"] for t in tiles)
    region_counts = Counter(t["region"] for t in tiles)

    lines = [
        "",
        "═" * 60,
        "  TILE PLAN SUMMARY",
        "═" * 60,
        f"  Total tiles : {len(tiles):>10,}",
        f"  Grid size   : {TILE_SIZE_M:.0f} m × {TILE_SIZE_M:.0f} m  "
        f"({TILE_PIXELS}×{TILE_PIXELS} px @ {SCALE} m/px)",
        f"  CRS         : {CRS}",
        f"  Min overlap : {MIN_AOI_OVERLAP_FRAC*100:.0f}% of tile inside a single AOI",
        "",
        "  By biome:",
    ]
    for b, n in biome_counts.most_common():
        lines.append(f"    {b:<45} {n:>8,}  ({100*n/len(tiles):5.1f}%)")
    lines += ["", "  By region:"]
    for r, n in region_counts.most_common():
        lines.append(f"    {r:<30} {n:>8,}  ({100*n/len(tiles):5.1f}%)")
    lines += ["═" * 60, ""]
    return "\n".join(lines)


def _init_gee():
    ee.Initialize(
        ee.ServiceAccountCredentials(None, os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        project=GEE_PROJECT,
    )

    global esa_wc, esa_trees, esa_crop, gem_treecrop, jrc, nat_forest
    global meta_ch, DW, srtm_slope, glulc_2015, glulc_2020, TREE_CLASSES

    esa_wc = ee.Image("ESA/WorldCover/v100/2020")
    esa_trees = esa_wc.eq(10).unmask(0)
    esa_crop = esa_wc.eq(40).unmask(0)
    gem_treecrop = (
        ee.ImageCollection("projects/sat-io/open-datasets/GEM-Forest/GEM-Forest_2020")
        .mosaic()
        .select("b1")
        .eq(2)
        .unmask(0)
        .toFloat()
    )
    jrc = ee.Image("JRC/GFC2020_subtypes/V1")
    nat_forest = (
        ee.ImageCollection(
            "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
        )
        .mosaic()
        .select("B0")
        .divide(250)
        .unmask(0)
    )
    meta_ch = (
        ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
        .mosaic()
        .select("cover_code")
        .unmask(0)
    )
    DW = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    srtm_slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    glulc_2015 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").select([0])
    glulc_2020 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").select([0])
    TREE_CLASSES = ee.List.sequence(25, 96).cat(ee.List.sequence(125, 196))


def _tile_geom(tile: dict) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [tile["min_lon"], tile["min_lat"], tile["max_lon"], tile["max_lat"]],
        proj=ee.Projection("EPSG:4326"),
        geodesic=False,
    )


def _crs_transform(tile: dict) -> list:
    return [SCALE, 0, tile["x_min_m"], 0, -SCALE, tile["y_max_m"]]


def build_gain_layer(geom):
    ones = ee.List.repeat(1, TREE_CLASSES.length())
    tree_2015 = glulc_2015.clip(geom).remap(TREE_CLASSES, ones, 0)
    tree_2020 = glulc_2020.clip(geom).remap(TREE_CLASSES, ones, 0)
    gain = tree_2020.And(tree_2015.Not())
    clean = gain.updateMask(gain).focal_max(1).focal_min(1)
    validated = clean.And(esa_trees.clip(geom))
    return validated, validated.unmask(0).rename("gain")


def _mask_s2_scl(img):
    scl = img.select("SCL")
    return img.updateMask(
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(0))
    )


def _s2_avail(geom, year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .map(_mask_s2_scl)
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    )
    return ic.map(lambda i: i.mask().reduce(ee.Reducer.min())).reduce(ee.Reducer.max())


def build_full_valid(geom):
    return (
        _s2_avail(geom, 2016)
        .And(_s2_avail(geom, 2020))
        .And(_s2_avail(geom, 2025))
        .selfMask()
        .rename("valid")
    )


def _add_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5*((NIR-RED)/(NIR+6.0*RED-7.5*BLUE+1.0))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    return img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"]).addBands([ndvi, evi])


def s2_composite(geom, year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "NDVI", "EVI"]
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(_add_indices)
    )
    fallback = (
        ee.Image.constant([0] * 9)
        .rename([b + "_p25" for b in bands])
        .updateMask(ee.Image.constant(0))
    )
    reduced = ee.Image(
        ee.Algorithms.If(
            ic.size().eq(0), fallback, ic.reduce(ee.Reducer.percentile([25]))
        )
    )
    return reduced.select([b + "_p25" for b in bands], bands)


def s2_peak(geom, year):
    centroid = ee.Geometry(geom).centroid(maxError=1)
    north = ee.Number(centroid.coordinates().get(1)).gt(0)
    if year == 2016:
        start = ee.String(ee.Algorithms.If(north, "2015-05-01", "2015-11-01"))
        end = ee.String(ee.Algorithms.If(north, "2016-09-30", "2017-03-31"))
    else:
        start = ee.String(ee.Algorithms.If(north, f"{year}-05-01", f"{year}-11-01"))
        end = ee.String(ee.Algorithms.If(north, f"{year}-09-30", f"{year+1}-03-31"))
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(_add_indices)
        .select(["NDVI", "EVI"])
        .median()
    )


def s1_composite(geom, year):
    med = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .median()
    )
    return med.addBands(med.select("VV").divide(med.select("VH")).rename("VVVH"))


def dw_composite(geom, year):
    return (
        DW.filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .select(["trees", "crops", "built"])
        .median()
    )


def _stack(s2, s1, dw, prefix):
    bands = [
        f"{prefix}_{b}"
        for b in [
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "NDVI",
            "EVI",
            "VV",
            "VH",
            "VVVH",
            "DW_trees",
            "DW_crops",
            "DW_built",
        ]
    ]
    return s2.addBands(s1).addBands(dw).rename(bands)


def build_pseudo_labels(geom, gain_validated):
    gm = gain_validated.selfMask()
    jrc_planted = jrc.eq(20).unmask(0).toFloat()
    jrc_natreg = jrc.eq(1).unmask(0).toFloat()
    low_nat = ee.Image(1.0).subtract(nat_forest)

    ch_std = (
        meta_ch.updateMask(gm)
        .unmask(0)
        .reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(3, "pixels"))
        .divide(10)
        .min(1.0)
    )
    ch_uni = ee.Image(1.0).subtract(ch_std)

    def dw_mean(y0, y1, band):
        return DW.filterDate(y0, y1).filterBounds(geom).select(band).mean().unmask(0)

    dw_trees_pre = dw_mean("2015-01-01", "2016-12-31", "trees")
    dw_crops_pre = dw_mean("2015-01-01", "2016-12-31", "crops")
    dw_crops_post = dw_mean("2020-01-01", "2020-12-31", "crops")
    annual = [dw_mean(f"{y}-01-01", f"{y}-12-31", "trees") for y in range(2016, 2021)]
    dw_stack = ee.ImageCollection(annual).toBands()
    dw_slope = dw_stack.reduce(ee.Reducer.linearFit()).select("scale").max(0).min(1.0)
    dw_std = dw_stack.reduce(ee.Reducer.stdDev())

    s_agro = (
        gem_treecrop.updateMask(gm)
        .unmask(0)
        .multiply(esa_crop.toFloat())
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(dw_crops_pre.multiply(0.4))
            .add(dw_crops_post.multiply(0.4))
            .min(2.0)
        )
        .rename("score_agrocrop")
    )
    s_nat = (
        nat_forest.multiply(dw_trees_pre)
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(jrc_natreg.multiply(0.5))
            .add(dw_std.multiply(2.0).min(0.5))
            .min(2.0)
        )
        .rename("score_nat_regen")
    )
    s_plant = (
        jrc_planted.multiply(ch_uni)
        .pow(0.5)
        .multiply(
            ee.Image(1.0)
            .add(low_nat.multiply(0.5))
            .add(dw_trees_pre.multiply(0.3))
            .min(2.0)
        )
        .rename("score_plantation")
    )
    s_rest = (
        jrc_planted.multiply(ch_std)
        .multiply(low_nat)
        .pow(ee.Image(1.0).divide(3))
        .multiply(
            ee.Image(1.0)
            .add(dw_slope.multiply(0.5))
            .add(srtm_slope.divide(30).min(1.0).multiply(0.3))
            .min(2.0)
        )
        .rename("score_restoration")
    )

    scores = ee.Image.cat([s_agro, s_nat, s_plant, s_rest])
    dominant = scores.toArray().argmax().arrayGet(0).rename("dominant_class").toFloat()
    confidence = (
        scores.reduce(ee.Reducer.max())
        .divide(scores.reduce(ee.Reducer.sum()).max(1e-6))
        .rename("label_confidence")
    )
    return ee.Image.cat([scores, dominant, confidence]).toFloat()


def score_viability(geom, gain_validated) -> dict:
    gm = gain_validated.selfMask()
    ndvi_d = (
        s2_peak(geom, 2020)
        .select("NDVI")
        .subtract(s2_peak(geom, 2016).select("NDVI"))
        .updateMask(gm)
    )
    nd_stats = ndvi_d.reduceRegion(
        ee.Reducer.median(), geom, SCALE, CRS, maxPixels=1e13
    )
    ch_stats = meta_ch.updateMask(gm).reduceRegion(
        ee.Reducer.mean(), geom, SCALE, CRS, maxPixels=1e13
    )
    nd_val = ee.Number(ee.Algorithms.If(nd_stats.get("NDVI"), nd_stats.get("NDVI"), 0))
    ch_val = ee.Number(
        ee.Algorithms.If(ch_stats.get("cover_code"), ch_stats.get("cover_code"), 0)
    )
    return {"ndvi_delta": nd_val.getInfo(), "gain_canopy_mean": ch_val.getInfo()}


def build_full_stack(tile, geom, gain_validated, full_valid):
    fabdem = (
        ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
        .filterBounds(geom)
        .mosaic()
        .clip(geom)
    )
    gain_height = meta_ch.updateMask(gain_validated.selfMask()).rename(
        "canopy_gain_height"
    )
    return (
        _stack(
            s2_composite(geom, 2016),
            s1_composite(geom, 2016),
            dw_composite(geom, 2016),
            "T0",
        )
        .addBands(
            _stack(
                s2_composite(geom, 2020),
                s1_composite(geom, 2020),
                dw_composite(geom, 2020),
                "T1",
            )
        )
        .addBands(
            _stack(
                s2_composite(geom, 2025),
                s1_composite(geom, 2025),
                dw_composite(geom, 2025),
                "T2",
            )
        )
        .addBands(fabdem.rename("DEM"))
        .addBands(ee.Terrain.slope(fabdem).rename("slope"))
        .addBands(gain_height)
        .addBands(jrc.rename("jrc_forest_type"))
        .addBands(nat_forest.rename("natural_forest_prob"))
        .addBands(gain_validated.unmask(0).rename("gain_mask"))
        .addBands(build_pseudo_labels(geom, gain_validated))
        .updateMask(full_valid)
        .toFloat()
    )


def _rclone(tile_id: str, hpc_path: str):
    return subprocess.run(
        [
            "rclone",
            "moveto",
            "--drive-use-trash=false",
            f"{DRIVE_REMOTE}:{DRIVE_FOLDER}/{tile_id}.tif",
            f"{hpc_path}/{tile_id}.tif",
        ],
        capture_output=True,
        text=True,
    )


def monitor(
    submitted: dict, registry: dict, hpc_path: str | None, logger: logging.Logger
):
    """
    Poll GEE tasks for all submitted tiles.
    submitted: {tile_id: ee.Task}
    Updates registry status in place and flushes on every change.
    """
    remaining = dict(submitted)

    while remaining:
        done_this_round = []
        for tile_id, task in remaining.items():
            state = task.status()["state"]

            if state == "COMPLETED":
                if hpc_path:
                    result = _rclone(tile_id, hpc_path)
                    if result.returncode != 0:
                        logger.warning(
                            f"rclone failed for {tile_id}: {result.stderr[:80]}"
                        )
                    else:
                        logger.info(f"complete + transferred: {tile_id}")
                update_tile(
                    registry,
                    tile_id,
                    status=STATUS_COMPLETE,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                done_this_round.append(tile_id)

            elif state == "FAILED":
                err = task.status().get("error_message", "unknown")
                logger.error(f"GEE failed: {tile_id} — {err}")
                update_tile(registry, tile_id, status=STATUS_FAILED, error=err)
                done_this_round.append(tile_id)

        for t in done_this_round:
            remaining.pop(t)

        counts = Counter(registry[t]["status"] for t in submitted)
        logger.info(
            f"  complete={counts[STATUS_COMPLETE]}  "
            f"failed={counts[STATUS_FAILED]}  "
            f"pending={len(remaining)}"
        )
        if remaining:
            time.sleep(POLL_INTERVAL)


def process_tile(
    tile: dict, registry: dict, hpc_path: str | None, logger: logging.Logger
) -> str:
    """
    Submit a single tile to GEE, run viability checks, and update the registry.
    Returns the final status string for the tile.
    Called from both local thread workers and HPC process workers.
    """
    tile_id = tile["tile_id"]
    geom = _tile_geom(tile)
    ct = _crs_transform(tile)

    try:
        gain_validated, gain_binary = build_gain_layer(geom)

        gain_stats = gain_binary.reduceRegion(
            ee.Reducer.mean(), geom, SCALE, CRS, crsTransform=ct, maxPixels=1e9
        )
        gain_pct = (
            ee.Number(
                ee.Algorithms.If(gain_stats.get("gain"), gain_stats.get("gain"), 0)
            )
            .multiply(100)
            .getInfo()
        )

        if gain_pct < GAIN_PCT_MIN:
            reason = f"gain_pct={gain_pct:.3f} < {GAIN_PCT_MIN}"
            logger.info(f"reject (low gain {gain_pct:.2f}%): {tile_id}")
            update_tile(
                registry, tile_id, status=STATUS_REJECTED, rejection_reason=reason
            )
            return STATUS_REJECTED

        viability = score_viability(geom, gain_validated)
        if (
            viability["ndvi_delta"] <= NDVI_DELTA_MIN
            or viability["gain_canopy_mean"] < GAIN_CANOPY_MIN
        ):
            reason = f"viability={viability}"
            logger.info(f"reject (viability): {tile_id} {viability}")
            update_tile(
                registry, tile_id, status=STATUS_REJECTED, rejection_reason=reason
            )
            return STATUS_REJECTED

        full_valid = build_full_valid(geom)
        stack = build_full_stack(tile, geom, gain_validated, full_valid)

        task = ee.batch.Export.image.toDrive(
            image=stack,
            description=tile_id,
            folder=DRIVE_FOLDER,
            fileNamePrefix=tile_id,
            region=geom,
            scale=SCALE,
            crs=CRS,
            crsTransform=ct,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        update_tile(
            registry,
            tile_id,
            status=STATUS_SUBMITTED,
            gee_task_id=task.id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"submitted: {tile_id}  task={task.id}")

        # Poll this single task to completion so the caller gets a terminal status
        while True:
            state = task.status()["state"]
            if state == "COMPLETED":
                if hpc_path:
                    result = _rclone(tile_id, hpc_path)
                    if result.returncode != 0:
                        logger.warning(f"rclone failed {tile_id}: {result.stderr[:80]}")
                update_tile(
                    registry,
                    tile_id,
                    status=STATUS_COMPLETE,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                return STATUS_COMPLETE
            elif state == "FAILED":
                err = task.status().get("error_message", "unknown")
                logger.error(f"GEE task failed: {tile_id} — {err}")
                update_tile(registry, tile_id, status=STATUS_FAILED, error=err)
                return STATUS_FAILED
            elif state in ("CANCELLED", "CANCEL_REQUESTED"):
                update_tile(
                    registry, tile_id, status=STATUS_FAILED, error="task cancelled"
                )
                return STATUS_FAILED
            time.sleep(POLL_INTERVAL)

    except Exception as e:
        logger.error(f"error processing {tile_id}: {e}")
        update_tile(registry, tile_id, status=STATUS_FAILED, error=str(e))
        return STATUS_FAILED


def stratified_sample(
    candidates: list[dict],
    key: str,
    limit: int,
    mode: str = "prop",
) -> list[dict]:
    """
    Draw up to `limit` tiles from `candidates` stratified by `key`
    (typically "biome" or "region").

    mode="prop"  — each stratum gets floor(limit * stratum_share) tiles,
                   remainder allocated to largest strata first.
    mode="equal" — each stratum gets floor(limit / n_strata) tiles,
                   remainder allocated to largest strata first.

    Strata with fewer tiles than their allocation receive all available tiles.
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for tile in candidates:
        buckets[tile.get(key, "Unknown")].append(tile)

    n_strata = len(buckets)
    total = len(candidates)

    if mode == "equal":
        base_alloc = {k: limit // n_strata for k in buckets}
    else:  # prop
        base_alloc = {k: int(limit * len(v) / total) for k, v in buckets.items()}

    # Distribute any remainder to the largest strata
    allocated = sum(base_alloc.values())
    remainder = limit - allocated
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        if remainder <= 0:
            break
        base_alloc[k] += 1
        remainder -= 1

    sampled = []
    for k, alloc in base_alloc.items():
        sampled.extend(buckets[k][:alloc])

    return sampled


def run_local(candidates: list[dict], registry: dict, logger: logging.Logger):
    """Simple sequential processing — suitable for local testing."""
    hpc_path = f"{HPC_BASE}/{DRIVE_FOLDER}" if HPC_BASE else None
    total = len(candidates)

    for i, tile in enumerate(candidates, 1):
        logger.info(f"Tile {i}/{total}: {tile['tile_id']}")
        process_tile(tile, registry, hpc_path, logger)
        time.sleep(0.2)  # brief pause to avoid hammering GEE


def _mp_worker(tile_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
    _creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    _credentials = ee.ServiceAccountCredentials(None, _creds_path)
    time.sleep(worker_id * 5)
    ee.Initialize(_credentials, project=GEE_PROJECT)

    # Each process needs its own registry reference; results are sent via queue
    local_registry: dict = {}

    while True:
        item = tile_queue.get()
        if item is None:
            break

        tile = item
        tile_id = tile["tile_id"]
        local_registry[tile_id] = dict(tile)

        hpc_path = f"{HPC_BASE}/{DRIVE_FOLDER}" if HPC_BASE else None
        logger = logging.getLogger(f"gee.worker.{worker_id}")

        for attempt in range(8):
            status = process_tile(tile, local_registry, hpc_path, logger)
            if status != STATUS_FAILED:
                break
            err = local_registry[tile_id].get("error", "")
            if any(k in err.lower() for k in ("429", "concurrent", "quota", "memory")):
                wait = (2**attempt) + random.uniform(0, 2)
                logger.warning(
                    f"Worker {worker_id} | {tile_id} | retry {attempt+1}/8 in {wait:.1f}s"
                )
                time.sleep(wait)
            else:
                break
        else:
            logger.error(f"Worker {worker_id} | {tile_id}: exhausted retries")
            status = STATUS_FAILED

        result_queue.put((tile_id, local_registry[tile_id]))


def _mp_writer(
    result_queue: mp.Queue, total: int, registry: dict, logger: logging.Logger
):
    done = 0
    t0 = time.time()

    while done < total:
        try:
            tile_id, entry = result_queue.get(timeout=600)
        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total})")
            continue

        registry[tile_id] = entry
        done += 1

        if done % 20 == 0:
            save_registry(registry)
            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0
            complete = sum(
                1 for e in registry.values() if e["status"] == STATUS_COMPLETE
            )
            rejected = sum(
                1 for e in registry.values() if e["status"] == STATUS_REJECTED
            )
            failed = sum(1 for e in registry.values() if e["status"] == STATUS_FAILED)
            logger.info(
                f"Progress {done}/{total} | complete={complete} "
                f"rejected={rejected} failed={failed} | "
                f"{rate:.1f} tiles/min | {elapsed:.1f}min elapsed"
            )

    save_registry(registry)
    logger.info("Writer complete — registry flushed")


def run_hpc(candidates: list[dict], registry: dict, logger: logging.Logger):
    tile_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = [
        mp.Process(target=_mp_worker, args=(tile_queue, result_queue, i))
        for i in range(NUM_WORKERS)
    ]
    writer_thread = threading.Thread(
        target=_mp_writer,
        args=(result_queue, len(candidates), registry, logger),
        daemon=False,
    )

    for w in workers:
        w.start()
    writer_thread.start()

    logger.info(f"Started {NUM_WORKERS} HPC workers + 1 writer thread")

    for tile in candidates:
        tile_queue.put(tile)
    for _ in range(NUM_WORKERS):
        tile_queue.put(None)

    logger.info(f"Queued {len(candidates)} tiles")

    for w in workers:
        w.join()
    writer_thread.join()


def cmd_plan(args):
    logger = setup_logging("plan")

    logger.info(f"Loading valid AOIs from {VALID_AOIS_PATH}…")
    with open(VALID_AOIS_PATH) as f:
        valid_aois = json.load(f)
    logger.info(f"  {len(valid_aois):,} valid AOIs")

    tiles = build_global_grid(valid_aois, logger)
    print(plan_summary(tiles))

    registry = load_registry()
    new_count = 0
    for t in tiles:
        if t["tile_id"] not in registry:
            registry[t["tile_id"]] = t
            new_count += 1

    save_registry(registry)
    logger.info(
        f"Registry: {new_count:,} new tiles added, "
        f"{len(tiles)-new_count:,} already existed → {REGISTRY_PATH}"
    )

    # Write initial AOI audit
    audit = build_aoi_audit(registry, valid_aois)
    save_aoi_audit(audit)
    logger.info(f"AOI audit written → {AOI_AUDIT_PATH}")
    print(audit_summary(audit))


def cmd_status(args):
    registry = load_registry()
    if not registry:
        print("Registry is empty. Run `python gee.py plan` first.")
        return
    print(registry_summary(registry))


def cmd_audit(args):
    registry = load_registry()
    if not registry:
        print("Registry is empty. Run `python gee.py plan` first.")
        return
    logger = setup_logging("audit")
    logger.info(f"Loading valid AOIs from {VALID_AOIS_PATH}…")
    with open(VALID_AOIS_PATH) as f:
        valid_aois = json.load(f)
    audit = build_aoi_audit(registry, valid_aois)
    save_aoi_audit(audit)
    print(audit_summary(audit))
    uncovered = [aoi_id for aoi_id, v in audit.items() if not v["has_coverage"]]
    logger.info(
        f"AOIs with no complete tiles: {len(uncovered):,} — "
        f"see {AOI_AUDIT_PATH} for full breakdown"
    )


def cmd_run(args):
    logger = setup_logging("run")

    registry = load_registry()
    if not registry:
        logger.error("Registry is empty. Run `python gee.py plan` first.")
        return

    target_status = args.status or STATUS_PENDING
    # rejected tiles are never retried unless explicitly targeted
    if target_status == STATUS_REJECTED:
        logger.warning(
            "Targeting rejected tiles — these failed viability checks "
            "and will likely be rejected again unless thresholds have changed."
        )

    candidates = [e for e in registry.values() if e["status"] == target_status]

    if args.aoi_id:
        candidates = [e for e in candidates if args.aoi_id in e["aoi_ids"]]
        logger.info(f"Filtered to AOI {args.aoi_id}: {len(candidates):,} tiles")
    if args.biome:
        candidates = [
            e for e in candidates if args.biome.lower() in e.get("biome", "").lower()
        ]
        logger.info(f"Filtered to biome '{args.biome}': {len(candidates):,} tiles")
    if args.region:
        candidates = [
            e for e in candidates if args.region.lower() in e.get("region", "").lower()
        ]
        logger.info(f"Filtered to region '{args.region}': {len(candidates):,} tiles")

    # Stratified sampling must happen before a plain --limit truncation
    if args.stratify and args.limit:
        candidates = stratified_sample(
            candidates, args.stratify, args.limit, args.stratify_mode
        )
        strata_counts = Counter(t.get(args.stratify, "Unknown") for t in candidates)
        logger.info(
            f"Stratified sample ({args.stratify_mode}) by {args.stratify}: "
            f"{len(candidates):,} tiles across {len(strata_counts)} strata"
        )
        for stratum, n in strata_counts.most_common():
            logger.info(f"  {stratum:<45} {n:>6,}")
    elif args.limit:
        candidates = candidates[: args.limit]
        logger.info(f"Limited to {args.limit} tiles")

    if not candidates:
        logger.info("No tiles match the given filters.")
        return

    complete_count = sum(1 for e in registry.values() if e["status"] == STATUS_COMPLETE)
    logger.info(
        f"Processing {len(candidates):,} tiles  "
        f"(registry total: {len(registry):,}  complete: {complete_count:,})"
    )

    _init_gee()

    if USE_HPC:
        logger.info(f"Mode: HPC | workers={NUM_WORKERS}")
        run_hpc(candidates, registry, logger)
    else:
        logger.info("Mode: local sequential")
        run_local(candidates, registry, logger)

    print(registry_summary(registry))

    # Refresh AOI audit after every run
    if VALID_AOIS_PATH.exists():
        with open(VALID_AOIS_PATH) as f:
            valid_aois = json.load(f)
        audit = build_aoi_audit(registry, valid_aois)
        save_aoi_audit(audit)
        print(audit_summary(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forest-gain tile export pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python gee.py plan                              # build tile registry + AOI audit
        python gee.py status                            # print registry progress
        python gee.py audit                             # report AOIs with no complete tiles
        python gee.py run                               # process all pending tiles
        python gee.py run --limit 500                   # next 500 pending tiles
        python gee.py run --biome "Boreal Forests"      # pending tiles in biome
        python gee.py run --region Neotropic            # pending tiles in region
        python gee.py run --aoi-id aoi_-73.25_-52.75   # single AOI (debug)
        python gee.py run --status failed               # retry GEE-error tiles
        python gee.py run --status rejected             # re-examine viability rejects
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Build tile registry and AOI audit (no GEE calls)")
    sub.add_parser("status", help="Print current registry progress")
    sub.add_parser("audit", help="Report AOIs with no complete tiles")

    run_p = sub.add_parser("run", help="Submit and monitor export tasks")
    run_p.add_argument("--aoi-id", default=None, help="Filter by AOI ID")
    run_p.add_argument("--biome", default=None, help="Filter by biome (substring)")
    run_p.add_argument("--region", default=None, help="Filter by region (substring)")
    run_p.add_argument("--limit", default=None, type=int, help="Max tiles to process")
    run_p.add_argument(
        "--status",
        default=STATUS_PENDING,
        choices=[STATUS_PENDING, STATUS_FAILED, STATUS_REJECTED],
        help="Which tile status to target (default: pending)",
    )

    args = parser.parse_args()

    dispatch = {
        "plan": cmd_plan,
        "status": cmd_status,
        "audit": cmd_audit,
        "run": cmd_run,
    }
    dispatch[args.command](args)
