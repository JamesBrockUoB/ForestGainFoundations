"""
generate_aois.py

AOI generation and forest-gain class prior estimation.

Runs in two modes controlled by the USE_HPC environment variable:
  USE_HPC=0 (default) — single-process, sequential batches, suitable for local dev
  USE_HPC=1           — multiprocess workers + dedicated writer thread for HPC/SLURM

Usage:
  # Local
  python generate_aois.py

  # HPC
  USE_HPC=1 NUM_WORKERS=32 sbatch submit_aoi_generation.sh

────────────────────────────────────────────────────────────────────────────────
Datasets
────────────────────────────────────────────────────────────────────────────────
GEM Forest / Tree Crops         Tree crop vs forest — agrocrop anchor
Global Mangrove Watch (GMW)     Mangrove biome extent (no gain — extent only)
ESA WorldCover 2020             Mangrove (95), cropland (40), trees (10)
JRC GFC2020 subtypes            Planted (20) / nat regen (1) / primary (10)
Global Nat+Planted Forests      Second planted/natural opinion
Nature-Trace natural forest     Continuous naturalness probability
Dynamic World 2015–2020         Pre/post land cover + annual tree % trajectory
ESRI WorldCover 2020            Second cropland opinion
SRTM slope                      Terrain marginality
UMD GLCLUC 2015/2020            Forest gain definition (primary source)
────────────────────────────────────────────────────────────────────────────────

Classification approach
────────────────────────────────────────────────────────────────────────────────
Hard class assigned via priority decision tree (all GEE server-side ee.Number):
  0 = agrocrop     GEM tree-crop > 0.25 OR high pre-gain DW crops
  1 = nat_regen    JRC nat-regen OR high naturalness AND had pre-gain trees
  2 = plantation   JRC planted OR GNPF planted AND not natural AND not agrocrop
  3 = restoration  Bare/degraded baseline AND forest gain AND not other class
 -1 = ambiguous    Doesn't fit any pattern clearly

Soft probabilities set per hard class; ambiguous cells get balanced weights.
Mangrove cells: p_agrocrop = p_plantation = 0 by design.
────────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import ee
from dotenv import load_dotenv

load_dotenv("../")

# ── Configuration ──────────────────────────────────────────────────────────────

GEE_PROJECT = os.getenv("GEE_PROJECT")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))
OUTPUT_FILE = OUTPUT_DIR / os.getenv("OUTPUT_FILE", "aois/valid_aois.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTPUT_FILE.parent / "aoi_filter_checkpoint.json"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))
AOI_STEP = float(os.getenv("AOI_STEP", 0.25))
USE_HPC = os.getenv("USE_HPC", "0") == "1"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

MIN_VEG_FRACTION = 0.01  # trees OR mangrove (ESA)
MANGROVE_THRESHOLD = 0.15  # fraction of cell to trigger mangrove biome handling

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_DIR = OUTPUT_DIR / "logs"
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

# ── GEE initialisation ─────────────────────────────────────────────────────────

ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
logger.info(f"GEE initialised | project={GEE_PROJECT} | HPC={USE_HPC}")

# ── Dataset loading ────────────────────────────────────────────────────────────

land = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

# ESA WorldCover 2020
esa_wc = ee.Image("ESA/WorldCover/v100/2020")
esa_trees = esa_wc.eq(10).unmask(0)
esa_mangrove = esa_wc.eq(95).unmask(0)
esa_crop = esa_wc.eq(40).unmask(0)
esa_veg = esa_trees.Or(esa_mangrove).unmask(0)

# ESRI WorldCover 2020 (class 7 = crops)
esri_wc = (
    ee.ImageCollection(
        "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
    )
    .filterDate("2020-01-01", "2020-12-31")
    .mosaic()
)
esri_crop = esri_wc.eq(7).unmask(0)

# JRC GFC2020 subtypes: 1=nat_regen, 10=primary, 20=planted
jrc = ee.Image("JRC/GFC2020_subtypes/V1")

# Global Natural and Planted Forests
# b1=total cover, b2=planted fraction, b3=natural fraction, scaled 0-127; 127=nodata
gnpf = ee.ImageCollection(
    "projects/sat-io/open-datasets/GLOBAL-NATURAL-PLANTED-FORESTS"
).mosaic()
gnpf_valid = gnpf.updateMask(gnpf.select("b1").neq(127))
gnpf_planted_frac = gnpf_valid.select("b2").divide(127).unmask(0)

# Nature-Trace natural forest probability (B0 scaled 0-250)
nat_forest = (
    ee.ImageCollection(
        "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
    )
    .mosaic()
    .select("B0")
    .divide(250)
    .unmask(0)
)

# GEM Forest / Tree Crops — b1: 1=forest, 2=tree crop
gem = (
    ee.ImageCollection("projects/sat-io/open-datasets/GEM-Forest/GEM-Forest_2020")
    .mosaic()
    .select("b1")
    .unmask(0)
)
gem_treecrop = gem.eq(2).unmask(0)

# Global Mangrove Watch 2020 — extent only
gmw_2020 = (
    ee.ImageCollection("projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020")
    .mosaic()
    .unmask(0)
    .rename("mangrove_2020")
)

# Meta canopy height v1 — cover_code band, metres
meta_ch = (
    ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
    .mosaic()
    .select("cover_code")
    .unmask(0)
)

# SRTM slope
srtm_slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))

# Dynamic World
DW = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")

# UMD GLCLUC forest gain
glulc_2015 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").select([0])
glulc_2020_i = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").select([0])
TREE_CLASSES = ee.List.sequence(25, 96).cat(ee.List.sequence(125, 196))

logger.info("All GEE datasets loaded")


# ── Helpers ────────────────────────────────────────────────────────────────────


def safe_num(val, default=0):
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val, None), default, val))


def ee_if(condition, true_val, false_val):
    """Convenience wrapper for ee.Algorithms.If with ee.Number output."""
    return ee.Number(ee.Algorithms.If(condition, true_val, false_val))


def flag_if(flag, true_val, false_val):
    """
    Conditional branch on an ee.Number flag (0 or 1).
    Uses .eq(1) so GEE server-side evaluation works correctly —
    passing an ee.Number directly to ee.Algorithms.If does not
    treat numeric 1 as True on the server.
    """
    return ee.Number(ee.Algorithms.If(flag.eq(1), true_val, false_val))


def reduce_mean(image, geom, scale=1000):
    val = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9
    ).get(image.bandNames().get(0))
    return safe_num(val, 0)


def canopy_height_stddev_in_gain(geom, gain_mask, scale=100):
    """
    Spatial stddev of canopy height within gain pixels, normalised to [0,1].
    Normalised by dividing by 10m (typical plantation cohort height) then clamped.

    Low stddev  → uniform even-aged canopy → plantation signal
    High stddev → mixed heights            → restoration or nat_regen signal
    """
    ch_gain = meta_ch.updateMask(gain_mask)
    val = ch_gain.reduceRegion(
        reducer=ee.Reducer.stdDev(), geometry=geom, scale=scale, maxPixels=1e9
    ).get("cover_code")
    raw = safe_num(val, 0)
    return raw.divide(10).min(1.0)  # normalised [0,1]


def dw_band_mean(geom, band, start, end, scale=1000):
    BAND_MAP = {
        "shrub": "shrub_and_scrub",
        "bare": "bare",
        "trees": "trees",
        "crops": "crops",
    }
    b = BAND_MAP.get(band, band)
    coll = DW.filterDate(start, end).filterBounds(geom).select(b)
    img = ee.Image(
        ee.Algorithms.If(coll.size().gt(0), coll.mean(), ee.Image(0).rename(b))
    )
    val = img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9
    ).get(b)
    return safe_num(val, 0)


def s2_scene_count(geom, year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .size()
    )


def forest_gain_fraction_umd(geom, scale=30):
    """
    Fraction of cell with UMD forest gain 2015→2020, plus the raw gain mask.
    Returns (fg_frac, gain_mask). Caller is responsible for checking fg_frac > 0.
    """
    ones = ee.List.repeat(1, TREE_CLASSES.length())
    tree2015 = glulc_2015.remap(TREE_CLASSES, ones, 0)
    tree2020 = glulc_2020_i.remap(TREE_CLASSES, ones, 0)
    gain_mask = tree2020.And(tree2015.Not()).select([0]).rename("gain")
    val = gain_mask.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9
    ).get("gain")
    return safe_num(val, 0), gain_mask


def rejection_reason_str(reason_code):
    """Convert rejection bitfield to human-readable string.
    Bits: 0x1=insufficient_veg, 0x2=missing_s2, 0x4=no_land, 0x8=no_forest_gain
    """
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


# ── Classification ─────────────────────────────────────────────────────────────


def classify_aoi(geom):
    """
    Compute a [0,1] ranking score for each of the four forest-gain classes.

    No labels, no classifier, no thresholds.  Each score is a direct arithmetic
    combination of GEE signals that definitionally or corroboratively measure
    that class.  Scores are independent — they do not sum to 1.  Use them for
    stratified sampling: sample AOIs ranked highest for each class.

    Scores
    ──────
    score_agrocrop    GEM tree-crop × crops_pre × crops_post boost
    score_nat_regen   Nature-Trace × GNPF_natural × trees_pre boost
    score_plantation  GNPF_planted × JRC_planted × low CH-std boost
    score_restoration bare_pre × gain_frac × low trees_pre boost

    Geometric mean of definitional signals ensures the score is only high
    when multiple independent lines of evidence agree.  Corroborating signals
    provide a multiplicative boost (capped so they cannot flip a near-zero
    definitional score into a high rank).

    Mangrove cells: agrocrop and plantation scores are zeroed out.
    """

    fg_frac, gain_mask = forest_gain_fraction_umd(geom)

    # ── Evidence signals ───────────────────────────────────────────────────

    # GEM tree-crop (definitional for agrocrop)
    ev_gem = reduce_mean(gem_treecrop.updateMask(gain_mask), geom, scale=100)

    # JRC forest type fractions (whole cell)
    ev_jrc_planted = reduce_mean(jrc.eq(20).unmask(0), geom)
    ev_jrc_natregen = reduce_mean(jrc.eq(1).unmask(0), geom)

    # Nature-Trace naturalness probability (definitional for nat_regen)
    ev_nat = reduce_mean(nat_forest, geom)

    # GNPF planted/natural fractions
    ev_gnpf_planted = reduce_mean(gnpf_planted_frac, geom, scale=100)
    ev_gnpf_natural = ee.Number(1.0).subtract(ev_gnpf_planted)

    # DW pre-gain land cover (what was there before trees appeared)
    ev_trees_pre = dw_band_mean(geom, "trees", "2015-01-01", "2016-12-31")
    ev_crops_pre = dw_band_mean(geom, "crops", "2015-01-01", "2016-12-31")
    ev_bare_pre = (
        dw_band_mean(geom, "bare", "2015-01-01", "2016-12-31")
        .add(dw_band_mean(geom, "shrub", "2015-01-01", "2016-12-31"))
        .min(1.0)
    )

    # DW post-gain crops (confirms agrocrop persists after tree establishment)
    ev_crops_post = dw_band_mean(geom, "crops", "2020-01-01", "2020-12-31")

    # DW annual trees trajectory 2016-2020
    years = [2016, 2017, 2018, 2019, 2020]
    annual = [dw_band_mean(geom, "trees", f"{y}-01-01", f"{y}-12-31") for y in years]
    mean_a = ee.Number(0)
    for v in annual:
        mean_a = mean_a.add(v)
    mean_a = mean_a.divide(5)
    var = ee.Number(0)
    for v in annual:
        var = var.add(v.subtract(mean_a).pow(2))
    ev_dw_std = var.divide(5).sqrt()
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    cov = ee.Number(0)
    for x, v in zip(xs, annual):
        cov = cov.add(ee.Number(x).multiply(v.subtract(mean_a)))
    ev_dw_slope = cov.divide(10.0).max(0).min(1.0)

    # ESA / ESRI cropland fractions (corroborating for agrocrop)
    ev_esa_crop = reduce_mean(esa_crop.toFloat(), geom)
    ev_esri_crop = reduce_mean(esri_crop.toFloat(), geom)

    # Canopy height spatial stddev in gain pixels (normalised /10)
    # Low = uniform even-aged canopy → plantation signal
    ev_ch_std = canopy_height_stddev_in_gain(geom, gain_mask)

    # Mangrove extent
    ev_esa_mng = reduce_mean(esa_mangrove.toFloat(), geom)
    ev_gmw = reduce_mean(gmw_2020.toFloat(), geom)
    ev_mangrove = ev_esa_mng.max(ev_gmw)

    # Slope (raw degrees — not used in class scores but recorded)
    ev_slope = safe_num(
        srtm_slope.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=1000, maxPixels=1e9
        ).get("slope"),
        0,
    )

    # ── Class scores ───────────────────────────────────────────────────────
    # Each score = geometric_mean(definitional signals) * corroborating_boost
    # Geometric mean of N signals = (s1 * s2 * ... * sN) ^ (1/N)
    # ── Class scores ───────────────────────────────────────────────────────
    # Each score = geometric_mean(definitional signals) * corroborating_boost
    #
    # Geometric mean ensures score is only high when ALL definitional signals agree.
    # Boost = 1 + weighted corroborating signals, capped at 2x.
    #
    # GNPF is excluded as a definitional signal — it reads 0.6–0.98 "natural"
    # even for known plantations at 0.25° resolution and does not discriminate.

    slope_norm = ev_slope.divide(30).min(1.0)

    # AGROCROP
    # Definitional: GEM tree-crop (purpose-built) × ESA cropland fraction
    # Two independent products must agree. Corroborated by ESRI crop, DW crops.
    agro_def = ev_gem.multiply(ev_esa_crop).pow(0.5)
    agro_boost = (
        ee.Number(1.0)
        .add(ev_esri_crop.multiply(0.4))
        .add(ev_crops_pre.multiply(0.3))
        .add(ev_crops_post.multiply(0.3))
        .min(2.0)
    )
    score_agrocrop = agro_def.multiply(agro_boost)

    # NAT_REGEN
    # Definitional: Nature-Trace naturalness × DW trees pre-gain
    # Cell must read as natural forest AND have had tree cover before gain
    # (genuine succession, not planting into bare land).
    # Corroborated by JRC nat-regen fraction and DW temporal stddev (patchy succession).
    natregen_def = ev_nat.multiply(ev_trees_pre).pow(0.5)
    natregen_boost = (
        ee.Number(1.0)
        .add(ev_jrc_natregen.multiply(0.5))
        .add(ev_dw_std.multiply(0.5))
        .min(2.0)
    )
    score_nat_regen = natregen_def.multiply(natregen_boost)

    # PLANTATION
    # Definitional: JRC planted fraction × low CH stddev (uniform even-aged canopy)
    # JRC is the best available planted-forest map; canopy uniformity confirms
    # monoculture structure. Low stddev = commercial timber cohort.
    # Corroborated by low Nature-Trace naturalness and DW trees pre (established stand).
    ch_uniformity = ee.Number(1.0).subtract(ev_ch_std)
    plantation_def = ev_jrc_planted.multiply(ch_uniformity).pow(0.5)
    plantation_boost = (
        ee.Number(1.0)
        .add(ee.Number(1.0).subtract(ev_nat).multiply(0.5))
        .add(ev_trees_pre.multiply(0.3))
        .min(2.0)
    )
    score_plantation = plantation_def.multiply(plantation_boost)

    # RESTORATION
    # Definitional: JRC planted × high CH stddev × low Nature-Trace
    # Separates restoration from:
    #   plantation → same JRC planted, but plantation has LOW ch_std (uniform)
    #   nat_regen  → both have high ch_std, but nat_regen has HIGH Nature-Trace
    #   agrocrop   → no JRC planted signal
    # Corroborated by positive DW trees slope (active recruitment) and slope
    # (marginal terrain avoided by commercial forestry).
    restoration_def = (
        ev_jrc_planted.multiply(ev_ch_std)
        .multiply(ee.Number(1.0).subtract(ev_nat))
        .pow(ee.Number(1.0).divide(3))
    )
    restoration_boost = (
        ee.Number(1.0)
        .add(ev_dw_slope.multiply(0.5))
        .add(slope_norm.multiply(0.3))
        .min(2.0)
    )
    score_restoration = restoration_def.multiply(restoration_boost)

    # ── Mangrove override ──────────────────────────────────────────────────
    # Mangroves cannot be agrocrop or plantation.  Zero those scores out.
    is_mangrove = ev_mangrove.gt(MANGROVE_THRESHOLD)
    score_agrocrop = ee.Number(ee.Algorithms.If(is_mangrove.eq(1), 0.0, score_agrocrop))
    score_plantation = ee.Number(
        ee.Algorithms.If(is_mangrove.eq(1), 0.0, score_plantation)
    )

    # ── Dominant class = argmax of scores ─────────────────────────────────
    hard_class = ee.Number(
        ee.Algorithms.If(
            score_agrocrop.gte(score_nat_regen)
            .And(score_agrocrop.gte(score_plantation))
            .And(score_agrocrop.gte(score_restoration)),
            0,
            ee.Algorithms.If(
                score_nat_regen.gt(score_agrocrop)
                .And(score_nat_regen.gte(score_plantation))
                .And(score_nat_regen.gte(score_restoration)),
                1,
                ee.Algorithms.If(
                    score_plantation.gt(score_agrocrop)
                    .And(score_plantation.gt(score_nat_regen))
                    .And(score_plantation.gte(score_restoration)),
                    2,
                    3,
                ),
            ),
        )
    )

    return hard_class, {
        # Raw evidence
        "ev_gem_treecrop": ev_gem,
        "ev_jrc_planted": ev_jrc_planted,
        "ev_jrc_natregen": ev_jrc_natregen,
        "ev_nat_forest": ev_nat,
        "ev_gnpf_planted": ev_gnpf_planted,
        "ev_gnpf_natural": ev_gnpf_natural,
        "ev_dw_trees_pre": ev_trees_pre,
        "ev_dw_crops_pre": ev_crops_pre,
        "ev_dw_crops_post": ev_crops_post,
        "ev_dw_bare_pre": ev_bare_pre,
        "ev_dw_trees_std": ev_dw_std,
        "ev_dw_trees_slope": ev_dw_slope,
        "ev_esa_crop_frac": ev_esa_crop,
        "ev_esri_crop_frac": ev_esri_crop,
        "ev_esa_mangrove": ev_esa_mng,
        "ev_gmw_frac": ev_gmw,
        "ev_ch_std": ev_ch_std,
        "ev_slope": ev_slope,
        "forest_gain_frac": fg_frac,
        # Class scores (independent [0,1] rankings, not probabilities)
        "score_agrocrop": score_agrocrop,
        "score_nat_regen": score_nat_regen,
        "score_plantation": score_plantation,
        "score_restoration": score_restoration,
    }


# ── Per-AOI validation ─────────────────────────────────────────────────────────


def aoi_is_valid(f):
    geom = f.geometry()

    has_land = land.filterBounds(geom).size().gt(0)
    veg_frac = safe_num(
        esa_veg.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=1000, maxPixels=1e9
        ).get("Map"),
        0,
    )
    # Relax veg threshold if cell has confirmed UMD forest gain —
    # UMD (30m) and ESA (aggregated to 1km) measure at different scales,
    # so a cell can have real gain pixels but low ESA veg fraction.
    fg_frac_pre, _ = forest_gain_fraction_umd(geom)
    has_confirmed_gain = fg_frac_pre.gt(0.001)
    veg_threshold = ee.Number(
        ee.Algorithms.If(has_confirmed_gain.eq(1), 0.005, MIN_VEG_FRACTION)
    )
    has_veg = veg_frac.gt(veg_threshold)

    s2_2016 = s2_scene_count(geom, 2016)
    s2_2020 = s2_scene_count(geom, 2020)
    s2_2025 = s2_scene_count(geom, 2025)
    has_s2 = s2_2020.gt(0).And(s2_2025.gt(0))

    # Forest gain check — reuse pre-computed value from veg threshold step
    has_gain = has_confirmed_gain

    is_valid = has_land.And(has_veg).And(has_s2).And(has_gain)

    # Rejection reason bitfield: 0x1=veg, 0x2=s2, 0x4=land, 0x8=no_gain
    rejection_reason = (
        ee.Number(0)
        .add(ee.Number(1).multiply(has_veg.Not()))
        .add(ee.Number(2).multiply(has_s2.Not()))
        .add(ee.Number(4).multiply(has_land.Not()))
        .add(ee.Number(8).multiply(has_gain.Not()))
    )

    hard_class, evidence = classify_aoi(geom)

    return f.set(
        {
            "valid": is_valid,
            "rejection_reason": rejection_reason,
            "veg_fraction": veg_frac,
            "s2_count_2016": s2_2016,
            "s2_count_2020": s2_2020,
            "s2_count_2025": s2_2025,
            "hard_class": hard_class,
            **evidence,
        }
    )


# ── AOI grid ───────────────────────────────────────────────────────────────────


def generate_global_aois(step=AOI_STEP):
    aois = []
    lat = -90.0
    while lat < 90.0:
        lon = -180.0
        while lon < 180.0:
            aois.append(
                {
                    "id": f"aoi_{round(lon,4)}_{round(lat,4)}",
                    "minLon": round(lon, 4),
                    "minLat": round(lat, 4),
                    "maxLon": round(min(lon + step, 180.0), 4),
                    "maxLat": round(min(lat + step, 90.0), 4),
                }
            )
            lon = round(lon + step, 4)
        lat = round(lat + step, 4)
    return aois


# ── Batch processing ───────────────────────────────────────────────────────────


def process_batch(batch):
    features = [
        ee.Feature(
            ee.Geometry.Rectangle([a["minLon"], a["minLat"], a["maxLon"], a["maxLat"]]),
            a,
        )
        for a in batch
    ]
    fc = ee.FeatureCollection(features)
    fc_validated = fc.map(aoi_is_valid)
    valid_fc = fc_validated.filter(ee.Filter.gt("valid", 0))
    rejected_fc = fc_validated.filter(ee.Filter.eq("valid", 0))
    return valid_fc.getInfo()["features"], rejected_fc.getInfo()["features"]


# ── Execution modes ────────────────────────────────────────────────────────────


def run_local(remaining):
    valid_aois = []
    rejected_aois = []
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        try:
            valid_batch, rejected_batch = process_batch(batch)
            valid_aois.extend([f["properties"] for f in valid_batch])
            rejected_aois.extend([f["properties"] for f in rejected_batch])
        except Exception as e:
            logger.error(f"Batch error (offset {i}): {e} — will retry next run")
        with open(CHECKPOINT, "w") as f:
            json.dump({"valid": valid_aois, "rejected": rejected_aois}, f)
        logger.info(
            f"  {i + len(batch)}/{len(remaining)} processed — {len(valid_aois)} valid"
        )
        time.sleep(0.2)
    return valid_aois, rejected_aois


def _worker(batch_queue, result_queue, worker_id):
    while True:
        item = batch_queue.get()
        if item is None:
            break
        batch_idx, batch = item
        try:
            valid, rejected = process_batch(batch)
            result_queue.put(("valid", batch_idx, valid))
            result_queue.put(("rejected", batch_idx, rejected))
        except Exception as e:
            logger.error(f"Worker {worker_id} | Batch {batch_idx}: {e}")
            result_queue.put(("error", batch_idx, str(e)))


def _writer(result_queue, total_batches):
    valid_aois = []
    rejected_aois = []
    done = 0
    t0 = time.time()
    while done < total_batches:
        try:
            msg_type, batch_idx, data = result_queue.get(timeout=300)
        except Exception:
            logger.warning(f"Result queue timeout ({done}/{total_batches})")
            continue
        if msg_type == "valid":
            valid_aois.extend([f["properties"] for f in data])
        elif msg_type == "rejected":
            rejected_aois.extend([f["properties"] for f in data])
        elif msg_type == "error":
            logger.warning(f"Batch {batch_idx} error: {data}")
        done += 1
        if done % 10 == 0:
            with open(CHECKPOINT, "w") as f:
                json.dump({"valid": valid_aois, "rejected": rejected_aois}, f)
            elapsed = (time.time() - t0) / 60
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                f"Checkpoint {done}/{total_batches} | {len(valid_aois)} valid | "
                f"{rate:.1f} batches/min | {elapsed:.1f}min elapsed"
            )
    with open(OUTPUT_FILE, "w") as f:
        json.dump(valid_aois, f, indent=2)
    logger.info(f"✓ Final output: {len(valid_aois)} valid AOIs → {OUTPUT_FILE}")
    return valid_aois, rejected_aois


def run_hpc(remaining):
    batches = [
        remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)
    ]
    batch_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = [
        mp.Process(target=_worker, args=(batch_queue, result_queue, i))
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()
    writer_thread = threading.Thread(
        target=_writer, args=(result_queue, len(batches)), daemon=False
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


# ── Output summary ─────────────────────────────────────────────────────────────


def print_summary(valid_aois, rejected_aois):
    CLASS_NAMES = {
        0: "agrocrop",
        1: "nat_regen",
        2: "plantation",
        3: "restoration",
        -1: "ambiguous",
    }
    counts = {k: 0 for k in CLASS_NAMES}
    for a in valid_aois:
        c = a.get("hard_class")
        if c is not None:
            counts[int(c)] = counts.get(int(c), 0) + 1
    # Also report mean scores per class across all valid AOIs
    score_sums = {
        k: 0.0 for k in ["agrocrop", "nat_regen", "plantation", "restoration"]
    }
    for a in valid_aois:
        for cls in score_sums:
            v = a.get(f"score_{cls}", 0) or 0
            score_sums[cls] += float(v)

    rejection_counts = {}
    for a in rejected_aois:
        reason = rejection_reason_str(int(a.get("rejection_reason", 0)))
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done — {len(valid_aois)} valid AOIs → {OUTPUT_FILE}")
    logger.info("Hard class distribution (dominant score):")
    for k, name in CLASS_NAMES.items():
        n = counts.get(k, 0)
        pct = 100 * n / len(valid_aois) if valid_aois else 0
        mean_score = score_sums.get(name, 0) / len(valid_aois) if valid_aois else 0
        logger.info(f"  {name:14s}: {n:7d} ({pct:5.1f}%)  mean_score={mean_score:.3f}")

    total_rejected = sum(rejection_counts.values())
    logger.info(f"\nRejections ({total_rejected} total):")
    for reason, n in sorted(rejection_counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total_rejected if total_rejected else 0
        logger.info(f"  {reason:30s}: {n:7d} ({pct:5.1f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_aois = generate_global_aois()
    logger.info(f"Total {AOI_STEP}° cells: {len(all_aois)}")

    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            data = json.load(f)
            loaded_valid = data.get("valid", [])
            loaded_rejected = data.get("rejected", [])
        already_done = {a["id"] for a in loaded_valid} | {
            a["id"] for a in loaded_rejected
        }
        remaining = [a for a in all_aois if a["id"] not in already_done]
        logger.info(
            f"Resuming — {len(loaded_valid)} valid, {len(loaded_rejected)} rejected, {len(remaining)} remaining"
        )
    else:
        remaining = all_aois
        loaded_valid = []
        loaded_rejected = []
        logger.info(f"Starting fresh — {len(remaining)} cells to process")

    if USE_HPC:
        logger.info(f"Mode: HPC | workers={NUM_WORKERS} | batch_size={BATCH_SIZE}")
        run_hpc(remaining)
        with open(OUTPUT_FILE) as f:
            valid_aois = json.load(f)
        rejected_aois = loaded_rejected
    else:
        logger.info(f"Mode: local | batch_size={BATCH_SIZE}")
        valid_aois, rejected_batch = run_local(remaining)
        rejected_aois = loaded_rejected + rejected_batch
        with open(OUTPUT_FILE, "w") as f:
            json.dump(valid_aois, f, indent=2)

    print_summary(valid_aois, rejected_aois)
