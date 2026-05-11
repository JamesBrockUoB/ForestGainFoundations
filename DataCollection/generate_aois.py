import json
import os
import time
from pathlib import Path

import ee
from dotenv import load_dotenv

load_dotenv()

GEE_PROJECT = os.getenv("GEE_PROJECT")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))
OUTPUT_FILE = OUTPUT_DIR / os.getenv("OUTPUT_FILE", "aois/valid_aois.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTPUT_FILE.parent / "aoi_filter_checkpoint.json"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))
AOI_STEP = float(os.getenv("AOI_STEP", 1.0))
TILE_PIXELS = int(os.getenv("TILE_PIXELS", 128))
TILE_SCALE = int(os.getenv("TILE_SCALE", 10))

MIN_TREE_COVER_FRACTION = 0.01

ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)

land = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
worldcover = ee.Image("ESA/WorldCover/v100/2020")
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
is_forest = worldcover.eq(10)
forest_px = is_forest.unmask(0)


def s2_scene_count(geom, year):
    """Count S2 L2A scenes for a year, used as availability proxy."""
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .size()
    )


def dw_crops_mean(geom):
    """Mean DynamicWorld crops probability 2016-2018 as agroforestry prior."""
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate("2016-01-01", "2018-12-31")
        .filterBounds(geom)
        .select("crops")
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9,
        )
        .get("crops", 0)
    )


def aoi_is_valid(f):
    geom = f.geometry()

    # Land presence
    has_land = land.filterBounds(geom).size().gt(0)

    # Tree cover fraction at 1km — require > MIN_TREE_COVER_FRACTION
    tree_stats = forest_px.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=1000,
        maxPixels=1e9,
    )
    tree_fraction = ee.Number(tree_stats.get("Map", 0))
    has_sufficient_tree = tree_fraction.gt(MIN_TREE_COVER_FRACTION)

    s2_2016 = s2_scene_count(geom, 2016)
    s2_2020 = s2_scene_count(geom, 2020)
    s2_2025 = s2_scene_count(geom, 2025)

    has_s2_all = s2_2016.gt(0).And(s2_2020.gt(0)).And(s2_2025.gt(0))

    is_valid = has_land.And(has_sufficient_tree).And(has_s2_all)

    # JRC mode: 1=natural regen, 10=primary, 20=planted
    jrc_mode_num = ee.Number(
        jrc.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9,
        ).get("Map", -1)
    )

    # Nature Trace natural forest probability mean
    nat_mean_num = ee.Number(
        nat_forest.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9,
        ).get("B0", 0)
    )

    # DynamicWorld crops fraction as agroforestry prior
    crops_mean_num = ee.Number(dw_crops_mean(geom))

    # Soft dominant class label based on feature combination:
    #   plantation:       jrc_mode == 20
    #   agroforestry:     crops_mean > 0.2
    #   natural_regen:    jrc_mode == 1 AND nat_mean > 0.3
    #   restoration:      nat_mean <= 0.3 AND jrc_mode != 20 AND crops_mean <= 0.2
    # Expressed as integer: 0=plantation, 1=agroforestry, 2=natural_regen, 3=restoration
    jrc_mode_num = ee.Number(ee.Algorithms.If(jrc_mode, jrc_mode, -1))

    nat_mean_num = ee.Number(ee.Algorithms.If(nat_mean, nat_mean, 0))

    crops_mean_num = ee.Number(ee.Algorithms.If(crops_mean, crops_mean, 0))

    is_plantation = jrc_mode_num.eq(20)
    is_agroforestry = crops_mean_num.gt(0.2).And(is_plantation.Not())
    is_nat_regen = (
        jrc_mode_num.eq(1)
        .And(nat_mean_num.gt(0.3))
        .And(is_plantation.Not())
        .And(is_agroforestry.Not())
    )
    # restoration is the residual
    dominant_class = (
        ee.Number(0)
        .multiply(is_plantation)
        .add(ee.Number(1).multiply(is_agroforestry))
        .add(ee.Number(2).multiply(is_nat_regen))
        .add(
            ee.Number(3).multiply(
                is_plantation.Not().And(is_agroforestry.Not()).And(is_nat_regen.Not())
            )
        )
    )

    return f.set(
        {
            "valid": is_valid,
            "tree_fraction": tree_fraction,
            "s2_count_2016": s2_2016,
            "s2_count_2020": s2_2020,
            "s2_count_2025": s2_2025,
            "jrc_mode": jrc_mode_num,
            "nat_mean": nat_mean_num,
            "crops_mean": crops_mean_num,
            "dominant_class": dominant_class,
        }
    )


def generate_global_aois(step=AOI_STEP):
    aois = []
    lat = -90.0
    while lat < 90.0:
        lon = -180.0
        while lon < 180.0:
            aois.append(
                {
                    "id": f"aoi_{round(lon, 4)}_{round(lat, 4)}",
                    "minLon": round(lon, 4),
                    "minLat": round(lat, 4),
                    "maxLon": round(min(lon + step, 180.0), 4),
                    "maxLat": round(min(lat + step, 90.0), 4),
                }
            )
            lon += step
        lat += step
    return aois


all_aois = generate_global_aois()
print(f"Total 1x1 degree cells: {len(all_aois)}")

if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        data = json.load(f)
        valid_aois = data.get("valid", [])
        rejected_ids = set(data.get("rejected", []))
    already_done = {a["id"] for a in valid_aois} | rejected_ids
    remaining = [a for a in all_aois if a["id"] not in already_done]
    print(
        f"Resuming — {len(valid_aois)} valid, {len(rejected_ids)} rejected, {len(remaining)} remaining"
    )
else:
    valid_aois = []
    rejected_ids = set()
    remaining = all_aois
    print(f"Starting fresh — {len(remaining)} cells to process")


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


for i in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[i : i + BATCH_SIZE]
    try:
        valid_batch, rejected_batch = process_batch(batch)
        for f in valid_batch:
            valid_aois.append(f["properties"])
        for f in rejected_batch:
            rejected_ids.add(f["properties"]["id"])
    except Exception as e:
        print(f"Batch error ({i}): {e} — skipping batch")
        for a in batch:
            rejected_ids.add(a["id"])

    with open(CHECKPOINT, "w") as f:
        json.dump({"valid": valid_aois, "rejected": list(rejected_ids)}, f, indent=2)

    processed = i + len(batch)
    print(f"  {processed}/{len(remaining)} processed — {len(valid_aois)} valid so far")
    time.sleep(0.2)

with open(OUTPUT_FILE, "w") as f:
    json.dump(valid_aois, f, indent=2)

class_names = {0: "plantation", 1: "agroforestry", 2: "natural_regen", 3: "restoration"}
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for a in valid_aois:
    c = a.get("dominant_class")
    if c is not None:
        class_counts[int(c)] = class_counts.get(int(c), 0) + 1

print(f"\nDone. {len(valid_aois)} valid AOIs saved to {OUTPUT_FILE}")
print("\nDominant class distribution across valid AOIs:")
for k, name in class_names.items():
    print(f"  {name:20s}: {class_counts.get(k, 0)}")
print(
    f"\nEach AOI tiles into {TILE_PIXELS}x{TILE_PIXELS}px tiles "
    f"at {TILE_SCALE}m (~{TILE_PIXELS * TILE_SCALE / 1000:.2f}km patches)"
)
