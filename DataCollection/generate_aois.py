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

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1000))
AOI_STEP = float(os.getenv("AOI_STEP", 1.0))
TILE_PIXELS = int(os.getenv("TILE_PIXELS", 128))
TILE_SCALE = int(os.getenv("TILE_SCALE", 10))

ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)

land = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
treecover = ee.Image("ESA/WorldCover/v100/2020")


def aoi_is_valid(aoi_geom):
    has_land = land.filterBounds(aoi_geom).size().gt(0)
    has_tree = (
        treecover.eq(10)
        .reduceRegion(
            reducer=ee.Reducer.anyNonZero(),
            geometry=aoi_geom,
            scale=1000,
            maxPixels=1e9,
        )
        .get("Map")
    )
    return ee.Number(has_land).And(ee.Number(has_tree).unmask(0).gt(0))


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

    processed_ids = {a["id"] for a in valid_aois}
    already_done = processed_ids | rejected_ids
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
    features = []
    for aoi_def in batch:
        geom = ee.Geometry.Rectangle(
            [aoi_def["minLon"], aoi_def["minLat"], aoi_def["maxLon"], aoi_def["maxLat"]]
        )
        features.append(ee.Feature(geom, aoi_def))

    fc = ee.FeatureCollection(features)

    def add_valid_flag(f):
        return f.set("valid", aoi_is_valid(f.geometry()))

    fc_validated = fc.map(add_valid_flag)

    valid_fc = fc_validated.filter(ee.Filter.gt("valid", 0))
    rejected_fc = fc_validated.filter(ee.Filter.eq("valid", 0))

    valid = valid_fc.getInfo()["features"]
    rejected = rejected_fc.getInfo()["features"]

    return valid, rejected


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
        json.dump(
            {
                "valid": valid_aois,
                "rejected": list(rejected_ids),
            },
            f,
            indent=2,
        )

    processed = i + len(batch)
    print(f"  {processed}/{len(remaining)} processed — {len(valid_aois)} valid so far")

    time.sleep(0.2)


with open(OUTPUT_FILE, "w") as f:
    json.dump(valid_aois, f, indent=2)

print(f"\nDone. {len(valid_aois)} valid AOIs saved to {OUTPUT_FILE}")
print(
    f"Each AOI will be tiled into {TILE_PIXELS}x{TILE_PIXELS} pixel tiles at {TILE_SCALE}m scale (~{(TILE_PIXELS * TILE_SCALE / 1000):.2f}km patches)"
)
