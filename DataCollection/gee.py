import json
import os
import subprocess
import time
from pathlib import Path

import ee
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GEE_PROJECT = os.getenv("GEE_PROJECT")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))
DRIVE_REMOTE = os.getenv("DRIVE_REMOTE", "gdrive")
HPC_BASE = os.getenv("HPC_BASE")
POLL_INTERVAL = 30

TILE_PIXELS = 128
SCALE = 10
TILE_METERS = TILE_PIXELS * SCALE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)


def aoi_to_folder(aoi_id: str) -> str:
    return f"forest_gain_{aoi_id.replace(' ', '_').lower()}"


def build_aoi(bounds):
    raw = ee.Geometry.Rectangle(bounds)
    coords = ee.List(raw.coordinates().get(0))

    raw_min_lon = ee.Number(ee.List(coords.get(0)).get(0))
    raw_min_lat = ee.Number(ee.List(coords.get(0)).get(1))
    raw_max_lon = ee.Number(ee.List(coords.get(2)).get(0))
    raw_max_lat = ee.Number(ee.List(coords.get(2)).get(1))

    tile_deg_lat = ee.Number(TILE_METERS).divide(111320)
    center_lat = raw_min_lat.add(raw_max_lat).divide(2)
    lat_cos = center_lat.multiply(3.141592653589793 / 180).cos()
    tile_deg_lon = ee.Number(TILE_METERS).divide(111320).divide(lat_cos)

    min_lon = raw_min_lon.divide(tile_deg_lon).floor().multiply(tile_deg_lon)
    min_lat = raw_min_lat.divide(tile_deg_lat).floor().multiply(tile_deg_lat)
    max_lon = raw_max_lon.divide(tile_deg_lon).ceil().multiply(tile_deg_lon)
    max_lat = raw_max_lat.divide(tile_deg_lat).ceil().multiply(tile_deg_lat)

    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    return aoi, min_lon, min_lat, max_lon, max_lat, tile_deg_lon, tile_deg_lat


def build_gain_layer(aoi):
    worldcover = ee.Image("ESA/WorldCover/v100/2020").clip(aoi)
    is_forest = worldcover.eq(10)
    m15 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").clip(aoi)
    m20 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").clip(aoi)
    tree_classes = ee.List.sequence(25, 96).cat(ee.List.sequence(125, 196))
    ones = ee.List.repeat(1, tree_classes.length())
    tree2015 = m15.remap(tree_classes, ones, 0)
    tree2020 = m20.remap(tree_classes, ones, 0)
    forest_gain = tree2020.And(tree2015.Not())
    clean_gain = forest_gain.updateMask(forest_gain).focal_max(1).focal_min(1)
    gain_validated = clean_gain.And(is_forest)
    gain_binary = gain_validated.unmask(0).rename("gain")
    return gain_validated, gain_binary


def s2_availability(aoi, year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(bands)
    )
    return ic.map(lambda img: img.mask().reduce(ee.Reducer.min())).reduce(
        ee.Reducer.max()
    )


def build_full_valid(aoi):
    return (
        s2_availability(aoi, 2016)
        .And(s2_availability(aoi, 2020))
        .And(s2_availability(aoi, 2025))
        .selfMask()
        .rename("valid")
    )


def build_grid(min_lon, min_lat, max_lon, max_lat, tile_deg_lon, tile_deg_lat):
    def make_tiles(lon):
        lon = ee.Number(lon)

        def inner(lat):
            lat = ee.Number(lat)
            tile_id = (
                ee.String("tile_")
                .cat(lon.multiply(1e6).round().format("%d"))
                .cat("_")
                .cat(lat.multiply(1e6).round().format("%d"))
            )
            return ee.Feature(
                ee.Geometry.Rectangle(
                    [lon, lat, lon.add(tile_deg_lon), lat.add(tile_deg_lat)]
                ),
                {"tile_id": tile_id},
            )

        return ee.List.sequence(min_lat, max_lat, tile_deg_lat).map(inner)

    return ee.FeatureCollection(
        ee.List.sequence(min_lon, max_lon, tile_deg_lon).map(make_tiles).flatten()
    )


def build_valid_tiles(gain_binary, full_valid, grid):
    tile_area_pixels = ee.Number(TILE_METERS).divide(SCALE).pow(2)
    gain_count = gain_binary.reduceRegions(
        collection=grid, reducer=ee.Reducer.sum(), scale=SCALE, tileScale=4
    )
    valid_tiles = (
        full_valid.unmask(0)
        .reduceRegions(
            collection=gain_count, reducer=ee.Reducer.min(), scale=SCALE, tileScale=4
        )
        .filter(ee.Filter.eq("min", 1))
    )
    return valid_tiles, tile_area_pixels


def score_tile_viability(tile, gain_validated, gain_height):
    geom = tile.geometry()
    gain_mask = gain_validated.selfMask()

    s2_t0 = s2_peak_composite(geom, 2016)
    s2_t1 = s2_peak_composite(geom, 2020)

    ndvi_diff = (
        s2_t1.select("NDVI")
        .subtract(s2_t0.select("NDVI"))
        .updateMask(gain_mask)
        .rename("NDVI_diff")
    )

    stats = ndvi_diff.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=geom,
        scale=SCALE,
        maxPixels=1e13,
    )

    ndvi_delta = ee.Number(
        ee.Algorithms.If(stats.get("NDVI_diff"), stats.get("NDVI_diff"), 0)
    )

    canopy_stats = gain_height.updateMask(gain_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=SCALE,
        maxPixels=1e13,
    )

    canopy_mean = ee.Number(
        ee.Algorithms.If(
            canopy_stats.get("canopy_gain_height"),
            canopy_stats.get("canopy_gain_height"),
            0,
        )
    )

    return tile.set(
        {
            "ndvi_delta": ndvi_delta,
            "gain_canopy_mean": canopy_mean,
        }
    )


def build_ancillary(aoi, gain_validated):
    fabdem = (
        ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
        .filterBounds(aoi)
        .mosaic()
        .clip(aoi)
    )
    slope = ee.Terrain.slope(fabdem)
    canopy_raw = ee.ImageCollection(
        "projects/sat-io/open-datasets/facebook/meta-canopy-height"
    ).mosaic()

    canopy = canopy_raw.clip(aoi).rename("canopy_height").updateMask(canopy_raw.gte(0))

    gain_height = canopy.updateMask(gain_validated).rename("canopy_gain_height")

    jrc = ee.Image("JRC/GFC2020_subtypes/V1").clip(aoi).rename("jrc_forest_type")
    nat_forest = (
        ee.ImageCollection(
            "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
        )
        .mosaic()
        .select("B0")
        .divide(250)
        .clip(aoi)
        .unmask(0)
        .rename("natural_forest_prob")
    )
    return fabdem, slope, gain_height, jrc, nat_forest


def enrich_tile(tile, tile_area_pixels):
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    biomes = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    centroid = tile.geometry().centroid(maxError=1)
    coords = tile.geometry().bounds().coordinates().get(0)
    gain_pct = ee.Number(tile.get("sum")).divide(tile_area_pixels).multiply(100)
    return tile.set(
        {
            "tile_id": tile.get("tile_id"),
            "country": countries.filterBounds(centroid).first().get("country_na"),
            "biome": biomes.filterBounds(centroid).first().get("BIOME_NAME"),
            "minLon": ee.List(ee.List(coords).get(0)).get(0),
            "minLat": ee.List(ee.List(coords).get(0)).get(1),
            "maxLon": ee.List(ee.List(coords).get(2)).get(0),
            "maxLat": ee.List(ee.List(coords).get(2)).get(1),
            "gainPct": gain_pct,
            "is_selected": gain_pct.gte(1.0),
        }
    )


def add_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    return img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"]).addBands([ndvi, evi])


def s2_composite(geom, year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(add_indices)
    )
    reduced = ic.reduce(ee.Reducer.percentile([25]))
    fallback = (
        ee.Image.constant(0)
        .rename("B2_p25")
        .addBands(ee.Image.constant(0).rename("B3_p25"))
        .addBands(ee.Image.constant(0).rename("B4_p25"))
        .addBands(ee.Image.constant(0).rename("B5_p25"))
        .addBands(ee.Image.constant(0).rename("B6_p25"))
        .addBands(ee.Image.constant(0).rename("B7_p25"))
        .addBands(ee.Image.constant(0).rename("B8_p25"))
        .addBands(ee.Image.constant(0).rename("NDVI_p25"))
        .addBands(ee.Image.constant(0).rename("EVI_p25"))
        .updateMask(ee.Image.constant(0))
    )
    reduced = ee.Image(ee.Algorithms.If(ic.size().eq(0), fallback, reduced))
    return reduced.select(
        [
            "B2_p25",
            "B3_p25",
            "B4_p25",
            "B5_p25",
            "B6_p25",
            "B7_p25",
            "B8_p25",
            "NDVI_p25",
            "EVI_p25",
        ],
        ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "NDVI", "EVI"],
    )


def s2_peak_composite(geom, year):
    centroid = ee.Geometry(geom).centroid(maxError=1)
    lat = ee.Number(centroid.coordinates().get(1))
    north = lat.gt(0)

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
        .map(add_indices)
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
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .select(["trees", "crops", "built"])
        .median()
    )


def build_stack(s2, s1, dw, prefix):
    return (
        s2.addBands(s1)
        .addBands(dw)
        .rename(
            [
                f"{prefix}_B2",
                f"{prefix}_B3",
                f"{prefix}_B4",
                f"{prefix}_B5",
                f"{prefix}_B6",
                f"{prefix}_B7",
                f"{prefix}_B8",
                f"{prefix}_NDVI",
                f"{prefix}_EVI",
                f"{prefix}_VV",
                f"{prefix}_VH",
                f"{prefix}_VVVH",
                f"{prefix}_DW_trees",
                f"{prefix}_DW_crops",
                f"{prefix}_DW_built",
            ]
        )
    )


def build_full_stack(
    geom,
    fabdem,
    slope,
    gain_height,
    jrc,
    nat_forest,
    gain_validated,
    full_valid,
):
    return (
        build_stack(
            s2_composite(geom, 2016),
            s1_composite(geom, 2016),
            dw_composite(geom, 2016),
            "T0",
        )
        .addBands(
            build_stack(
                s2_composite(geom, 2020),
                s1_composite(geom, 2020),
                dw_composite(geom, 2020),
                "T1",
            )
        )
        .addBands(
            build_stack(
                s2_composite(geom, 2025),
                s1_composite(geom, 2025),
                dw_composite(geom, 2025),
                "T2",
            )
        )
        .addBands(fabdem.rename("DEM"))
        .addBands(slope.rename("slope"))
        .addBands(gain_height)
        .addBands(jrc)
        .addBands(nat_forest)
        .addBands(gain_validated.rename("gain_mask"))
        .updateMask(full_valid)
    )


def save_tasks(tasks_file, task_ids):
    with open(tasks_file, "w") as f:
        json.dump(task_ids, f)


def load_tasks(tasks_file):
    if not tasks_file.exists():
        return {}
    with open(tasks_file) as f:
        return json.load(f)


def rclone_to_hpc(tile_id, drive_folder, hpc_remote):
    return subprocess.run(
        [
            "rclone",
            "moveto",
            "--drive-use-trash=false",
            f"{DRIVE_REMOTE}:{drive_folder}/{tile_id}.tif",
            f"{hpc_remote}/{tile_id}.tif",
        ],
        capture_output=True,
        text=True,
    )


def monitor(tasks, n, drive_folder, hpc_remote):
    uploaded = set()
    failed = set()

    while True:
        for task, tile_id in tasks.items():
            if tile_id in uploaded or tile_id in failed:
                continue

            state = task.status()["state"]

            if state == "COMPLETED":
                print(f"{tile_id} complete — moving to HPC...")
                result = rclone_to_hpc(tile_id, drive_folder, hpc_remote)
                if result.returncode == 0:
                    print(f"{tile_id} on HPC")
                    uploaded.add(tile_id)
                else:
                    print(f"rclone error for {tile_id}: {result.stderr}")
                    failed.add(tile_id)

            elif state == "FAILED":
                print(f"{tile_id} FAILED: {task.status().get('error_message')}")
                failed.add(tile_id)

        print(
            f"Uploaded: {len(uploaded)}/{n} | Failed: {len(failed)} | Pending: {n - len(uploaded) - len(failed)}"
        )

        if len(uploaded) + len(failed) == n:
            return uploaded, failed

        time.sleep(POLL_INTERVAL)


def run(aoi_id: str, aoi_bounds: list):
    drive_folder = aoi_to_folder(aoi_id)
    hpc_remote = f"{HPC_BASE}/{drive_folder}"
    tasks_file = OUTPUT_DIR / f"tasks_{drive_folder}.json"

    aoi, min_lon, min_lat, max_lon, max_lat, tile_deg_lon, tile_deg_lat = build_aoi(
        aoi_bounds
    )
    gain_validated, gain_binary = build_gain_layer(aoi)
    full_valid = build_full_valid(aoi)
    grid = build_grid(min_lon, min_lat, max_lon, max_lat, tile_deg_lon, tile_deg_lat)
    valid_tiles, tile_area_pixels = build_valid_tiles(gain_binary, full_valid, grid)
    fabdem, slope, gain_height, jrc, nat_forest = build_ancillary(aoi, gain_validated)

    GAIN_PCT_MIN = 0.01
    NDVI_DELTA_MIN = 0.0
    GAIN_CANOPY_MIN = 3.0

    full_grid_index = valid_tiles.map(lambda t: enrich_tile(t, tile_area_pixels))

    gain_tiles = full_grid_index.filter(ee.Filter.gte("gainPct", GAIN_PCT_MIN))

    scored_tiles = gain_tiles.map(
        lambda t: score_tile_viability(t, gain_validated, gain_height)
    )

    filtered_tiles = scored_tiles.filter(
        ee.Filter.And(
            ee.Filter.gt("ndvi_delta", NDVI_DELTA_MIN),
            ee.Filter.gte("gain_canopy_mean", GAIN_CANOPY_MIN),
        )
    )

    saved = load_tasks(tasks_file)

    if saved:
        print(f"Resuming {len(saved)} tasks for {aoi_id}...")
        all_ee_tasks = {t.id: t for t in ee.batch.Task.list()}
        tasks = {}
        for tile_id, task_id in saved.items():
            if task_id in all_ee_tasks:
                tasks[all_ee_tasks[task_id]] = tile_id
            else:
                print(
                    f"  Warning: task {task_id} for {tile_id} not found in EE task list"
                )
        n = len(tasks)
    else:
        ee.batch.Export.table.toDrive(
            collection=full_grid_index,
            description=f"{drive_folder}_index",
            folder=drive_folder,
            fileNamePrefix=f"{drive_folder}_index",
            fileFormat="CSV",
        ).start()

        tile_list = filtered_tiles.toList(filtered_tiles.size())
        n = filtered_tiles.size().getInfo()
        print(f"Submitting {n} export tasks for {aoi_id}...")

        tasks = {}
        task_ids = {}
        for i in tqdm(range(n), desc=f"Submitting {aoi_id}", smoothing=0):
            tile = ee.Feature(tile_list.get(i))
            geom = tile.geometry()
            tile_id = tile.get("tile_id").getInfo()

            stack = build_full_stack(
                geom,
                fabdem,
                slope,
                gain_height,
                jrc,
                nat_forest,
                gain_validated,
                full_valid,
            )

            task = ee.batch.Export.image.toDrive(
                image=stack.clip(geom).toFloat(),
                description=tile_id,
                folder=drive_folder,
                fileNamePrefix=tile_id,
                region=geom,
                scale=SCALE,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            task.start()
            tasks[task] = tile_id
            task_ids[tile_id] = task.id

        save_tasks(tasks_file, task_ids)

    uploaded, failed = monitor(tasks, n, drive_folder, hpc_remote)

    if not failed:
        tasks_file.unlink(missing_ok=True)

    print(f"Done. {len(uploaded)} tiles on HPC, {len(failed)} failed.")
    if failed:
        print(f"Failed tiles: {failed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export forest gain tiles for an AOI.")
    parser.add_argument("--aoi-id", required=True, help="AOI identifier, e.g. 'wales'")
    parser.add_argument(
        "--bounds",
        required=True,
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    args = parser.parse_args()

    run(aoi_id=args.aoi_id, aoi_bounds=args.bounds)

# example usage: python gee.py --aoi-id wales --bounds -4.04 51.54 -3.32 51.75

# p1: Mato Grosso do Sul eucalyptus plantation python gee.py --aoi-id mgs_plantation --bounds -52.05 -20.90 -51.96 -20.81

# p2: Araucania Region pine plantation python gee.py --aoi-id chile_pine --bounds -72.85 -37.85 -72.76 -37.76

# a1 South-west Cote d'Ivoire cocoa agroforestry near Soubre python gee.py --aoi-id cdi_cocoa --bounds -6.65 5.72 -6.56 5.81

# a1 Xishuangbanna (西双版纳) rubber expansion python gee.py --aoi-id yunnan_rubber --bounds 100.85 21.85 100.94 21.94

# n1 Pontal do Paranapanema natural regen zone sao Paulo python gee.py --aoi-id brazil_nat_regen --bounds -52.45 -22.35 -52.36 -22.26

# n2 Eastern Usambara Mountains natural regen zone Tanzania python gee.py --aoi-id tanzania_nat_regen --bounds 38.60 4.90 38.69 4.99

# r1 Glen Affric native woodland restoration Scotland python gee.py --aoi-id glen_affric --bounds -4.95 57.18 -4.86 57.27

# r1 Loess plateua restoration zone near Yan'an (延安) python gee.py --aoi-id loess_restoration --bounds 109.35 36.45 109.44 36.54
