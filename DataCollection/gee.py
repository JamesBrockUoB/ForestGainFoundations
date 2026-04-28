import csv
import subprocess
import time
from pathlib import Path

import ee

ee.Authenticate()
ee.Initialize(project="symbolic-base-346316")

AOI = ee.Geometry.Rectangle([-4.04, 51.54, -3.32, 51.75])

tilePixels = 128
scale = 10
tileMeters = tilePixels * scale

DRIVE_FOLDER = "forest_gain_wales"
DRIVE_REMOTE = "gdrive"
HPC_REMOTE = (
    "isambard:/projects/b6be/jamesbrock.b6be/ForestGrowthFoundations/forest_gain_wales"
)
LOCAL_STAGING = Path("./staging")
INDEX_FILE = Path("tile_index.csv")
POLL_INTERVAL = 30

LOCAL_STAGING.mkdir(exist_ok=True)

rawCoords = ee.List(AOI.coordinates().get(0))
rawMinLon = ee.Number(ee.List(rawCoords.get(0)).get(0))
rawMinLat = ee.Number(ee.List(rawCoords.get(0)).get(1))
rawMaxLon = ee.Number(ee.List(rawCoords.get(2)).get(0))
rawMaxLat = ee.Number(ee.List(rawCoords.get(2)).get(1))

tileDegLat = ee.Number(tileMeters).divide(111320)
centerLat = rawMinLat.add(rawMaxLat).divide(2)
latCos = centerLat.multiply(3.141592653589793 / 180).cos()
tileDegLon = ee.Number(tileMeters).divide(111320).divide(latCos)

minLon = rawMinLon.divide(tileDegLon).floor().multiply(tileDegLon)
minLat = rawMinLat.divide(tileDegLat).floor().multiply(tileDegLat)
maxLon = rawMaxLon.divide(tileDegLon).ceil().multiply(tileDegLon)
maxLat = rawMaxLat.divide(tileDegLat).ceil().multiply(tileDegLat)

AOI = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat])

worldcover = ee.Image("ESA/WorldCover/v100/2020").clip(AOI)
isForest2020 = worldcover.eq(10)

m15 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").clip(AOI)
m20 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").clip(AOI)

treeClasses = ee.List.sequence(25, 96).cat(ee.List.sequence(125, 196))
ones = ee.List.repeat(1, treeClasses.length())

tree2015 = m15.remap(treeClasses, ones, 0)
tree2020 = m20.remap(treeClasses, ones, 0)

forestGain = tree2020.And(tree2015.Not())
cleanGain = forestGain.updateMask(forestGain).focal_max(1).focal_min(1)
gainValidated = cleanGain.And(isForest2020)
gainBinary = gainValidated.unmask(0).rename("gain")


def s2AvailabilityAllBands(year):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(AOI)
        .select(bands)
    )
    per_image_valid = ic.map(lambda img: img.mask().reduce(ee.Reducer.min()))
    return per_image_valid.reduce(ee.Reducer.max())


fullValid = (
    s2AvailabilityAllBands(2016)
    .And(s2AvailabilityAllBands(2020))
    .And(s2AvailabilityAllBands(2025))
    .selfMask()
    .rename("valid")
)


def make_tiles(lon):
    lon = ee.Number(lon)

    def inner(lat):
        lat = ee.Number(lat)
        t_tile_id = (
            ee.String("tile_")
            .cat(lon.multiply(1e6).round().format("%d"))
            .cat("_")
            .cat(lat.multiply(1e6).round().format("%d"))
        )
        return ee.Feature(
            ee.Geometry.Rectangle([lon, lat, lon.add(tileDegLon), lat.add(tileDegLat)]),
            {"tile_id": t_tile_id},
        )

    return ee.List.sequence(minLat, maxLat, tileDegLat).map(inner)


grid = ee.FeatureCollection(
    ee.List.sequence(minLon, maxLon, tileDegLon).map(make_tiles).flatten()
)

tileAreaPixels = ee.Number(tileMeters).divide(scale).pow(2)

gainCount = gainBinary.reduceRegions(
    collection=grid, reducer=ee.Reducer.sum(), scale=scale, tileScale=4
)

validTiles = (
    fullValid.unmask(0)
    .reduceRegions(
        collection=gainCount, reducer=ee.Reducer.min(), scale=scale, tileScale=4
    )
    .filter(ee.Filter.eq("min", 1))
)

countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
biomes = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

fabdem = (
    ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    .filterBounds(AOI)
    .mosaic()
    .clip(AOI)
)
slope = ee.Terrain.slope(fabdem)
canopyHeight = (
    ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
    .clip(AOI)
    .rename("canopy_height")
    .updateMask(ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").gte(0))
)
gainHeight = canopyHeight.updateMask(gainValidated).rename("canopy_gain_height")

jrcForestTypes = ee.Image("JRC/GFC2020_subtypes/V1").clip(AOI).rename("jrc_forest_type")
naturalForestProb = (
    ee.ImageCollection(
        "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
    )
    .mosaic()
    .select("B0")
    .divide(250)
    .clip(AOI)
    .unmask(0)
    .rename("natural_forest_prob")
)


def enrich_tile(tile):
    centroid = tile.geometry().centroid()
    coords = tile.geometry().bounds().coordinates().get(0)
    gain_pct = ee.Number(tile.get("sum")).divide(tileAreaPixels).multiply(100)
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


fullGridIndex = validTiles.map(enrich_tile)
filteredTiles = validTiles.filter(ee.Filter.gte("sum", tileAreaPixels.multiply(0.01)))


def addIndices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    return img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8"]).addBands([ndvi, evi])


def s2Composite(year, geom):
    start = "2015-01-01" if year == 2016 else f"{year}-01-01"
    end = "2016-12-31" if year == 2016 else f"{year}-12-31"
    reduced = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(addIndices)
        .reduce(ee.Reducer.percentile([25]))
    )
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


def s1Composite(year, geom):
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


def dwComposite(year, geom):
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .select(["trees", "crops", "built"])
        .median()
    )


def buildStack(s2, s1, dw, prefix):
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


def append_to_index(props):
    write_header = not INDEX_FILE.exists()
    with open(INDEX_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "country",
                "biome",
                "gain_pct",
                "minLon",
                "minLat",
                "maxLon",
                "maxLat",
                "is_selected",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "tile_id": props.get("tile_id"),
                "country": props.get("country", "unknown"),
                "biome": props.get("biome", "unknown"),
                "gain_pct": round(props.get("gainPct", 0), 4),
                "minLon": props.get("minLon"),
                "minLat": props.get("minLat"),
                "maxLon": props.get("maxLon"),
                "maxLat": props.get("maxLat"),
                "is_selected": props.get("is_selected"),
            }
        )


ee.batch.Export.table.toDrive(
    collection=fullGridIndex,
    description="full_grid_index_all_tiles",
    folder=DRIVE_FOLDER,
    fileNamePrefix="full_grid_index_all_tiles",
    fileFormat="CSV",
).start()

tile_list = filteredTiles.toList(filteredTiles.size())
n = filteredTiles.size().getInfo()
print(f"Submitting {n} export tasks to Drive...")

tasks = {}
for i in range(n):
    tile = ee.Feature(tile_list.get(i))
    geom = tile.geometry()
    tile_id = tile.get("tile_id").getInfo()

    fullStack = (
        buildStack(
            s2Composite(2016, geom),
            s1Composite(2016, geom),
            dwComposite(2016, geom),
            "T0",
        )
        .addBands(
            buildStack(
                s2Composite(2020, geom),
                s1Composite(2020, geom),
                dwComposite(2020, geom),
                "T1",
            )
        )
        .addBands(
            buildStack(
                s2Composite(2025, geom),
                s1Composite(2025, geom),
                dwComposite(2025, geom),
                "T2",
            )
        )
        .addBands(fabdem.rename("DEM"))
        .addBands(slope.rename("slope"))
        .addBands(gainHeight)
        .addBands(jrcForestTypes)
        .addBands(naturalForestProb)
        .addBands(gainValidated.rename("gain_mask"))
        .updateMask(fullValid)
    )

    task = ee.batch.Export.image.toDrive(
        image=fullStack.clip(geom).toFloat(),
        description=tile_id,
        folder=DRIVE_FOLDER,
        fileNamePrefix=tile_id,
        region=geom,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    tasks[task] = tile_id
    print(f"  Submitted {tile_id}")

uploaded = set()
failed = set()

print("Monitoring tasks...")
while True:
    for task, tile_id in tasks.items():
        if tile_id in uploaded or tile_id in failed:
            continue

        state = task.status()["state"]

        if state == "COMPLETED":
            print(f"{tile_id} complete — moving Drive -> HPC...")
            result = subprocess.run(
                [
                    "rclone",
                    "moveto",
                    f"{DRIVE_REMOTE}:{DRIVE_FOLDER}/{tile_id}.tif",
                    f"{HPC_REMOTE}/{tile_id}.tif",
                ],
                capture_output=True,
                text=True,
            )
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
        break

    time.sleep(POLL_INTERVAL)

print(f"Done. {len(uploaded)} tiles on HPC, {len(failed)} failed.")
if failed:
    print(f"Failed tiles: {failed}")
