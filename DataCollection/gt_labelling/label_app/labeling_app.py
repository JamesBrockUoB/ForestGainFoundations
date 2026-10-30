import base64
import copy
import hashlib
import io
import json
import os
import queue
import threading
from pathlib import Path

import ee
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
import streamlit.elements.image as _st_image
from folium.plugins import SideBySideLayers
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage
from streamlit_folium import st_folium


# --- streamlit-drawable-canvas fix --------------------------------------------------------------
# The canvas calls streamlit.elements.image.image_to_url(...) to show its background image. Recent Streamlit
# versions moved/changed that function, so the background (all your layers) silently never appears.
# Replace it with a data-URL version BEFORE the canvas is imported/used; this works on every version.
def _image_to_url(image, *args, **kwargs):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_st_image.image_to_url = _image_to_url
from streamlit_drawable_canvas import (  # noqa: E402  (must come after the patch)
    st_canvas,
)

# Page Setup
st.set_page_config(
    page_title="Woodland Ground Truth Segmentation Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# No vertical gap between widgets (Streamlit's default is 1rem). Raise the 0rem to add some back.
st.markdown(
    """<style>
    [data-testid="stColumn"] [data-testid="stVerticalBlock"] { gap: 0rem; }
    [data-testid="stColumn"] [data-testid="stSlider"] { padding-bottom: 0; }

    /* Keep the fixed-size drawing canvas centred in the middle column. */
    div[data-testid="stColumn"]:nth-child(2) [data-testid="stCustomComponentV1"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    div[data-testid="stColumn"]:nth-child(2) [data-testid="stCustomComponentV1"] iframe {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ==========================================
# CONFIG
# ==========================================
GEOJSON_PATH = "gain_tiles_10m_combined.geojson"
CACHE_DIR = Path(
    "./tile_cache"
)  # layers are generated from Earth Engine on demand, then cached here
LABELS_DIR = Path("./ground_truth_masks")
QUEUE_PATH = Path(
    "./annotation_queue.csv"
)  # the work queue (a stratified sample); survives restarts
PREFETCH_AHEAD = 2  # how many upcoming queue tiles to fetch in the background
LABELS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

EE_PROJECT = "symbolic-base-346316"
GAIN_ASSETS = [
    "projects/symbolic-base-346316/assets/hrl_gain_20m_eu",
    "projects/symbolic-base-346316/assets/hrl_gain_20m",
]
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_PLUS_BAND = "cs_cdf"
CLOUD_SCORE_THRESH = 0.6
SCENE_CLOUDY_PCT_MAX = 60
SNOW_NDSI_THRESH, SNOW_NIR_MIN, SNOW_GREEN_MIN = 0.4, 1100, 1000
LEAF_ON = ("05-01", "09-30")  # northern hemisphere only
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
YEAR_A, YEAR_B = 2017, 2024

# Big map comparison uses Esri World Imagery Wayback releases (not Sentinel-2).
# Release years are publication vintages; local capture dates can differ by location.
WAYBACK_CONFIG_URL = (
    "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
)

PX = 10.0  # metres per pixel
SIZE = 256  # tile raster size in pixels (2560 m at 10 m)
CANVAS = 512  # canvas size on screen = exactly 2 screen pixels per 10 m pixel
K = CANVAS // SIZE
NODATA = -9999.0
# "medoid": each pixel takes ALL its bands from the single clear image closest to the per-band median, so colours
# stay spectrally consistent (per-band medians of a few images mix dates and give coloured speckle).
# "median": plain per-band median. Changing this uses a new cache file, so tiles are re-fetched once.
S2_COMPOSITE = "medoid"
S2_VMIN, S2_VMAX = (
    0.0,
    3000.0,
)  # fixed display range for S2 reflectance; raise S2_VMAX for a flatter look

# Painting classes on the canvas: the colour of a stroke is its class
CLASSES = {
    "gain (add)": ((0, 200, 0), "#00C800"),
    "not gain (erase)": ((255, 0, 0), "#FF0000"),
    "ignore": ((0, 0, 255), "#0000FF"),
}

BASEMAPS = {
    "Satellite (Esri)": dict(
        style="white-bg",
        layers=[
            dict(
                below="traces",
                sourcetype="raster",
                sourceattribution="Esri, Maxar, Earthstar Geographics",
                source=[
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            )
        ],
    ),
    "OpenStreetMap": dict(style="open-street-map", layers=[]),
    "Light": dict(style="carto-positron", layers=[]),
}

S2_A, S2_B = f"S2 {YEAR_A} RGB", f"S2 {YEAR_B} RGB"
DW_A, DW_B = f"DW trees {YEAR_A} (prob)", f"DW trees {YEAR_B} (prob)"
HRL = "HRL gain (helper)"
NDVI = f"NDVI Delta ({YEAR_B} - {YEAR_A})"
SLOPE = "Slope"
LAYERS = {  # Earth Engine layers: name -> cache file stem
    S2_B: f"s2_{S2_COMPOSITE}_2024",
    S2_A: f"s2_{S2_COMPOSITE}_2017",
    DW_B: "dw_2024",
    DW_A: "dw_2017",
    NDVI: "ndvi_delta",
    HRL: "gain_conf",
    SLOPE: "slope",
}
START = "Starting mask (what you edit)"
DW_NT_A = (
    f"DW {YEAR_A} non-tree mask"  # derived from the DW layer with the threshold slider
)
DW_T_B = f"DW {YEAR_B} tree mask"
DERIVED = [
    START,
    DW_NT_A,
    DW_T_B,
]  # overlays computed from layers (no Dynamic World gain/loss calculation)


def dw_name(year):
    return DW_A if year == YEAR_A else DW_B


# ==========================================
# EARTH ENGINE: layer generation (same masking as your composite code, defined inline)
# ==========================================
@st.cache_resource
def init_ee():
    ee.Initialize(project=EE_PROJECT)
    return True


def _snow_mask(img):
    green, nir, swir = img.select("B3"), img.select("B8"), img.select("B11")
    ndsi = green.subtract(swir).divide(green.add(swir))
    return (
        ndsi.gt(SNOW_NDSI_THRESH)
        .And(nir.gt(SNOW_NIR_MIN))
        .And(green.gt(SNOW_GREEN_MIN))
    )


def _mask_clouds(img):
    return img.updateMask(img.select(CLOUD_SCORE_PLUS_BAND).gte(CLOUD_SCORE_THRESH))


def _mask_clouds_and_snow(img):
    clear = (
        img.select(CLOUD_SCORE_PLUS_BAND)
        .gte(CLOUD_SCORE_THRESH)
        .And(_snow_mask(img).Not())
    )
    return img.updateMask(clear)


def _masked_s2(geom, start, end, snow=True):
    ic = (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUDY_PCT_MAX))
    )
    cs = ee.ImageCollection(CLOUD_SCORE_PLUS).filterDate(start, end).filterBounds(geom)
    return ic.linkCollection(cs, [CLOUD_SCORE_PLUS_BAND]).map(
        _mask_clouds_and_snow if snow else _mask_clouds
    )


def _s2_rgb(geom, year):
    """Full-year composite (medoid or median, see S2_COMPOSITE), cloud- and snow-masked. Bands B4, B3, B2."""
    bands = ["B4", "B3", "B2"]
    ic = _masked_s2(geom, f"{year}-01-01", f"{year + 1}-01-01", snow=True).select(bands)
    med = ic.median()
    if S2_COMPOSITE == "median":
        return med

    def _with_score(
        img,
    ):  # negative squared distance to the median over the three bands
        dist = img.subtract(med).pow(2).reduce(ee.Reducer.sum()).multiply(-1)
        return img.addBands(dist.rename("score"))

    return ic.map(_with_score).qualityMosaic("score").select(bands)


@st.cache_resource(
    ttl=24 * 3600, show_spinner="Loading Esri Wayback imagery releases..."
)
def _wayback_releases():
    """Fetch Esri's Wayback config and tolerate its real nested config schemas.

    Esri's public config is not guaranteed to expose releaseDateLabel/date fields:
    many entries identify the release via itemTitle (e.g. "World Imagery (Wayback
    2024-01-25)") and use the numeric release ID as the dictionary key. Parse those
    documented fields rather than treating a schema variation as no imagery.
    """
    response = requests.get(WAYBACK_CONFIG_URL, timeout=25)
    response.raise_for_status()
    payload = response.json()
    import re
    from datetime import date as _date

    found = {}
    date_re = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")
    layer_re = re.compile(r"WB_(20\d{2})_R(\d+)", re.IGNORECASE)
    tile_re = re.compile(r"/tile/(\d+)/(?:\{level\}|\{z\})", re.IGNORECASE)

    def walk(obj):
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    yield from walk(value)
        elif isinstance(obj, list):
            for value in obj:
                if isinstance(value, (dict, list)):
                    yield from walk(value)

    for item in walk(payload):
        title = str(
            item.get("itemTitle") or item.get("title") or item.get("name") or ""
        )
        layer_id = str(item.get("layerIdentifier") or item.get("layerId") or "")
        date_value = (
            item.get("releaseDateLabel")
            or item.get("releaseDate")
            or item.get("date")
            or item.get("releaseDateString")
        )
        match = date_re.search(str(date_value or "")) or date_re.search(title)
        layer_match = layer_re.search(layer_id)
        if match:
            release_date = match.group(1)
        elif layer_match:
            # Some older config entries only expose WB_YYYY_Rnn. The release date
            # is unknown in that schema; do not fabricate a specific publication date.
            release_date = f"{layer_match.group(1)}-07-01"
        else:
            continue

        release_num = (
            item.get("releaseNum") or item.get("releaseNumber") or item.get("release")
        )
        if release_num is None:
            item_url = str(item.get("itemURL") or item.get("url") or "")
            url_match = tile_re.search(item_url)
            if url_match:
                release_num = url_match.group(1)
        if release_num is None:
            # The config commonly uses the numeric release number as the object key.
            for key, value in item.items():
                if (
                    isinstance(value, (str, int))
                    and str(key).isdigit()
                    and int(key) > 0
                ):
                    release_num = key
                    break
        if release_num is None:
            continue
        try:
            release_num = int(release_num)
            # Validate actual dates; skip placeholder values that aren't valid.
            _date.fromisoformat(release_date)
        except (TypeError, ValueError):
            continue
        found[(release_date, release_num)] = (release_date, release_num)

    releases = sorted(found.values())
    if not releases:
        # Keep the real response diagnostic so a network/config failure is not
        # misreported as proof that the Wayback archive does not exist.
        content_type = response.headers.get("content-type", "unknown")
        preview = response.text[:180].replace("\n", " ")
        raise RuntimeError(
            f"Esri Wayback config fetched (HTTP {response.status_code}, "
            f"content-type {content_type}) but its schema could not be parsed. "
            f"Response begins: {preview}"
        )
    return releases


@st.cache_resource(ttl=24 * 3600, show_spinner="Preparing Esri historical imagery...")
def map_imagery_url(year):
    """XYZ URL for an Esri World Imagery Wayback publication vintage."""
    releases = _wayback_releases()
    year_releases = [r for r in releases if r[0].startswith(f"{int(year)}-")]
    if not year_releases:
        raise RuntimeError(f"No Esri World Imagery Wayback release found for {year}")
    # Pick the release nearest mid-year, giving a stable representative vintage.
    chosen_date, release_num = min(
        year_releases,
        key=lambda r: abs(int(r[0][5:7]) * 31 + int(r[0][8:10]) - (6 * 31 + 15)),
    )
    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{release_num}/{{z}}/{{y}}/{{x}}"
    return url, chosen_date


def _peak_ndvi(geom, year):
    """Leaf-on NDVI (May-Sep), cloud-masked only - same as your s2_peak_ndvi."""
    return (
        _masked_s2(geom, f"{year}-{LEAF_ON[0]}", f"{year}-{LEAF_ON[1]}", snow=False)
        .map(
            lambda im: im.addBands(im.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        )
        .select(["NDVI"])
        .median()
    )


def _dw_trees(geom, year):
    """Dynamic World 'trees' probability, mean over the leaf-on months (clear observations only)."""
    return (
        ee.ImageCollection(DW_COLLECTION)
        .filterDate(f"{year}-{LEAF_ON[0]}", f"{year}-{LEAF_ON[1]}")
        .filterBounds(geom)
        .select("trees")
        .mean()
        .rename("p")
    )


def _gain_asset():
    for a in GAIN_ASSETS:
        try:
            ee.data.getAsset(a)
            return ee.Image(a)
        except Exception:
            continue
    raise RuntimeError("gain asset not found: " + ", ".join(GAIN_ASSETS))


def _layer_image(layer, geom):
    """(ee.Image, band names) for one layer."""
    if layer == S2_B:
        return _s2_rgb(geom, YEAR_B), ["B4", "B3", "B2"]
    if layer == S2_A:
        return _s2_rgb(geom, YEAR_A), ["B4", "B3", "B2"]
    if layer == DW_B:
        return _dw_trees(geom, YEAR_B), ["p"]
    if layer == DW_A:
        return _dw_trees(geom, YEAR_A), ["p"]
    if layer == NDVI:
        return _peak_ndvi(geom, YEAR_B).subtract(_peak_ndvi(geom, YEAR_A)).rename(
            "dndvi"
        ), ["dndvi"]
    if layer == HRL:
        a = _gain_asset()
        gain = (
            a.select(0).gt(0).unmask(0).toFloat().resample("bilinear").gt(0.5)
        )  # bilinear 10 m mask
        conf = a.select(1).unmask(0).toFloat().max(1)  # >= 1 so every gain px shows
        return gain.multiply(conf).rename("gain_conf"), ["gain_conf"]
    if layer == SLOPE:
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
        mosaic = dem.mosaic().setDefaultProjection(dem.first().select(0).projection())
        return ee.Terrain.slope(mosaic).resample("bilinear").rename("slope"), ["slope"]
    raise KeyError(layer)


def generate_layer(layer, x_min, y_min, x_max, y_max):
    """Pull one layer for the tile from Earth Engine as a (bands, 256, 256) float32 array, masked pixels = NaN.
    Needs init_ee() to have run already (it is also called from the prefetch thread, which must not touch st.*).
    """
    geom = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], "EPSG:3035", False)
    img, bands = _layer_image(layer, geom)
    req = dict(
        expression=img.toFloat().unmask(NODATA),
        fileFormat="NUMPY_NDARRAY",
        grid=dict(
            dimensions=dict(width=SIZE, height=SIZE),
            affineTransform=dict(
                scaleX=PX,
                shearX=0,
                translateX=x_min,
                shearY=0,
                scaleY=-PX,
                translateY=y_max,
            ),
            crsCode="EPSG:3035",
        ),
    )
    arr = ee.data.computePixels(req)
    out = np.stack([np.asarray(arr[b], dtype="float32") for b in bands])
    out[out <= NODATA + 1] = np.nan
    return out


def layer_path(tile_id, layer):
    return CACHE_DIR / tile_id / f"{LAYERS[layer]}.npy"


def fetch_layer(tile_id, layer, x_min, y_min, x_max, y_max):
    """Generate one layer and cache it atomically (safe to call from a thread; no st.* calls)."""
    p = layer_path(tile_id, layer)
    arr = generate_layer(layer, x_min, y_min, x_max, y_max)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, p)  # readers never see a half-written file
    return arr


def get_layer(tile_id, layer, x_min, y_min, x_max, y_max):
    """Cached layer array, generated on first use. Returns (array or None, error text or None)."""
    p = layer_path(tile_id, layer)
    if p.exists():
        return np.load(p), None
    try:
        with st.spinner(f"Generating {layer} for {tile_id} from Earth Engine..."):
            arr = fetch_layer(tile_id, layer, x_min, y_min, x_max, y_max)
        return arr, None
    except Exception as e:
        return None, f"{layer}: {str(e)[:300]}"


class _Prefetcher:
    """Background worker that fetches the layers needed for labelling the next queue tiles."""

    LAYERS = (S2_B, S2_A, DW_B, DW_A)

    def __init__(self):
        self.jobs, self.pending, self.lock = queue.Queue(), set(), threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, tile_id, bounds):
        for layer in self.LAYERS:
            key = (tile_id, layer)
            if layer_path(tile_id, layer).exists():
                continue
            with self.lock:
                if key in self.pending:
                    continue
                self.pending.add(key)
            self.jobs.put((key, bounds))

    def _run(self):
        while True:
            (tile_id, layer), bounds = self.jobs.get()
            try:
                if not layer_path(tile_id, layer).exists():
                    fetch_layer(tile_id, layer, *bounds)
            except Exception:
                pass  # the foreground get_layer retries and reports errors when the tile is opened
            finally:
                with self.lock:
                    self.pending.discard((tile_id, layer))


@st.cache_resource
def get_prefetcher():
    return _Prefetcher()


# ==========================================
# CHANGE LOGIC (same as dw_tree_change.py / ls_export_masks.py)
# ==========================================
def drop_small(mask, min_px):
    if min_px <= 1 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return np.isin(lab, np.flatnonzero(sizes >= min_px) + 1)


def saved_path(tile_id):
    return LABELS_DIR / f"{tile_id}_gain.npz"


def write_tile_tif(tile_id, bounds):
    """ground_truth_masks/<tile>.tif: 2 bands, forest gain 2017->2024 and ignore."""
    z = np.load(saved_path(tile_id))
    bands = [("gain", z["gain"]), ("ignore", z["ignore"])]
    with rasterio.open(
        LABELS_DIR / f"{tile_id}.tif",
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=len(bands),
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(bounds[0], bounds[3], PX, PX),
        compress="deflate",
    ) as dst:
        for i, (name, arr) in enumerate(bands, 1):
            dst.write(arr.astype("uint8"), i)
            dst.set_band_description(i, name)
    return int(z["gain"].sum())


# ==========================================
# TILE INDEX
# ==========================================
@st.cache_data
def load_tile_index(path):
    if not Path(path).exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3035")
    # Woodland-creation polygons count as gain just like HRL gain: the working gain_pct is the larger of the two.
    gdf["hrl_gain_pct"] = gdf["gain_pct"]
    if "woodland_pct" in gdf.columns:
        gdf["gain_pct"] = np.fmax(
            gdf["gain_pct"].fillna(0).to_numpy(float),
            gdf["woodland_pct"].fillna(0).to_numpy(float),
        )
    gdf["x_center"] = (gdf["x_min_m"] + gdf["x_max_m"]) / 2
    gdf["y_center"] = (gdf["y_min_m"] + gdf["y_max_m"]) / 2
    to_ll = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    gdf["lon"], gdf["lat"] = to_ll.transform(
        gdf["x_center"].to_numpy(), gdf["y_center"].to_numpy()
    )
    return gdf


@st.cache_data(show_spinner=False)
def marker_geojson(path):
    """Point GeoJSON for the map: one feature per tile with a precomputed colour and tooltip."""
    gdf = load_tile_index(path)
    gain = pd.to_numeric(gdf["gain_pct"], errors="coerce").fillna(0).to_numpy(float)
    t = np.clip(gain / 50.0, 0.0, 1.0)
    g, b = (220 * (1 - t)).astype(int), (40 * (1 - t)).astype(int)
    hrl = gdf["hrl_gain_pct"].fillna(0).to_numpy(float)
    wood = (
        gdf["woodland_pct"].to_numpy(float)
        if "woodland_pct" in gdf.columns
        else np.full(len(gdf), np.nan)
    )
    feats = []
    for tid, lon, lat, gv, hv, wv, gi, bi in zip(
        gdf["tile_id"].astype(str), gdf["lon"], gdf["lat"], gain, hrl, wood, g, b
    ):
        tip = f"{tid} · Gain {gv:.1f}%"
        if not np.isnan(wv):
            tip += f" (HRL {hv:.1f}% · woodland {wv:.1f}%)"
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "tile_id": tid,
                    "tip": tip,
                    "col": f"#ff{gi:02x}{bi:02x}",
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


COUNTRY_CACHE = CACHE_DIR / "tile_countries.json"


@st.cache_data(show_spinner=False)
def lookup_tile_countries(tile_ids, lon_values, lat_values):
    """Assign countries in small server-side batches to stay below GEE query limits."""
    init_ee()
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    result = {}
    batch_size = 300

    for offset in range(0, len(tile_ids), batch_size):
        ids = tile_ids[offset : offset + batch_size]
        lons = lon_values[offset : offset + batch_size]
        lats = lat_values[offset : offset + batch_size]
        points = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point([float(lon), float(lat)]), {"tile_id": str(tid)}
                )
                for tid, lon, lat in zip(ids, lons, lats)
            ]
        )
        joined = ee.FeatureCollection(
            ee.Join.saveFirst("country_feature").apply(
                primary=points,
                secondary=countries,
                condition=ee.Filter.intersects(leftField=".geo", rightField=".geo"),
            )
        )

        def add_country(feature):
            country_feature = feature.get("country_feature")
            country_name = ee.Algorithms.If(
                ee.Algorithms.IsEqual(country_feature, None),
                "Unknown",
                ee.Feature(country_feature).get("country_na"),
            )
            return feature.set("country", country_name)

        rows = (
            joined.map(add_country)
            .select(["tile_id", "country"])
            .getInfo()
            .get("features", [])
        )
        for row in rows:
            props = row.get("properties", {})
            result[str(props.get("tile_id", ""))] = props.get("country") or "Unknown"

        # If a join omits a point with no matching country, keep it in the output.
        for tid in ids:
            result.setdefault(str(tid), "Unknown")

    return result


def ensure_country_column(gdf):
    """Use a country attribute or locally cached LSIB lookup; defer new lookups until sampling."""
    for col in ("country", "COUNTRY_NA", "country_na", "Country", "COUNTRY"):
        if col in gdf.columns:
            gdf["country"] = gdf[col].fillna("Unknown").astype(str)
            return gdf
    if COUNTRY_CACHE.exists():
        try:
            country_map = json.loads(COUNTRY_CACHE.read_text())
        except Exception:
            country_map = {}
    else:
        country_map = {}
    gdf["country"] = gdf["tile_id"].astype(str).map(country_map).fillna("Unknown")
    return gdf


def lookup_missing_countries(gdf):
    """Fill uncached tile countries in one GEE LSIB spatial join, then persist the result."""
    if any(
        col in gdf.columns for col in ("COUNTRY_NA", "country_na", "Country", "COUNTRY")
    ):
        return gdf
    try:
        country_map = (
            json.loads(COUNTRY_CACHE.read_text()) if COUNTRY_CACHE.exists() else {}
        )
    except Exception:
        country_map = {}
    missing = ~gdf["tile_id"].astype(str).isin(country_map)
    if missing.any():
        with st.spinner(
            f"Looking up countries for {int(missing.sum()):,} tile centroids in GEE LSIB…"
        ):
            looked_up = lookup_tile_countries(
                gdf.loc[missing, "tile_id"].astype(str).tolist(),
                gdf.loc[missing, "lon"].astype(float).tolist(),
                gdf.loc[missing, "lat"].astype(float).tolist(),
            )
            country_map.update(looked_up)
            COUNTRY_CACHE.write_text(json.dumps(country_map, indent=2))
    gdf["country"] = gdf["tile_id"].astype(str).map(country_map).fillna("Unknown")
    return gdf


gdf_projected = load_tile_index(GEOJSON_PATH)
if gdf_projected is None:
    st.error(f"GeoJSON file `{GEOJSON_PATH}` not found.")
    st.stop()

gdf_projected = ensure_country_column(gdf_projected)

if "selected_tile_id" not in st.session_state:
    st.session_state.selected_tile_id = gdf_projected["tile_id"].iloc[0]

# A save (or map click) that moves to another tile is applied here, before any widget exists.
for _k, _v in st.session_state.pop("_pending", {}).items():
    st.session_state[_k] = _v

# ==========================================
# WORK QUEUE (the stratified sample; a tile is "done" when its gain mask is saved)
# ==========================================
TILE_SET = set(gdf_projected["tile_id"])


def set_queue(tile_ids):
    ids = [t for t in dict.fromkeys(tile_ids) if t in TILE_SET]
    st.session_state["queue_ids"] = ids
    pd.DataFrame({"tile_id": ids}).to_csv(QUEUE_PATH, index=False)


if "queue_ids" not in st.session_state:
    st.session_state["queue_ids"] = []
    if QUEUE_PATH.exists():
        st.session_state["queue_ids"] = [
            t for t in pd.read_csv(QUEUE_PATH)["tile_id"].astype(str) if t in TILE_SET
        ]


SAVED = {p.name[: -len("_gain.npz")] for p in LABELS_DIR.glob("*_gain.npz")}


def tile_status(t):
    """1 = labelled (gain mask saved), 0 = to do."""
    return 1 if t in SAVED else 0


def next_unlabelled(current, step=1):
    """Next queue tile (in queue order, wrapping) that is still to do (not labelled)."""
    q = st.session_state["queue_ids"]
    if not q:
        return None
    i = q.index(current) if current in q else (-1 if step > 0 else 0)
    for k in range(1, len(q) + 1):
        t = q[(i + step * k) % len(q)]
        if t != current and tile_status(t) == 0:
            return t
    return None


def go_to(t):
    if t:
        st.session_state.selected_tile_id = t


def go_next():
    go_to(next_unlabelled(st.session_state.selected_tile_id, 1))


def go_prev():
    q = st.session_state["queue_ids"]
    cur = st.session_state.selected_tile_id
    if cur in q and q.index(cur) > 0:
        go_to(q[q.index(cur) - 1])


def on_search():
    text = st.session_state.get("search_id", "")
    if text:
        m = gdf_projected[
            gdf_projected["tile_id"].str.contains(text, case=False, na=False)
        ]
        if not m.empty:
            go_to(m["tile_id"].iloc[0])


def on_select():
    go_to(st.session_state.tile_select)


# ==========================================
# SIDEBAR: NAVIGATION & SEARCH
# ==========================================
st.sidebar.title("🌲 Tile Navigation")
st.sidebar.markdown(f"**Total Indexed Tiles:** {len(gdf_projected):,}")

st.sidebar.text_input("Search Tile ID", "", key="search_id", on_change=on_search)

queue_ids = st.session_state["queue_ids"]
if queue_ids:
    n_done = sum(tile_status(t) == 1 for t in queue_ids)
    st.sidebar.progress(
        min(n_done / max(len(queue_ids), 1), 1.0),
        text=f"{n_done} / {len(queue_ids)} done",
    )
    nav_prev, nav_next = st.sidebar.columns(2)
    nav_prev.button("⬅ Previous", on_click=go_prev, use_container_width=True)
    nav_next.button(
        "Next unlabelled ➡",
        on_click=go_next,
        use_container_width=True,
        type="primary",
        disabled=next_unlabelled(st.session_state.selected_tile_id) is None,
    )
    show_all = st.sidebar.checkbox(
        "Show all tiles in the selector", key="show_all_tiles"
    )
    tile_options = gdf_projected["tile_id"].tolist() if show_all else list(queue_ids)
    if st.session_state.selected_tile_id not in tile_options:
        tile_options.insert(
            0, st.session_state.selected_tile_id
        )  # opened from the map / search
    icons = {0: "⬜", 1: "✅"}
else:
    tile_options = gdf_projected["tile_id"].tolist()
    icons = None
st.session_state["tile_select"] = st.session_state.selected_tile_id
st.sidebar.selectbox(
    ("Select Tile ID" if not queue_ids else "Tile (work queue: ⬜ to do · ✅ done)"),
    tile_options,
    key="tile_select",
    on_change=on_select,
    format_func=(lambda t: f"{icons[tile_status(t)]} {t}") if icons else str,
)

with st.sidebar.expander("🧪 Build a representative annotation sample", expanded=False):
    st.caption(
        "Randomly sample tiles within country × HRL-gain-percentage strata. Country is taken from an existing country field or looked up from GEE LSIB."
    )
    sample_n = st.number_input(
        "Number of tiles",
        min_value=1,
        max_value=max(1, len(gdf_projected)),
        value=min(100, max(1, len(gdf_projected))),
        step=10,
    )
    gain_bins_text = st.text_input("Gain % bin edges", "0, 1, 5, 10, 25, 50, 100")
    sample_seed = st.number_input(
        "Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1
    )
    new_queue = False
    if st.button("Generate stratified tile list"):
        try:
            gdf_projected = lookup_missing_countries(gdf_projected)
            tile_row = gdf_projected[
                gdf_projected["tile_id"] == st.session_state.selected_tile_id
            ].iloc[0]
            edges = sorted(
                set(float(x.strip()) for x in gain_bins_text.split(",") if x.strip())
            )
            if len(edges) < 2:
                raise ValueError("Provide at least two numeric bin edges.")
            gains = pd.to_numeric(gdf_projected["gain_pct"], errors="coerce")
            # tiles whose country could not be determined (centroid in water / not in LSIB) are not sampled
            known = gdf_projected["country"].notna() & ~gdf_projected["country"].astype(
                str
            ).str.strip().str.lower().isin(["unknown", "", "nan", "none"])
            strata = gdf_projected[known].copy()
            if strata.empty:
                raise ValueError("No tiles with a known country.")
            strata["gain_bin"] = pd.cut(
                gains.clip(lower=edges[0], upper=edges[-1]),
                bins=edges,
                include_lowest=True,
                right=True,
            ).astype(str)
            strata["stratum"] = (
                strata["country"].astype(str) + " | " + strata["gain_bin"]
            )
            rng = np.random.default_rng(int(sample_seed))
            strata["_random"] = rng.random(len(strata))
            populations = (
                strata.groupby("stratum", dropna=False)
                .size()
                .rename("stratum_population")
            )
            strata = strata.sort_values("_random")
            # Proportional allocation with at least one tile per stratum where possible.
            counts = populations.to_dict()
            allocation = {s: 0 for s in counts}
            remaining = min(int(sample_n), len(strata))
            if remaining >= len(counts):
                for s in counts:
                    allocation[s] = 1
                remaining -= len(counts)
            while remaining > 0:
                eligible = [s for s, n in counts.items() if allocation[s] < n]
                if not eligible:
                    break
                total = sum(counts[s] for s in eligible)
                weights = np.array([counts[s] / total for s in eligible], dtype=float)
                picks = rng.multinomial(remaining, weights)
                used = 0
                for s, k in zip(eligible, picks):
                    take = min(int(k), counts[s] - allocation[s])
                    allocation[s] += take
                    used += take
                if used == 0:
                    s = max(eligible, key=lambda z: counts[z] - allocation[z])
                    allocation[s] += 1
                    used = 1
                remaining -= used
            parts = []
            for s, n_take in allocation.items():
                if n_take:
                    part = strata[strata["stratum"] == s].head(n_take).copy()
                    part["stratum_population"] = int(counts[s])
                    part["sample_probability"] = n_take / counts[s]
                    parts.append(part)
            sample = (
                pd.concat(parts, ignore_index=True) if parts else strata.head(0).copy()
            )
            cols = [
                c
                for c in [
                    "tile_id",
                    "country",
                    "gain_pct",
                    "gain_bin",
                    "stratum",
                    "stratum_population",
                    "sample_probability",
                ]
                if c in sample.columns
            ]
            st.session_state["tile_sample_csv"] = (
                sample[cols].sort_values(["country", "gain_pct"]).to_csv(index=False)
            )
            st.session_state["tile_sample_preview"] = sample[cols].sort_values(
                ["country", "gain_pct"]
            )
            set_queue(sample.sort_values(["country", "gain_pct"])["tile_id"].tolist())
            new_queue = True
        except Exception as exc:
            st.error(f"Could not generate sample: {exc}")
    if "tile_sample_csv" in st.session_state:
        st.download_button(
            "Download sample CSV",
            st.session_state["tile_sample_csv"],
            "stratified_tile_sample.csv",
            "text/csv",
        )
        st.dataframe(
            st.session_state["tile_sample_preview"],
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Generating a sample makes it your work queue (saved to `annotation_queue.csv`). Or load an existing list with a `tile_id` column:"
    )
    uploaded = st.file_uploader("Load queue CSV", type="csv", key="queue_upload")
    if uploaded is not None and st.session_state.get("_queue_sig") != (
        uploaded.name,
        uploaded.size,
    ):
        st.session_state["_queue_sig"] = (uploaded.name, uploaded.size)
        set_queue(pd.read_csv(uploaded)["tile_id"].astype(str))
        new_queue = True
    if st.session_state["queue_ids"] and st.button("Clear work queue"):
        st.session_state["queue_ids"] = []
        QUEUE_PATH.unlink(missing_ok=True)
        new_queue = True
if new_queue:
    st.rerun()  # so the tile selector / progress above pick up the new queue

basemap_name = st.sidebar.selectbox("Map background", list(BASEMAPS), index=0)
recentre = st.sidebar.checkbox("Re-centre map when the tile changes", value=False)

tile_id = st.session_state.selected_tile_id
tile_row = gdf_projected[gdf_projected["tile_id"] == tile_id].iloc[0]
BOUNDS = tuple(float(tile_row[k]) for k in ("x_min_m", "y_min_m", "x_max_m", "y_max_m"))

# Link the selected tile to the ArcGIS Living Atlas Wayback viewer (a plain link; the app makes no Esri calls for it),
# centered on the tile centroid (longitude, latitude) at a useful local zoom.
tile_lon = float(tile_row["lon"])
tile_lat = float(tile_row["lat"])
esri_wayback_url = (
    "https://livingatlas.arcgis.com/wayback/#mapCenter="
    f"{tile_lon:.5f}%2C{tile_lat:.5f}%2C14"
    "&mode=updates&active=6543&updatesMode=vivid-advanced%7Clast-year%7C"
)
st.markdown(
    f"**Selected tile:** `{tile_id}` · "
    f"Centre: `{tile_lat:.5f}, {tile_lon:.5f}` · "
    f"[Open this tile in Esri Living Atlas Wayback ↗]({esri_wayback_url})"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ What you are labelling")
st.sidebar.caption(
    f"Forest gain over the whole period {YEAR_A} → {YEAR_B}: pixels that are not tree in {YEAR_A} and tree in {YEAR_B}."
)
is_saved = saved_path(tile_id).exists()
st.sidebar.write("Saved for this tile: " + ("yes" if is_saved else "not yet"))
start_options = ["Dynamic World", "Blank"] + (["Previously saved"] if is_saved else [])
start_mode = st.sidebar.radio("Start from", start_options, horizontal=True)
with st.sidebar.expander("Dynamic World thresholds"):
    DW_THR = st.slider("Tree if probability >", 0.0, 1.0, 0.5, 0.05)
    MIN_PX = st.slider("Smallest Dynamic World gain patch to propose (px)", 1, 50, 8)


# ==========================================
# COMPOSITING
# ==========================================
def _blend(base, rgb, alpha):
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    return base * (1 - a) + rgb * a


def _tint(mask, colour):
    out = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    out[:] = np.array(colour, dtype=np.float32) / 255.0
    return out, mask.astype(np.float32)


def starting_masks(arrays):
    """(gain, ignore) bool (256, 256) that the canvas edits start from."""
    blank = np.zeros((SIZE, SIZE), bool)
    if start_mode == "Previously saved" and is_saved:
        z = np.load(saved_path(tile_id))
        return z["gain"], z["ignore"]
    if (
        start_mode == "Dynamic World"
        and arrays.get(DW_A) is not None
        and arrays.get(DW_B) is not None
    ):
        a, b = arrays[DW_A][0], arrays[DW_B][0]
        valid = np.isfinite(a) & np.isfinite(b)
        cand = (np.nan_to_num(a) <= DW_THR) & (np.nan_to_num(b) > DW_THR) & valid
        return (
            drop_small(cand, MIN_PX),
            blank,
        )  # DW candidate: non-tree in 2017 and tree in 2024
    return blank, blank


def build_composite_layer(enabled, op):
    """Returns (PIL image, errors, (start_gain, start_ignore))."""
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    errors, arrays = [], {}

    def load(name):
        if name not in arrays:
            arr, err = get_layer(tile_id, name, *BOUNDS)
            if err:
                errors.append(err)
            arrays[name] = arr
        return arrays[name]

    # Dynamic World for the label year is always needed (starting mask); both years for the change layer
    need = {name for name, on in enabled.items() if on and name in LAYERS}
    if start_mode == "Dynamic World":
        need |= {DW_A, DW_B}
    if enabled.get(DW_NT_A):
        need.add(DW_A)
    if enabled.get(DW_T_B):
        need.add(DW_B)
    if need & {S2_A, S2_B}:
        need |= {
            S2_A,
            S2_B,
        }  # both years are fetched so they can share one contrast stretch
    for name in LAYERS:
        if name in need:
            load(name)

    # Fixed reflectance stretch (same convention as your notebook / GEE: 0-3000 of the 0-10000 SR scale,
    # linear). A per-tile percentile stretch re-scales every tile to its own contrast, which looks oversaturated.
    s2_lo, s2_hi = S2_VMIN, S2_VMAX
    for name in (S2_A, S2_B):
        if enabled.get(name) and arrays.get(name) is not None:
            raw = np.moveaxis(arrays[name], 0, -1).astype(np.float32)
            valid = np.isfinite(raw).all(axis=-1)
            rgb = np.clip(
                (np.nan_to_num(raw, nan=s2_lo) - s2_lo) / (s2_hi - s2_lo), 0, 1
            ).astype(np.float32)
            rgb[~valid] = 0.0
            canvas = _blend(canvas, rgb, op[name])

    if enabled.get(NDVI) and arrays.get(NDVI) is not None:
        d = arrays[NDVI][0]
        colour = plt.get_cmap("RdYlGn")(np.clip(np.nan_to_num(d) + 0.5, 0, 1))[..., :3]
        canvas = _blend(canvas, colour, op[NDVI] * np.isfinite(d))

    for name in (DW_A, DW_B):
        if enabled.get(name) and arrays.get(name) is not None:
            p = arrays[name][0]
            colour = plt.get_cmap("viridis")(np.clip(np.nan_to_num(p), 0, 1))[..., :3]
            canvas = _blend(canvas, colour, op[name] * np.isfinite(p))

    # Binary DW masks at the same threshold as the starting mask: 2017 non-tree (magenta), 2024 tree (cyan).
    # Where both show, DW says the pixel went non-tree -> tree (a gain candidate).
    for name, dw, want_tree, colour in (
        (DW_NT_A, DW_A, False, (255, 0, 255)),
        (DW_T_B, DW_B, True, (0, 255, 255)),
    ):
        if enabled.get(name) and arrays.get(dw) is not None:
            p = arrays[dw][0]
            valid = np.isfinite(p)
            m = (
                (np.nan_to_num(p) > DW_THR)
                if want_tree
                else (np.nan_to_num(p) <= DW_THR)
            )
            col, a = _tint(m & valid, colour)
            canvas = _blend(canvas, col, a * op[name])

    if enabled.get(HRL) and arrays.get(HRL) is not None:
        g = np.nan_to_num(arrays[HRL][0])
        colour = plt.get_cmap("Greens")(0.35 + 0.65 * np.clip(g / 100.0, 0, 1))[..., :3]
        canvas = _blend(canvas, colour, op[HRL] * (g > 0))  # only where there is gain

    if enabled.get(SLOPE) and arrays.get(SLOPE) is not None:
        s = np.nan_to_num(arrays[SLOPE][0])
        canvas = _blend(
            canvas, plt.get_cmap("terrain")(np.clip(s / 45.0, 0, 1))[..., :3], op[SLOPE]
        )

    s_gain, s_ign = starting_masks(arrays)
    if enabled.get(START):
        for m, c in ((s_gain, (255, 220, 0)), (s_ign, (0, 120, 255))):
            col, a = _tint(m, c)
            canvas = _blend(canvas, col, a * op[START])

    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    return img.resize((CANVAS, CANVAS), Image.NEAREST), errors, (s_gain, s_ign)


# ==========================================
# MAIN APP INTERFACE
# ==========================================
st.title("🗺️ EPSG:3035 Spatial Grid & Segmentation Labelling Suite")


# Interactive Leaflet component: tile clicks return a tile ID to Streamlit.
import streamlit.components.v1 as components

with st.expander("📍 Tile map (swipe 2017 / 2024 imagery)", expanded=True):
    try:
        imagery_2017_url, wayback_2017_release = map_imagery_url(YEAR_A)
        imagery_2024_url, wayback_2024_release = map_imagery_url(YEAR_B)
    except Exception as e:
        imagery_2017_url = imagery_2024_url = None
        wayback_2017_release = wayback_2024_release = "Unknown"
        st.warning(f"Could not prepare Esri Wayback comparison imagery: {str(e)[:200]}")

    if imagery_2017_url and imagery_2024_url:
        # Compact notice banner displaying the exact imagery dates
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; background-color: #262730; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; font-size: 13px; color: #ffffff;">
                <span>👈 <b>2017 View:</b> Esri Wayback Release <code>{wayback_2017_release}</code></span>
                <span>👉 <b>2024 View:</b> Esri Wayback Release <code>{wayback_2024_release}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create native Folium map centered on the selected tile
        m = folium.Map(
            location=[float(tile_row["lat"]), float(tile_row["lon"])],
            zoom_start=11,
            tiles=(
                "CartoDB positron"
                if basemap_name == "Light"
                else "OpenStreetMap" if basemap_name == "OpenStreetMap" else None
            ),
        )

        # Add Esri Satellite basemap if selected
        if basemap_name == "Satellite (Esri)":
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, Maxar, Earthstar Geographics",
                name="Esri Satellite",
                control=False,
            ).add_to(m)

        # 2017 Tile Layer (Left side of swipe)
        layer_2017 = folium.TileLayer(
            tiles=imagery_2017_url,
            attr="2017 Imagery",
            name="2017",
            overlay=True,
            control=False,
            max_zoom=18,
        )
        layer_2017.add_to(m)

        # 2024 Tile Layer (Right side of swipe)
        layer_2024 = folium.TileLayer(
            tiles=imagery_2024_url,
            attr="2024 Imagery",
            name="2024",
            overlay=True,
            control=False,
            max_zoom=18,
        )
        layer_2024.add_to(m)

        # Native Folium Side-by-Side Swipe Plugin
        SideBySideLayers(layer_left=layer_2017, layer_right=layer_2024).add_to(m)

        # All tile markers go in ONE GeoJSON layer (built once and cached) - thousands of individual
        # folium.CircleMarker objects were the main reason every rerun was so slow.
        folium.GeoJson(
            marker_geojson(GEOJSON_PATH),
            marker=folium.CircleMarker(radius=4, fill=True, fill_opacity=0.9),
            style_function=lambda f: {
                "color": f["properties"]["col"],
                "fillColor": f["properties"]["col"],
                "weight": 1,
            },
            tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False),
        ).add_to(m)
        folium.CircleMarker(  # the selected tile, highlighted
            location=[float(tile_row["lat"]), float(tile_row["lon"])],
            radius=8,
            color="#ffffff",
            weight=3,
            fill=True,
            fill_color="#ff0000",
            fill_opacity=0.9,
            tooltip=f"{tile_id} (selected)",
        ).add_to(m)

        # Render map natively in Streamlit and listen for clicks
        map_data = st_folium(
            m, width="100%", height=490, returned_objects=["last_active_drawing"]
        )

        # Handle marker selection change
        if map_data and map_data.get("last_active_drawing"):
            clicked_id = (map_data["last_active_drawing"].get("properties") or {}).get(
                "tile_id"
            )
            if clicked_id and clicked_id in TILE_SET and clicked_id != tile_id:
                st.session_state["_pending"] = {"selected_tile_id": clicked_id}
                st.rerun()

        st.caption(
            "Click any tile marker to open it for labelling. Drag the swipe divider to compare imagery versions."
        )
    else:
        st.info(
            "The comparison map is unavailable until both imagery layers can be prepared."
        )

st.markdown("---")

# Equal-width side panels keep the centre column's midpoint fixed relative to both panels,
# even when the available main-page width changes as the sidebar is resized.
col_layers, col_canvas, col_tools = st.columns([1.15, 2.2, 1.15])

# --- LAYER COMPOSITOR PANEL ---
with col_layers:
    st.subheader("🎛️ Layer Compositor")
    st.caption(
        "Layers are generated from Earth Engine the first time you switch them on for a tile, then cached."
    )

    default_on = {S2_B, START}
    default_op = {START: 0.4, HRL: 0.7}
    enabled_layers, opacities = {}, {}
    for layer in list(LAYERS) + DERIVED:
        # full-width checkbox (long names wrap instead of truncating) with its opacity slider underneath
        enabled_layers[layer] = st.checkbox(
            layer, value=layer in default_on, key=f"en_{layer}"
        )
        opacities[layer] = st.slider(
            f"{layer} Opacity",
            0.0,
            1.0,
            default_op.get(layer, 1.0),
            key=f"op_{layer}",
            label_visibility="collapsed",
        )
    st.caption(
        "HRL gain is only a helper layer: keep it off while you label so it does not steer the masks."
    )

    with st.expander("Cached layers for this tile"):
        for layer in LAYERS:
            st.write(("✅ " if layer_path(tile_id, layer).exists() else "⬜ ") + layer)
        if st.button("Delete this tile's cache (regenerate)"):
            for layer in LAYERS:
                layer_path(tile_id, layer).unlink(missing_ok=True)
            st.rerun()


# --- SEGMENTATION TOOLS PANEL ---
with col_tools:
    st.subheader("🛠️ Segmentation Tools")
    st.caption(
        "Yellow = the starting gain mask (non-tree in 2017, tree in 2024). Paint corrections on top: green adds gain, "
        "red removes gain, blue marks cloud / unclear areas to ignore. Where strokes overlap, erase beats add and add beats ignore."
    )
    tool_cols = st.columns([1.25, 1, 1])
    with tool_cols[0]:
        paint_class = st.selectbox("Class", list(CLASSES), key="paint_class")
    with tool_cols[1]:
        drawing_mode = st.selectbox(
            "Tool",
            ["freedraw", "polygon", "rect", "point (fill)"],
            index=0,
            key="tool",
        )
    with tool_cols[2]:
        stroke_width = st.slider("Brush", 1, 25, 6, key="brush")
    fill_tolerance = st.slider(
        "Fill colour tolerance",
        0,
        100,
        18,
        help="For point (fill): include connected pixels whose displayed background colour is within this RGB distance of the clicked pixel.",
    )
    view_quadrant = st.selectbox(
        "View / zoom",
        [
            "Full tile",
            "Top-left quarter",
            "Top-right quarter",
            "Bottom-left quarter",
            "Bottom-right quarter",
        ],
        key=f"view_quadrant_{tile_id}",
        help="Label one quarter at a time, then switch back to Full tile or another quarter. Edits are saved in full-tile coordinates.",
    )
    rgb_c, hex_c = CLASSES[paint_class]
    if drawing_mode == "point (fill)":
        canvas_drawing_mode = "point"
        st.caption(
            "Choose a class, then click a region on the image. Fill follows connected pixels of similar displayed colour."
        )
    else:
        canvas_drawing_mode = drawing_mode


# --- strokes are stored in FULL-tile canvas coordinates, whatever view they were drawn in ---------------------------
# A quarter view shows one 256x256-canvas-px quadrant of the 512x512 full canvas magnified 2x. Strokes drawn there are
# converted back to full-tile coordinates before storing, and the stored strokes are converted into the current view
# before they are drawn, so drawings survive zooming in and back out (and appear magnified when you zoom in).
# Every Fabric.js object is positioned by (left, top) and sized by (scaleX, scaleY), so the conversion only needs to
# remap those - the path/rect/circle geometry inside the object is left alone.
def _remap_drawing(drawing, scale, ox, oy, to_view):
    """Copy of a Fabric.js canvas JSON with every object moved/scaled between full-tile and view coordinates.
    to_view=True : full-tile -> view   (p' = scale * (p - origin))
    to_view=False: view -> full-tile   (p  = p' / scale + origin)"""
    out = copy.deepcopy(drawing)
    f = scale if to_view else 1.0 / scale
    for obj in out.get("objects", []):
        for key, o in (("left", ox), ("top", oy)):
            if key in obj:
                obj[key] = (obj[key] - o) * scale if to_view else obj[key] / scale + o
        for key in ("scaleX", "scaleY"):
            obj[key] = float(obj.get(key, 1.0)) * f
        if obj.get("strokeUniform") and "strokeWidth" in obj:
            obj["strokeWidth"] = (
                float(obj["strokeWidth"]) * f
            )  # not scaled with the object, so scale it by hand
    return out


# --- CANVAS DISPLAY ---
# A fragment: drawing a stroke reruns only this panel instead of the whole page.
@st.fragment
def canvas_panel():
    st.subheader(f"🎨 Labelling forest gain {YEAR_A} → {YEAR_B}: `{tile_id}`")
    if "_flash" in st.session_state:
        st.success(st.session_state.pop("_flash"))

    try:
        init_ee()
        comp_key = (
            tile_id,
            start_mode,
            DW_THR,
            MIN_PX,
            is_saved,
            tuple(layer_path(tile_id, l).exists() for l in LAYERS),
            json.dumps([enabled_layers, opacities], sort_keys=True),
        )
        memo = st.session_state.get("_comp_memo")
        if memo is not None and memo[0] == comp_key:
            bg_img, errors, (start_gain, start_ign) = memo[1]
        else:
            result = build_composite_layer(enabled_layers, opacities)
            st.session_state["_comp_memo"] = (comp_key, result)
            bg_img, errors, (start_gain, start_ign) = result
    except Exception as e:
        st.error(
            f"Earth Engine could not be started: {e}\n\nRun `earthengine authenticate` once, then restart the app."
        )
        st.stop()
    for err in errors:
        st.warning("Could not generate " + err)

    # Crop the displayed tile for quarter-level labelling. The underlying labels remain
    # full-resolution (256x256); each quarter is enlarged to the same 512x512 canvas.
    half_canvas = CANVAS // 2
    quadrant_boxes = {
        "Full tile": (0, 0, CANVAS, CANVAS),
        "Top-left quarter": (0, 0, half_canvas, half_canvas),
        "Top-right quarter": (half_canvas, 0, CANVAS, half_canvas),
        "Bottom-left quarter": (0, half_canvas, half_canvas, CANVAS),
        "Bottom-right quarter": (half_canvas, half_canvas, CANVAS, CANVAS),
    }
    qx0, qy0, qx1, qy1 = quadrant_boxes[view_quadrant]
    if view_quadrant == "Full tile":
        canvas_bg_img = bg_img
    else:
        canvas_bg_img = bg_img.crop((qx0, qy0, qx1, qy1)).resize(
            (CANVAS, CANVAS), Image.NEAREST
        )

    state = json.dumps(
        [
            tile_id,
            start_mode,
            DW_THR,
            MIN_PX,
            enabled_layers,
            opacities,
            view_quadrant,
        ],
        sort_keys=True,
    )
    canvas_key = "canvas_" + hashlib.md5(state.encode()).hexdigest()[:12]

    stroke_store = st.session_state.setdefault(
        "_strokes", {}
    )  # tile_id -> drawing in FULL-tile coordinates
    view_scale = CANVAS / (qx1 - qx0)  # 1 for the full tile, 2 for a quarter

    # FIX: Only load initial_drawing when the canvas key changes (mount/layer change).
    # On subsequent reruns triggered by drawing strokes, pass None so st_canvas doesn't blink or reset.
    if (
        "_last_canvas_key" not in st.session_state
        or st.session_state["_last_canvas_key"] != canvas_key
    ):
        st.session_state["_last_canvas_key"] = canvas_key
        stored = stroke_store.get(tile_id)
        initial_draw_payload = (
            _remap_drawing(stored, view_scale, qx0, qy0, to_view=True)
            if stored
            else None
        )
    else:
        initial_draw_payload = None

    canvas_result = st_canvas(
        initial_drawing=initial_draw_payload,
        fill_color=f"rgba({rgb_c[0]}, {rgb_c[1]}, {rgb_c[2]}, 0.5)",
        stroke_width=stroke_width,
        stroke_color=hex_c,
        background_image=canvas_bg_img,
        update_streamlit=True,
        height=CANVAS,
        width=CANVAS,
        drawing_mode=canvas_drawing_mode,
        point_display_radius=max(2, stroke_width),
        key=canvas_key,
    )

    drawn = canvas_result.json_data
    if drawn is not None and drawn.get("objects"):
        stroke_store[tile_id] = _remap_drawing(
            drawn, view_scale, qx0, qy0, to_view=False
        )
        st.session_state["_strokes_seen"] = canvas_key

    if st.button("💾 Save forest gain mask", type="primary"):
        if canvas_result.image_data is None:
            st.warning("Canvas not ready yet.")
        else:
            rgba = canvas_result.image_data
            r, g, b, a = (rgba[..., i].astype(np.int32) for i in range(4))
            painted = a > 0

            def to_tile(m):
                # Reduce the visible canvas to its underlying pixel resolution.
                side = SIZE if view_quadrant == "Full tile" else SIZE // 2
                scale = CANVAS // side
                return m.reshape(side, scale, side, scale).mean(axis=(1, 3)) >= 0.25

            add_canvas = painted & (g > r) & (g > b)
            erase_canvas = painted & (r > g) & (r > b)
            ign_canvas = painted & (b > r) & (b > g)
            if drawing_mode == "point (fill)" and painted.any():
                # Use the centre of the first point mark as the seed; flood-fill the connected
                # region of the displayed composite with a configurable RGB tolerance.
                ys, xs = np.where(painted)
                seed_y, seed_x = int(np.median(ys)), int(np.median(xs))
                background = np.asarray(
                    canvas_bg_img.convert("RGB").resize(
                        (CANVAS, CANVAS), Image.NEAREST
                    ),
                    dtype=np.int16,
                )
                seed_colour = background[seed_y, seed_x]
                distance = np.sqrt(np.sum((background - seed_colour) ** 2, axis=2))
                candidate = distance <= float(fill_tolerance) * np.sqrt(3.0)
                labels, _ = ndimage.label(candidate)
                seed_label = labels[seed_y, seed_x]
                fill_region = (
                    (labels == seed_label)
                    if seed_label
                    else np.zeros((CANVAS, CANVAS), bool)
                )
                if paint_class == "gain (add)":
                    add_canvas = fill_region
                    erase_canvas = np.zeros_like(fill_region)
                    ign_canvas = np.zeros_like(fill_region)
                elif paint_class == "not gain (erase)":
                    erase_canvas = fill_region
                    add_canvas = np.zeros_like(fill_region)
                    ign_canvas = np.zeros_like(fill_region)
                else:
                    ign_canvas = fill_region
                    add_canvas = np.zeros_like(fill_region)
                    erase_canvas = np.zeros_like(fill_region)
            add_local = to_tile(add_canvas)
            erase_local = to_tile(erase_canvas)
            ign_local = to_tile(ign_canvas)
            if view_quadrant == "Full tile":
                add, erase, ign = add_local, erase_local, ign_local
            else:
                # Map local quarter edits back to the correct 128x128 region of the full tile.
                add = np.zeros((SIZE, SIZE), dtype=bool)
                erase = np.zeros((SIZE, SIZE), dtype=bool)
                ign = np.zeros((SIZE, SIZE), dtype=bool)
                row0 = 0 if "Top" in view_quadrant else SIZE // 2
                col0 = 0 if "left" in view_quadrant else SIZE // 2
                region = (slice(row0, row0 + SIZE // 2), slice(col0, col0 + SIZE // 2))
                add[region] = add_local
                erase[region] = erase_local
                ign[region] = ign_local
            # When revisiting another quarter, start from the already-saved full-tile
            # masks so edits elsewhere are retained. On the first save, use the chosen
            # starting masks (Dynamic World / blank / previously saved).
            existing_path = saved_path(tile_id)
            if existing_path.exists():
                with np.load(existing_path) as previous:
                    gain_base = previous["gain"].astype(bool)
                    ignore_base = previous["ignore"].astype(bool)
            else:
                gain_base, ignore_base = start_gain.copy(), start_ign.copy()
            gain = (gain_base | add) & ~erase
            ignore = (ignore_base | ign) & ~erase & ~add
            gain &= ~ignore
            np.savez_compressed(saved_path(tile_id), gain=gain, ignore=ignore)
            SAVED.add(tile_id)
            write_tile_tif(tile_id, BOUNDS)
            with open(LABELS_DIR / f"{tile_id}_meta.json", "w") as f:
                json.dump(
                    {
                        "tile_id": tile_id,
                        "x_min_m": BOUNDS[0],
                        "y_min_m": BOUNDS[1],
                        "x_max_m": BOUNDS[2],
                        "y_max_m": BOUNDS[3],
                        "hrl_gain_pct": float(
                            tile_row.get("hrl_gain_pct", tile_row["gain_pct"])
                        ),
                        "woodland_pct": (
                            None
                            if pd.isna(tile_row.get("woodland_pct"))
                            else float(tile_row.get("woodland_pct"))
                        ),
                        "gain_pct": float(tile_row["gain_pct"]),
                        "country": str(tile_row.get("country", "Unknown")),
                        "period": [YEAR_A, YEAR_B],
                        "start": start_mode,
                        "dw_thr": DW_THR,
                        "min_px": MIN_PX,
                    },
                    f,
                    indent=2,
                )
            msg = (
                f"Saved: gain {100 * gain.mean():.1f}% of tile ({int(gain.sum())} px), "
                f"ignore {100 * ignore.mean():.1f}% (written to `{tile_id}.tif`)."
            )
            target = next_unlabelled(tile_id) if view_quadrant == "Full tile" else None
            if target is None:
                st.success(msg)
            else:
                # Auto-advance to the next unlabelled queue tile (full-tile saves only).
                stroke_store.pop(tile_id, None)
                st.session_state["_pending"] = {
                    "selected_tile_id": target,
                    "_flash": f"{msg} Next: `{target}`.",
                }
                st.rerun(scope="app")


with col_canvas:
    canvas_panel()

# --- prefetch the next queue tiles' Earth Engine layers while you label this one ---
upcoming, _t = [], tile_id
for _ in range(PREFETCH_AHEAD):
    _t = next_unlabelled(_t)
    if _t is None or _t == tile_id or _t in upcoming:
        break
    upcoming.append(_t)
for _t in upcoming:
    _r = gdf_projected[gdf_projected["tile_id"] == _t].iloc[0]
    _b = tuple(float(_r[k]) for k in ("x_min_m", "y_min_m", "x_max_m", "y_max_m"))
    get_prefetcher().submit(_t, _b)
