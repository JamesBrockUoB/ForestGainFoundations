"""Run with: streamlit run tile_inspector.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import folium
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import streamlit as st
from config import settings
from inspector.service import assess_metrics, point_centred_tile, tile_corners_lonlat
from shapely.geometry import box
from streamlit_folium import st_folium

st.set_page_config(page_title="Forest Gain tile inspector", layout="wide")

EXTENT_PATH = settings.data_dir / "aois" / "aoi_footprint_europe" / "aoi_footprint.shp"

REJECTION_CODE = {
    "no_land": 1,
    "insufficient_veg": 2,
    "no_forest_gain": 3,
    "missing_imagery": 4,
    "valid": 5,
}
REJECTION_COLOR = {
    1: "#4a4a4a",
    2: "#e07b39",
    3: "#c0392b",
    4: "#8e44ad",
}
VALID_COLOR = "#27ae60"
GRID_STEP = 0.25
GRID_LON_MIN = -180.0
GRID_LAT_MIN = -60.0


@st.cache_resource
def load_product_extent() -> tuple[dict, object, tuple[float, float, float, float]]:
    footprint = gpd.read_file(EXTENT_PATH)
    if footprint.empty:
        raise ValueError(f"Forest Gain extent is empty: {EXTENT_PATH}")
    if footprint.crs is None:
        raise ValueError(f"Forest Gain extent has no CRS: {EXTENT_PATH}")

    footprint_wgs84 = footprint.to_crs("EPSG:4326")
    footprint_6933 = footprint.to_crs(settings.crs)
    return (
        footprint_wgs84.__geo_interface__,
        footprint_6933.geometry.unary_union,
        tuple(footprint_wgs84.total_bounds),
    )


@st.cache_data
def load_aoi_checkpoint(period: str) -> tuple[list[dict], list[dict]]:
    path = settings.data_dir / "aois" / f"aoi_filter_checkpoint_{period}.json"
    if not path.exists():
        return [], []

    with open(path) as f:
        checkpoint = json.load(f)

    def props(records):
        return [r.get("properties", r) for r in records]

    return props(checkpoint.get("valid", [])), props(checkpoint.get("rejected", []))


@st.cache_data
def build_aoi_geojson(period: str):
    valid_aois, rejected_aois = load_aoi_checkpoint(period)

    def records_to_features(records, include_reason=False):
        features = []

        for p in records:
            min_lon = p.get("minLon")
            max_lon = p.get("maxLon")
            min_lat = p.get("minLat")
            max_lat = p.get("maxLat")

            if None in (min_lon, max_lon, min_lat, max_lat):
                continue

            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]],
            }

            properties = {
                "id": p.get("id"),
                "rejection_reason": p.get("rejection_reason"),
                "land_frac": p.get("land_frac"),
                "veg_fraction": p.get("veg_fraction"),
                "forest_gain_frac": p.get("forest_gain_frac"),
                "has_imagery": p.get("has_imagery"),
            }

            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    return {
        "valid": records_to_features(valid_aois),
        "rejected": records_to_features(rejected_aois),
    }


def run_worker(action: str, period: str, payload: dict) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "inspector.worker", action],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env={**os.environ, "PERIOD": period},
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Inspector worker failed")
    return json.loads(result.stdout)


st.title("Forest Gain tile inspector")
period = st.segmented_control("Period", options=("p1", "p2"), default="p1")

if st.session_state.get("map_period") != period:
    st.session_state.map_period = period
    st.session_state.selected_point = None
    st.session_state.pop("metrics", None)
    st.session_state.pop("metrics_tile_id", None)

period_years = "2017–2020" if period == "p1" else "2020–2024"
is_p1 = period == "p1"
st.caption(
    f"{period}: {period_years} · "
    f"{settings.tile_size_m / 1000:.2f} km point-centred EPSG:6933 tile"
)

with st.sidebar:
    st.header("Validity thresholds")
    gain_min = st.number_input(
        "Minimum gain (%)", 0.0, 100.0, float(settings.gain_pct_min), 0.1
    )
    ndvi_min = st.number_input(
        "Minimum NDVI trend",
        value=float(settings.ndvi_trend_min),
        step=0.001,
        format="%.4f",
    )
    forty_min = st.number_input(
        "Minimum ForTy coverage over gain (%)",
        0.0,
        100.0,
        float(settings.min_pseudo_gain_frac * 100),
        0.1,
        disabled=not is_p1,
    )
    imagery_min = st.number_input(
        "Minimum valid S1/S2 pixels per year (%)",
        0.0,
        100.0,
        float(settings.imagery_min_valid_frac * 100),
        0.1,
    )
    st.caption("Threshold changes re-evaluate fetched metrics locally.")

    st.header("AOI validity overlay")
    show_valid_aois = st.checkbox("Show valid AOIs", value=False)
    show_rejected_aois = st.checkbox("Show rejected AOIs", value=False)
    aoi_overlay_opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.6, 0.05)

extent_geojson, extent_6933, (west, south, east, north) = load_product_extent()

point = st.session_state.get(
    "selected_point",
    ((west + east) / 2, (south + north) / 2),
)

if point is None:
    point = ((west + east) / 2, (south + north) / 2)

tile = point_centred_tile(*point, period)
corners = tile_corners_lonlat(tile)

tile_box_6933 = box(
    tile["x_min_m"],
    tile["y_min_m"],
    tile["x_max_m"],
    tile["y_max_m"],
)

tile_is_in_extent = extent_6933.covers(tile_box_6933)

map_ = folium.Map(
    location=[(south + north) / 2, (west + east) / 2],
    zoom_start=4,
    control_scale=True,
)

map_.fit_bounds([[south, west], [north, east]])

map_layers = folium.FeatureGroup(name="Dynamic map layers")

folium.GeoJson(
    extent_geojson,
    name="Forest Gain product extent",
    style_function=lambda _: {
        "color": "#2e7d32",
        "weight": 2,
        "fill": False,
    },
).add_to(map_layers)

if show_valid_aois or show_rejected_aois:
    overlays = build_aoi_geojson(period)

    if show_valid_aois:
        folium.GeoJson(
            overlays["valid"],
            name="Valid AOIs",
            style_function=lambda feature: {
                "color": VALID_COLOR,
                "weight": 1,
                "fillColor": VALID_COLOR,
                "fillOpacity": aoi_overlay_opacity,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["id", "forest_gain_frac", "veg_fraction"],
                aliases=["AOI", "Forest gain", "Vegetation"],
                localize=True,
            ),
        ).add_to(map_layers)

    if show_rejected_aois:
        def rejected_style(feature):
            reason = feature["properties"].get("rejection_reason")
            code = REJECTION_CODE.get(reason)
            colour = REJECTION_COLOR.get(code, "#666666")

            return {
                "color": colour,
                "weight": 1,
                "fillColor": colour,
                "fillOpacity": aoi_overlay_opacity,
            }

        folium.GeoJson(
            overlays["rejected"],
            name="Rejected AOIs",
            style_function=rejected_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["id", "rejection_reason"],
                aliases=["AOI", "Reason"],
                localize=True,
            ),
        ).add_to(map_layers)

folium.CircleMarker(
    [point[1], point[0]],
    radius=5,
    color="#1565c0",
    fill=True,
    fill_opacity=1,
    tooltip="Selected tile centre",
).add_to(map_layers)

folium.Polygon(
    [(lat, lon) for lon, lat in corners],
    color="#1565c0" if tile_is_in_extent else "#d62728",
    fill=True,
    fill_opacity=0.12,
).add_to(map_layers)

event = st_folium(
    map_,
    height=780,
    use_container_width=True,
    key="tile-inspector-map",
    feature_group_to_add=map_layers,
    returned_objects=["last_clicked"],
)

clicked = event.get("last_clicked") if event else None

if clicked:
    selected = (
        round(float(clicked["lng"]), 8),
        round(float(clicked["lat"]), 8),
    )

    if selected != st.session_state.get("selected_point"):
        st.session_state.selected_point = selected
        st.session_state.pop("metrics", None)
        st.session_state.pop("metrics_tile_id", None)
        st.rerun()

st.code(
    f"{tile['tile_id']}\ncentre: {point[1]:.6f}, {point[0]:.6f}\n"
    f"EPSG:6933 bounds: {tile['x_min_m']:.3f}, {tile['y_min_m']:.3f}, "
    f"{tile['x_max_m']:.3f}, {tile['y_max_m']:.3f}",
    language="text",
)

if not tile_is_in_extent:
    st.error("The complete 2.56 km tile must lie inside the Forest Gain footprint.")

if st.button(
    "Fetch raw validity metrics",
    type="primary",
    disabled=not tile_is_in_extent,
):
    with st.spinner("Fetching gain, ForTy, NDVI, and annual S1/S2 coverage…"):
        st.session_state.metrics = run_worker(
            "fetch",
            period,
            {"tile": tile},
        )
        st.session_state.metrics_tile_id = tile["tile_id"]

metrics = st.session_state.get("metrics")

if metrics and st.session_state.get("metrics_tile_id") == tile["tile_id"]:
    checks = assess_metrics(
        metrics,
        gain_pct_min=gain_min,
        ndvi_trend_min=ndvi_min,
        pseudo_gain_pct_min=forty_min,
        imagery_min_valid_frac=imagery_min / 100,
        pseudo_labels_available=is_p1,
    )

    valid = all(passed for _, passed, _ in checks)

    (st.success if valid else st.warning)(
        "Valid with these thresholds"
        if valid
        else "Not valid with these thresholds"
    )

    st.dataframe(
        [
            {
                "Check": name,
                "Pass": "✓" if passed else "✗",
                "Value": value,
            }
            for name, passed, value in checks
        ],
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("Export for investigation")

output_root = settings.data_dir / "inspector_tiles" / period
st.caption(f"Writes to `{output_root}` without updating registry data")

confirmed = st.checkbox(
    "I understand this submits Earth Engine export tasks."
)

if st.button(
    "Export this tile locally",
    disabled=not (confirmed and tile_is_in_extent),
):
    try:
        with st.spinner("Submitting exports and downloading products…"):
            output_dir = run_worker(
                "export",
                period,
                {
                    "tile": tile,
                    "output_root": str(output_root),
                },
            )["output_dir"]

        st.success(f"Export complete: {output_dir}")

    except Exception as exc:
        st.exception(exc)