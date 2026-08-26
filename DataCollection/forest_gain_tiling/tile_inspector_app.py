"""Run with: streamlit run tile_inspector.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import folium
import geopandas as gpd
import streamlit as st
from config import settings
from inspector import display
from inspector.service import assess_metrics, point_centred_tile, tile_corners_lonlat
from shapely.geometry import box
from streamlit_folium import st_folium

st.set_page_config(page_title="Forest Gain tile inspector", layout="wide")

EXTENT_PATH = settings.data_dir / "aois" / "aoi_footprint_europe" / "aoi_footprint.shp"


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

    def records_to_features(records):
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

            features.append({"type": "Feature", "geometry": geometry, "properties": properties})

        return {"type": "FeatureCollection", "features": features}

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


@st.cache_data
def list_exported_tiles(period: str) -> list[str]:
    root = settings.data_dir / "inspector_tiles" / period
    if not root.exists():
        return []

    return sorted(p.name for p in root.iterdir() if p.is_dir())

st.title("Forest Gain tile inspector")

period = st.segmented_control("Period", options=("p1", "p2"), default="p1")

if st.session_state.get("map_period") != period:
    st.session_state.map_period = period
    st.session_state.selected_point = None
    st.session_state.pop("metrics", None)
    st.session_state.pop("metrics_tile_id", None)

period_years = "2017-2020" if period == "p1" else "2020-2024"
is_p1 = period == "p1"

st.caption(
    f"{period}: {period_years} · "
    f"{settings.tile_size_m / 1000:.2f} km "
    f"point-centred EPSG:6933 tile"
)

with st.sidebar:
    st.header("Validity thresholds")

    gain_min = st.number_input("Minimum gain (%)", 0.0, 100.0, float(settings.gain_pct_min), 0.1)
    ndvi_min = st.number_input(
        "Minimum NDVI trend", value=float(settings.ndvi_trend_min), step=0.001, format="%.4f"
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

    valid_aois, rejected_aois = load_aoi_checkpoint(period)

    if not valid_aois and not rejected_aois:
        st.info(f"No AOI data is available for {period}.")
        show_valid_aois = False
        show_rejected_aois = False
        aoi_overlay_opacity = 0.6
    else:
        show_valid_aois = st.checkbox("Show valid AOIs", value=False)
        show_rejected_aois = st.checkbox("Show rejected AOIs", value=False)
        aoi_overlay_opacity = st.slider(
            "Overlay opacity",
            0.0,
            1.0,
            0.6,
            0.05,
        )

extent_geojson, extent_6933, (west, south, east, north) = load_product_extent()

point = st.session_state.get("selected_point", ((west + east) / 2, (south + north) / 2))
if point is None:
    point = ((west + east) / 2, (south + north) / 2)

tile = point_centred_tile(*point, period)
corners = tile_corners_lonlat(tile)

tile_box_6933 = box(tile["x_min_m"], tile["y_min_m"], tile["x_max_m"], tile["y_max_m"])
tile_is_in_extent = extent_6933.covers(tile_box_6933)

tab_select, tab_viewer = st.tabs(["Tile selection & validity", "Exported tile viewer"])

with tab_select:
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
        style_function=lambda _: {"color": "#2e7d32", "weight": 2, "fill": False},
    ).add_to(map_layers)

    if show_valid_aois or show_rejected_aois:
        overlays = build_aoi_geojson(period)

        if show_valid_aois:
            valid_features = overlays["valid"].get("features", [])

            if valid_features:
                folium.GeoJson(
                    overlays["valid"],
                    name="Valid AOIs",
                    style_function=lambda _: display.valid_aoi_style(
                        aoi_overlay_opacity
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=["id", "forest_gain_frac", "veg_fraction"],
                        aliases=["AOI", "Forest gain", "Vegetation"],
                        localize=True,
                    ),
                ).add_to(map_layers)

        if show_rejected_aois:
            rejected_features = overlays["rejected"].get("features", [])

            if rejected_features:
                folium.GeoJson(
                    overlays["rejected"],
                    name="Rejected AOIs",
                    style_function=lambda feature: display.rejected_aoi_style(
                        feature,
                        aoi_overlay_opacity,
                    ),
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
        selected = (round(float(clicked["lng"]), 8), round(float(clicked["lat"]), 8))

        if selected != st.session_state.get("selected_point"):
            st.session_state.selected_point = selected
            st.session_state.pop("metrics", None)
            st.session_state.pop("metrics_tile_id", None)
            st.rerun()

    st.code(
        f"{tile['tile_id']}\n"
        f"centre: {point[1]:.6f}, {point[0]:.6f}\n"
        f"EPSG:6933 bounds: "
        f"{tile['x_min_m']:.3f}, {tile['y_min_m']:.3f}, "
        f"{tile['x_max_m']:.3f}, {tile['y_max_m']:.3f}",
        language="text",
    )

    if not tile_is_in_extent:
        st.error("The complete 2.56 km tile must lie inside the Forest Gain footprint.")

    if st.button("Fetch raw validity metrics", type="primary", disabled=not tile_is_in_extent):
        with st.spinner("Fetching gain, ForTy, NDVI, and annual S1/S2 coverage…"):
            st.session_state.metrics = run_worker("fetch", period, {"tile": tile})
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
            "Valid with these thresholds" if valid else "Not valid with these thresholds"
        )

        st.dataframe(
            [{"Check": name, "Pass": "✓" if passed else "✗", "Value": value} for name, passed, value in checks],
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Export for investigation")

    output_root = settings.data_dir / "inspector_tiles" / period

    st.caption(f"Writes to `{output_root}` without updating registry data")

    confirmed = st.checkbox("I understand this submits Earth Engine export tasks.")

    if st.button("Export this tile locally", disabled=not (confirmed and tile_is_in_extent)):
        try:
            with st.spinner("Submitting exports and downloading products…"):
                output_dir = run_worker(
                    "export", period, {"tile": tile, "output_root": str(output_root)}
                )["output_dir"]

            st.success(f"Export complete: {output_dir}")

        except Exception as exc:
            st.exception(exc)

with tab_viewer:
    viewer_period = st.selectbox("Period", options=("p1", "p2"), key="viewer_period")

    exported_tiles = list_exported_tiles(viewer_period)

    if not exported_tiles:
        st.info(f"No exported tiles found in {settings.data_dir / 'inspector_tiles' / viewer_period}.")

    else:
        selected_tile_id = st.selectbox("Tile", exported_tiles, key="selected_exported_tile")

        tile_dir = settings.data_dir / "inspector_tiles" / viewer_period / selected_tile_id

        composites_dir = tile_dir / "composites"
        embeddings_dir = tile_dir / "embeddings"

        s1s2_by_year = {display.raster_year(p): p for p in sorted(composites_dir.glob("s1s2_*.tif"))}
        aee_by_year = {display.raster_year(p): p for p in sorted(embeddings_dir.glob("aee_*.tif"))}
        tessera_by_year = {display.raster_year(p): p for p in sorted(embeddings_dir.glob("tessera_*.tif"))}

        aee_raw = {}
        for year, path in aee_by_year.items():
            try:
                aee_raw[year] = display.read_embedding(str(path))
            except Exception as exc:
                st.warning(f"Could not read AEE {year}: {exc}")

        tessera_raw = {}
        for year, path in tessera_by_year.items():
            try:
                tessera_raw[year] = display.read_embedding(str(path))
            except Exception as exc:
                st.warning(f"Could not read TESSERA {year}: {exc}")

        aee_stretched = {}
        if aee_raw:
            years_sorted = sorted(aee_raw, key=int)
            aee_stretched = dict(
                zip(years_sorted, display.shared_rgb_stretch([aee_raw[y] for y in years_sorted]))
            )

        tessera_stretched = {}
        if tessera_raw:
            years_sorted = sorted(tessera_raw, key=int)
            tessera_stretched = dict(
                zip(years_sorted, display.shared_rgb_stretch([tessera_raw[y] for y in years_sorted]))
            )

        s1_images = {}
        s2_images = {}
        for year, path in s1s2_by_year.items():
            try:
                s1_images[year], s2_images[year] = display.read_composite_s1_s2(str(path))
            except Exception as exc:
                st.error(f"Could not read S1/S2 {year}: {exc}")

        with st.expander("Yearly products (S1 / S2 / AEE / TESSERA)", expanded=True):
            display.render_yearly_products(s1_images, s2_images, aee_stretched, tessera_stretched)

        with st.expander("Static layers"):
            static_paths = {
                "Slope": tile_dir / "static" / "slope.tif",
                "Protected area": tile_dir / "static" / "protected_area.tif",
                "FABDEM": tile_dir / "static" / "fabdem.tif",
            }
            static_colours = {"Slope": "viridis", "Protected area": "Greens", "FABDEM": "terrain"}

            static_display = st.radio(
                "Static layer display",
                options=("Separate", "Overlay on S2"),
                horizontal=True,
                key="static_display_mode",
            )

            static_opacity = st.slider(
                "Static overlay opacity",
                0.1,
                1.0,
                0.65,
                0.05,
                key="static_overlay_opacity",
                disabled=static_display != "Overlay on S2",
            )

            display.render_static_layers(
                static_paths, static_colours, s2_images, static_display, static_opacity
            )

        with st.expander("Gain & pseudo-label diagnostics"):
            labels_dir = tile_dir / "labels"
            gain_confidence_path = labels_dir / "gain_confidence.tif"
            pseudo_labels_path = labels_dir / "pseudo_labels.tif"

            base_options = {
                name: images
                for name, images in (
                    ("S2", s2_images),
                    ("S1", s1_images),
                    ("AEE", aee_stretched),
                    ("TESSERA", tessera_stretched),
                )
                if images
            }

            if not base_options:
                st.info("No imagery available to use as a base for the diagnostics.")
            else:
                base_cols = st.columns(2)

                with base_cols[0]:
                    base_product = st.selectbox(
                        "Base imagery",
                        options=list(base_options.keys()),
                        key="gain_pseudo_base_product",
                    )

                available_base_years = sorted(base_options[base_product], key=int)

                with base_cols[1]:
                    base_year = st.selectbox(
                        "Year",
                        options=available_base_years,
                        index=len(available_base_years) - 1,
                        key="gain_pseudo_base_year",
                    )

                base_image = base_options[base_product][base_year]

                gain_confidence = None
                if gain_confidence_path.exists():
                    try:
                        gain_confidence = display.read_gain_confidence(str(gain_confidence_path))
                    except Exception as exc:
                        st.error(f"Could not read gain confidence: {exc}")

                dominant = confidence = None
                if pseudo_labels_path.exists():
                    try:
                        dominant, confidence = display.read_pseudo_labels(str(pseudo_labels_path))
                    except Exception as exc:
                        st.error(f"Could not read pseudo labels: {exc}")

                display.render_gain_pseudo_diagnostics(
                    base_image, f"{base_product} {base_year}", gain_confidence, dominant, confidence
                )

        with st.expander("Tile metadata"):
            metadata = display.load_tile_metadata(tile_dir)

            if metadata is None:
                st.info("No metadata.json found for this tile.")
            else:
                display.render_tile_metadata(metadata)