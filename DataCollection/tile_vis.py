"""
Interactive visualisation of forest gain tiles.
Usage: python tile_vis.py --tile path/to/tile.tif
       python tiles.py --dir path/to/tiles/  (loads all .tif files)
"""

import argparse
import base64
import webbrowser
from io import BytesIO
from pathlib import Path

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.warp import transform_bounds

BANDS = {
    "T0_B4": 2,
    "T0_B3": 1,
    "T0_B2": 0,  # T0 RGB
    "T1_B4": 17,
    "T1_B3": 16,
    "T1_B2": 15,  # T1 RGB
    "T2_B4": 32,
    "T2_B3": 31,
    "T2_B2": 30,  # T2 RGB
    "DEM": 45,
    "slope": 46,
    "canopy_gain_height": 47,
    "jrc_forest_type": 48,
    "natural_forest_prob": 49,
    "gain_mask": 50,
}

JRC_COLORS = {
    1: ("#006400", "Primary forest"),
    2: ("#228B22", "Secondary forest"),
    3: ("#32CD32", "Planted forest"),
    4: ("#90EE90", "Other wooded land"),
}

CANOPY_CMAP = LinearSegmentedColormap.from_list(
    "canopy", ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
)
NFP_CMAP = LinearSegmentedColormap.from_list("nfp", ["#ffffff", "#1a9641"])


def read_band(src, band_idx):
    data = src.read(band_idx + 1).astype(float)
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def normalise_rgb(r, g, b, p_low=2, p_high=98):
    def norm(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return arr
        lo, hi = np.nanpercentile(arr, p_low), np.nanpercentile(arr, p_high)
        return np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1)

    return norm(r), norm(g), norm(b)


def array_to_png_b64(rgba):
    """Convert RGBA uint8 array to base64 PNG."""
    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_rgb_overlay(src, b4_idx, b3_idx, b2_idx, bounds_wgs84):
    r = read_band(src, b4_idx)
    g = read_band(src, b3_idx)
    b = read_band(src, b2_idx)
    r, g, b = normalise_rgb(r, g, b)

    alpha = (~np.isnan(r)).astype(float)
    rgba = np.stack(
        [np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b), alpha], axis=-1
    )
    rgba = (rgba * 255).astype(np.uint8)
    return array_to_png_b64(rgba), bounds_wgs84


def make_gain_overlay(src, bounds_wgs84):
    gain = read_band(src, BANDS["gain_mask"])
    r = np.zeros_like(gain)
    g = np.where(gain == 1, 1.0, 0.0)
    b = np.zeros_like(gain)
    alpha = np.where(gain == 1, 0.8, 0.0)
    rgba = np.stack([r, g, b, alpha], axis=-1)
    rgba = (rgba * 255).astype(np.uint8)
    return array_to_png_b64(rgba), bounds_wgs84


def make_canopy_overlay(src, bounds_wgs84):
    ch = read_band(src, BANDS["canopy_gain_height"])
    valid_mask = ~np.isnan(ch)
    norm = plt.Normalize(vmin=0, vmax=40)
    rgba = CANOPY_CMAP(norm(np.nan_to_num(ch)))
    rgba[..., 3] = np.where(valid_mask, 0.85, 0.0)
    rgba = (rgba * 255).astype(np.uint8)
    return array_to_png_b64(rgba), bounds_wgs84


def make_nfp_overlay(src, bounds_wgs84):
    nfp = read_band(src, BANDS["natural_forest_prob"])
    valid_mask = ~np.isnan(nfp) & (nfp > 0)
    norm = plt.Normalize(vmin=0, vmax=1)
    rgba = NFP_CMAP(norm(np.nan_to_num(nfp)))
    rgba[..., 3] = np.where(valid_mask, 0.8, 0.0)
    rgba = (rgba * 255).astype(np.uint8)
    return array_to_png_b64(rgba), bounds_wgs84


def make_jrc_overlay(src, bounds_wgs84):
    jrc = read_band(src, BANDS["jrc_forest_type"])
    h, w = jrc.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for val, (hex_col, _) in JRC_COLORS.items():
        mask = jrc == val
        rgb = mcolors.to_rgb(hex_col)
        rgba[mask, 0] = int(rgb[0] * 255)
        rgba[mask, 1] = int(rgb[1] * 255)
        rgba[mask, 2] = int(rgb[2] * 255)
        rgba[mask, 3] = 200
    return array_to_png_b64(rgba), bounds_wgs84


def get_wgs84_bounds(src):
    bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    # folium expects [[south, west], [north, east]]
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def add_image_overlay(fmap, b64_png, bounds, name, show=True):
    img_data = f"data:image/png;base64,{b64_png}"
    folium.raster_layers.ImageOverlay(
        image=img_data,
        bounds=bounds,
        name=name,
        opacity=1.0,
        show=show,
        cross_origin=False,
    ).add_to(fmap)


def tile_stats(src):
    stats = {}
    for name, idx in [
        ("canopy_height_m", BANDS["canopy_gain_height"]),
        ("natural_forest_prob", BANDS["natural_forest_prob"]),
        ("gain_coverage_%", BANDS["gain_mask"]),
    ]:
        data = read_band(src, idx)
        valid = data[~np.isnan(data)]
        if name == "gain_coverage_%":
            pct = 100 * np.sum(valid == 1) / valid.size if valid.size > 0 else 0
            stats[name] = f"{pct:.2f}%"
        else:
            if len(valid) > 0:
                stats[name] = f"mean={np.nanmean(valid):.2f} max={np.nanmax(valid):.2f}"
            else:
                stats[name] = "no data"
    return stats


def visualise_tile(tif_path, fmap=None):
    tif_path = Path(tif_path)
    print(f"Loading {tif_path.name}...")

    with rasterio.open(tif_path) as src:
        bounds = get_wgs84_bounds(src)
        center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
        ]

        if fmap is None:
            fmap = folium.Map(location=center, zoom_start=14, tiles=None)
            folium.TileLayer(
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Satellite",
                overlay=False,
                control=True,
            ).add_to(fmap)
            folium.TileLayer(
                "OpenStreetMap", name="OSM", overlay=False, control=True
            ).add_to(fmap)

        # RGB composites
        for label, b4, b3, b2, show in [
            (
                f"{tif_path.stem} — T0 (2016)",
                BANDS["T0_B4"],
                BANDS["T0_B3"],
                BANDS["T0_B2"],
                True,
            ),
            (
                f"{tif_path.stem} — T1 (2020)",
                BANDS["T1_B4"],
                BANDS["T1_B3"],
                BANDS["T1_B2"],
                False,
            ),
            (
                f"{tif_path.stem} — T2 (2025)",
                BANDS["T2_B4"],
                BANDS["T2_B3"],
                BANDS["T2_B2"],
                False,
            ),
        ]:
            b64, bds = make_rgb_overlay(src, b4, b3, b2, bounds)
            add_image_overlay(fmap, b64, bds, label, show=show)

        # Gain mask
        b64, bds = make_gain_overlay(src, bounds)
        add_image_overlay(fmap, b64, bds, f"{tif_path.stem} — Gain mask", show=True)

        # Canopy height
        b64, bds = make_canopy_overlay(src, bounds)
        add_image_overlay(
            fmap, b64, bds, f"{tif_path.stem} — Canopy height", show=False
        )

        # Natural forest probability
        b64, bds = make_nfp_overlay(src, bounds)
        add_image_overlay(
            fmap, b64, bds, f"{tif_path.stem} — Natural forest prob", show=False
        )

        # JRC forest type
        b64, bds = make_jrc_overlay(src, bounds)
        add_image_overlay(
            fmap, b64, bds, f"{tif_path.stem} — JRC forest type", show=False
        )

        # Tile boundary
        folium.Rectangle(
            bounds=bounds,
            color="#00FF00",
            fill=False,
            weight=2,
            tooltip=tif_path.stem,
            popup=folium.Popup(
                "<br>".join(f"<b>{k}</b>: {v}" for k, v in tile_stats(src).items()),
                max_width=250,
            ),
        ).add_to(fmap)

    return fmap


def add_legends(fmap):
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:rgba(20,20,20,0.92);
                padding:14px 18px;border-radius:8px;font-family:monospace;font-size:12px;color:#eee;
                border:1px solid #444;min-width:180px;">
      <b style="font-size:13px;">Canopy Height (m)</b><br>
      <div style="background:linear-gradient(to right,#f7fcf5,#c7e9c0,#74c476,#238b45,#00441b);
                  height:12px;width:150px;margin:6px 0;border-radius:2px;"></div>
      <span>0</span><span style="float:right">40</span>
      <br><br>
      <b style="font-size:13px;">Natural Forest Prob</b><br>
      <div style="background:linear-gradient(to right,#ffffff,#1a9641);
                  height:12px;width:150px;margin:6px 0;border-radius:2px;"></div>
      <span>0</span><span style="float:right">1</span>
      <br><br>
      <b style="font-size:13px;">JRC Forest Type</b><br>
      <span style="color:#006400;">&#9632;</span> Primary &nbsp;
      <span style="color:#228B22;">&#9632;</span> Secondary<br>
      <span style="color:#32CD32;">&#9632;</span> Planted &nbsp;
      <span style="color:#90EE90;">&#9632;</span> Other<br>
      <br>
      <span style="color:#00FF00;">&#9632;</span> <b>Gain mask</b>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


def main():
    parser = argparse.ArgumentParser(
        description="Visualise forest gain tiles interactively."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tile", help="Path to a single .tif file")
    group.add_argument("--dir", help="Directory containing .tif files")
    parser.add_argument("--out", default="tile_viewer.html", help="Output HTML file")
    args = parser.parse_args()

    fmap = None

    if args.tile:
        fmap = visualise_tile(args.tile, fmap)
    else:
        tifs = sorted(Path(args.dir).glob("*.tif"))
        if not tifs:
            print(f"No .tif files found in {args.dir}")
            return
        print(f"Found {len(tifs)} tiles")
        for tif in tifs:
            fmap = visualise_tile(tif, fmap)

    add_legends(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = Path(args.out)
    fmap.save(str(out))
    print(f"Saved to {out.resolve()}")
    webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
