import json
import re
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

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

PSEUDO_CLASS_NAMES = ["AGROCROP", "NAT_REGEN", "PLANTATION", "PLANTED"]
PSEUDO_CLASS_COLORS = ["#e6ab02", "#1b9e77", "#7570b3", "#d95f02"]
PSEUDO_LABEL_NODATA = -9999


def valid_aoi_style(opacity: float) -> dict:
    return {
        "color": VALID_COLOR,
        "weight": 1,
        "fillColor": VALID_COLOR,
        "fillOpacity": opacity,
    }


def rejected_aoi_style(feature: dict, opacity: float) -> dict:
    reason = feature["properties"].get("rejection_reason")
    code = REJECTION_CODE.get(reason)
    colour = REJECTION_COLOR.get(code, "#666666")

    return {
        "color": colour,
        "weight": 1,
        "fillColor": colour,
        "fillOpacity": opacity,
    }


def raster_year(path: Path) -> str:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    return match.group(0) if match else path.stem


def stretch_to_unit(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((arr.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)


def shared_rgb_stretch(
    arrays: list[np.ndarray],
    low: float = 2,
    high: float = 98,
) -> list[np.ndarray]:
    stacked = np.concatenate(
        [arr.reshape(-1, arr.shape[-1]) for arr in arrays],
        axis=0,
    )

    bounds = []
    for band in range(stacked.shape[-1]):
        valid = stacked[:, band]
        valid = valid[np.isfinite(valid)]

        if valid.size:
            lo, hi = np.nanpercentile(valid, [low, high])
        else:
            lo, hi = 0.0, 1.0

        if hi <= lo:
            hi = lo + 1.0

        bounds.append((lo, hi))

    result = []
    for arr in arrays:
        out = np.empty_like(arr, dtype=np.float32)

        for band, (lo, hi) in enumerate(bounds):
            out[..., band] = np.clip((arr[..., band] - lo) / (hi - lo), 0, 1)

        result.append(out)

    return result


def read_composite_s1_s2(path: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)

    s2 = data[[2, 1, 0]].transpose(1, 2, 0)
    s2 = stretch_to_unit(s2, 0, 3000)

    vv = data[10]
    vh = data[11]

    s1 = np.stack(
        [
            stretch_to_unit(vv, -25, 0),
            stretch_to_unit(vh, -30, -5),
            stretch_to_unit(vv - vh, 0, 15),
        ],
        axis=-1,
    )

    return s1, s2


def read_embedding(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3]).astype(np.float32)

    return data.transpose(1, 2, 0)


def static_colour_preview(path: str, cmap_name: str) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    if not valid.any():
        return np.zeros((*arr.shape, 3), dtype=np.float32)

    lo, hi = np.nanpercentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1.0

    normalised = np.clip((arr - lo) / (hi - lo), 0, 1)

    cmap = colormaps[cmap_name]
    rgb = cmap(normalised)[..., :3].astype(np.float32)
    rgb[~valid] = 0

    plt.close("all")

    return rgb


def load_tile_metadata(tile_dir: Path) -> dict | None:
    metadata_path = tile_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)


def _render_overlay(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    opacity: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base_image)
    ax.imshow(overlay_image, alpha=opacity)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout(pad=0)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_yearly_products(
    s1_images: dict[str, np.ndarray],
    s2_images: dict[str, np.ndarray],
    aee_stretched: dict[str, np.ndarray],
    tessera_stretched: dict[str, np.ndarray],
) -> None:
    """Show each yearly product (S1 / S2 / AEE / TESSERA) as plain images.

    No overlay between products here -- stacking, say, AEE on top of S2
    doesn't mean anything. Overlays live in render_gain_pseudo_diagnostics,
    where they're actually meaningful (gain/pseudo-label maps over imagery).
    """
    products = {
        "S1": s1_images,
        "S2": s2_images,
        "AEE": aee_stretched,
        "TESSERA": tessera_stretched,
    }

    for product_name, images_by_year in products.items():
        st.subheader(product_name)

        available_years = sorted(images_by_year, key=int)

        if not available_years:
            st.info(f"No {product_name} yearly products found for this tile.")
            continue

        cols = st.columns(len(available_years))

        for col, year in zip(cols, available_years):
            with col:
                st.markdown(f"### {year}")

                try:
                    st.image(images_by_year[year], use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not display {product_name} {year}: {exc}")


def render_static_layers(
    static_paths: dict[str, Path],
    static_colours: dict[str, str],
    s2_images: dict[str, np.ndarray],
    display_mode: str,
    opacity: float,
) -> None:
    cols = st.columns(len(static_paths))

    for col, (layer_name, static_path) in zip(cols, static_paths.items()):
        with col:
            st.markdown(f"### {layer_name}")

            if not static_path.exists():
                st.info(f"No {layer_name} raster found.")
                continue

            try:
                static_image = static_colour_preview(
                    str(static_path), static_colours[layer_name]
                )

                if display_mode == "Overlay on S2" and s2_images:
                    first_year = sorted(s2_images, key=int)[0]
                    _render_overlay(
                        s2_images[first_year],
                        static_image,
                        opacity,
                        f"{layer_name} / S2 {first_year}",
                    )
                else:
                    st.image(static_image, use_container_width=True)

            except Exception as exc:
                st.error(f"Could not display {layer_name}: {exc}")


def read_gain_confidence(path: str) -> np.ndarray:
    """Read the continuous gain-confidence raster (band 1).

    Valid pixels hold canopy-cover % at year_end (always above the
    TREE_THRESHOLD used to build the gain mask); pixels outside gain
    coverage are NaN, which also defines the binary gain mask.
    """
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def read_pseudo_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the dominant-class (band 5) and confidence (band 6) bands.

    Nodata (-9999) is converted to NaN so masking downstream is a plain
    isnan check.
    """
    with rasterio.open(path) as src:
        dominant, confidence = src.read([5, 6]).astype(np.float32)

    dominant[dominant == PSEUDO_LABEL_NODATA] = np.nan
    confidence[confidence == PSEUDO_LABEL_NODATA] = np.nan

    return dominant, confidence


def render_gain_pseudo_diagnostics(
    base_image: np.ndarray,
    base_label: str,
    gain_confidence: np.ndarray | None,
    dominant: np.ndarray | None,
    confidence: np.ndarray | None,
) -> None:
    """Render gain-mask / pseudo-label panels over a chosen imagery composite.

    Shows, wherever the underlying raster is available: the dominant
    pseudo-class, per-pixel pseudo-label confidence, continuous gain
    confidence, and the binary gain mask derived from it -- each drawn
    as its own panel, only over pixels with valid gain coverage.
    """
    import matplotlib.pyplot as plt

    gain_valid = ~np.isnan(gain_confidence) if gain_confidence is not None else None

    panels = []

    if dominant is not None and gain_valid is not None:
        panels.append(("dominant", dominant))
    if confidence is not None and gain_valid is not None:
        panels.append(("confidence", confidence))
    if gain_confidence is not None:
        panels.append(("gain_confidence", gain_confidence))
    if gain_valid is not None:
        panels.append(("gain_mask", gain_valid.astype(np.float32)))

    if not panels:
        st.info("No gain confidence or pseudo-label rasters found for this tile.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    fig.suptitle(f"Gain & pseudo-label diagnostics — {base_label}")

    cmap_classes = ListedColormap(PSEUDO_CLASS_COLORS)
    norm_classes = BoundaryNorm(
        [i - 0.5 for i in range(len(PSEUDO_CLASS_NAMES) + 1)],
        cmap_classes.N,
    )

    for ax, (kind, arr) in zip(axes, panels):
        ax.imshow(base_image)

        if kind == "dominant":
            masked = np.ma.masked_where(~gain_valid | np.isnan(arr), arr)
            im = ax.imshow(masked, cmap=cmap_classes, norm=norm_classes, alpha=0.8)
            ax.set_title("Dominant pseudo-class\n(gain pixels)")
            cbar = fig.colorbar(
                im, ax=ax, ticks=range(len(PSEUDO_CLASS_NAMES)), fraction=0.046, pad=0.04
            )
            cbar.ax.set_yticklabels(PSEUDO_CLASS_NAMES)

        elif kind == "confidence":
            masked = np.ma.masked_where(~gain_valid | np.isnan(arr), arr)
            im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=1, alpha=0.85)
            ax.set_title("Pseudo-label confidence\n(gain pixels)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ForTy confidence")

        elif kind == "gain_confidence":
            masked = np.ma.masked_where(np.isnan(arr), arr)
            im = ax.imshow(masked, cmap="plasma", vmin=50, vmax=100, alpha=0.85)
            ax.set_title("Gain confidence mask\n(continuous)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                "Gain confidence (canopy cover %, year_end)"
            )

        else:  # gain_mask
            masked = np.ma.masked_where(arr == 0, arr)
            ax.imshow(masked, cmap="viridis", vmin=0, vmax=1, alpha=0.8)
            ax.set_title("Gain mask\n(binary)")

        ax.axis("off")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if gain_valid is not None:
        n_gain = int(gain_valid.sum())
        st.caption(f"Gain pixels: {n_gain} ({100 * n_gain / gain_valid.size:.2f}% of tile)")


def display_climate(metadata: dict) -> None:
    climate = metadata.get("climate_yearly")
    if not climate:
        return

    import matplotlib.pyplot as plt

    years = sorted(climate, key=int)
    precip = [climate[year].get("precip_sum", np.nan) for year in years]
    temp = [climate[year].get("temp_mean", np.nan) for year in years]

    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(years, precip, marker="o", linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual precipitation (mm)")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(years, temp, marker="o", linewidth=2, linestyle="--")
    ax2.set_ylabel("Mean temperature (°C)")
    ax1.set_title("Yearly climate")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    climate_rows = [
        {
            "Year": int(year),
            "Precipitation (mm)": climate[year].get("precip_sum"),
            "Precip min (mm)": climate[year].get("precip_min"),
            "Precip max (mm)": climate[year].get("precip_max"),
            "Mean temperature (°C)": climate[year].get("temp_mean"),
            "Temp min (°C)": climate[year].get("temp_min"),
            "Temp max (°C)": climate[year].get("temp_max"),
            "Source": climate[year].get("climate_source"),
        }
        for year in years
    ]

    st.dataframe(climate_rows, hide_index=True, use_container_width=True)


def render_tile_metadata(metadata: dict) -> None:
    bounds = metadata.get("bounds", {})
    soil = metadata.get("soil", {})
    pseudo = metadata.get("pseudo_labels", {})
    slope_metadata = metadata.get("slope_deg", {})

    overview_cols = st.columns(4)

    with overview_cols[0]:
        gain_pct = metadata.get("gain_pct")
        st.metric("Forest gain", f"{gain_pct:.2f}%" if gain_pct is not None else "—")
        st.write(f"**Biome:** {metadata.get('biome', '—')}")
        st.write(f"**Region:** {metadata.get('region', '—')}")
        st.write(f"**Country:** {metadata.get('country', '—')}")

    with overview_cols[1]:
        st.write("**Soil**")
        st.write(f"SOC: {soil.get('soc', '—')}")
        st.write(f"Clay: {soil.get('clay_pct', '—')}%")
        st.write(f"pH: {soil.get('ph', '—')}")

    with overview_cols[2]:
        st.write("**Pseudo labels**")
        st.write(f"Dominant class: {pseudo.get('dominant_class', '—')}")

        confidence = pseudo.get("mean_confidence")
        st.write(
            f"Mean confidence: {confidence:.3f}"
            if confidence is not None
            else "Mean confidence: —"
        )

        labelled_fraction = pseudo.get("labelled_gain_pixel_fraction")
        st.write(
            f"Labelled gain: {labelled_fraction * 100:.1f}%"
            if labelled_fraction is not None
            else "Labelled gain: —"
        )

    with overview_cols[3]:
        st.write("**Slope**")

        mean_slope = slope_metadata.get("mean")
        p90_slope = slope_metadata.get("p90")

        st.write(f"Mean: {mean_slope:.2f}°" if mean_slope is not None else "Mean: —")
        st.write(f"P90: {p90_slope:.2f}°" if p90_slope is not None else "P90: —")
        st.write(f"Exported: {metadata.get('exported_at', '—')}")

    st.write("**Bounds**")
    bounds_cols = st.columns(4)

    with bounds_cols[0]:
        st.write(f"CRS: {bounds.get('crs', '—')}")

    with bounds_cols[1]:
        st.write(f"X: {bounds.get('x_min_m', '—')} → {bounds.get('x_max_m', '—')}")

    with bounds_cols[2]:
        st.write(f"Y: {bounds.get('y_min_m', '—')} → {bounds.get('y_max_m', '—')}")

    with bounds_cols[3]:
        st.write(f"Lon: {bounds.get('min_lon', '—')} → {bounds.get('max_lon', '—')}")
        st.write(f"Lat: {bounds.get('min_lat', '—')} → {bounds.get('max_lat', '—')}")

    if metadata.get("climate_yearly"):
        st.subheader("Climate by year")
        display_climate(metadata)

    with st.expander("Pseudo-label class counts"):
        class_counts = pseudo.get("class_pixel_counts", {})
        if class_counts:
            st.dataframe(
                [{"Class": name, "Pixels": count} for name, count in class_counts.items()],
                hide_index=True,
                use_container_width=True,
            )