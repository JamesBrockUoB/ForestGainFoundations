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

# Units shown on static-layer colorbars, keyed by the layer names used in
# tile_inspector.py's static_paths dict. A missing/blank entry just means
# no unit suffix.
STATIC_LAYER_UNITS = {
    "Slope": "°",
    "Protected area": "",
    "FABDEM": "m",
}


def _fmt(value, decimals: int = 2, suffix: str = "") -> str:
    """Format a number to a fixed number of decimal places, or '—' if missing.

    Centralises the 2dp convention used throughout the metadata summary so
    it isn't repeated (and isn't inconsistent) across every st.write call.
    """
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _round2(value):
    """Round a number to 2dp for tabular display, passing through None."""
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return value


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


def static_colour_data(path: str) -> tuple[np.ndarray, float, float]:
    """Read a static raster and return it alongside its display bounds.

    Unlike a pre-baked RGB preview, the raw (nodata-masked) array is kept
    scalar so callers can render with imshow(..., cmap=..., vmin=, vmax=)
    and attach a matching colorbar as a legend.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    arr = np.where(valid, arr, np.nan)

    if not valid.any():
        return arr, 0.0, 1.0

    vmin, vmax = np.nanpercentile(arr[valid], [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0

    return arr, float(vmin), float(vmax)


def load_tile_metadata(tile_dir: Path) -> dict | None:
    metadata_path = tile_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)


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
    """Render the three static layers side-by-side."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if not static_paths:
        return

    fig, axes = plt.subplots(
        1,
        len(static_paths),
        figsize=(5 * len(static_paths), 5),
    )

    if len(static_paths) == 1:
        axes = [axes]

    for ax, (layer_name, static_path) in zip(axes, static_paths.items()):
        if not static_path.exists():
            ax.set_title(layer_name)
            ax.axis("off")
            continue

        try:
            arr, vmin, vmax = static_colour_data(str(static_path))

            if display_mode == "Overlay on S2" and s2_images:
                first_year = sorted(s2_images, key=int)[0]
                ax.imshow(s2_images[first_year])

            if layer_name == "Protected area":
                cmap = ListedColormap([
                    "#d9d9d9",
                    "#1b9e77",
                ])

                norm = BoundaryNorm(
                    [-0.5, 0.5, 1.5],
                    cmap.N,
                )

                masked = np.ma.masked_invalid(arr)

                im = ax.imshow(
                    masked,
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                    alpha=(
                        opacity
                        if display_mode == "Overlay on S2"
                        else 1.0
                    ),
                )

                ax.set_title("Protected area")

                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    ticks=[0, 1],
                    fraction=0.046,
                    pad=0.04,
                )
                cbar.ax.set_yticklabels([
                    "Not protected",
                    "Protected",
                ])

            else:
                # Continuous layer.
                cmap_name = static_colours[layer_name]
                unit = STATIC_LAYER_UNITS.get(layer_name, "")

                im = ax.imshow(
                    arr,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    alpha=(
                        opacity
                        if display_mode == "Overlay on S2"
                        else 1.0
                    ),
                )

                ax.set_title(layer_name)

                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )

                cbar.set_label(
                    f"{layer_name}{f' ({unit})' if unit else ''}"
                )

            ax.axis("off")

        except Exception as exc:
            ax.set_title(layer_name)
            ax.axis("off")
            st.error(f"Could not display {layer_name}: {exc}")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
    ax1.set_ylabel("Annual precipitation")
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
            "Precipitation": _round2(climate[year].get("precip_sum")),
            "Precip min": _round2(climate[year].get("precip_min")),
            "Precip max": _round2(climate[year].get("precip_max")),
            "Mean temperature": _round2(climate[year].get("temp_mean")),
            "Temp min": _round2(climate[year].get("temp_min")),
            "Temp max": _round2(climate[year].get("temp_max")),
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
        st.metric("Forest gain", _fmt(gain_pct, suffix="%"))
        st.write(f"**Biome:** {metadata.get('biome', '—')}")
        st.write(f"**Region:** {metadata.get('region', '—')}")
        st.write(f"**Country:** {metadata.get('country', '—')}")

    with overview_cols[1]:
        st.write("**Soil**")
        st.write(f"SOC: {_fmt(soil.get('soc'))}")
        st.write(f"Clay: {_fmt(soil.get('clay_pct'), suffix='%')}")
        st.write(f"pH: {_fmt(soil.get('ph'))}")

    with overview_cols[2]:
        st.write("**Pseudo labels**")
        st.write(f"Dominant class: {pseudo.get('dominant_class', '—')}")
        st.write(f"Mean confidence: {_fmt(pseudo.get('mean_confidence'))}")

        labelled_fraction = pseudo.get("labelled_gain_pixel_fraction")
        labelled_pct = labelled_fraction * 100 if labelled_fraction is not None else None
        st.write(f"Labelled gain: {_fmt(labelled_pct, suffix='%')}")

    with overview_cols[3]:
        st.write("**Slope**")
        st.write(f"Mean: {_fmt(slope_metadata.get('mean'), suffix='°')}")
        st.write(f"P90: {_fmt(slope_metadata.get('p90'), suffix='°')}")
        st.write(f"Exported: {metadata.get('exported_at', '—')}")

    st.write("**Bounds**")
    bounds_cols = st.columns(4)

    with bounds_cols[0]:
        st.write(f"CRS: {bounds.get('crs', '—')}")

    with bounds_cols[1]:
        st.write(f"X: {_fmt(bounds.get('x_min_m'))} → {_fmt(bounds.get('x_max_m'))}")

    with bounds_cols[2]:
        st.write(f"Y: {_fmt(bounds.get('y_min_m'))} → {_fmt(bounds.get('y_max_m'))}")

    with bounds_cols[3]:
        st.write(f"Lon: {_fmt(bounds.get('min_lon'))} → {_fmt(bounds.get('max_lon'))}")
        st.write(f"Lat: {_fmt(bounds.get('min_lat'))} → {_fmt(bounds.get('max_lat'))}")

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

    with st.expander("Full metadata JSON"):
        st.json(metadata)