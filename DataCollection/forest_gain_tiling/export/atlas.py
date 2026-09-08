from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ee
import rasterio
from config import settings
from rasterio.transform import Affine


@dataclass(frozen=True)
class AtlasCell:
    tile_id: str
    row: int
    col: int
    x_offset_px: int
    y_offset_px: int
    cell_x_min_m: float
    cell_y_min_m: float


@dataclass(frozen=True)
class AtlasLayout:
    cells: list[AtlasCell]
    cols: int
    rows: int
    tile_pixels: int
    canvas_width_px: int
    canvas_height_px: int

    def cell_for(self, tile_id: str) -> AtlasCell:
        for c in self.cells:
            if c.tile_id == tile_id:
                return c
        raise KeyError(f"{tile_id} not in this atlas layout")


def build_atlas_layout(tiles: list[dict]) -> AtlasLayout:
    """
    Assigns each tile a unique, non-overlapping cell in a synthetic
    packing grid, purely in pixel space. Has nothing to do with the
    tiles' real-world adjacency — batched tiles can be arbitrarily far
    apart / stratified across the globe. The atlas's own georeferencing
    is discarded after export and replaced per-tile with each tile's
    real geometry at split time (see split_atlas_file).
    """
    tile_pixels = settings.tile_pixels
    tile_size_m = settings.tile_size_m
    n = len(tiles)
    if n == 0:
        raise ValueError("build_atlas_layout requires at least one tile")

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cells = []
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        cell_x_min_m = col * tile_size_m
        cell_y_max_m = -row * tile_size_m
        cell_y_min_m = cell_y_max_m - tile_size_m
        cells.append(
            AtlasCell(
                tile_id=tile["tile_id"],
                row=row,
                col=col,
                x_offset_px=col * tile_pixels,
                y_offset_px=row * tile_pixels,
                cell_x_min_m=cell_x_min_m,
                cell_y_min_m=cell_y_min_m,
            )
        )

    return AtlasLayout(
        cells=cells,
        cols=cols,
        rows=rows,
        tile_pixels=tile_pixels,
        canvas_width_px=cols * tile_pixels,
        canvas_height_px=rows * tile_pixels,
    )


def atlas_crs_transform() -> list[float]:
    """Affine transform for the atlas canvas — origin (0,0) in
    settings.crs_wkt, north-up. Not a real place; a packing frame only."""
    scale = settings.scale
    return [scale, 0, 0, 0, -scale, 0]


def atlas_region_for(layout: AtlasLayout) -> ee.Geometry:
    width_m = layout.canvas_width_px * settings.scale
    height_m = layout.canvas_height_px * settings.scale
    return ee.Geometry.Rectangle(
        [0, -height_m, width_m, 0], proj=settings.crs_wkt, geodesic=False
    )


def build_atlas_image(
    tiles: list[dict],
    layout: AtlasLayout,
    image_builder: Callable[[dict], ee.Image],
) -> ee.Image:
    """
    image_builder(tile) -> an ee.Image already built over that tile's
    real geometry (composite, static layer, label, whatever). Each
    tile's image is shifted via .translate() so its real bounding box
    lands exactly on its assigned atlas cell, then all shifted images
    are mosaicked — safe since cells never overlap by construction.

    NOTE: verify .translate()'s sign convention against a small 2-tile
    test batch before running this at scale (split the result and diff
    against normal single-tile exports for the same two tiles).
    """
    translated = []
    for tile in tiles:
        cell = layout.cell_for(tile["tile_id"])
        try:
            img = image_builder(tile)
        except Exception:
            # One bad tile shouldn't take down the whole batch's export —
            # skip its cell; the caller falls that tile back to a normal
            # single-tile export afterward.
            continue

        dx = cell.cell_x_min_m - tile["x_min_m"]
        dy = cell.cell_y_min_m - tile["y_min_m"]

        translated.append(img.translate(dx, dy, "meters", proj=settings.crs_wkt))

    if not translated:
        raise RuntimeError("build_atlas_image: every tile's builder failed")

    return ee.ImageCollection(translated).mosaic()


def submit_atlas_export(
    tiles: list[dict],
    layout: AtlasLayout,
    image_builder: Callable[[dict], ee.Image],
    *,
    batch_id: str,
    category: str,
    name: str,
) -> ee.batch.Task:
    """
    One export task covering every tile in `tiles`. Filename is keyed by
    batch_id, not tile_id, since it holds many tiles — split_atlas_file()
    is what turns it back into normal per-tile files after download.
    """
    image = build_atlas_image(tiles, layout, image_builder).toFloat()
    prefix = f"atlas__{batch_id}__{category}__{name}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=prefix,
        folder=settings.drive_folder,
        fileNamePrefix=prefix,
        region=atlas_region_for(layout),
        crs=settings.crs_wkt,
        crsTransform=atlas_crs_transform(),
        maxPixels=10_000_000_000_000,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def split_atlas_file(
    atlas_path: Path,
    layout: AtlasLayout,
    tiles_by_id: dict[str, dict],
    output_paths: Callable[[str], Path],
) -> None:
    """Slice a downloaded atlas GeoTIFF into one real, correctly
    georeferenced file per tile. output_paths(tile_id) -> destination Path.

    Reads one tile-sized window at a time rather than the whole atlas
    into memory -- the atlas canvas grows with batch size (cols x rows
    x tile_pixels, times band count), so a full src.read() can easily
    hit multiple GB per file and is the main OOM risk in batched export.
    """
    from rasterio.windows import Window

    tile_pixels = layout.tile_pixels
    scale = settings.scale

    with rasterio.open(atlas_path) as src:
        profile = src.profile.copy()

        for cell in layout.cells:
            tile = tiles_by_id[cell.tile_id]
            window = Window(
                cell.x_offset_px, cell.y_offset_px, tile_pixels, tile_pixels
            )
            block = src.read(window=window)  # only this tile's pixels

            real_transform = Affine(
                scale, 0, tile["x_min_m"], 0, -scale, tile["y_max_m"]
            )

            out_profile = profile.copy()
            out_profile.update(
                height=tile_pixels, width=tile_pixels, transform=real_transform
            )

            dest = output_paths(cell.tile_id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dest, "w", **out_profile) as dst:
                dst.write(block)
