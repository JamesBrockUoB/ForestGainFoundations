from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path

import ee
import rasterio


def chunk_years(years: list[int], size: int) -> list[list[int]]:
    size = max(1, size)
    return [years[i : i + size] for i in range(0, len(years), size)]


def chunk_name(base: str, years: list[int]) -> str:
    """('s1s2', [2017, 2018]) -> 's1s2_2017-2018'. A single year gives the
    plain per-year name ('s1s2_2017'), so size 1 needs no splitting."""
    return f"{base}_{'-'.join(str(y) for y in years)}"


def parse_chunk_name(name: str) -> tuple[str, list[int]]:
    """Returns (base, years); years is empty for names like 'fabdem' or
    'protected_area' that aren't year-suffixed at all."""
    base, sep, year_part = name.rpartition("_")
    if not sep:
        return name, []
    try:
        return base, [int(y) for y in year_part.split("-")]
    except ValueError:
        return name, []


def is_chunked(name: str) -> bool:
    return len(parse_chunk_name(name)[1]) > 1


def expand_product_keys(keys: list[str]) -> list[str]:
    """Task keys -> the per-year files that must exist after splitting."""
    out: list[str] = []
    for key in keys:
        category, name = key.split("/", 1)
        base, years = parse_chunk_name(name)
        if len(years) > 1:
            out.extend(f"{category}/{base}_{y}" for y in years)
        else:
            out.append(key)
    return out


def prefix_bands(image: ee.Image, year: int) -> ee.Image:
    prefix = f"{year}_"
    return image.rename(
        image.bandNames().map(lambda b: ee.String(prefix).cat(ee.String(b)))
    )


def split_chunk_file(path: Path, logger: logging.Logger) -> list[Path]:
    """Split a multi-year bundle into one GeoTIFF per year.

    Streams one block window at a time (all bands), so peak memory is a
    single window, never the full raster. The bundle is deleted only after
    every per-year file has been written. Bands are assigned to years by
    order (each year contributes the same number of bands, year-major), so
    this doesn't depend on band names surviving the export; if names are
    present they're restored with the year prefix stripped.
    """
    base, years = parse_chunk_name(path.stem)

    with rasterio.open(path) as src, ExitStack() as stack:
        if src.count % len(years) != 0:
            raise ValueError(
                f"{path.name}: {src.count} bands don't divide evenly "
                f"into {len(years)} years"
            )
        per_year = src.count // len(years)

        profile = src.profile.copy()
        profile.update(count=per_year)

        out_paths = [path.with_name(f"{base}_{y}.tif") for y in years]
        dsts = [
            stack.enter_context(rasterio.open(p, "w", **profile)) for p in out_paths
        ]

        for i, (year, dst) in enumerate(zip(years, dsts)):
            names = src.descriptions[i * per_year : (i + 1) * per_year]
            for band_idx, desc in enumerate(names, start=1):
                if desc:
                    dst.set_band_description(band_idx, desc.removeprefix(f"{year}_"))

        for _, window in src.block_windows(1):
            data = src.read(window=window)
            for i, dst in enumerate(dsts):
                dst.write(data[i * per_year : (i + 1) * per_year], window=window)

    path.unlink()
    logger.info(f"split {path.name}")
    return out_paths
