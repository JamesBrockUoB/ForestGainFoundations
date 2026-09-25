from __future__ import annotations

import ee


class Datasets:
    def __init__(self) -> None:
        self.forty = ee.ImageCollection(
            "projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection"
        ).mosaic()

        self.dt_cover: dict[int, ee.Image | None] = {
            year: (
                ee.Image(
                    f"projects/symbolic-base-346316/assets/dt_tree_cover_{year}_v2"
                )
                .select(0)
                .divide(2.55)
                .rename("tree_cover_pct")
            )
            for year in range(2017, 2025)
        }

    def get_dt_cover(self, year: int) -> ee.Image:
        image = self.dt_cover.get(year)

        if image is None:
            raise RuntimeError(f"Canopy-cover asset for {year} is not available yet")

        return image
