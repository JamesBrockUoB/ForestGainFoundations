from __future__ import annotations

import ee


class Datasets:
    def __init__(self) -> None:
        self.esa_wc = ee.Image("ESA/WorldCover/v100/2020")
        self.esa_trees = self.esa_wc.eq(10).unmask(0)

        self.forty = ee.ImageCollection(
            "projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection"
        ).mosaic()

        # p1 (2017->2020): yearly tree-cover mosaics, all uploaded.
        self.dt_cover: dict[int, ee.Image | None] = {
            year: (
                ee.Image(
                    f"projects/symbolic-base-346316/assets/dt_tree_cover_{year}_mosaic"
                )
                .select(0)
                .divide(2.55)
                .rename("tree_cover_pct")
            )
            for year in [2017, 2018, 2019, 2020]
        }

        # p2 (2020->2024): not yet uploaded. Uncomment and fill in the
        # real asset IDs once available — do not guess a path or point
        # these at the 2020 mosaic as a placeholder, since that would
        # silently produce zero gain everywhere for p2 rather than
        # failing loudly.
        #
        # for year in [2021, 2022, 2023, 2024]:
        #     self.dt_cover[year] = (
        #         ee.Image(
        #             f"projects/symbolic-base-346316/assets/dt_tree_cover_{year}_mosaic"
        #         )
        #         .select(0)
        #         .divide(2.55)
        #         .rename("tree_cover_pct")
        #     )
