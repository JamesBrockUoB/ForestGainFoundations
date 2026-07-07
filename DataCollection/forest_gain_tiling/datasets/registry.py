from __future__ import annotations

import ee


class Datasets:
    def __init__(self) -> None:
        self.esa_wc = ee.Image("ESA/WorldCover/v100/2020")
        self.esa_trees = self.esa_wc.eq(10).unmask(0)

        self.forty = ee.ImageCollection(
            "projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection"
        ).mosaic()

        self.meta_ch = (
            ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
            )
            .mosaic()
            .select("cover_code")
            .unmask(0)
        )

        self.dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")

        self.dt_cover_2017 = (
            ee.Image("projects/symbolic-base-346316/assets/dt_tree_cover_2017_mosaic")
            .select(0)
            .divide(2.55)
            .rename("tree_cover_pct")
        )

        self.dt_cover_2020 = (
            ee.Image("projects/symbolic-base-346316/assets/dt_tree_cover_2020_mosaic")
            .select(0)
            .divide(2.55)
            .rename("tree_cover_pct")
        )

        # p2 (2020->2024) not yet available: the 2024 cover mosaic hasn't
        # been uploaded. Deliberately absent rather than pointed at a
        # guessed asset path — build_gain_layer raises a clear error if
        # something tries to run p2 before this is set.
        self.dt_cover_2024 = None
