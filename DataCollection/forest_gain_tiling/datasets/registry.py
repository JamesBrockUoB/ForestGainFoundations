from __future__ import annotations

import ee


class Datasets:
    def __init__(self) -> None:
        self.esa_wc = ee.Image("ESA/WorldCover/v100/2020")
        self.esa_trees = self.esa_wc.eq(10).unmask(0)
        self.esa_crop = self.esa_wc.eq(40).unmask(0)

        self.gem_treecrop = (
            ee.ImageCollection(
                "projects/sat-io/open-datasets/GEM-Forest/GEM-Forest_2020"
            )
            .mosaic()
            .select("b1")
            .eq(2)
            .unmask(0)
            .toFloat()
        )

        self.jrc = ee.Image("JRC/GFC2020_subtypes/V1")

        self.nat_forest = (
            ee.ImageCollection(
                "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
            )
            .mosaic()
            .select("B0")
            .divide(250)
            .unmask(0)
        )

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
