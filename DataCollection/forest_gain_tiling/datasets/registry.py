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

        self.glulc_2015 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2015").select([0])
        self.glulc_2020 = ee.Image("projects/glad/GLCLU2020/v2/LCLUC_2020").select([0])

        self.tree_classes: ee.List = ee.List.sequence(25, 96).cat(
            ee.List.sequence(125, 196)
        )
