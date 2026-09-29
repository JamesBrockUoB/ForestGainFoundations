from __future__ import annotations

import ee


class Datasets:
    def __init__(self) -> None:
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

        self.dynamic_world = "GOOGLE/DYNAMICWORLD/V1"
        self.esri_lulc = (
            "projects/sat-io/open-datasets/landcover/" "ESRI_Global-LULC_10m_TS"
        )

        self.sentinel_2 = "COPERNICUS/S2_SR_HARMONIZED"
        self.cloud_score_plus = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
        self.sentinel_1 = "COPERNICUS/S1_GRD"
        self.satellite_embedding = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

        self.soilgrids_soc = "projects/soilgrids-isric/soc_mean"
        self.soilgrids_clay = "projects/soilgrids-isric/clay_mean"
        self.soilgrids_ph = "projects/soilgrids-isric/phh2o_mean"

        self.era5_land_monthly = "ECMWF/ERA5_LAND/MONTHLY_AGGR"

        self.fabdem = "projects/sat-io/open-datasets/FABDEM"
        self.wdpa = "WCMC/WDPA/201707/polygons"

    def get_dt_cover(self, year: int) -> ee.Image:
        image = self.dt_cover.get(year)

        if image is None:
            raise RuntimeError(f"Canopy-cover asset for {year} is not available yet")

        return image
