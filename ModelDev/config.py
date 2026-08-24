from DataCollection.forest_gain_tiling.config import settings

CLASS_NAMES = ["AGROCROP", "NAT_REGEN", "PLANTATION", "PLANTED"]
NUM_CLASSES = len(CLASS_NAMES)

BACKBONE_BAND_INDICES = {
    # name -> 1-based band index in composites/s1s2_<year>.tif
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8A": 8,
    "B11": 9,
    "B12": 10,
}

PERIOD_YEARS = {
    "p1": [2017, 2018, 2019, 2020],
    "p2": [2020, 2021, 2022, 2023, 2024],
}
PERIOD_HAS_TYPOLOGY = {
    "p1": True,
    "p2": False,
}

BACKBONE_NAME = "prithvi_eo_v2_300"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["qkv", "proj"]
