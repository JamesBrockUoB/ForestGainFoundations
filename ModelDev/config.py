"""
Configuration settings aligned with GEE export outputs.
"""

# 1-based band indices mapping directly to exported s1s2_<year>.tif
BACKBONE_BAND_INDICES = {
    "S1_VV": 11,
    "S1_VH": 12,
    "S1_VVVH": 13,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B8A": 8,
    "B11": 9,
    "B12": 10,
}

VALID_MASK_BAND_INDEX = 14  # s2_valid_<year> embedded directly in the GeoTIFF
NUM_INPUT_CHANNELS = len(BACKBONE_BAND_INDICES)

PERIOD_YEARS = {
    "p1": [2017, 2018, 2019, 2020],  # T = 4
    "p2": [2020, 2021, 2022, 2023, 2024],  # T = 5
}

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
