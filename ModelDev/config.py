"""
Configuration settings aligned with GEE export outputs.
"""

BACKBONE_BAND_INDICES = {
    # Sentinel-2
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B5": 5,
    "B6": 6,
    "B7": 7,
    "B8A": 8,
    "B11": 9,
    "B12": 10,
    # Sentinel-1 (dB scale)
    "S1_VV": 11,
    "S1_VH": 12,
    "S1_VVVH": 13,
}

VALID_MASK_BAND_INDEX = 14
S1_BANDS = ("S1_VV", "S1_VH", "S1_VVVH")
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")

NUM_INPUT_CHANNELS = len(BACKBONE_BAND_INDICES)

YEARS = list(range(2017, 2025))  # 2017 - 2024 inclusive

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
