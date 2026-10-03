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

S2_SCALE = 10000.0  # DN -> reflectance

NORM_STATS: dict[str, tuple[float, float]] = {
    "B2": (0.012300, 0.668800),
    "B3": (0.018900, 0.593800),
    "B4": (0.014200, 0.600300),
    "B5": (0.021950, 0.618900),
    "B6": (0.022467, 0.598000),
    "B7": (0.023360, 0.569150),
    "B8": (0.021492, 0.579740),
    "B8A": (0.022500, 0.546600),
    "B11": (0.011675, 0.299050),
    "B12": (0.009650, 0.228000),
    "S1_VV": (-19.533228, -6.894751),
    "S1_VH": (-26.175932, -13.108449),
    "S1_VVVH": (4.263482, 9.991885),
}

NORM_CLIP = (-1.0, 2.0)
NORM_NAN_FILL = 0.5  # normalised value used for NaN/inf pixels (mid-range)

NUM_INPUT_CHANNELS = len(BACKBONE_BAND_INDICES)

YEARS = list(range(2017, 2025))  # 2017 - 2024 inclusive

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
