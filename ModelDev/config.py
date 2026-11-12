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

# ----------------------------------------------------------------- embeddings
EMB_DIR = "embeddings"
AE_BANDS = tuple(f"AE_{i:02d}" for i in range(64))
TESSERA_BANDS = tuple(f"TS_{i:03d}" for i in range(128))

# source -> (file template relative to tile dir, band names)
SOURCES = {
    "s1": ("composites/s1s2_{year}.tif", S1_BANDS),
    "s2": ("composites/s1s2_{year}.tif", S2_BANDS),
    "alphaearth": (f"{EMB_DIR}/aee_{{year}}.tif", AE_BANDS),
    "tessera": (f"{EMB_DIR}/tessera_{{year}}.tif", TESSERA_BANDS),
}
EMBEDDING_SOURCES = ("alphaearth", "tessera")

# 1-based band index of each band within its own file
BAND_FILE_INDEX = {
    **BACKBONE_BAND_INDICES,
    **{b: i + 1 for i, b in enumerate(AE_BANDS)},
    **{b: i + 1 for i, b in enumerate(TESSERA_BANDS)},
}

S2_SCALE = 10000.0  # DN -> reflectance

NORM_CLIP = (-1.0, 2.0)
NORM_NAN_FILL = 0.5

NUM_INPUT_CHANNELS = len(BACKBONE_BAND_INDICES)  # default S1+S2 only

YEARS = list(range(2017, 2025))  # 2017 - 2024 inclusive

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4

EMBEDDING_ABS_MAX = 30.0  # Tessera legit max ~20
