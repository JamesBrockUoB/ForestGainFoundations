CLASS_NAMES = {0: "agrocrop", 1: "nat_regen", 2: "plantation", 3: "restoration"}

TEST_AOIS = [
    # ── AGROCROP / AGROFORESTRY ────────────────────────────────────────────────
    # Signals: high GEM tree-crop, high pre/post-gain DW crops, ESA/ESRI cropland
    {
        "name": "Côte d'Ivoire cocoa",
        "bounds": [-6.65, 5.72, -6.56, 5.81],
        "expected_class": 0,
        "reason": "Cocoa agroforestry belt — high pre-gain DW crops, GEM tree-crop signal",
    },
    {
        "name": "Yunnan rubber (China)",
        "bounds": [100.85, 21.85, 100.94, 21.94],
        "expected_class": 0,
        "reason": "Rubber expansion into agricultural transition zone, strong tree-crop prior",
    },
    {
        "name": "Sumatra oil palm (Indonesia)",
        "bounds": [103.20, 0.50, 103.29, 0.59],
        "expected_class": 0,
        "reason": "Industrial oil palm — maximum GEM tree-crop signal, high ESA cropland adjacency",
    },
    {
        "name": "Ghana cocoa-forest mosaic",
        "bounds": [-2.10, 6.80, -2.01, 6.89],
        "expected_class": 0,
        "reason": "Cocoa-forest transition zone, persistent DW crops signal through gain period",
    },
    # ── NATURAL REGENERATION ───────────────────────────────────────────────────
    # Signals: high nat_forest, high DW trees stddev, JRC nat_regen, low pre-gain crops
    {
        "name": "Western Amazon (Peru)",
        "bounds": [-75.05, -2.05, -74.96, -1.96],
        "expected_class": 1,
        "reason": "Dense primary / nat-regen, very high nat_forest probability, tall heterogeneous canopy",
    },
    {
        "name": "Congo Basin (DRC)",
        "bounds": [24.40, 0.40, 24.49, 0.49],
        "expected_class": 1,
        "reason": "Intact tropical nat-regen, high naturalness, low human footprint, high DW trees stddev",
    },
    {
        "name": "Borneo lowland forest (Malaysia)",
        "bounds": [117.20, 4.50, 117.29, 4.59],
        "expected_class": 1,
        "reason": "Secondary nat-regen adjacent to primary forest, high naturalness and canopy heterogeneity",
    },
    {
        "name": "Atlantic Forest secondary (Brazil)",
        "bounds": [-42.55, -19.55, -42.46, -19.46],
        "expected_class": 1,
        "reason": "Secondary Atlantic Forest, JRC nat_regen dominant, high temporal DW stddev",
    },
    # ── PLANTATION ────────────────────────────────────────────────────────────
    # Signals: high JRC planted + GNPF planted, low nat_forest, low CH stddev, mid CH mean
    {
        "name": "Araucania pine (Chile)",
        "bounds": [-72.85, -37.85, -72.76, -37.76],
        "expected_class": 2,
        "reason": "Industrial pine, even-aged stand, high JRC planted, low canopy stddev",
    },
    {
        "name": "New Zealand radiata pine",
        "bounds": [176.20, -38.30, 176.29, -38.21],
        "expected_class": 2,
        "reason": "Pinus radiata plantation, homogeneous mid-height canopy, low naturalness",
    },
    {
        "name": "Landes maritime pine (France)",
        "bounds": [-0.90, 44.30, -0.81, 44.39],
        "expected_class": 2,
        "reason": "Largest European plantation — even-aged maritime pine, very low CH stddev",
    },
    {
        "name": "Uruguay eucalyptus",
        "bounds": [-57.20, -32.50, -57.11, -32.41],
        "expected_class": 2,
        "reason": "Industrial eucalyptus, high GNPF planted, uniform mid canopy, flat terrain",
    },
    # ── RESTORATION ───────────────────────────────────────────────────────────
    # Signals: high slope, high CH stddev, high pre-gain bare/shrub, positive DW slope,
    #          JRC planted but with degraded baseline distinguishing from commercial
    {
        "name": "Loess Plateau restoration (China)",
        "bounds": [109.35, 36.45, 109.44, 36.54],
        "expected_class": 3,
        "reason": "Terraced afforestation on severely degraded slopes — high slope, high bare pre-gain",
    },
    {
        "name": "Ethiopian highlands restoration",
        "bounds": [38.20, 9.00, 38.29, 9.09],
        "expected_class": 3,
        "reason": "Community restoration on degraded highland, short young canopy, steep terrain",
    },
    {
        "name": "Mozambique coastal restoration",
        "bounds": [35.30, -17.20, 35.39, -17.11],
        "expected_class": 3,
        "reason": "Conservation planting on degraded coastal land, bare pre-gain baseline, rising DW trees",
    },
    {
        "name": "Pakistan northern reforestation",
        "bounds": [73.50, 34.80, 73.59, 34.89],
        "expected_class": 3,
        "reason": "Billion Tree programme — steep terrain, short mixed canopy, strong positive DW slope",
    },
]
