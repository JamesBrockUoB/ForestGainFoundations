"""
test_aois.py

16-AOI test suite — 4 per class — validating forest-gain typology classification.

Classes:
  0 = agrocrop     (tree crops, pre-gain agriculture)
  1 = nat_regen    (natural regeneration, low human footprint)
  2 = plantation   (commercial timber, even-aged, homogeneous canopy)
  3 = restoration  (conservation/biodiversity planting, marginal land)

Run:
    python test_aois.py
"""

import sys
from dotenv import load_dotenv

load_dotenv("../")

from generate_aois import process_batch

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


def format_cell(bounds):
    minLon, minLat, maxLon, maxLat = bounds
    return {
        "id": f"aoi_{round(minLon,2)}_{round(minLat,2)}",
        "minLon": minLon,
        "minLat": minLat,
        "maxLon": maxLon,
        "maxLat": maxLat,
    }


def run_tests():
    results = []
    passed = 0
    failed = 0
    n = len(TEST_AOIS)

    print("=" * 80)
    print(f"FOREST-GAIN TYPOLOGY TEST SUITE ({n} AOIs × 4 classes)")
    print("=" * 80)

    for i, aoi in enumerate(TEST_AOIS, 1):
        print(f"\n[{i}/{n}] {aoi['name']}")
        print(f"       Expected : {CLASS_NAMES[aoi['expected_class']]}")
        print(f"       Reason   : {aoi['reason']}")
        print(f"       Bounds   : {aoi['bounds']}")
        print("-" * 80)

        try:
            valid_batch, rejected_batch = process_batch([format_cell(aoi["bounds"])])

            if valid_batch:
                r = valid_batch[0]["properties"]
                actual = int(r["hard_class"])
                match = actual == aoi["expected_class"]
                symbol = "✓" if match else "✗"
                status = "PASS" if match else "FAIL"
                if match:
                    passed += 1
                else:
                    failed += 1

                print(
                    f"Result: {symbol} {status} — classified as {CLASS_NAMES[actual]}"
                )
                print("  Scores (independent rankings, not probabilities):")
                for k in [
                    "score_agrocrop",
                    "score_nat_regen",
                    "score_plantation",
                    "score_restoration",
                ]:
                    marker = " ◀" if k == f"score_{CLASS_NAMES[actual]}" else ""
                    print(f"    {k:25s}: {r.get(k, 0):.4f}{marker}")
                print(
                    f"  Forest gain: {r['forest_gain_frac']:.4f}"
                    f" | Mangrove ESA: {r['ev_esa_mangrove']:.4f}"
                    f" | GMW: {r['ev_gmw_frac']:.4f}"
                    f" | CH std: {r.get('ev_ch_std', 0):.2f}"
                )
                print(
                    f"  DW trees std: {r['ev_dw_trees_std']:.4f}"
                    f" | slope: {r['ev_dw_trees_slope']:.4f}"
                    f" | bare pre: {r['ev_dw_bare_pre']:.4f}"
                )

                # Top 5 evidence signals by absolute value
                ev = sorted(
                    {k: v for k, v in r.items() if k.startswith("ev_")}.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
                print("  Top evidence:")
                for k, v in ev:
                    print(f"    {k}: {v:.4f}")

                results.append(
                    {
                        "name": aoi["name"],
                        "expected": aoi["expected_class"],
                        "actual": actual,
                        "match": match,
                        "scores": {
                            k: r.get(k, 0)
                            for k in [
                                "score_agrocrop",
                                "score_nat_regen",
                                "score_plantation",
                                "score_restoration",
                            ]
                        },
                    }
                )

            else:
                # Surface rejection reason from bitfield if available
                rej_props = rejected_batch[0]["properties"] if rejected_batch else {}
                reason_code = int(rej_props.get("rejection_reason", 0))
                from generate_aois import rejection_reason_str

                reason_str = rejection_reason_str(reason_code)
                print(f"Result: ✗ REJECTED — {reason_str}")
                failed += 1
                results.append(
                    {
                        "name": aoi["name"],
                        "expected": aoi["expected_class"],
                        "actual": None,
                        "match": False,
                        "reason": reason_str,
                    }
                )

        except Exception as e:
            print(f"Result: ✗ ERROR — {str(e)[:120]}")
            failed += 1
            results.append(
                {
                    "name": aoi["name"],
                    "expected": aoi["expected_class"],
                    "actual": None,
                    "match": False,
                    "reason": str(e)[:120],
                }
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    classified = [r for r in results if r.get("actual") is not None]
    rejected = [r for r in results if r.get("actual") is None]
    n_class = len(classified)

    print(f"Passed:   {passed}/{n_class} classified")
    print(f"Failed:   {failed}/{n_class} classified")
    print(f"Rejected: {len(rejected)}/{n} (no_gain / insufficient_veg / missing_s2)")
    if n_class:
        print(f"Success rate (classified only): {passed/n_class*100:.1f}%")
    print("\nPer-class performance (classified cells only):")
    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_classified = [r for r in classified if r["expected"] == cls_idx]
        cls_rejected = [r for r in rejected if r["expected"] == cls_idx]
        cls_passed = sum(1 for r in cls_classified if r["match"])
        rej_str = f" ({len(cls_rejected)} rejected)" if cls_rejected else ""
        print(f"  {cls_name:14s}: {cls_passed}/{len(cls_classified)} correct{rej_str}")

    return passed, failed, results


if __name__ == "__main__":
    passed, failed, results = run_tests()
    sys.exit(0 if failed == 0 else 1)
