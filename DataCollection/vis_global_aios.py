import json
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/aois"))
VALID_FILE = OUTPUT_DIR / os.getenv("OUTPUT_FILE", "valid_aois.json")
REJECTED_FILE = OUTPUT_DIR / "aoi_filter_rejected.json"
CHECKPOINT = OUTPUT_DIR / "aoi_filter_checkpoint.json"
MAP_FILE = OUTPUT_DIR / "aoi_map.png"

AOI_STEP = float(os.getenv("AOI_STEP", 1.0))

# ── Load valid AOIs ───────────────────────────────────────────────────────────
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        valid_aois = json.load(f)
    print(f"Loaded {len(valid_aois)} valid AOIs from checkpoint")
elif VALID_FILE.exists():
    with open(VALID_FILE) as f:
        valid_aois = json.load(f)
    print(f"Loaded {len(valid_aois)} valid AOIs from output file")
else:
    valid_aois = []
    print("No valid AOI data found yet")

if REJECTED_FILE.exists():
    with open(REJECTED_FILE) as f:
        rejected_ids = set(json.load(f))
    print(f"Loaded {len(rejected_ids)} rejected AOIs")
else:
    rejected_ids = set()
    print("No rejected AOI file found yet")

# ── Reconstruct rejected AOI bounds from IDs ──────────────────────────────────
rejected_aois = []
for rid in rejected_ids:
    try:
        _, lon, lat = rid.split("_", 2)
        lon, lat = float(lon), float(lat)
        rejected_aois.append(
            {
                "minLon": lon,
                "minLat": lat,
                "maxLon": lon + AOI_STEP,
                "maxLat": lat + AOI_STEP,
            }
        )
    except Exception:
        continue

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.set_xlabel("Longitude", color="white")
ax.set_ylabel("Latitude", color="white")
ax.tick_params(colors="white")
ax.set_title(f"Valid: {len(valid_aois)} | Rejected: {len(rejected_ids)}", color="white")

for spine in ax.spines.values():
    spine.set_edgecolor("white")

ax.axhline(0, color="white", linewidth=0.3, alpha=0.3)
ax.axvline(0, color="white", linewidth=0.3, alpha=0.3)

for aoi in rejected_aois:
    rect = patches.Rectangle(
        (aoi["minLon"], aoi["minLat"]),
        aoi["maxLon"] - aoi["minLon"],
        aoi["maxLat"] - aoi["minLat"],
        linewidth=0,
        facecolor="#ff4444",
        alpha=0.4,
    )
    ax.add_patch(rect)

for aoi in valid_aois:
    rect = patches.Rectangle(
        (aoi["minLon"], aoi["minLat"]),
        aoi["maxLon"] - aoi["minLon"],
        aoi["maxLat"] - aoi["minLat"],
        linewidth=0,
        facecolor="#00ff88",
        alpha=0.6,
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig(MAP_FILE, dpi=150, bbox_inches="tight")
print(f"Map saved to {MAP_FILE}")
