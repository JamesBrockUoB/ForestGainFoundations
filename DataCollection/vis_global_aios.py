import json
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "aois/aoi_filter_checkpoint.json")
AOI_STEP = float(os.getenv("AOI_STEP", 0.25))

CHECKPOINT = OUTPUT_DIR / OUTPUT_FILE
MAP_FILE = CHECKPOINT.parent / "aoi_map.png"

# Must match dominant_class codes in generate_aois.py
CLASS_COLORS = {
    0: "#ffcc00",  # agrocrop
    1: "#00cc66",  # nat_regen
    2: "#0099ff",  # plantation
    3: "#cc44ff",  # restoration
}

CLASS_NAMES = {
    0: "agrocrop",
    1: "nat_regen",
    2: "plantation",
    3: "restoration",
}

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

with open(CHECKPOINT) as f:
    checkpoint = json.load(f)

valid_aois = checkpoint.get("valid", [])
rejected_ids = set(checkpoint.get("rejected", []))

print(f"Loaded checkpoint: {len(valid_aois)} valid | {len(rejected_ids)} rejected")

# Reconstruct rejected AOI bounds from IDs ("aoi_{minLon}_{minLat}")
rejected_aois = []
for rid in rejected_ids:
    try:
        parts = rid.split("_")  # ["aoi", lon, lat]  (lon/lat may be negative)
        # Handle negative coords: "aoi_-60.0_-3.0" → ["aoi", "", "60.0", "", "3.0"]
        # Safer: strip prefix and split on the second underscore-before-digit boundary
        stripped = rid[len("aoi_") :]  # "-60.0_-3.0"  or  "10.0_-3.0"
        # Find split point: last underscore that is preceded by a digit or dot
        idx = None
        for i in range(len(stripped) - 1, 0, -1):
            if stripped[i] == "_" and stripped[i - 1] not in ("_",):
                idx = i
                break
        if idx is None:
            raise ValueError("could not find split point")
        lon = float(stripped[:idx])
        lat = float(stripped[idx + 1 :])
        rejected_aois.append(
            {
                "minLon": lon,
                "minLat": lat,
                "maxLon": round(lon + AOI_STEP, 4),
                "maxLat": round(lat + AOI_STEP, 4),
            }
        )
    except Exception as e:
        print(f"Could not parse AOI id '{rid}': {e}")

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")

ax.set_xlabel("Longitude", color="white")
ax.set_ylabel("Latitude", color="white")
ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_edgecolor("white")

ax.axhline(0, color="white", linewidth=0.3, alpha=0.3)
ax.axvline(0, color="white", linewidth=0.3, alpha=0.3)

# Rejected cells
for aoi in rejected_aois:
    ax.add_patch(
        patches.Rectangle(
            (aoi["minLon"], aoi["minLat"]),
            aoi["maxLon"] - aoi["minLon"],
            aoi["maxLat"] - aoi["minLat"],
            linewidth=0,
            facecolor="#ff4444",
            alpha=0.35,
        )
    )

# Valid cells — coloured by dominant class
class_counts = {k: 0 for k in CLASS_NAMES}

for aoi in valid_aois:
    cls = int(aoi.get("dominant_class", 3))
    class_counts[cls] = class_counts.get(cls, 0) + 1
    ax.add_patch(
        patches.Rectangle(
            (aoi["minLon"], aoi["minLat"]),
            aoi["maxLon"] - aoi["minLon"],
            aoi["maxLat"] - aoi["minLat"],
            linewidth=0,
            facecolor=CLASS_COLORS.get(cls, "#ffffff"),
            alpha=0.85,
        )
    )

ax.set_title(
    f"Valid: {len(valid_aois)} | Rejected: {len(rejected_ids)}",
    color="white",
)

# Legend
legend_items = [
    patches.Patch(
        color=CLASS_COLORS[cls],
        label=f"{name} ({class_counts.get(cls, 0)})",
    )
    for cls, name in CLASS_NAMES.items()
]
legend_items.append(
    patches.Patch(color="#ff4444", label=f"rejected ({len(rejected_ids)})")
)

ax.legend(
    handles=legend_items,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    facecolor="#1a1a2e",
    edgecolor="white",
    labelcolor="white",
)

plt.tight_layout()
MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(MAP_FILE, dpi=150, bbox_inches="tight")
print(f"Map saved to {MAP_FILE}")
