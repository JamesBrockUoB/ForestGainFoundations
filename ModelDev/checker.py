from pathlib import Path

import torch
from config import NUM_INPUT_CHANNELS
from datasets import MultiTemporalGainDataset
from lightning_module import GainDetectionTask
from torch.utils.data import DataLoader

tile_root = Path("../DataCollection/data/test_tiles")
tile_dirs = sorted(
    [p for p in tile_root.iterdir() if p.is_dir() and p.name.endswith("_p1")]
)

ds = MultiTemporalGainDataset(tile_dirs, period="p1")
loader = DataLoader(ds, batch_size=4, shuffle=False)

task = GainDetectionTask(model_type="tsvit", in_channels=NUM_INPUT_CHANNELS)
task.eval()

batch = next(iter(loader))

with torch.no_grad():
    logits = task(batch["pixels"])

print("logits finite:", torch.isfinite(logits).all().item())
print("logits min/max:", logits.min().item(), logits.max().item())
print("logits shape:", logits.shape)

print("gain_mask shape:", batch["gain_mask"].shape)
print("gain_valid sum:", batch["gain_valid"].sum().item())
print(
    "gain_weight min/max:",
    batch["gain_weight"].min().item(),
    batch["gain_weight"].max().item(),
)

loss = task._compute_loss(
    logits, batch["gain_mask"], batch["gain_valid"], batch["gain_weight"]
)
print("loss:", loss.item())
