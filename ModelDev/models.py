import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from terratorch.registry import BACKBONE_REGISTRY

from .config import (
    BACKBONE_BAND_INDICES,
    BACKBONE_NAME,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    NUM_CLASSES,
)


class MultiTaskGainModel(nn.Module):
    """LoRA-adapted Prithvi backbone with a segmentation head and an
    optional classification head. Two independent instances are built,
    one for p1 (seg + cls) and one for p2 (seg only) -- see train.py.

    num_frames isn't a constructor argument: Prithvi's positional
    encoding is 3D sin/cos, computed from input shape rather than a
    learned fixed-T parameter, and the official TerraTorch implementation
    doesn't constrain timestep count -- so the same model class handles
    p1's T=4 or p2's T=5 batches without any special-casing, even though
    the two periods get separate trained instances.

    include_classification_head=False skips building the classification
    head entirely for the p2 model, since it would never be supervised.

    The segmentation head is a minimal two-layer conv decoder over
    last-layer tokens, not TerraTorch's UperNetDecoder -- swap that in
    once you've confirmed the neck/decoder import paths against your
    installed version.
    """

    def __init__(
        self,
        backbone_name=BACKBONE_NAME,
        num_classes=NUM_CLASSES,
        include_classification_head=True,
        freeze_backbone_except_lora=True,
    ):
        super().__init__()

        self.include_classification_head = include_classification_head

        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=True,
            bands=list(BACKBONE_BAND_INDICES.keys()),
        )

        if freeze_backbone_except_lora:
            lora_cfg = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            # peft freezes everything except LoRA params by default;
            # confirm with self.backbone.print_trainable_parameters()

        embed_dim = getattr(self.backbone, "embed_dim", 1024)  # 1024 for the 300M model

        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        if include_classification_head:
            self.cls_head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Linear(256, num_classes),
            )
        else:
            self.cls_head = None

    def forward(self, pixels: torch.Tensor):
        # pixels: (B, C, T, H, W) -- confirm this matches your installed
        # backbone's expected signature before trusting shapes downstream.
        tokens = self.backbone.forward_features(pixels)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        b, n, d = tokens.shape
        h = w = int(n**0.5)  # assumes a square token grid; adjust if not
        spatial = tokens.transpose(1, 2).reshape(b, d, h, w)

        seg_logits = self.seg_head(spatial)
        seg_logits = F.interpolate(
            seg_logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False
        )
        seg_logits = seg_logits.squeeze(1)

        if self.cls_head is None:
            return seg_logits, None

        pooled = tokens.mean(dim=1)
        cls_logits = self.cls_head(pooled)
        return seg_logits, cls_logits
