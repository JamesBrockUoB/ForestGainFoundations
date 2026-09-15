import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """Collapses temporal sequence (B * N_patches, T, Dim) -> (B * N_patches, Dim)

    using learned temporal attention weights across N input frames.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)


class ProgressiveUpBlock(nn.Module):
    """2x Progressive upsampling block to prevent patch boundary artifacts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ProgressiveDecoder(nn.Module):
    """Progressively decodes low-res spatial patch grid (B, Dim, H/P, W/P)

    to full resolution change map (B, 1, H, W).
    """

    def __init__(
        self, embed_dim: int = 128, patch_size: int = 16, out_channels: int = 1
    ):
        super().__init__()
        num_up_stages = int(math.log2(patch_size))

        layers = []
        curr_dim = embed_dim
        for _ in range(num_up_stages):
            next_dim = max(32, curr_dim // 2)
            layers.append(ProgressiveUpBlock(curr_dim, next_dim))
            curr_dim = next_dim

        self.up_blocks = nn.Sequential(*layers)
        self.final_head = nn.Conv2d(curr_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.up_blocks(x)
        return self.final_head(feat)


class TSViT(nn.Module):
    """Time-Series Vision Transformer (TSViT) for N-timestep single binary gain mask output.

    Input:  (B, T, C, H, W) where T = N variable sequence length
    Output: (B, H, W) binary change logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth_temporal: int = 2,
        depth_spatial: int = 4,
        num_heads: int = 4,
        max_frames: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.h_p = img_size // patch_size
        self.w_p = img_size // patch_size
        self.num_patches = self.h_p * self.w_p

        # Patch Projection
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional Embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Factorized Encoders
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=depth_temporal
        )
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=depth_spatial
        )

        # Progressive Decoder Head
        self.decoder = ProgressiveDecoder(
            embed_dim=embed_dim, patch_size=patch_size, out_channels=1
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.temporal_pos_embed, std=0.02)
        nn.init.normal_(self.spatial_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        x_flat = x.view(b * t, c, h, w)
        patches = self.patch_embed(x_flat)
        patches = patches.flatten(2).transpose(1, 2)

        # 1. Temporal Attention across N sequence frames
        temporal_tokens = patches.view(b, t, self.num_patches, -1)
        temporal_tokens = (
            temporal_tokens.permute(0, 2, 1, 3)
            .contiguous()
            .view(b * self.num_patches, t, -1)
        )
        temporal_tokens = temporal_tokens + self.temporal_pos_embed[:, :t, :]
        temporal_features = self.temporal_transformer(temporal_tokens)
        pooled_temporal = self.temporal_pool(temporal_features)

        # 2. Spatial Attention across Patch Grid
        spatial_tokens = pooled_temporal.view(b, self.num_patches, -1)
        spatial_tokens = spatial_tokens + self.spatial_pos_embed
        spatial_features = self.spatial_transformer(spatial_tokens)

        # 3. Progressive Decoder Head
        spatial_grid = spatial_features.transpose(1, 2).view(b, -1, self.h_p, self.w_p)
        logits = self.decoder(spatial_grid)
        return logits.squeeze(1)
