import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionModule(nn.Module):
    """Multi-Head Temporal Attention Module for Satellite Image Time Series (SITS).

    Aggregates features across T timesteps per pixel/location.
    """

    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # Reshape for per-pixel temporal sequence attention: (B * H * W, T, C)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)

        q = (
            self.q(x_flat)
            .view(b * h * w, t, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(x_flat)
            .view(b * h * w, t, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(x_flat)
            .view(b * h * w, t, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*H*W, heads, T, T)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B*H*W, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(b * h * w, t, c)
        out = self.out_proj(out)

        # Compute attention-weighted sum over sequence dimension T -> 1
        attn_weights = F.softmax(out.mean(dim=-1), dim=-1).unsqueeze(
            -1
        )  # (B*H*W, T, 1)
        pooled = (out * attn_weights).sum(dim=1)  # (B*H*W, C)

        return pooled.view(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)


class ConvBlock(nn.Module):
    """Double 2D Convolution block with BatchNorm and GELU activations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
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


class UpBlock(nn.Module):
    """Bilinear Upsampling -> Skip connection concat -> ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SitsSCD(nn.Module):
    """SITS-SCD Architecture for N-timestep single binary change/gain mask output.

    Input:  (B, T, C, H, W)
    Output: (B, H, W) binary change logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        f = base_features

        # 1. Multi-scale Shared Spatial Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # 2. Multi-scale Temporal Attention Modules
        self.tae1 = TemporalAttentionModule(f, num_heads=num_heads)
        self.tae2 = TemporalAttentionModule(f * 2, num_heads=num_heads)
        self.tae3 = TemporalAttentionModule(f * 4, num_heads=num_heads)
        self.tae4 = TemporalAttentionModule(f * 8, num_heads=num_heads)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.tae_bottleneck = TemporalAttentionModule(f * 16, num_heads=num_heads)

        # 3. Progressive Decoder
        self.dec4 = UpBlock(in_channels=f * 16, skip_channels=f * 8, out_channels=f * 8)
        self.dec3 = UpBlock(in_channels=f * 8, skip_channels=f * 4, out_channels=f * 4)
        self.dec2 = UpBlock(in_channels=f * 4, skip_channels=f * 2, out_channels=f * 2)
        self.dec1 = UpBlock(in_channels=f * 2, skip_channels=f, out_channels=f)

        self.final_head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        e1_seq, e2_seq, e3_seq, e4_seq, btn_seq = [], [], [], [], []

        # Extract features across timesteps
        for step in range(t):
            frame = x[:, step]

            e1 = self.enc1(frame)  # (B, f, H, W)
            e2 = self.enc2(self.pool1(e1))  # (B, f*2, H/2, W/2)
            e3 = self.enc3(self.pool2(e2))  # (B, f*4, H/4, W/4)
            e4 = self.enc4(self.pool3(e3))  # (B, f*8, H/8, W/8)
            btn = self.bottleneck(self.pool4(e4))  # (B, f*16, H/16, W/16)

            e1_seq.append(e1)
            e2_seq.append(e2)
            e3_seq.append(e3)
            e4_seq.append(e4)
            btn_seq.append(btn)

        # Stack temporal dimensions -> (B, T, C_i, H_i, W_i)
        e1_stack = torch.stack(e1_seq, dim=1)
        e2_stack = torch.stack(e2_seq, dim=1)
        e3_stack = torch.stack(e3_seq, dim=1)
        e4_stack = torch.stack(e4_seq, dim=1)
        btn_stack = torch.stack(btn_seq, dim=1)

        # Temporal Attention Fusion per scale
        s1 = self.tae1(e1_stack)  # (B, f, H, W)
        s2 = self.tae2(e2_stack)  # (B, f*2, H/2, W/2)
        s3 = self.tae3(e3_stack)  # (B, f*4, H/4, W/4)
        s4 = self.tae4(e4_stack)  # (B, f*8, H/8, W/8)
        b_fused = self.tae_bottleneck(btn_stack)  # (B, f*16, H/16, W/16)

        # Decode to full resolution change mask
        d4 = self.dec4(b_fused, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits = self.final_head(d1)  # (B, 1, H, W)
        return logits.squeeze(1)  # (B, H, W)
