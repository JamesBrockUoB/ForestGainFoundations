import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalAttentionLayer(nn.Module):
    """Pre-norm residual multi-head self-attention over the T axis of (N, T, C)."""

    def __init__(self, channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, c = x.shape
        y = self.norm(x)

        def split(z):
            return z.view(n, t, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split(self.q(y)),
            split(self.k(y)),
            split(self.v(y)),
            dropout_p=self.dropout if self.training else 0.0,
        )  # (N, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(n, t, c)
        return x + self.drop(self.out_proj(out))


class TemporalAttentionModule(nn.Module):
    """Multi-Head Temporal Attention Module for Satellite Image Time Series.

    Aggregates features across T timesteps per pixel/location with stacked
    attention layers followed by attention-weighted pooling over T.

    Spatial sequences are processed in chunks to avoid oversized
    efficient-attention kernel batch dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_heads < 1 or in_channels % num_heads:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if depth < 1:
            raise ValueError("depth (temporal) must be at least 1")

        self.num_heads = num_heads

        self.layers = nn.ModuleList(
            _TemporalAttentionLayer(in_channels, num_heads, dropout)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # One independent temporal sequence per spatial location.
        # (B, T, C, H, W) -> (B*H*W, T, C)
        out = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)

        n = out.shape[0]

        # Bound the effective attention batch, accounting for head count.
        # The margin also avoids operating exactly at the kernel limit.
        max_sequences = min(8192, 65535 // self.num_heads)

        chunks = []

        for start in range(0, n, max_sequences):
            chunk = out[start : start + max_sequences]

            for layer in self.layers:
                chunk = layer(chunk)

            chunks.append(chunk)

        out = torch.cat(chunks, dim=0)

        # Attention-weighted temporal pooling.
        attn_weights = F.softmax(out.mean(dim=-1), dim=-1).unsqueeze(-1)

        pooled = (out * attn_weights).sum(dim=1)

        return pooled.view(b, h, w, c).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    """`n_convs` x (Conv2D -> BatchNorm -> GELU), then optional Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_convs < 1:
            raise ValueError("n_convs must be at least 1")
        layers = []
        c = in_channels
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(c, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ]
            c = out_channels
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Bilinear Upsampling -> Skip connection concat -> ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        n_convs: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(
            in_channels + skip_channels, out_channels, n_convs, dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SitsSCD(nn.Module):
    """SITS-SCD Architecture for N-timestep single binary change/gain mask output.

    Tunable: base_features (width), num_heads, depth_temporal (stacked temporal attention
    layers per scale), convs_per_block (convs in every encoder/decoder block), dropout
    (Dropout2d in conv blocks, attention/residual dropout in temporal layers).

    Input:  (B, T, C, H, W)
    Output: (B, H, W) binary change logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        depth_temporal: int = 1,
        convs_per_block: int = 2,
        **kwargs,
    ):
        super().__init__()
        if base_features < 1:
            raise ValueError("base_features must be positive")
        f = base_features
        n, p = convs_per_block, dropout

        # 1. Multi-scale Shared Spatial Encoder
        self.enc1 = ConvBlock(in_channels, f, n, p)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f, f * 2, n, p)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f * 2, f * 4, n, p)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(f * 4, f * 8, n, p)
        self.pool4 = nn.MaxPool2d(2)

        # 2. Multi-scale Temporal Attention Modules
        def tae(c):
            return TemporalAttentionModule(c, num_heads, depth_temporal, dropout)

        self.tae1 = tae(f)
        self.tae2 = tae(f * 2)
        self.tae3 = tae(f * 4)
        self.tae4 = tae(f * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, n, p)
        self.tae_bottleneck = tae(f * 16)

        # 3. Progressive Decoder
        self.dec4 = UpBlock(f * 16, f * 8, f * 8, n, p)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4, n, p)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2, n, p)
        self.dec1 = UpBlock(f * 2, f, f, n, p)

        self.final_head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        e1_seq, e2_seq, e3_seq, e4_seq, btn_seq = [], [], [], [], []

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

        e1_stack = torch.stack(e1_seq, dim=1)
        e2_stack = torch.stack(e2_seq, dim=1)
        e3_stack = torch.stack(e3_seq, dim=1)
        e4_stack = torch.stack(e4_seq, dim=1)
        btn_stack = torch.stack(btn_seq, dim=1)

        s1 = self.tae1(e1_stack)
        s2 = self.tae2(e2_stack)
        s3 = self.tae3(e3_stack)
        s4 = self.tae4(e4_stack)
        b_fused = self.tae_bottleneck(btn_stack)

        d4 = self.dec4(b_fused, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits = self.final_head(d1)  # (B, 1, H, W)
        return logits.squeeze(1)  # (B, H, W)
