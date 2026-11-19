import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> GELU, repeated n_convs times, then optional Dropout2d."""

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
        channels = in_channels
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ]
            channels = out_channels
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Bilinear upsampling to the skip's size -> skip concatenation -> ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        n_convs: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels + skip_channels, out_channels, n_convs, dropout
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SitsSCD(nn.Module):
    """Multi-temporal U-Net with bottleneck temporal attention shared across scales.

    A shared encoder runs on every frame. Temporal attention is computed once at the
    bottleneck (per location, over T, with ordinal position encoding), upsampled, and used
    to mix the frame features at every scale. Only the first and last frame are decoded
    (they are all the output head uses), and their features are fused as
    [first, last, last - first].

    Input:  (B, T, C, H, W), T >= 2, H and W divisible by 16
    Output: (B, H, W) change logits

    num_heads and depth_temporal are accepted for config compatibility and ignored.
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        dropout: float = 0.0,
        convs_per_block: int = 2,
        attention_channels: int = 64,
    ):
        super().__init__()
        if base_features < 1:
            raise ValueError("base_features must be positive")

        f = base_features
        n, p = convs_per_block, dropout

        # Shared spatial encoder
        self.enc1 = ConvBlock(in_channels, f, n, p)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f, f * 2, n, p)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f * 2, f * 4, n, p)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(f * 4, f * 8, n, p)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16, n, p)

        # Bottleneck temporal query / key projections
        self.query = nn.Conv2d(f * 16, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(f * 16, attention_channels, kernel_size=1)

        # Shared decoder
        self.dec4 = UpBlock(f * 16, f * 8, f * 8, n, p)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4, n, p)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2, n, p)
        self.dec1 = UpBlock(f * 2, f, f, n, p)

        # Ordered endpoint fusion
        self.output_head = nn.Sequential(
            ConvBlock(f * 3, f, n, p),
            nn.Conv2d(f, 1, kernel_size=1),
        )

    def positional_encoding(self, x, positions=None):
        """Sinusoidal encoding of frame positions, added to (B, T, C, H, W).

        positions: optional (T,) tensor, e.g. acquisition day offsets as in the paper.
        Defaults to ordinal indices 0..T-1.
        """
        _, T, C, _, _ = x.shape
        if positions is None:
            position = torch.arange(T, device=x.device, dtype=x.dtype)
        else:
            position = positions.to(device=x.device, dtype=x.dtype)
        div_term = torch.exp(
            torch.arange(0, C, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / C)
        )
        encoding = torch.zeros(T, C, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(position[:, None] * div_term)
        if C > 1:
            n_cos = encoding[:, 1::2].shape[1]
            encoding[:, 1::2] = torch.cos(position[:, None] * div_term[:n_cos])
        return x + encoding[None, :, :, None, None]

    def compute_attention(self, x):
        """Temporal attention at every bottleneck location. x: (B, T, C, h, w) -> (B, h*w, T, T)."""
        B, T, C, h, w = x.shape
        flat = x.reshape(B * T, C, h, w)
        q = self.query(flat)
        k = self.key(flat)
        d = q.shape[1]
        q = q.reshape(B, T, d, h * w).permute(0, 3, 1, 2)  # (B, hw, T, d)
        k = k.reshape(B, T, d, h * w).permute(0, 3, 1, 2)
        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        return torch.softmax(attention, dim=-1)

    @staticmethod
    def resize_attention(attention, source_size, target_size):
        """Bilinearly resample (B, h*w, Tq, Tk) attention to a new spatial size.
        Rows stay convex combinations, so they still sum to 1."""
        B, _, Tq, Tk = attention.shape
        sh, sw = source_size
        th, tw = target_size
        if (sh, sw) == (th, tw):
            return attention
        a = attention.permute(0, 2, 3, 1).reshape(B * Tq * Tk, 1, sh, sw)
        a = F.interpolate(a, size=(th, tw), mode="bilinear", align_corners=False)
        return a.reshape(B, Tq, Tk, th * tw).permute(0, 3, 1, 2)

    @staticmethod
    def mix(features, attention):
        """features (B, T, C, h, w), attention (B, h*w, Tq, T) -> (B, Tq, C, h, w)."""
        B, T, C, h, w = features.shape
        f = features.reshape(B, T, C, h * w).permute(0, 3, 1, 2)  # (B, hw, T, C)
        out = torch.matmul(attention, f)  # (B, hw, Tq, C)
        Tq = out.shape[2]
        return out.permute(0, 2, 3, 1).reshape(B, Tq, C, h, w)

    def forward(self, x, positions=None):
        B, T, C, H, W = x.shape
        if T < 2:
            raise ValueError("SitsSCD needs at least 2 frames")
        if H % 16 or W % 16:
            raise ValueError(f"H, W ({H}, {W}) must be divisible by 16")

        # Shared encoder over all frames
        frames = x.reshape(B * T, C, H, W)
        e1 = self.enc1(frames)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        btn = self.bottleneck(self.pool4(e4))

        def restore(z):
            return z.reshape(B, T, z.shape[1], z.shape[2], z.shape[3])

        e1, e2, e3, e4, btn = map(restore, (e1, e2, e3, e4, btn))

        btn = self.positional_encoding(btn, positions)

        # Attention from position-encoded bottleneck features; values stay unencoded
        source_size = btn.shape[-2:]
        attention = self.compute_attention(btn)
        attention = attention[
            :, :, [0, T - 1], :
        ]  # rows for the first and last frame only

        def fuse(features):
            att = self.resize_attention(attention, source_size, features.shape[-2:])
            return self.mix(features, att)  # (B, 2, C, h, w)

        def flat(z):
            return z.reshape(B * 2, z.shape[2], z.shape[3], z.shape[4])

        b, s4, s3, s2, s1 = (fuse(z) for z in (btn, e4, e3, e2, e1))

        d = self.dec4(flat(b), flat(s4))
        d = self.dec3(d, flat(s3))
        d = self.dec2(d, flat(s2))
        d = self.dec1(d, flat(s1))
        d = d.reshape(B, 2, d.shape[1], H, W)

        first, last = d[:, 0], d[:, 1]
        features = torch.cat([first, last, last - first], dim=1)
        return self.output_head(features).squeeze(1)
