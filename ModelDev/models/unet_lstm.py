import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM cell for spatio-temporal feature aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape

        if hx is None:
            h_state = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
            c_state = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
        else:
            h_state, c_state = hx

        combined = torch.cat([x, h_state], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i_gate = torch.sigmoid(i)
        f_gate = torch.sigmoid(f)
        o_gate = torch.sigmoid(o)
        g_candidate = torch.tanh(g)

        c_next = f_gate * c_state + i_gate * g_candidate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling -> Concatenation with skip connection -> DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetLSTM(nn.Module):
    """U-Net with ConvLSTM Bottleneck for N-timestep single binary gain mask output.

    Input:  (B, T, C, H, W)
    Output: (B, H, W) binary change logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        **kwargs,
    ):
        super().__init__()
        f = base_features

        # 1. Shared Spatial Encoder Stage
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # 2. Bottleneck ConvLSTM (processes T frames sequentially)
        self.bottleneck_conv = DoubleConv(f * 8, f * 16)
        self.conv_lstm = ConvLSTMCell(in_channels=f * 16, hidden_channels=f * 16)

        # 3. Spatial Decoder Stages
        self.dec4 = UpBlock(in_channels=f * 16, skip_channels=f * 8, out_channels=f * 8)
        self.dec3 = UpBlock(in_channels=f * 8, skip_channels=f * 4, out_channels=f * 4)
        self.dec2 = UpBlock(in_channels=f * 4, skip_channels=f * 2, out_channels=f * 2)
        self.dec1 = UpBlock(in_channels=f * 2, skip_channels=f, out_channels=f)

        self.out_head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        e1_list, e2_list, e3_list, e4_list = [], [], [], []
        hx = None

        # --- Temporal Forward Pass through Encoder & ConvLSTM ---
        for step in range(t):
            frame = x[:, step]  # (B, C, H, W)

            # Spatial Encoding
            e1 = self.enc1(frame)  # (B, f, H, W)
            e2 = self.enc2(self.pool1(e1))  # (B, f*2, H/2, W/2)
            e3 = self.enc3(self.pool2(e2))  # (B, f*4, H/4, W/4)
            e4 = self.enc4(self.pool3(e3))  # (B, f*8, H/8, W/8)

            bottleneck = self.bottleneck_conv(self.pool4(e4))  # (B, f*16, H/16, W/16)

            # Recurrent temporal update
            hx = self.conv_lstm(bottleneck, hx)

            # Store skip connections across timesteps
            e1_list.append(e1)
            e2_list.append(e2)
            e3_list.append(e3)
            e4_list.append(e4)

        # Final ConvLSTM hidden state carries temporal sequence history
        bottleneck_fused = hx[0]  # (B, f*16, H/16, W/16)

        # Temporal Mean Pooling across skip connections
        s1 = torch.stack(e1_list, dim=1).mean(dim=1)  # (B, f, H, W)
        s2 = torch.stack(e2_list, dim=1).mean(dim=1)  # (B, f*2, H/2, W/2)
        s3 = torch.stack(e3_list, dim=1).mean(dim=1)  # (B, f*4, H/4, W/4)
        s4 = torch.stack(e4_list, dim=1).mean(dim=1)  # (B, f*8, H/8, W/8)

        # --- Spatial Decoding to Binary Change Mask ---
        d4 = self.dec4(bottleneck_fused, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits = self.out_head(d1)  # (B, 1, H, W)
        return logits.squeeze(1)  # (B, H, W)
