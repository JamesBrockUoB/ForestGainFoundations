import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM cell for spatio-temporal feature aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape

        if hx is None:
            h_state = x.new_zeros(b, self.hidden_channels, h, w)
            c_state = x.new_zeros(b, self.hidden_channels, h, w)
        else:
            h_state, c_state = hx

        gates = self.conv(torch.cat([x, h_state], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        c_next = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class DoubleConv(nn.Module):
    """`n_convs` x (Conv2D -> BatchNorm -> ReLU), then optional Dropout2d."""

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
                nn.ReLU(inplace=True),
            ]
            c = out_channels
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling -> Concatenation with skip connection -> DoubleConv."""

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
        self.conv = DoubleConv(
            in_channels + skip_channels, out_channels, n_convs, dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetLSTM(nn.Module):
    """U-Net with a (stacked) ConvLSTM bottleneck for N-timestep single binary gain mask output.

    Tunable: base_features (width), depth_temporal (stacked ConvLSTM layers), lstm_kernel
    (ConvLSTM kernel size), convs_per_block (convs in every encoder/decoder block), dropout
    (Dropout2d in conv blocks and between ConvLSTM layers).

    Input:  (B, T, C, H, W)
    Output: (B, H, W) binary change logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        dropout: float = 0.0,
        depth_temporal: int = 1,
        lstm_kernel: int = 3,
        convs_per_block: int = 2,
        **kwargs,
    ):
        super().__init__()
        if base_features < 1:
            raise ValueError("base_features must be positive")
        if depth_temporal < 1:
            raise ValueError("depth_temporal must be at least 1")
        f = base_features
        n, p = convs_per_block, dropout

        # 1. Shared Spatial Encoder Stage
        self.enc1 = DoubleConv(in_channels, f, n, p)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f, f * 2, n, p)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f * 2, f * 4, n, p)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(f * 4, f * 8, n, p)
        self.pool4 = nn.MaxPool2d(2)

        # 2. Bottleneck ConvLSTM stack (processes T frames sequentially)
        self.bottleneck_conv = DoubleConv(f * 8, f * 16, n, p)
        self.conv_lstm = nn.ModuleList(
            ConvLSTMCell(f * 16, f * 16, lstm_kernel) for _ in range(depth_temporal)
        )
        self.lstm_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 3. Spatial Decoder Stages
        self.dec4 = UpBlock(f * 16, f * 8, f * 8, n, p)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4, n, p)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2, n, p)
        self.dec1 = UpBlock(f * 2, f, f, n, p)

        self.out_head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        e1_list, e2_list, e3_list, e4_list = [], [], [], []
        states = [None] * len(self.conv_lstm)

        for step in range(t):
            frame = x[:, step]  # (B, C, H, W)

            e1 = self.enc1(frame)  # (B, f, H, W)
            e2 = self.enc2(self.pool1(e1))  # (B, f*2, H/2, W/2)
            e3 = self.enc3(self.pool2(e2))  # (B, f*4, H/4, W/4)
            e4 = self.enc4(self.pool3(e3))  # (B, f*8, H/8, W/8)

            inp = self.bottleneck_conv(self.pool4(e4))  # (B, f*16, H/16, W/16)

            # Recurrent temporal update through the ConvLSTM stack
            for i, cell in enumerate(self.conv_lstm):
                states[i] = cell(inp, states[i])
                inp = states[i][0]
                if i < len(self.conv_lstm) - 1:
                    inp = self.lstm_drop(inp)

            e1_list.append(e1)
            e2_list.append(e2)
            e3_list.append(e3)
            e4_list.append(e4)

        # Final hidden state of the top ConvLSTM layer carries the sequence history
        bottleneck_fused = states[-1][0]  # (B, f*16, H/16, W/16)

        # Temporal mean pooling across skip connections
        s1 = torch.stack(e1_list, dim=1).mean(dim=1)
        s2 = torch.stack(e2_list, dim=1).mean(dim=1)
        s3 = torch.stack(e3_list, dim=1).mean(dim=1)
        s4 = torch.stack(e4_list, dim=1).mean(dim=1)

        d4 = self.dec4(bottleneck_fused, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits = self.out_head(d1)  # (B, 1, H, W)
        return logits.squeeze(1)  # (B, H, W)
