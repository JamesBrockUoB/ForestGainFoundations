import torch.nn as nn

from .sits_scd import SitsSCD
from .tsvit import TSViT
from .unet_lstm import UNetLSTM


def build_model(
    model_type: str,
    in_channels: int,
    img_size: int,
    **kwargs,
) -> nn.Module:
    """Factory function to instantiate change detection models by string name."""
    model_type = model_type.lower()

    if model_type == "tsvit":
        return TSViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=kwargs.get("patch_size", 16),
            embed_dim=kwargs.get("embed_dim", 128),
            depth_temporal=kwargs.get("depth_temporal", 2),
            depth_spatial=kwargs.get("depth_spatial", 4),
            num_heads=kwargs.get("num_heads", 4),
            max_frames=kwargs.get("max_frames", 8),
        )
    elif model_type == "unet_lstm":
        return UNetLSTM(
            in_channels=in_channels,
            base_features=kwargs.get("base_features", 64),
        )
    elif model_type == "sits_scd":
        return SitsSCD(
            in_channels=in_channels,
            base_features=kwargs.get("base_features", 64),
            num_heads=kwargs.get("num_heads", 4),
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Supported options: ['tsvit', 'unet_lstm', 'sits_scd']"
        )


__all__ = ["TSViT", "UNetLSTM", "SitsSCD", "build_model"]
