import torch
from torch import nn

from .attention import ChannelAttention, SpatialAttention


class FusionBlock(nn.Module):
    """Spatial attention (per branch) -> concat -> channel attention."""

    def __init__(self, channels_cnn: int, channels_trans: int) -> None:
        super().__init__()
        self.spatial_cnn = SpatialAttention()
        self.spatial_trans = SpatialAttention()
        self.channel_attention = ChannelAttention(channels_cnn + channels_trans)
        self.out_channels = channels_cnn + channels_trans

    def forward(self, feat_cnn, feat_trans):
        fused = torch.cat(
            [self.spatial_cnn(feat_cnn), self.spatial_trans(feat_trans)],
            dim=1,
        )
        return self.channel_attention(fused)
