from torch import nn


class TransformerBranch(nn.Module):
    """Transformer encoder for global temporal dependencies."""

    def __init__(self, in_channels=64,num_layers=6,num_heads=4,mlp_ratio=4,dropout=0.1,) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.downsample = nn.Conv1d(64,512,kernel_size=8,stride=8,padding=4) # 下采样和CNN分支提取的特征维度一样，要基于CNN分支得到的尺寸设计
        self.out_channels = in_channels

    def forward(self, x):
        # (B, C, L) -> (B, L, C) -> encoder -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        return self.downsample(x)
