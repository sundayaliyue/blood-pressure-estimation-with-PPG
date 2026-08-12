from torch import nn


class Regressor(nn.Module):
    """Global average pooling + two FC layers -> [SBP, DBP]."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, output_dim: int = 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        return self.head(x)
