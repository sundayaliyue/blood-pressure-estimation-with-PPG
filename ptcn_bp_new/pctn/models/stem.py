from torch import nn


class Stem(nn.Module):
    """Shallow feature extraction: Conv1d -> BN -> ReLU -> MaxPool."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 15,
        stride: int = 2,
        padding: int = 7,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.pool(x)
