from torch import nn
from torchvision.models.resnet import Bottleneck

class Bottleneck1D(Bottleneck):
    """ResNet bottleneck adapted for 1D signals."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__(inplanes, planes, stride=stride, downsample=downsample, norm_layer=nn.BatchNorm1d,)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)


class CNNBranch(nn.Module):
    """ResNet-50 style pyramid CNN for local feature extraction."""

    def __init__(self, in_channels: int = 64, layers=None, channels=None) -> None:
        super().__init__()
        layers = layers or [3, 4, 6, 3]
        channels = channels or [64, 128, 256, 512]

        self.layer1 = self._make_layer(in_channels, channels[0], layers[0])
        self.layer2 = self._make_layer(channels[0] * 4, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1] * 4, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2] * 4, channels[3], layers[3], stride=2)
        self.out_channels = channels[3] * 4

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * Bottleneck.expansion),
            )

        block_layers = [Bottleneck1D(inplanes, planes, stride, downsample)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(blocks - 1):
            block_layers.append(Bottleneck1D(inplanes, planes))
        return nn.Sequential(*block_layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
