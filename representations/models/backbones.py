import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights



class ResNet50Backbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        return features.view(features.size(0), -1)  # flatten


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class BasicCNNBackbone(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        return features.view(features.size(0), -1)  # flatten