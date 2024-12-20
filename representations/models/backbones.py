import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b1, EfficientNet_B1_Weights


class BaseBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        pretrained: bool
    ):
        super(BaseBackbone, self).__init__()

        net = self.get_net(weights=self.pretrained_weights if pretrained else None)
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(self.cnn_feature_dim, self.feature_dim)
    
    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        return self.fc(flattened)


class EfficientNetBackbone(BaseBackbone):
    @property
    def get_net(self):
        return efficientnet_b1

    @property
    def weights(self):
        return EfficientNet_B1_Weights.DEFAULT
    
    @property
    def cnn_feature_dim(self):
        return 1280


class ResNet50Backbone(BaseBackbone):
    @property
    def get_net(self):
        return resnet50

    @property
    def weights(self):
        return ResNet50_Weights.DEFAULT
    
    @property
    def cnn_feature_dim(self):
        return 2048
