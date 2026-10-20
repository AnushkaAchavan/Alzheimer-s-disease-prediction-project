import torch
import torch.nn as nn
import torchvision.models as models


class FeatureConcatEnsemble(nn.Module):
    """
    Ensemble architecture from the source GitHub project.

    The backbones are created without downloading pretrained weights so the
    app can initialize in offline environments before loading your .pth file.
    """

    def __init__(self, num_classes: int = 3, freeze_backbones: bool = False):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet_out = 2048

        efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet = efficientnet.features
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        eff_out = 1280

        densenet = models.densenet121(weights=None)
        self.densenet = densenet.features
        self.dense_pool = nn.AdaptiveAvgPool2d(1)
        dense_out = 1024

        if freeze_backbones:
            for backbone in (self.resnet, self.efficientnet, self.densenet):
                for param in backbone.parameters():
                    param.requires_grad = False

        combined_dim = resnet_out + eff_out + dense_out
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_features = self.resnet(x).view(x.size(0), -1)

        efficientnet_features = self.efficientnet(x)
        efficientnet_features = self.eff_pool(efficientnet_features).view(x.size(0), -1)

        densenet_features = self.densenet(x)
        densenet_features = torch.relu(densenet_features)
        densenet_features = self.dense_pool(densenet_features).view(x.size(0), -1)

        combined = torch.cat(
            [resnet_features, efficientnet_features, densenet_features],
            dim=1,
        )
        return self.head(combined)
