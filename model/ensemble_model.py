import torch
import torch.nn as nn
import torchvision.models as models


class FeatureConcatEnsemble(nn.Module):
    """
    Feature concatenation ensemble:
    ResNet50 + EfficientNetB0 + DenseNet121
    → concat features → Dense head → 3-class softmax
    """

    def __init__(self, num_classes=3, freeze_backbones=True):
        super().__init__()

        # ── ResNet50 ──────────────────────────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        resnet_out  = 2048

        # ── EfficientNetB0 ────────────────────────────────────
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet = eff.features   # keep only feature extractor
        self.eff_pool     = nn.AdaptiveAvgPool2d(1)
        eff_out           = 1280

        # ── DenseNet121 ───────────────────────────────────────
        dense = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.densenet  = dense.features
        self.dense_pool = nn.AdaptiveAvgPool2d(1)
        dense_out       = 1024

        # Freeze backbones for phase 1
        if freeze_backbones:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            for param in self.densenet.parameters():
                param.requires_grad = False

        # ── Fusion head ───────────────────────────────────────
        combined_dim = resnet_out + eff_out + dense_out  # 4352
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # ResNet
        x1 = self.resnet(x)
        x1 = x1.view(x1.size(0), -1)           # (B, 2048)

        # EfficientNet
        x2 = self.efficientnet(x)
        x2 = self.eff_pool(x2)
        x2 = x2.view(x2.size(0), -1)           # (B, 1280)

        # DenseNet
        x3 = self.densenet(x)
        x3 = torch.relu(x3)
        x3 = self.dense_pool(x3)
        x3 = x3.view(x3.size(0), -1)           # (B, 1024)

        # Concatenate and classify
        combined = torch.cat([x1, x2, x3], dim=1)  # (B, 4352)
        return self.head(combined)

    def unfreeze_top_layers(self, n=30):
        """Unfreeze last n layers of each backbone for Phase 2 fine-tuning."""
        for backbone in [self.resnet, self.efficientnet, self.densenet]:
            layers = list(backbone.parameters())
            for param in layers[-n:]:
                param.requires_grad = True
        print(f"✅ Unfroze last {n} parameter groups per backbone")