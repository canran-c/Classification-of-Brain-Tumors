from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, resnet50, vgg11

try:
    from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, ResNet50_Weights, VGG11_Weights
except ImportError:
    EfficientNet_B0_Weights = None
    ResNet18_Weights = None
    ResNet50_Weights = None
    VGG11_Weights = None


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class SimpleCNNDeeper(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int, use_pretrained: bool):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "simple_cnn_deeper":
        return SimpleCNNDeeper(num_classes=num_classes)

    if model_name == "resnet18":
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            model = resnet18(weights=weights)
        else:
            model = resnet18(pretrained=use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet50":
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.DEFAULT if use_pretrained else None
            model = resnet50(weights=weights)
        else:
            model = resnet50(pretrained=use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "vgg11":
        if VGG11_Weights is not None:
            weights = VGG11_Weights.DEFAULT if use_pretrained else None
            model = vgg11(weights=weights)
        else:
            model = vgg11(pretrained=use_pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        if EfficientNet_B0_Weights is not None:
            weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            model = efficientnet_b0(weights=weights)
        else:
            model = efficientnet_b0(pretrained=use_pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model '{model_name}'. Choose from: simple_cnn, simple_cnn_deeper, resnet18, resnet50, vgg11, efficientnet_b0."
    )
