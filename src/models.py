"""
Deep Learning Backbones: ResNet-101 and DenseNet-201
Configured for transfer learning on Knee Osteoporosis Radiographs.
"""

import torch
import torch.nn as nn
from torchvision import models

def build_resnet101(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Constructs ResNet-101 with ImageNet weights and customized classification head.
    """
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weights=weights)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(256, num_classes)
    )
    return model

def build_densenet201(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Constructs DenseNet-201 with ImageNet weights and customized classification head.
    """
    weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
    model = models.densenet201(weights=weights)

    in_features = model.classifier.in_features  # 1920
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(256, num_classes)
    )
    return model

def freeze_backbone(model: nn.Module, model_name: str = "resnet101"):
    """
    Freezes all feature extraction layers, keeping only the final classifier head trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    if "resnet" in model_name.lower():
        for param in model.fc.parameters():
            param.requires_grad = True
    elif "densenet" in model_name.lower():
        for param in model.classifier.parameters():
            param.requires_grad = True

def unfreeze_backbone(model: nn.Module):
    """
    Unfreezes all layers for full end-to-end fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True
