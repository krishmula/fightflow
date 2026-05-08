#!/usr/bin/env python3
"""Train and evaluate ResNet-18 with transfer learning on punch classification data."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torchvision.models as models

from ..base.pipeline import train as base_train
from ..base.pipeline import test_only as base_test_only
from ..base.pipeline import validate_only as base_validate_only


class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on motion-relevant regions."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class ResNet18PunchClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, use_attention: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # Original: 512 input features → 1000 classes
        # New: 512 input features → num_classes
        num_features = self.resnet.fc.in_features
        
        # Add spatial attention before classifier if enabled
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SpatialAttention(num_features)
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout for regularization
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from ResNet backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            # Reshape for attention (add spatial dimensions)
            b, c = x.shape
            h, w = int(c**0.5), int(c**0.5)
            if h * w == c:  # Only apply if we can reshape to spatial
                x_spatial = x.view(b, c, 1, 1)
                x_spatial = self.attention(x_spatial)
                x = x_spatial.view(b, c)
        
        # Final classification
        x = self.resnet.fc(x)
        return x

    def freeze_backbone(self):
        """Freeze all layers except the final classifier for initial training."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        if self.use_attention:
            for param in self.attention.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True


def train(args):
    """Train the ResNet-18 model."""
    return base_train(args, ResNet18PunchClassifier)


def test_only(args):
    """Test the trained ResNet-18 model."""
    return base_test_only(args, ResNet18PunchClassifier)


def validate_only(args):
    """Validate the trained ResNet-18 model."""
    return base_validate_only(args, ResNet18PunchClassifier)