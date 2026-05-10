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


class ChannelAttention(nn.Module):
    """Channel Attention Module (SE-style) for focusing on motion-relevant features."""
    
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

        # ResNet18 final layer features
        num_features = self.resnet.fc.in_features
        
        # Add channel attention before the global average pool
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = ChannelAttention(num_features)
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Flag to keep Batch Normalization layers in eval mode
        self.freeze_bn = False

    def train(self, mode: bool = True):
        """Override train to keep BatchNorm layers in eval mode if freeze_bn is True."""
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional features (after attention and pooling)."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.resnet.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features
        x = self.get_features(x)
        # Final classification
        x = self.resnet.fc(x)
        return x

    def freeze_backbone(self):
        """Freeze all layers except the final classifier for initial training."""
        self.freeze_bn = True # Keep BN frozen during initial phase
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        if self.use_attention:
            for param in self.attention.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning, keeping BatchNorm frozen for stability."""
        self.freeze_bn = True # Keep BN frozen even during fine-tuning for better stability
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

    def unfreeze_partial(self):
        """Surgical unfreeze: Only Layer 4 and Attention (keeps BN frozen)."""
        self.freeze_bn = True
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze Layer 4
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        # Unfreeze classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
        # Unfreeze attention
        if self.use_attention:
            for param in self.attention.parameters():
                param.requires_grad = True
                
        # Ensure BatchNorm stays frozen even in Layer 4
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False


def train(args):
    """Train the ResNet-18 model."""
    return base_train(args, ResNet18PunchClassifier)


def test_only(args):
    """Test the trained ResNet-18 model."""
    return base_test_only(args, ResNet18PunchClassifier)


def validate_only(args):
    """Validate the trained ResNet-18 model."""
    return base_validate_only(args, ResNet18PunchClassifier)