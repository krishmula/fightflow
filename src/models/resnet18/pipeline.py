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


class ResNet18PunchClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        # Original: 512 input features → 1000 classes
        # New: 512 input features → num_classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout for regularization
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def freeze_backbone(self):
        """Freeze all layers except the final classifier for initial training."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
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