#!/usr/bin/env python3
"""Train and evaluate VGG-16 with transfer learning on punch classification data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from ..base.pipeline import train as base_train
from ..base.pipeline import test_only as base_test_only
from ..base.pipeline import validate_only as base_validate_only


class VGG16PunchClassifier(nn.Module):
    """VGG-16 optimized for punch classification data."""

    def __init__(self, num_classes: int, pretrained: bool = True, use_attention: bool = False):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.vgg16 = models.vgg16(weights=weights)
        self.use_attention = use_attention

        # Replace final Linear(4096 → 1000) with dropout + head for punch classes.
        cls = self.vgg16.classifier
        in_features = cls[6].in_features
        cls[6] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg16.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze convolutional backbone and classifier stem; train head."""
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg16.classifier[6].parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.vgg16.parameters():
            param.requires_grad = True


def train(args):
    """Train the VGG-16 model."""
    return base_train(args, VGG16PunchClassifier)


def test_only(args):
    """Test the trained VGG-16 model."""
    return base_test_only(args, VGG16PunchClassifier)


def validate_only(args):
    """Validate the trained VGG-16 model."""
    return base_validate_only(args, VGG16PunchClassifier)
