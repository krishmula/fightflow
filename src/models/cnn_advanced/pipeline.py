#!/usr/bin/env python3
"""Train and evaluate an advanced CNN on folder-based frame classification data."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from ..base.pipeline import train as base_train
from ..base.pipeline import test_only as base_test_only
from ..base.pipeline import validate_only as base_validate_only

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Initial block: Larger kernel to capture broader motion context
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Deeper residual blocks for better feature extraction
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128, 256, stride=2)
        self.res3 = ResidualBlock(256, 512, stride=2)
        
        # Attention mechanism for focusing on motion-relevant features
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        
        # Enhanced feature processing
        self.features = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # More sophisticated classifier with better regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Apply attention
        att_weights = self.attention(x)
        att_weights = att_weights.view(x.size(0), 512, 1, 1)
        x = x * att_weights
        
        x = self.features(x)
        return self.classifier(x)

def train(args: argparse.Namespace) -> None:
    base_train(args, AdvancedCNN)

def test_only(args: argparse.Namespace) -> None:
    base_test_only(args, AdvancedCNN)

def validate_only(args: argparse.Namespace) -> None:
    base_validate_only(args, AdvancedCNN)
