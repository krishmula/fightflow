#!/usr/bin/env python3
"""Train and evaluate a baseline CNN on folder-based frame classification data."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from ..base.pipeline import train as base_train
from ..base.pipeline import test_only as base_test_only
from ..base.pipeline import validate_only as base_validate_only

class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

def train(args: argparse.Namespace) -> None:
    base_train(args, BaselineCNN)

def test_only(args: argparse.Namespace) -> None:
    base_test_only(args, BaselineCNN)

def validate_only(args: argparse.Namespace) -> None:
    base_validate_only(args, BaselineCNN)
