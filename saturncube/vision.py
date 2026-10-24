"""Image models. Input is a float array of shape (n, channels, height, width) (PyTorch convention),
(n, height, width) for grayscale, or (n, height, width, channels) with channels_last=True."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel, register


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        self.shortcut = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return self.act(y + self.shortcut(x))


class CNNNet(nn.Module):
    """Stem convolution, `stages` residual stages (channels double and resolution halves after the first),
    global average pooling and a linear head. Works for any image size that survives the downsampling."""

    def __init__(self, in_channels, d_out, width=16, stages=3, blocks_per_stage=1, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
        cin = width
        for s in range(stages):
            cout = width * 2 ** s
            for b in range(blocks_per_stage):
                layers.append(_ResBlock(cin, cout, 2 if (s > 0 and b == 0) else 1))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.pool, self.drop, self.head = nn.AdaptiveAvgPool2d(1), nn.Dropout(dropout), nn.Linear(cin, d_out)

    def forward(self, x):                       # x arrives channels-last (B, H, W, C) from the trainer
        x = self.features(x.permute(0, 3, 1, 2))
        return self.head(self.drop(self.pool(x).flatten(1)))


@register("cnn", family="vision")
class CNN(BaseModel):
    """Compact ResNet-style convolutional network for image classification or regression.

        model = CNN(width=16, stages=3).fit(X_train, y_train)     # X: (n, C, H, W) or (n, H, W)
        model.predict(X_test); model.evaluate(X_test, y_test)

    Inputs are standardised per channel using the training statistics."""
    family = "vision"

    def __init__(self, width=16, stages=3, blocks_per_stage=1, dropout=0.0, channels_last=False, **kw):
        super().__init__(**kw)
        self.width, self.stages, self.blocks_per_stage = width, stages, blocks_per_stage
        self.dropout, self.channels_last = dropout, channels_last

    def _as_x(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:                                    # grayscale (n, H, W)
            return X[..., None]
        if X.ndim == 4:
            return X if self.channels_last else np.ascontiguousarray(X.transpose(0, 2, 3, 1))
        raise ValueError("images must have shape (n, C, H, W) or (n, H, W)")

    def build(self, d_in, d_out):
        return CNNNet(d_in, d_out, self.width, self.stages, self.blocks_per_stage, self.dropout)
