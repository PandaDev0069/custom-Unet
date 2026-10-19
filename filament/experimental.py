"""Native-resolution U-Net worker; launched with torchrun from the full-image notebook.

No tiles or spatial resampling are used. DDP replicates the model on each GPU;
it does not combine the cards into one larger VRAM pool.
"""
import argparse
from contextlib import nullcontext
import csv
import gc
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        groups = math.gcd(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=False),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class GlobalContext(nn.Module):
    """A pooled image summary makes every output depend on the complete input."""
    def __init__(self, channels):
        super().__init__()
        self.local = DoubleConv(channels, channels, dilation=2)
        self.global_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs):
        local = self.local(inputs)
        return local + self.global_projection(F.adaptive_avg_pool2d(local, 1))


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, 2, stride=2)
        self.conv = DoubleConv(skip_channels * 2, skip_channels)

    def forward(self, inputs, skip):
        up = self.up(inputs)
        dy, dx = skip.shape[-2]-up.shape[-2], skip.shape[-1]-up.shape[-1]
        up = F.pad(up, [0, dx, 0, dy])
        return self.conv(torch.cat((up, skip), dim=1))


class FullImageUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=16,
                 depth=7, max_channels=256, checkpoint_blocks=True):
        super().__init__()
        if min(in_channels, out_channels, base_channels, depth, max_channels) < 1:
            raise ValueError('Channel counts and depth must be positive integers.')
        self.depth = int(depth)
        self.image_size = 2048
        self.checkpoint_blocks = bool(checkpoint_blocks)
        channels = [min(base_channels * 2**index, max_channels)
                    for index in range(self.depth + 1)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for channel in channels[:-1]:
            self.encoders.append(DoubleConv(current, channel))
            current = channel
        self.bottleneck = nn.Sequential(DoubleConv(current, channels[-1]),
                                        GlobalContext(channels[-1]))
        self.decoders = nn.ModuleList()
        current = channels[-1]
        for channel in reversed(channels[:-1]):
            self.decoders.append(DecoderStage(current, channel))
            current = channel
        self.output = nn.Conv2d(current, out_channels, 1)

    def _block(self, block, *inputs):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            # Checkpointing the complete decoder stage also releases up/cat tensors.
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, inputs):
        divisor = 2**self.depth
        if inputs.ndim != 4 or min(inputs.shape[-2:]) < divisor:
            raise ValueError(f'Input must be BCHW, with H,W >= {divisor}.')
        skips = []
        for encoder in self.encoders:
            inputs = self._block(encoder, inputs)
            skips.append(inputs)
            inputs = F.max_pool2d(inputs, 2)
        inputs = self._block(self.bottleneck, inputs)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            inputs = self._block(decoder, inputs, skip)
        return self.output(inputs)


