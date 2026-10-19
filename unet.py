"""Configurable version of the user's four-pool ModifiedUNet (raw logits)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """Two 3x3 convolutions; GroupNorm is stable with batch size one."""
    def __init__(self, in_channels: int, out_channels: int,
                 normalization: str = "group", activation: str = "relu"):
        super().__init__()
        def norm():
            if normalization == "group":
                groups = math.gcd(8, out_channels)
                if groups == out_channels and groups > 1: groups //= 2
                return nn.GroupNorm(groups, out_channels)
            if normalization == "batch": return nn.BatchNorm2d(out_channels)
            if normalization == "none": return nn.Identity()
            raise ValueError(f"Unknown normalization: {normalization}")
        def act():
            if activation == "relu": return nn.ReLU(inplace=False)
            if activation == "silu": return nn.SiLU(inplace=False)
            if activation == "leaky_relu": return nn.LeakyReLU(0.1, inplace=False)
            raise ValueError(f"Unknown activation: {activation}")
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), norm(), act(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), norm(), act())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ModifiedUNet(nn.Module):
    """Original four-pool encoder and transpose-convolution decoder.

    For width f and 2048-square input: f@2048, 2f@1024, 4f@512,
    8f@256, 16f@128. Decoder reverses these through aligned skip connections.
    Channels 0/1 are foreground/separation; a third head is experimental.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 init_features: int = 16, normalization: str = "group",
                 activation: str = "relu", checkpoint_blocks: bool = True):
        super().__init__()
        if min(in_channels, out_channels, init_features) < 1:
            raise ValueError("Channel counts must be positive")
        if checkpoint_blocks and normalization == "batch":
            raise ValueError("BatchNorm statistics update twice with checkpointing; disable it or use GroupNorm")
        self.checkpoint_blocks, self.in_channels = checkpoint_blocks, in_channels
        def conv(a, b): return DoubleConv(a, b, normalization, activation)
        f = init_features
        self.inc = conv(in_channels, f)
        for i in range(1, 5):
            setattr(self, f"down{i}", nn.Sequential(nn.MaxPool2d(2), conv(f*2**(i-1), f*2**i)))
        for i, scale in enumerate((8, 4, 2, 1), 1):
            setattr(self, f"up{i}", nn.ConvTranspose2d(f*scale*2, f*scale, 2, stride=2))
            setattr(self, f"conv_up{i}", conv(f*scale*2, f*scale))
        self.outc = nn.Conv2d(f, out_channels, 1)

    def _block(self, block, *args):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    @staticmethod
    def _match_and_concat(upsampled: torch.Tensor, bypass: torch.Tensor) -> torch.Tensor:
        dy, dx = bypass.shape[-2] - upsampled.shape[-2], bypass.shape[-1] - upsampled.shape[-1]
        upsampled = F.pad(upsampled, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return torch.cat([bypass, upsampled], dim=1)

    def _decode(self, x, skip, index):
        return getattr(self, f"conv_up{index}")(
            self._match_and_concat(getattr(self, f"up{index}")(x), skip))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 16:
            raise ValueError("Expected BCHW input with configured channels and H,W >= 16")
        skips = [self._block(self.inc, x)]
        for i in range(1, 5): skips.append(self._block(getattr(self, f"down{i}"), skips[-1]))
        x = skips[-1]
        for i, skip in enumerate(reversed(skips[:-1]), 1):
            def block(a, b, index=i): return self._decode(a, b, index)
            x = self._block(block, x, skip)
        return self.outc(x)
