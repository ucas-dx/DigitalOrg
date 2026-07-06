"""
ConvNeXtV2: PyTorch 实现（官方风格、含 GRN），并在 forward 时返回四个 stage 的特征图。

- 输入:  (N, 3, H, W)
- 输出:  (x0, x1, x2, x3)
    x0: stage0 输出，stride=4，通道=dims[0]
    x1: stage1 输出，stride=8，通道=dims[1]
    x2: stage2 输出，stride=16，通道=dims[2]
    x3: stage3 输出，stride=32，通道=dims[3]

可选：提供一个简单的 head（GAP + Linear）用于分类；但默认 forward 返回四个阶段特征，
如需 logits，可调用 .forward_head(x3) 或 model(x, return_logits=True)。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (N,1,1,1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0 / 1
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    """LayerNorm for channels_first (N,C,H,W)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GRN(nn.Module):
    """Global Response Normalization (channels_last: N,H,W,C)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# -----------------------------
# ConvNeXtV2 Block
# -----------------------------
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)  # channels_last LN
        self.pwconv1    = nn.Linear(dim, 4 * dim)
        self.act    = nn.GELU()
        self.grn    = GRN(4 * dim)
        self.pwconv2    = nn.Linear(4 * dim, dim)
        self.drop   = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)            # (N,C,H,W)
        x = x.permute(0, 2, 3, 1)     # -> (N,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)     # -> (N,C,H,W)
        x = shortcut + self.drop(x)
        return x

# -----------------------------
# ConvNeXtV2 Backbone (features)
# -----------------------------
class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXtV2 主干网络，返回四个 stage 的特征图。

    Args:
        in_chans: 输入通道数（默认 3）
        depths:   四个 stage 各自包含的 block 数
        dims:     四个 stage 的通道数
        drop_path_rate: 随机深度率（stochastic depth），沿 block 线性递增

    Forward:
        返回 (x0, x1, x2, x3)
    """
    def __init__(self,
                 in_chans: int = 3,
                 depths: Sequence[int] = (3, 3, 9, 3),
                 dims:   Sequence[int] = (96, 192, 384, 768),
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.depths = list(depths)
        self.dims   = list(dims)

        # stem / downsample_layers
        self.downsample_layers = nn.ModuleList()
        # stem: 4x4 stride=4 conv + LN
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        # 后续三个下采样（2x2 stride=2 conv），前置 LN
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            )

        # stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        idx = 0
        for i in range(4):
            blocks = [ConvNeXtV2Block(dims[i], drop_path=dp_rates[idx + j]) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            idx += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)

    @torch.no_grad()
    def feature_info(self) -> List[dict]:
        """返回四个特征图的 stride 与通道信息。"""
        return [
            {"num_chs": self.dims[0], "reduction": 4,  "module": "stages.0"},
            {"num_chs": self.dims[1], "reduction": 8,  "module": "stages.1"},
            {"num_chs": self.dims[2], "reduction": 16, "module": "stages.2"},
            {"num_chs": self.dims[3], "reduction": 32, "module": "stages.3"},
        ]

    @torch.no_grad()
    def output_strides(self) -> Tuple[int, int, int, int]:
        return (4, 8, 16, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # stage0
        x = self.downsample_layers[0](x)
        x0 = self.stages[0](x)
        # stage1
        x = self.downsample_layers[1](x0)
        x1 = self.stages[1](x)
        # stage2
        x = self.downsample_layers[2](x1)
        x2 = self.stages[2](x)
        # stage3
        x = self.downsample_layers[3](x2)
        x3 = self.stages[3](x)
        return x0, x1, x2, x3

# -----------------------------
# 带分类头的封装（可选）
# -----------------------------
class ConvNeXtV2(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 in_chans: int = 3,
                 depths: Sequence[int] = (3, 3, 9, 3),
                 dims:   Sequence[int] = (96, 192, 384, 768),
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = ConvNeXtV2Backbone(in_chans=in_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate)
        self.num_features = dims[-1]
        self.head_norm = nn.LayerNorm(self.num_features, eps=1e-6)
        self.head = nn.Linear(self.num_features, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def forward_head(self, x3: torch.Tensor) -> torch.Tensor:
        # x3: (N,C,H/32,W/32)
        x = x3.permute(0, 2, 3, 1)                 # N,H,W,C
        x = self.head_norm(x)
        x = x.mean(dim=(1, 2))                     # GAP -> (N,C)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        x0, x1, x2, x3 = self.forward_features(x)
        if return_logits:
            logits = self.forward_head(x3)
            return (x0, x1, x2, x3), logits
        return x0, x1, x2, x3

# -----------------------------
# 预设配置
# -----------------------------
VARIANTS = {
    "atto":  dict(depths=(2, 2, 6, 2),   dims=(40,  80, 160, 320)),
    "femto": dict(depths=(2, 2, 6, 2),   dims=(48,  96, 192, 384)),
    "pico":  dict(depths=(2, 2, 6, 2),   dims=(64, 128, 256, 512)),
    "nano":  dict(depths=(2, 2, 8, 2),   dims=(80, 160, 320, 640)),
    "tiny":  dict(depths=(3, 3, 9, 3),   dims=(96, 192, 384, 768)),
    "small": dict(depths=(3, 3, 27, 3),  dims=(96, 192, 384, 768)),
    "base":  dict(depths=(3, 3, 27, 3),  dims=(128,256, 512,1024)),
    "large": dict(depths=(3, 3, 27, 3),  dims=(192,384, 768,1536)),
    "huge":  dict(depths=(3, 3, 27, 3),  dims=(352,704,1408,2816)),
}

def convnextv2_backbone(variant: str = "tiny", in_chans: int = 3, drop_path_rate: float = 0.0) -> ConvNeXtV2Backbone:
    cfg = VARIANTS[variant]
    return ConvNeXtV2Backbone(in_chans=in_chans, depths=cfg["depths"], dims=cfg["dims"], drop_path_rate=drop_path_rate)


def convnextv2(variant: str = "tiny", num_classes: int = 1000, in_chans: int = 3, drop_path_rate: float = 0.0) -> ConvNeXtV2:
    cfg = VARIANTS[variant]
    return ConvNeXtV2(num_classes=num_classes, in_chans=in_chans, depths=cfg["depths"], dims=cfg["dims"], drop_path_rate=drop_path_rate)

# -----------------------------
# 快速自测
# -----------------------------
if __name__ == "__main__":
    model = convnextv2_backbone("tiny", in_chans=3, drop_path_rate=0.1)
    print(model.state_dict().keys())
    print(torch.load(r'C:\localtask\Point2Org\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth').keys())
    x = torch.randn(2, 3, 224, 224)
    x0, x1, x2, x3 = model(x)
    print("x0:", x0.shape)  # (2, 96, 56, 56)
    print("x1:", x1.shape)  # (2, 192, 28, 28)
    print("x2:", x2.shape)  # (2, 384, 14, 14)
    print("x3:", x3.shape)  # (2, 768, 7, 7)

    # 带分类头：
    clf = convnextv2("tiny", num_classes=1000)
    (f0, f1, f2, f3), logits = clf(torch.randn(2, 3, 224, 224), return_logits=True)
    print("logits:", logits.shape)  # (2, 1000)
