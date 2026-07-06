# ultralytics/nn/modules/convnextv1.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union
import re

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """LayerNorm for channels_first (N,C,H,W)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtV1Block(nn.Module):
    """Standard ConvNeXt block without GRN."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale is not None else None
        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop(x)
        return x


class ConvNeXtV1Backbone(nn.Module):
    """ConvNeXt-V1 backbone returning four-stage feature maps."""

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stage_blocks = []
        idx = 0
        for i, depth in enumerate(depths):
            blocks = [
                ConvNeXtV1Block(dims[i], drop_path=dp_rates[idx + j], layer_scale=layer_scale)
                for j in range(depth)
            ]
            stage_blocks.append(nn.Sequential(*blocks))
            idx += depth
        self.stages = nn.ModuleList(stage_blocks)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)

    @torch.no_grad()
    def feature_info(self) -> List[dict]:
        return [
            {"num_chs": self.dims[0], "reduction": 4, "module": "stages.0"},
            {"num_chs": self.dims[1], "reduction": 8, "module": "stages.1"},
            {"num_chs": self.dims[2], "reduction": 16, "module": "stages.2"},
            {"num_chs": self.dims[3], "reduction": 32, "module": "stages.3"},
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.downsample_layers[0](x)
        x0 = self.stages[0](x)

        x = self.downsample_layers[1](x0)
        x1 = self.stages[1](x)

        x = self.downsample_layers[2](x1)
        x2 = self.stages[2](x)

        x = self.downsample_layers[3](x2)
        x3 = self.stages[3](x)
        return x0, x1, x2, x3


class ConvNeXtV1(nn.Module):
    """Wrapper returning multi-scale features from ConvNeXt-V1 backbone."""

    VARIANT_DIMS = {
        "atto": (40, 80, 160, 320),
        "femto": (48, 96, 192, 384),
        "pico": (64, 128, 256, 512),
        "nano": (80, 160, 320, 640),
        "tiny": (96, 192, 384, 768),
        "small": (96, 192, 384, 768),
        "base": (128, 256, 512, 1024),
        "large": (192, 384, 768, 1536),
        "huge": (352, 704, 1408, 2816),
    }

    VARIANT_DEPTHS = {
        "atto": (2, 2, 6, 2),
        "femto": (2, 2, 6, 2),
        "pico": (2, 2, 6, 2),
        "nano": (2, 2, 8, 2),
        "tiny": (3, 3, 9, 3),
        "small": (3, 3, 27, 3),
        "base": (3, 3, 27, 3),
        "large": (3, 3, 27, 3),
        "huge": (3, 3, 27, 3),
    }

    def __init__(
        self,
        variant: str = "tiny",
        in_chans: int = 3,
        ckpt: Optional[str] = None,
        out_indices: Sequence[int] = (1, 2, 3),
        out_channels: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.out_indices = tuple(out_indices)

        depths = self.VARIANT_DEPTHS.get(variant, self.VARIANT_DEPTHS["tiny"])
        dims = self.VARIANT_DIMS.get(variant, self.VARIANT_DIMS["tiny"])
        self.backbone = ConvNeXtV1Backbone(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale=layer_scale,
        )

        if ckpt and Path(ckpt).exists():
            try:
                sd = torch.load(ckpt, map_location="cpu")
                state = sd.get("model", sd.get("state_dict", sd))
                if isinstance(state, dict):
                    # Strip common nesting/prefix patterns
                    for pref in (
                        "model_ema.backbone.",
                        "teacher.backbone.",
                        "student.backbone.",
                        "backbone.",
                        "model.backbone.",
                        "model.",
                        "module.backbone.",
                        "module.",
                    ):
                        state = _maybe_strip_prefix(state, pref)
                safe_load_convnextv1_state_dict(self.backbone, state)
            except Exception as exc:
                print(f"[WARN] load ConvNeXt checkpoint failed: {exc}")

        if out_channels is None:
            out_channels = [dims[i] for i in self.out_indices]
        self.out_channels = tuple(int(c) for c in out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            return [feats[i] for i in self.out_indices]
        if isinstance(feats, dict):
            keys = list(sorted(feats.keys()))
            return [feats[keys[i]] for i in self.out_indices]
        raise RuntimeError("convnextv1_backbone should return list/tuple/dict of stage features.")


def safe_load_convnextv1_state_dict(
    backbone: nn.Module, raw_state: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str], List[str]]:
    """Safely load a variety of ConvNeXt-V1 checkpoints into this backbone.

    - Strips common prefixes (DDP/module, model/backbone wrappers).
    - Remaps OLS-style key names to this implementation's names.
    - Squeezes 1x1 conv weights for pwconv Linear layers.
    Falls back to partial loading with informative reporting.
    """
    model_state = backbone.state_dict()
    model_keys = set(model_state.keys())

    # 1) basic de-parallelization
    state = {k.replace("module.", ""): v for k, v in raw_state.items()}

    # 2) try additional prefix patterns if present
    for pref in (
        "model_ema.backbone.",
        "teacher.backbone.",
        "student.backbone.",
        "backbone.",
        "model.backbone.",
        "model.",
        "module.backbone.",
    ):
        if any(k.startswith(pref) for k in state):
            state = {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
            break

    # 3) build a remapped alternative for OLS-style naming
    remapped = _remap_ols_to_ultralytics(state)

    # choose mapping with better key overlap
    hits_orig = sum(1 for k in state if k in model_keys)
    hits_remap = sum(1 for k in remapped if k in model_keys)
    chosen = remapped if hits_remap >= hits_orig else state

    # 4) final pass: adjust pwconv weights from Conv(1x1) to Linear
    fixed_state: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in chosen.items():
        if k not in model_state:
            continue
        dst = model_state[k]
        vv = v
        if isinstance(vv, torch.Tensor) and vv.ndim == 4 and vv.shape[-2:] == (1, 1) and (
            k.endswith(".pwconv1.weight") or k.endswith(".pwconv2.weight")
        ):
            vv = vv.squeeze(-1).squeeze(-1)
        if isinstance(vv, torch.Tensor) and vv.shape == dst.shape:
            fixed_state[k] = vv.to(dtype=dst.dtype)
        else:
            skipped.append(f"{k}: {tuple(v.shape)} != {tuple(dst.shape)}")

    missing, unexpected = backbone.load_state_dict(fixed_state, strict=False)

    total_params = sum(param.numel() for param in model_state.values())
    loaded_params = sum(model_state[name].numel() for name in model_state.keys() if name not in missing)
    ratio = loaded_params / max(1, total_params)
    if skipped:
        print(f"[WARN] skipped {len(skipped)} mismatched keys (kept default init).")
    print(f"[INFO] ConvNeXt-V1 load ratio: {loaded_params} / {total_params} = {ratio:.2%}")
    return ([str(m) for m in missing], [str(u) for u in unexpected], skipped)


def _maybe_strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state


def _remap_ols_to_ultralytics(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap common OLS/ConvNeXt naming to this repo's ConvNeXtV1 names.

    Notes:
    - OLS uses two 3x3 convs for stem (downsample x4). This impl uses a single 4x4/4 conv.
      We only map stem LayerNorm (stem.2 -> downsample_layers.0.1) and skip stem.0/1 weights.
    - MLP 1x1 conv weights are reshaped later when loading.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        # blocks: stages.X.blocks.Y.*  -> stages.X.Y.*
        nk = re.sub(r"^stages\.(\d+)\.blocks\.(\d+)\.", r"stages.\1.\2.", nk)
        # depthwise conv
        nk = nk.replace(".conv_dw.", ".dwconv.")
        # MLP -> pointwise conv linears
        nk = nk.replace(".mlp.fc1.", ".pwconv1.")
        nk = nk.replace(".mlp.fc2.", ".pwconv2.")
        # inter-stage downsample blocks
        nk = re.sub(r"^stages\.(\d+)\.downsample\.0\.", r"downsample_layers.\1.0.", nk)
        nk = re.sub(r"^stages\.(\d+)\.downsample\.1\.", r"downsample_layers.\1.1.", nk)
        # stem norm -> first downsample norm
        nk = nk.replace("stem.2.", "downsample_layers.0.1.")
        # skip stem convs that don't have a direct equivalent
        if nk.startswith("stem.0.") or nk.startswith("stem.1."):
            continue
        out[nk] = v
    return out


__all__ = ["ConvNeXtV1", "ConvNeXtV1Backbone", "safe_load_convnextv1_state_dict"]
