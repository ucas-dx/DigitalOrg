# ultralytics/nn/modules/convnextv2.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    # 你的实现：确保该函数返回一个支持多尺度输出的 backbone
    from convnext import convnextv2_backbone  # pip/本地均可，只要可导入
except Exception as e:
    raise ImportError("请确保 convnext.py 在 PYTHONPATH 中，且实现了 convnextv2_backbone(...)") from e


# -------- ConvNeXt GRN 兼容加载（来自你的代码，稍作注解）--------
def safe_load_convnextv2_state_dict(
    backbone: nn.Module, raw_state: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str], List[str]]:
    model_state = backbone.state_dict()
    fixed_state: Dict[str, torch.Tensor] = {}
    converted: List[str] = []
    skipped: List[str] = []
    state = {k.replace("module.", ""): v for k, v in raw_state.items()}
    for k, v in state.items():
        if k not in model_state:
            continue
        dst = model_state[k]
        if v.shape == dst.shape:
            fixed_state[k] = v.to(dtype=dst.dtype)
            continue
        if (".grn.gamma" in k) or (".grn.beta" in k):
            if v.numel() == dst.numel():
                vv = v.reshape(dst.shape).to(dtype=dst.dtype)
                fixed_state[k] = vv
                converted.append(f"{k}: {tuple(v.shape)} -> {tuple(dst.shape)}")
                continue
            if v.shape[-1] == dst.shape[-1]:
                leading_ones = [1] * (len(dst.shape) - 1)
                vv = v.reshape(*leading_ones, dst.shape[-1]).to(dtype=dst.dtype)
                if vv.shape == dst.shape:
                    fixed_state[k] = vv
                    converted.append(f"{k}: {tuple(v.shape)} -> {tuple(dst.shape)}")
                    continue
        skipped.append(f"{k}: {tuple(v.shape)} != {tuple(dst.shape)}")
    missing, unexpected = backbone.load_state_dict(fixed_state, strict=False)
    total_params = sum(param.numel() for param in model_state.values())
    loaded_params = sum(
        model_state[name].numel()
        for name in model_state.keys()
        if name not in missing
    )
    load_ratio = loaded_params / max(1, total_params)
    if converted:
        print(f"[INFO] ConvNeXt GRN-compat loaded ({len(converted)} keys).")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} mismatched keys (kept default init).")
    print(f"[INFO] ConvNeXt load ratio: {loaded_params} / {total_params} = {load_ratio:.2%}")
    return ([str(m) for m in missing], [str(u) for u in unexpected], converted)


def _maybe_strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state


class ConvNeXtV2(nn.Module):
    """
    轻量包装：构建 ConvNeXtV2 骨干并输出多尺度特征序列。
    - variant: 'tiny'/'small'/...（与你的实现保持一致）
    - out_indices: 从骨干取哪些 stage 作为输出（例如 (1,2,3) -> stride 8/16/32）
    - ckpt: 可选权重路径，自动进行 GRN 兼容加载
    - out_channels: 对应各个 out_indices 的通道数（供 YAML/parse_model 做形状推断）
    """

    VARIANT_DIMS = {
        "atto":  (40,  80, 160, 320),
        "femto": (48,  96, 192, 384),
        "pico":  (64, 128, 256, 512),
        "nano":  (80, 160, 320, 640),
        "tiny":  (96, 192, 384, 768),
        "small": (96, 192, 384, 768),
        "base":  (128, 256, 512, 1024),
        "large": (192, 384, 768, 1536),
        "huge":  (352, 704, 1408, 2816),
    }

    def __init__(
        self,
        variant: str = "tiny",
        in_chans: int = 3,
        ckpt: Optional[str] = None,
        out_indices: Sequence[int] = (1, 2, 3),
        out_channels: Optional[Sequence[int]] = None,  # 可显式指定（否则按表推断）
    ):
        super().__init__()
        self.variant = variant
        self.out_indices = tuple(out_indices)
        self.backbone = convnextv2_backbone(variant, in_chans=in_chans)

        if ckpt and Path(ckpt).exists():
            try:
                sd = torch.load(ckpt, map_location="cpu")
                state = sd.get("model", sd.get("state_dict", sd))
                if isinstance(state, dict):
                    state = _maybe_strip_prefix(state, "model_ema.backbone.")
                    state = _maybe_strip_prefix(state, "teacher.backbone.")
                    state = _maybe_strip_prefix(state, "student.backbone.")
                safe_load_convnextv2_state_dict(self.backbone, state)
            except Exception as e:
                print(f"[WARN] load ConvNeXt ckpt failed: {e}")

        dims = self.VARIANT_DIMS.get(variant, self.VARIANT_DIMS["tiny"])
        if out_channels is None:
            out_channels = [dims[i] for i in self.out_indices]
        self.out_channels = tuple(int(c) for c in out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        # 兼容几种返回形式：list/tuple 或 dict
        if isinstance(feats, (list, tuple)):
            return [feats[i] for i in self.out_indices]
        if isinstance(feats, dict):
            # 尝试按 key 排序（如 'stage1','stage2',...）
            keys = list(sorted(feats.keys()))
            return [feats[keys[i]] for i in self.out_indices]
        raise RuntimeError("convnextv2_backbone 应返回序列或字典形式的多尺度特征。")
