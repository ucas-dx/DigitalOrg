"""Connected components utilities used by the video tracker.

原实现优先使用 cc_torch / Triton 加速，并在 CPU 回退时依赖 skimage。
在当前环境中：
  - Triton 扩展存在 PY_SSIZE_T_CLEAN 相关问题；
  - skimage 没有安装。

这里提供一个完全基于 numpy 的 CPU 实现，并在所有情况下走 CPU 路径，
避免对额外 C 扩展和 skimage 的依赖。视频帧数和分辨率在本项目中较小，
这种实现的性能足够。
"""

import logging
from collections import deque

import numpy as np
import torch


def connected_components_cpu_single(values: torch.Tensor):
    """简单的 4-邻域连通域标记，返回 (labels, counts)。"""
    assert values.dim() == 2
    # foreground: non-zero
    fg = values.detach().cpu().numpy() != 0  # (H, W) bool
    H, W = fg.shape
    labels = np.zeros((H, W), dtype=np.int32)
    counts = np.zeros((H, W), dtype=np.int32)

    current_label = 0
    # 4-neighborhood offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(H):
        for x in range(W):
            if not fg[y, x]:
                continue
            if labels[y, x] != 0:
                continue
            # 新连通域，BFS
            current_label += 1
            q = deque()
            q.append((y, x))
            labels[y, x] = current_label
            pixels = [(y, x)]

            while q:
                cy, cx = q.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        continue
                    if not fg[ny, nx]:
                        continue
                    if labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current_label
                    q.append((ny, nx))
                    pixels.append((ny, nx))

            area = len(pixels)
            for py, px in pixels:
                counts[py, px] = area

    labels_t = torch.from_numpy(labels)
    counts_t = torch.from_numpy(counts)
    return labels_t, counts_t


def connected_components_cpu(input_tensor: torch.Tensor):
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert (
            input_tensor.dim() == 3
        ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    batch_size = input_tensor.shape[0]
    if batch_size == 0:
        # 空 batch：直接返回全 0 的 labels / counts
        zeros = torch.zeros(out_shape, dtype=torch.int64, device=input_tensor.device)
        return zeros, zeros.clone()

    labels_list = []
    counts_list = []
    for b in range(batch_size):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


def connected_components(input_tensor: torch.Tensor):
    """
    Computes connected components labeling on a batch of 2D tensors, using the best available backend.

    Args:
        input_tensor (torch.Tensor): A BxHxW integer tensor or Bx1xHxW. Non-zero values are considered foreground. Bool tensor also accepted

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Both tensors have the same shape as input_tensor.
            - A tensor with dense labels. Background is 0.
            - A tensor with the size of the connected component for each pixel.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)

    assert (
        input_tensor.dim() == 4 and input_tensor.shape[1] == 1
    ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    # 统一走纯 CPU 实现，避免依赖 cc_torch / Triton / skimage。
    return connected_components_cpu(input_tensor)
