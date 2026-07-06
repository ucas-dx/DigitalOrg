from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .util import ROIRefineNet


@torch.no_grad()
def build_sam3_inst_model(
    bpe_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[object, Sam3Processor]:
    """
    Build SAM3 image model with instance interactivity enabled (SAM1-task mode),
    and wrap it with Sam3Processor.
    """
    device_t = torch.device(device)

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model, device=device_t)
    return model, processor


@torch.no_grad()
def infer_one_sam3_inst(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    boxes: List[List[float]],
) -> List[np.ndarray]:
    """
    Pure SAM3 + geometric box prompts (no text), via model.predict_inst.
    Returns a list of uint8 masks (H,W).
    """
    if len(boxes) == 0:
        return []

    image = Image.open(image_path).convert("RGB")

    # 禁用 autocast，强制在 float32 上跑，避免 dtype 冲突
    if torch.cuda.is_available():
        autocast_ctx = torch.cuda.amp.autocast
    else:
        class _DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        autocast_ctx = lambda enabled=False: _DummyCtx()

    with autocast_ctx(enabled=False):
        state = processor.set_image(image)
        boxes_np = np.array(boxes, dtype=np.float32)
        masks_np, scores_np, _ = model.predict_inst(
            state,
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False,
        )

    sam_masks = [(m.squeeze(0).astype(np.uint8) * 255) for m in masks_np]
    return sam_masks


@torch.no_grad()
def infer_one_sam3_with_refine(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    boxes: List[List[float]],
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    End-to-end: SAM3 coarse via predict_inst + ConvNeXtV1 refine.
    Returns (sam_masks, refine_masks).
    """
    device_t = torch.device(device)
    image = Image.open(image_path).convert("RGB")
    orig_H, orig_W = image.height, image.width

    # 确保模型和 processor 在同一个 device
    model.to(device_t)
    processor.device = device_t

    # coarse SAM3 masks
    sam_masks = infer_one_sam3_inst(model, processor, image_path, boxes)

    # build refine net
    refine_net = ROIRefineNet(
        variant="atto", convnext_ckpt=None, use_prompt=True, build_prompt=False
    )
    if refine_ckpt is not None and Path(refine_ckpt).exists():
        ckpt = torch.load(refine_ckpt, map_location="cpu")
        state_dict = ckpt.get("refine", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        refine_net.load_state_dict(state_dict, strict=False)
    refine_net.to(device_t).eval()

    # 1024×1024 canvas
    img_1024 = image.resize((1024, 1024))
    img_np_1024 = np.array(img_1024)
    img_bchw_1024 = (
        torch.from_numpy(img_np_1024)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device_t)
    )

    # scale boxes to 1024
    boxes_1024 = []
    for b in boxes:
        x1, y1, x2, y2 = b
        sx, sy = 1024.0 / orig_W, 1024.0 / orig_H
        bx1, bx2 = x1 * sx, x2 * sx
        by1, by2 = y1 * sy, y2 * sy
        w = bx2 - bx1
        h = by2 - by1
        px = w * roi_pad
        py = h * roi_pad
        bx1 = max(0.0, bx1 - px)
        by1 = max(0.0, by1 - py)
        bx2 = min(1024.0, bx2 + px)
        by2 = min(1024.0, by2 + py)
        boxes_1024.append([bx1, by1, bx2, by2])
    boxes_1024_t = torch.tensor(boxes_1024, device=device_t, dtype=torch.float32)

    # coarse prob 256 from sam_masks
    sam_masks_tensor = (
        torch.stack(
            [torch.from_numpy((m > 0).astype(np.float32)) for m in sam_masks], dim=0
        )
        .unsqueeze(1)
        .to(device_t)
    )
    sam_prob_256 = torch.nn.functional.interpolate(
        sam_masks_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )

    # crop ROIs
    roi_imgs = []
    roi_prompts = []
    for i in range(boxes_1024_t.shape[0]):
        x1, y1, x2, y2 = boxes_1024_t[i]
        x1i = max(0, min(1023, int(x1.item())))
        y1i = max(0, min(1023, int(y1.item())))
        x2i = max(x1i + 1, min(1024, int(x2.item())))
        y2i = max(y1i + 1, min(1024, int(y2.item())))
        crop_img = img_bchw_1024[..., y1i:y2i, x1i:x2i]
        crop_prob = sam_prob_256[i : i + 1]
        crop_img_256 = torch.nn.functional.interpolate(
            crop_img, size=(256, 256), mode="bilinear", align_corners=False
        )
        crop_prob_256 = torch.nn.functional.interpolate(
            crop_prob, size=(256, 256), mode="bilinear", align_corners=False
        )
        roi_imgs.append(crop_img_256)
        roi_prompts.append(crop_prob_256)
    roi_img_256 = torch.cat(roi_imgs, dim=0)
    roi_prompt_256 = torch.cat(roi_prompts, dim=0)

    logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=roi_prompt_256)
    refine_prob_256 = torch.sigmoid(logits_refine_256)

    # paste back to 1024 canvas
    refine_prob_1024 = []
    for i in range(refine_prob_256.shape[0]):
        canvas = torch.zeros((1, 1, 1024, 1024), device=device_t)
        x1, y1, x2, y2 = boxes_1024_t[i]
        x1i = max(0, min(1023, int(x1.item())))
        y1i = max(0, min(1023, int(y1.item())))
        x2i = max(x1i + 1, min(1024, int(x2.item())))
        y2i = max(y1i + 1, min(1024, int(y2.item())))
        roi_resized = torch.nn.functional.interpolate(
            refine_prob_256[i : i + 1],
            size=(y2i - y1i, x2i - x1i),
            mode="bilinear",
            align_corners=False,
        )
        canvas[..., y1i:y2i, x1i:x2i] = roi_resized
        refine_prob_1024.append(canvas)
    refine_prob_1024 = torch.cat(refine_prob_1024, dim=0)

    # resize to original resolution
    refine_masks = []
    for i in range(refine_prob_1024.shape[0]):
        p = refine_prob_1024[i, 0].detach().cpu().numpy()
        p_orig = cv2.resize(p, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        m = (p_orig > 0.5).astype(np.uint8) * 255
        refine_masks.append(m)

    return sam_masks, refine_masks


@torch.no_grad()
def infer_sam3_with_refine_batch(
    model: object,
    processor: Sam3Processor,
    image_paths: List[str],
    boxes_batch: List[List[List[float]]],
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    简单的 batch 封装：对多张图重复调用 infer_one_sam3_with_refine。
    每张图可以有不同数量的框。

    返回：
        sam_masks_batch:  List[ List[np.ndarray] ]  # 每张图的 coarse 掩码
        refine_masks_batch: 同上                       # 每张图的 refine 掩码
    """
    assert len(image_paths) == len(
        boxes_batch
    ), "image_paths 和 boxes_batch 长度必须一致"
    sam_batch: List[List[np.ndarray]] = []
    refine_batch: List[List[np.ndarray]] = []
    for img_path, boxes in zip(image_paths, boxes_batch):
        sam_masks, refine_masks = infer_one_sam3_with_refine(
            model=model,
            processor=processor,
            image_path=img_path,
            boxes=boxes,
            refine_ckpt=refine_ckpt,
            device=device,
            roi_pad=roi_pad,
        )
        sam_batch.append(sam_masks)
        refine_batch.append(refine_masks)
    return sam_batch, refine_batch

