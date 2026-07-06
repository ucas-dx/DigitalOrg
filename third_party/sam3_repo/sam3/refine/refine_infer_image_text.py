from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .util import ROIRefineNet
from .refine_infer_image_geo import infer_one_sam3_with_refine


@torch.no_grad()
def build_sam3_text_geo_model(
    bpe_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
) -> Tuple[object, Sam3Processor]:
    """
    Build SAM3 image model for text + geometric prompts, and wrap it with Sam3Processor.
    This uses the standard SAM3 grounding branch (forward_grounding), not the
    SAM1-style instance interactivity predictor.
    """
    device_t = torch.device(device)

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(
        model, device=device_t, confidence_threshold=confidence_threshold
    )
    return model, processor


def _boxes_xyxy_to_norm_cxcywh(
    boxes: List[List[float]], width: int, height: int
) -> List[List[float]]:
    """Convert pixel xyxy boxes to normalized cxcywh in [0, 1]."""
    norm_boxes: List[List[float]] = []
    w_f = float(width)
    h_f = float(height)
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) * 0.5) / w_f
        cy = ((y1 + y2) * 0.5) / h_f
        bw = (x2 - x1) / w_f
        bh = (y2 - y1) / h_f
        norm_boxes.append([cx, cy, bw, bh])
    return norm_boxes


def _compute_iou_xyxy(box: List[float], boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single xyxy box and an array of xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.clip(xx2 - xx1, a_min=0.0, a_max=None)
    inter_h = np.clip(yy2 - yy1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_box = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
    )
    union = area_box + area_boxes - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter / union


@torch.no_grad()
def infer_one_sam3_text_geo(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    text_prompt: str,
    boxes: List[List[float]],
) -> List[np.ndarray]:
    """
    SAM3 text + geometric box prompts (no instance interactivity), via forward_grounding.
    对于每一个输入的几何框，使用相同的文本提示，运行一次 grounding，
    并从所有预测中选取与该框 IoU 最大的 mask 作为该框的粗分割结果。

    Args:
        model: SAM3 image model
        processor: Sam3Processor
        image_path: path to the RGB image
        text_prompt: 文本提示（例如 “凋亡的类器官”）
        boxes: List[ [x1, y1, x2, y2] ] in pixel coordinates

    Returns:
        List of uint8 masks (H, W), one per input box.
    """
    if len(boxes) == 0:
        return []

    image = Image.open(image_path).convert("RGB")
    orig_W, orig_H = image.size

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

    sam_masks: List[np.ndarray] = []

    with autocast_ctx(enabled=False):
        # 先预计算图像与文本特征
        base_state = processor.set_image(image)
        if text_prompt:
            base_state = processor.set_text_prompt(prompt=text_prompt, state=base_state)

        norm_boxes = _boxes_xyxy_to_norm_cxcywh(boxes, orig_W, orig_H)

        for box_xyxy, box_norm in zip(boxes, norm_boxes):
            # 为每个几何框构建一个独立的 state，复用相同的 backbone_out
            state = {
                "original_height": base_state["original_height"],
                "original_width": base_state["original_width"],
                "backbone_out": base_state["backbone_out"],
            }

            # 添加几何提示（中心点 + 宽高，归一化到 [0,1]）
            state = processor.add_geometric_prompt(
                box=box_norm, label=True, state=state
            )

            if "masks_logits" not in state or "boxes" not in state:
                # 没有任何有效预测，返回空 mask
                sam_masks.append(
                    np.zeros((orig_H, orig_W), dtype=np.uint8)
                )
                continue

            masks_prob = state["masks_logits"]  # [N, 1, H, W]
            pred_boxes = state["boxes"]  # [N, 4] in xyxy (pixel)

            masks_prob_np = masks_prob.detach().cpu().numpy()
            pred_boxes_np = pred_boxes.detach().cpu().numpy()

            # 在所有预测中，选取与当前几何框 IoU 最大的一个
            ious = _compute_iou_xyxy(box_xyxy, pred_boxes_np)
            if ious.size == 0 or float(ious.max()) <= 0.0:
                sam_masks.append(
                    np.zeros((orig_H, orig_W), dtype=np.uint8)
                )
                continue

            best_idx = int(ious.argmax())
            prob = masks_prob_np[best_idx, 0]  # (H, W), 已经在原图尺寸
            mask = (prob > 0.5).astype(np.uint8) * 255
            sam_masks.append(mask)

    return sam_masks


@torch.no_grad()
def infer_one_sam3_text_geo_with_refine(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    text_prompt: str,
    boxes: List[List[float]],
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
    fuse_with_sam: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    End-to-end: SAM3 coarse via text+box prompts + ConvNeXtV1 refine.
    Returns (sam_masks, refine_masks).

    - 文本提示和几何框一起作为 SAM3 grounding 的 prompt；
    - coarse 掩码来自 forward_grounding；
    - refine 仍然使用 ROIRefineNet，对每个输入框进行 ROI 细化。
    """
    device_t = torch.device(device)
    image = Image.open(image_path).convert("RGB")
    orig_H, orig_W = image.height, image.width

    # 确保模型和 processor 在同一个 device（单一 device 流程）
    model.to(device_t)
    processor.device = device_t

    # coarse SAM3 masks（文本 + 几何提示）
    sam_masks = infer_one_sam3_text_geo(
        model=model,
        processor=processor,
        image_path=image_path,
        text_prompt=text_prompt,
        boxes=boxes,
    )
    if len(sam_masks) == 0:
        return [], []

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

    # scale input boxes to 1024
    boxes_1024: List[List[float]] = []
    for b in boxes:
        x1, y1, x2, y2 = b
        sx, sy = 1024.0 / float(orig_W), 1024.0 / float(orig_H)
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

    # coarse prob 256 from sam_masks（与 geo 版本保持一致：先二值化，再下采样到 256）
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
    # 同时构建 1024 尺度的 coarse prob，便于 fuse 时回填到 1024 canvas
    sam_prob_1024 = torch.nn.functional.interpolate(
        sam_masks_tensor, size=(1024, 1024), mode="bilinear", align_corners=False
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

    # paste back to 1024 canvas（可选与 coarse SAM prob fuse）
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

        if fuse_with_sam:
            # 从 coarse SAM prob 中提取对应 ROI，并回填到 1024 canvas
            sam_canvas = torch.zeros_like(canvas)
            sam_canvas[..., y1i:y2i, x1i:x2i] = sam_prob_1024[
                i : i + 1, :, y1i:y2i, x1i:x2i
            ]
            # 简单平均融合（可根据需要改为加权）
            canvas = 0.5 * canvas + 0.5 * sam_canvas

        refine_prob_1024.append(canvas)
    refine_prob_1024 = torch.cat(refine_prob_1024, dim=0)

    # resize to original resolution
    refine_masks: List[np.ndarray] = []
    for i in range(refine_prob_1024.shape[0]):
        p = refine_prob_1024[i, 0].detach().cpu().numpy()
        p_orig = cv2.resize(p, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        m = (p_orig > 0.5).astype(np.uint8) * 255
        refine_masks.append(m)

    return sam_masks, refine_masks


@torch.no_grad()
def infer_sam3_text_geo_with_refine_batch(
    model: object,
    processor: Sam3Processor,
    image_paths: List[str],
    text_prompts: List[str],
    boxes_batch: List[List[List[float]]],
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    简单的 batch 封装：对多张图重复调用 infer_one_sam3_text_geo_with_refine。
    每张图可以有不同数量的框和各自的文本提示。

    Args:
        image_paths: List[str]，每个元素是一张图像路径
        text_prompts: List[str]，每张图的文本提示
        boxes_batch: List[ List[ [x1, y1, x2, y2] ] ]，每张图一组框

    Returns:
        sam_masks_batch: List[ List[np.ndarray] ]
        refine_masks_batch: 同上
    """
    assert len(image_paths) == len(
        boxes_batch
    ), "image_paths 和 boxes_batch 长度必须一致"
    assert len(image_paths) == len(
        text_prompts
    ), "image_paths 和 text_prompts 长度必须一致"

    sam_batch: List[List[np.ndarray]] = []
    refine_batch: List[List[np.ndarray]] = []
    for img_path, prompt, boxes in zip(image_paths, text_prompts, boxes_batch):
        sam_masks, refine_masks = infer_one_sam3_text_geo_with_refine(
            model=model,
            processor=processor,
            image_path=img_path,
            text_prompt=prompt,
            boxes=boxes,
            refine_ckpt=refine_ckpt,
            device=device,
            roi_pad=roi_pad,
        )
        sam_batch.append(sam_masks)
        refine_batch.append(refine_masks)
    return sam_batch, refine_batch


@torch.no_grad()
def infer_sam3_text_only_with_refine(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    text_prompt: str,
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
    confidence_threshold: Optional[float] = None,
    mask_pixel_threshold: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    端到端纯文本 + refine 推理接口（中间不依赖 SAM3 的检测框作为几何提示）：

    1) 只用文本提示 text_prompt，在整张图上做一次 grounding，得到 SAM3 的粗分割 prob（masks_logits）；
    2) 从这些粗掩码直接计算每个实例的外接框（由掩码本身决定，而不是检测 head 输出的 boxes）；
    3) 在 1024×1024 canvas 上裁剪 ROI，并使用 ROIRefineNet 做细化；
    4) 返回 (coarse_masks, refine_masks)，二者都在原图分辨率下。

    注意：
    - 不使用任何标注边界框；
    - 几何 ROI 框由粗掩码本身计算得到；
    - 始终经过 refine（ROIRefineNet）。

    当前已知问题：
    - 在 ExampleDataset 等数据集上，粗掩码直接算外接框时，refine 结果有时会出现较多“离散小散点”；
      因此暂不作为默认/推荐的文本推理接口，仅保留以便后续进一步调试和改进。
    """
    if confidence_threshold is not None:
        processor.set_confidence_threshold(confidence_threshold)
    if mask_pixel_threshold is not None:
        processor.set_mask_threshold(mask_pixel_threshold)

    device_t = torch.device(device)

    image = Image.open(image_path).convert("RGB")
    orig_W, orig_H = image.size

    # 确保模型与 processor 在同一 device
    model.to(device_t)
    processor.device = device_t

    # 1) 纯文本提示，拿到 SAM3 的粗掩码（masks_logits 在原图尺寸）
    state: Dict[str, Any] = processor.set_image(image, state={})
    state = processor.set_text_prompt(prompt=text_prompt, state=state)

    masks_logits = state.get("masks_logits", None)
    if masks_logits is None or masks_logits.numel() == 0:
        return [], []

    # masks_logits: [N, 1, H, W] 概率
    sam_prob_orig = masks_logits.to(device_t)  # float, 已经是 [0,1]
    N = sam_prob_orig.shape[0]

    # 像素级阈值（用于从 prob 计算 coarse 掩码/ROI）
    pix_thr = float(mask_pixel_threshold) if mask_pixel_threshold is not None else 0.5

    # 2) 从粗掩码计算每个实例的外接框（在原图坐标系）
    boxes_orig: List[List[float]] = []
    coarse_masks: List[np.ndarray] = []
    for i in range(N):
        prob = sam_prob_orig[i, 0]  # (H, W)
        m_bin = (prob > pix_thr)
        ys, xs = torch.where(m_bin)
        if ys.numel() == 0 or xs.numel() == 0:
            # 没有前景，跳过这个实例
            continue
        ymin = ys.min().item()
        ymax = ys.max().item()
        xmin = xs.min().item()
        xmax = xs.max().item()

        w = float(xmax - xmin + 1)
        h = float(ymax - ymin + 1)
        px = w * float(roi_pad)
        py = h * float(roi_pad)

        x1 = max(0.0, float(xmin) - px)
        y1 = max(0.0, float(ymin) - py)
        x2 = min(float(orig_W), float(xmax + 1) + px)
        y2 = min(float(orig_H), float(ymax + 1) + py)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes_orig.append([x1, y1, x2, y2])

        # 保存粗掩码（二值）到原图分辨率
        m_np = (m_bin.detach().cpu().numpy().astype(np.uint8)) * 255
        coarse_masks.append(m_np)

    if not boxes_orig:
        return [], []

    # 3) 构造 1024×1024 image canvas，并将 boxes 映射到 1024 坐标系
    img_1024 = image.resize((1024, 1024))
    img_np_1024 = np.array(img_1024)
    img_bchw_1024 = (
        torch.from_numpy(img_np_1024)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device_t)
    )

    boxes_1024: List[List[float]] = []
    sx, sy = 1024.0 / float(orig_W), 1024.0 / float(orig_H)
    for (x1, y1, x2, y2) in boxes_orig:
        bx1, by1 = x1 * sx, y1 * sy
        bx2, by2 = x2 * sx, y2 * sy
        boxes_1024.append([bx1, by1, bx2, by2])
    boxes_1024_t = torch.tensor(boxes_1024, device=device_t, dtype=torch.float32)

    # 对应的 coarse prob 也 resize 到 1024，便于与上面的 boxes_1024 对齐裁剪
    sam_prob_1024 = torch.nn.functional.interpolate(
        sam_prob_orig, size=(1024, 1024), mode="bilinear", align_corners=False
    )

    # 4) 构造 refine 网络
    refine_net = ROIRefineNet(
        variant="atto", convnext_ckpt=None, use_prompt=True, build_prompt=False
    )
    if refine_ckpt is not None and Path(refine_ckpt).exists():
        ckpt = torch.load(refine_ckpt, map_location="cpu")
        state_dict = ckpt.get("refine", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        refine_net.load_state_dict(state_dict, strict=False)
    refine_net.to(device_t).eval()

    # 5) 根据 boxes_1024 裁剪 ROI 图像与 ROI coarse mask，到 256×256，送入 refine
    roi_imgs = []
    roi_prompts = []
    for i in range(len(boxes_1024)):
        x1, y1, x2, y2 = boxes_1024_t[i]
        x1i = max(0, min(1023, int(x1.item())))
        y1i = max(0, min(1023, int(y1.item())))
        x2i = max(x1i + 1, min(1024, int(x2.item())))
        y2i = max(y1i + 1, min(1024, int(y2.item())))

        crop_img = img_bchw_1024[..., y1i:y2i, x1i:x2i]
        crop_prob = sam_prob_1024[i : i + 1]  # [1,1,1024,1024] → ROI
        crop_prob = crop_prob[..., y1i:y2i, x1i:x2i]

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

    # 6) 将 refine 结果 paste 回 1024 canvas，再 resize 回原图分辨率
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

    refine_masks: List[np.ndarray] = []
    for i in range(refine_prob_1024.shape[0]):
        p = refine_prob_1024[i, 0].detach().cpu().numpy()
        p_orig = cv2.resize(p, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        m = (p_orig > 0.5).astype(np.uint8) * 255
        refine_masks.append(m)

    return coarse_masks, refine_masks


@torch.no_grad()
def infer_sam3_text_detector_geo_with_refine(
    model: object,
    processor: Sam3Processor,
    image_path: str,
    text_prompt: str,
    refine_ckpt: Optional[str],
    device: str = "cuda",
    roi_pad: float = 0.1,
    det_conf_thr: float = 0.25,
    fuse_with_sam: bool = False,
) -> Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:
    """
    端到端接口（按你描述的 1-6 步），当前推荐的“标准文本 + refine 推理接口”：

      1) 用文本提示做一次检测 -> 得到一批检测框 B_det；
      2) 把这些 B_det 原封不动地作为几何提示框（唯一几何提示）；
      3) 用 “几何框” 调用实例分割接口 infer_one_sam3_with_refine (predict_inst + refine)；
      4) 得到 coarse 掩码和 refine 掩码；
      5) 按比例 fuse (refine, coarse) 作为最终掩码；
      6) 返回 (B_det, coarse_masks, fused_masks)。

    注意：
      - 这里 coarse 分割完全来自几何提示的实例分割（predict_inst），
        文本只用于第 1 步的检测；
      - B_det 不会在分割阶段被更改或重新匹配。

    说明：
      - 相比 infer_sam3_text_only_with_refine，本接口在 ExampleDataset 项目中表现更稳定，
        掩码不容易出现大面积离散散点，因此暂时将其作为默认的文本推理接口使用；
        纯文本版本的接口保留用于后续问题排查和算法尝试。
    """
    device_t = torch.device(device)
    model.to(device_t)
    processor.device = device_t

    # 1) 文本检测，得到 B_det
    image = Image.open(image_path).convert("RGB")

    processor.set_confidence_threshold(det_conf_thr)
    state: Dict[str, Any] = processor.set_image(image, state={})
    state = processor.set_text_prompt(prompt=text_prompt, state=state)

    boxes_t = state.get("boxes", None)
    scores_t = state.get("scores", None)
    if boxes_t is None or scores_t is None or boxes_t.numel() == 0:
        return [], [], []

    boxes_xyxy: List[List[float]] = boxes_t.detach().cpu().tolist()

    # 2+3+4) 使用几何框调用实例分割 + refine（纯几何提示，无文本）
    sam_masks_coarse, refine_masks = infer_one_sam3_with_refine(
        model=model,
        processor=processor,
        image_path=image_path,
        boxes=boxes_xyxy,
        refine_ckpt=refine_ckpt,
        device=device,
        roi_pad=roi_pad,
    )

    # 5) 按比例 fuse coarse 与 refine
    fused_masks: List[np.ndarray] = []
    alpha = 0.5 if fuse_with_sam else 1.0
    for mc, mr in zip(sam_masks_coarse, refine_masks):
        c = mc.astype(np.float32) / 255.0
        r = mr.astype(np.float32) / 255.0
        p = alpha * r + (1.0 - alpha) * c
        m = (p > 0.5).astype(np.uint8) * 255
        fused_masks.append(m)

    return boxes_xyxy, sam_masks_coarse, fused_masks
