from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from .sam3_refine_infer import build_sam3_inst_model, infer_one_sam3_inst, infer_one_sam3_with_refine


# -----------------------------
# Basic tensor/image utilities
# -----------------------------
def _to_bchw(img: np.ndarray) -> torch.Tensor:
    """HWC (uint8) -> (1,C,H,W) float32"""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def _ensure_orig_size(mask: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
    """Ensure mask is exactly the original image size."""
    if mask.shape[0] != orig_H or mask.shape[1] != orig_W:
        mask = cv2.resize(mask, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
    return mask


# -----------------------------
# Resolution-agnostic geometry
# -----------------------------
def _scale_boxes_to_image(
    boxes_1024: torch.Tensor,            # [N,4], xyxy on the 1024 canvas
    height: int,                         # target canvas height
    width: int,                          # target canvas width
    sam_input_hw: Tuple[int, int],       # SAM processed image size (h_sam, w_sam); now (1024,1024)
) -> torch.Tensor:
    """Map SAM 1024-canvas boxes to an arbitrary (height,width) canvas."""
    if boxes_1024.numel() == 0:
        return boxes_1024
    in_h, in_w = sam_input_hw
    boxes = boxes_1024.clone().to(torch.float32)
    boxes[:, 0].clamp_(0.0, float(in_w))
    boxes[:, 2].clamp_(0.0, float(in_w))
    boxes[:, 1].clamp_(0.0, float(in_h))
    boxes[:, 3].clamp_(0.0, float(in_h))
    scale_x = width / max(1.0, float(in_w))
    scale_y = height / max(1.0, float(in_h))
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def _clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Clamp a float box to integer pixel box within (width,height)."""
    x1i = max(0, min(width - 1, int(np.floor(x1))))
    y1i = max(0, min(height - 1, int(np.floor(y1))))
    x2i = max(0, min(width, int(np.ceil(x2))))
    y2i = max(0, min(height, int(np.ceil(y2))))
    if x2i <= x1i:
        x2i = min(width, x1i + 1)
    if y2i <= y1i:
        y2i = min(height, y1i + 1)
    return x1i, y1i, x2i, y2i


def _expand_boxes_xyxy(
    boxes_xyxy: torch.Tensor,    # [N,4] on target canvas
    pad_ratio: float,
    width: int,
    height: int,
) -> torch.Tensor:
    """Pad xyxy boxes by ratio on each side, clamped to (width,height)."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    b = boxes_xyxy.clone().to(torch.float32)
    w = (b[:, 2] - b[:, 0]).clamp(min=1.0)
    h = (b[:, 3] - b[:, 1]).clamp(min=1.0)
    px = (w * float(pad_ratio))
    py = (h * float(pad_ratio))
    b[:, 0] = torch.clamp(b[:, 0] - px, 0.0, float(width))
    b[:, 1] = torch.clamp(b[:, 1] - py, 0.0, float(height))
    b[:, 2] = torch.clamp(b[:, 2] + px, 0.0, float(width))
    b[:, 3] = torch.clamp(b[:, 3] + py, 0.0, float(height))
    return b


def _binary_mask_to_box(mask_2d: torch.Tensor) -> Optional[Tuple[float, float, float, float]]:
    """
    mask_2d: torch.bool or torch.uint8 in (H,W)
    Return xyxy (float) or None if empty.
    """
    ys, xs = torch.where(mask_2d > 0)
    if ys.numel() == 0:
        return None
    y1 = ys.min().item()
    y2 = ys.max().item() + 1
    x1 = xs.min().item()
    x2 = xs.max().item()
    return float(x1), float(y1), float(x2 + 1), float(y2 + 1)


# -----------------------------
# Resolution-agnostic crop/paste
# -----------------------------
def _crop_resize_img(
    img_bchw: torch.Tensor,      # (1,C,H,W)
    boxes_xyxy: torch.Tensor,    # (N,4) on current img_bchw canvas
    out_hw: int = 256,
) -> torch.Tensor:
    """Crop each ROI from img_bchw and resize to out_hw."""
    _, _, H_img, W_img = img_bchw.shape
    rois: List[torch.Tensor] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.tolist()
        x1i, y1i, x2i, y2i = _clamp_box(x1, y1, x2, y2, W_img, H_img)
        crop = img_bchw[..., y1i:y2i, x1i:x2i]
        roi = F.interpolate(crop, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
        rois.append(roi)
    return torch.cat(rois, dim=0) if rois else img_bchw.new_zeros((0, img_bchw.shape[1], out_hw, out_hw))


def _crop_resize_prob(
    prob_full: torch.Tensor,     # (N,1,H,W) on some canvas
    boxes_xyxy: torch.Tensor,    # (N,4) on the same canvas
    out_hw: int = 256,
) -> torch.Tensor:
    """Crop each prob ROI and resize to out_hw (batched)."""
    N = prob_full.shape[0]
    rois: List[torch.Tensor] = []
    for i in range(N):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        H, W = prob_full.shape[2], prob_full.shape[3]
        x1i, y1i, x2i, y2i = _clamp_box(x1, y1, x2, y2, W, H)
        crop = prob_full[i:i+1, :, y1i:y2i, x1i:x2i]
        roi = F.interpolate(crop, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
        rois.append(roi)
    return torch.cat(rois, dim=0) if rois else prob_full.new_zeros((0, 1, out_hw, out_hw))


def _paste_roi_prob_back(
    prob_roi_256: torch.Tensor,  # (1,1,256,256)
    box_xyxy: torch.Tensor,      # (4,) on target canvas
    height: int,
    width: int,
) -> torch.Tensor:
    """Paste a 256×256 ROI prob map back to a (height,width) canvas at box_xyxy."""
    canvas = prob_roi_256.new_zeros((1, 1, height, width))
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1i, y1i, x2i, y2i = _clamp_box(x1, y1, x2, y2, width, height)
    if (x2i - x1i) <= 0 or (y2i - y1i) <= 0:
        return canvas
    roi_resized = F.interpolate(prob_roi_256, size=(y2i - y1i, x2i - x1i), mode="bilinear", align_corners=False)
    canvas[..., y1i:y2i, x1i:x2i] = roi_resized
    return canvas


# -----------------------------
# Helpers: prompts
# -----------------------------
def _boxes_to_masks_for_sam(
    shape_hw: Tuple[int, int],
    box_list: Optional[List[List[float]]],
    orig_H: int,
    orig_W: int,
    device_t: torch.device,
) -> torch.Tensor:
    """Rasterize original-image boxes to binary masks on (h_sam,w_sam)."""
    Hs, Ws = shape_hw
    masks_np: List[np.ndarray] = []
    if box_list:
        for b in box_list:
            mask = np.zeros((Hs, Ws), dtype=np.float32)
            x1, y1, x2, y2 = b
            sx, sy = Ws / float(orig_W), Hs / float(orig_H)
            x1s, y1s, x2s, y2s = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            xi1, yi1, xi2, yi2 = _clamp_box(x1s, y1s, x2s, y2s, Ws, Hs)
            if xi2 > xi1 and yi2 > yi1:
                mask[yi1:yi2, xi1:xi2] = 1.0
            masks_np.append(mask)
    return (
        torch.from_numpy(np.stack(masks_np)).to(device_t)
        if masks_np else torch.zeros((0, Hs, Ws), device=device_t)
    )


def _scale_points_to_sam(
    points: Optional[List[Tuple[float, float, int]]],
    orig_W: int,
    orig_H: int,
    w_sam: int,
    h_sam: int,
    device_t: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Scale (x,y,lbl) from original image coords to SAM canvas coords."""
    if not points:
        return None, None
    sx, sy = w_sam / float(orig_W), h_sam / float(orig_H)
    pts_xy: List[List[float]] = []
    pts_lbl: List[int] = []
    for x, y, lbl in points:
        pts_xy.append([x * sx, y * sy])
        pts_lbl.append(int(lbl))  # 1=positive, 0=negative
    pts_xy_t = torch.tensor(pts_xy, device=device_t, dtype=torch.float32).unsqueeze(0)      # [1,K,2]
    pts_lbl_t = torch.tensor(pts_lbl, device=device_t, dtype=torch.long).unsqueeze(0)       # [1,K]
    return pts_xy_t, pts_lbl_t

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PromptEntry:
    points: Optional[torch.Tensor]
    point_labels: Optional[torch.Tensor]
    boxes: Optional[torch.Tensor]
    mask_input: Optional[torch.Tensor]
    target_mask: Optional[torch.Tensor]


def prepare_prompts_from_boxes(
    boxes_1024: torch.Tensor,  # [N,4]
    device: torch.device,
    image_size: int,           # 这里就是 1024
) -> List[PromptEntry]:
    entries = []
    if boxes_1024.numel() == 0:
        return entries

    boxes_1024 = boxes_1024.to(torch.float32).to(device)
    # 一个简单的 dummy mask，全 0 即可
    dummy_mask = torch.zeros(1, 1, image_size, image_size, device=device)

    for i in range(boxes_1024.shape[0]):
        box = boxes_1024[i].unsqueeze(0)  # (1,4)
        entries.append(
            PromptEntry(
                points=None,
                point_labels=None,
                boxes=box,
                mask_input=None,
                target_mask=dummy_mask,  # 关键：不要是 None
            )
        )
    return entries


# -----------------------------
# Main inference (returns original-size masks)
# -----------------------------
def _infer_one_sam1(
    image_path: str,
    boxes: Optional[List[List[float]]] = None,
    points: Optional[List[Tuple[float, float, int]]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "vit_b_lm",
    sam_checkpoint: Optional[str] = None,
    refine_resume: Optional[str] = None,
    convnext_variant: str = "atto",
    convnext_ckpt: Optional[str] = None,
    fuse_with_sam: bool = False,
    crop_source: str = "prompt",
    roi_pad_256: float = 0.10,  # used when crop_source == "sam"（仍在原图尺度上应用）
    roi_pad: float = 0.10,      # used when crop_source == "prompt"（在 1024 上整型扩框）
    use_prompt: bool = True,
    input_preprocess: str = "orig",  # 参数保留，但本实现强制 1024×1024
    use_refine_net: bool = True,
    chunk_size: Optional[int] = None,
) -> List[np.ndarray]:
    device_t = torch.device(device)

    # --- Load image (original) ---
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_H, orig_W = img_rgb.shape[:2]
    img_bchw_orig = _to_bchw(img_rgb).to(device_t)

    # --- Build SAM + resize transform ---
    sam = base.build_plain_sam(model_type, sam_checkpoint, device_t)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)  # kept for base forward

    # --- Prepare SAM input image (force 1024x1024 square) ---
    target_len = 1024
    img_for_sam = cv2.resize(img_rgb, (target_len, target_len), interpolation=cv2.INTER_LINEAR)
    h_sam, w_sam = img_for_sam.shape[:2]  # (1024, 1024)
    img_bchw_sam = _to_bchw(img_for_sam).to(device_t)

    # --- Prepare prompts ---
    if not boxes and not points:
        raise ValueError("Boxes or points required")

    masks_for_sam = _boxes_to_masks_for_sam(
        (h_sam, w_sam), boxes, orig_H, orig_W, device_t
    )
    entries: List[dict] = []
    if masks_for_sam.numel() > 0:
        entries_boxes = base._prepare_prompts(
            masks=masks_for_sam,
            prompt_type="boxes",
            device=device_t,
            image_size=h_sam,
            mask_input_size=256,
        )
        entries.extend(entries_boxes)
    



    pts_xy_t, pts_lbl_t = _scale_points_to_sam(points, orig_W, orig_H, w_sam, h_sam, device_t)
    if pts_xy_t is not None and pts_lbl_t is not None:
        entries_points = base._prepare_prompts(
            points=pts_xy_t,
            point_labels=pts_lbl_t,
            prompt_type="points",
            device=device_t,
            image_size=h_sam,
            mask_input_size=256,
        )
        entries.extend(entries_points)

    if not entries:
        return []

    # --- SAM forward (on 1024 canvas) ---
    sam_out = base._sam_forward_batched_for_one_image(
        sam_model=sam,
        image_bchw=img_bchw_sam,
        entries=entries,
        resize_transform=resize_transform,
        input_hw=(h_sam, w_sam),           # (1024, 1024)
        mask_input_size=256,
        device=device_t,
        precomputed_img_emb=None,
        precomputed_image_pe=None,
    )
    low_res_logits_256: torch.Tensor = sam_out["low_res_logits_256"]   # (N,1,256,256)
    boxes_1024_from_prompt: torch.Tensor = sam_out["boxes_1024"].long()       # (N,4)
    sam_prob_256: torch.Tensor = torch.sigmoid(low_res_logits_256)     # (N,1,256,256)

    # --- Branch choice affects how we build "full-size" prob for no-refine & SAM ROI boxes ---
    crop_src = crop_source.lower()

    # Only compute postprocessed masks when we need exact original mapping (sam branch)
    prob_full_orig: Optional[torch.Tensor] = None
    if crop_src == "sam":
        masks_full_logits = sam.postprocess_masks(
            low_res_logits_256, input_size=(h_sam, w_sam), original_size=(orig_H, orig_W)
        )  # (N,1,orig_H,orig_W) logits
        prob_full_orig = torch.sigmoid(masks_full_logits)               # (N,1,orig_H,orig_W)

    # --- No refine path ---
    if not use_refine_net:
        masks_out: List[np.ndarray] = []
        if crop_src == "prompt":
            # 不调用 sam.postprocess_masks；用 cv2.resize 从 1024 拉回原图
            prob_1024 = F.interpolate(sam_prob_256, size=(1024, 1024), mode="bilinear", align_corners=False)
            for i in range(prob_1024.shape[0]):
                p1024 = prob_1024[i, 0].detach().cpu().numpy()
                p_orig = cv2.resize(p1024, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
                mask_np = (p_orig > 0.5).astype(np.uint8) * 255
                masks_out.append(mask_np)
            return masks_out
        else:
            # sam 分支：保持 postprocess 精确映射
            for i in range(prob_full_orig.shape[0]):
                mask_np = (prob_full_orig[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                masks_out.append(_ensure_orig_size(mask_np, orig_H, orig_W))
            return masks_out

    # --- Refine path ---
    if crop_src == "prompt":
        # 在 1024 上整型扩框 & 得到 256 框（train-like）
        x1y1x2y2 = boxes_1024_from_prompt.clone().to(torch.int64)  # [N,4]
        w = (x1y1x2y2[:, 2] - x1y1x2y2[:, 0] + 1).clamp(min=1)
        h = (x1y1x2y2[:, 3] - x1y1x2y2[:, 1] + 1).clamp(min=1)
        px = torch.round(w.float() * float(roi_pad)).to(torch.int64)
        py = torch.round(h.float() * float(roi_pad)).to(torch.int64)

        x1y1x2y2[:, 0] = torch.clamp(x1y1x2y2[:, 0] - px, 0, 1024)
        x1y1x2y2[:, 1] = torch.clamp(x1y1x2y2[:, 1] - py, 0, 1024)
        x1y1x2y2[:, 2] = torch.clamp(x1y1x2y2[:, 2] + px + 1, 0, 1024)
        x1y1x2y2[:, 3] = torch.clamp(x1y1x2y2[:, 3] + py + 1, 0, 1024)

        boxes_1024_all = x1y1x2y2.to(torch.float32)
        boxes_256_all = torch.stack([
            torch.tensor([
                max(0, int(b[0].item()) // 4),
                max(0, int(b[1].item()) // 4),
                min(256, (int(b[2].item()) + 3) // 4),
                min(256, (int(b[3].item()) + 3) // 4),
            ], device=device_t, dtype=torch.int64)
            for b in boxes_1024_all
        ], dim=0)

        total_instances = boxes_1024_all.shape[0]
        chunk = chunk_size if (chunk_size and chunk_size > 0) else total_instances

        refine_net = base.ROIRefineNet(
            variant=convnext_variant,
            convnext_ckpt=convnext_ckpt,
            use_prompt=use_prompt,
            build_prompt=False,
        ).to(device_t).eval()
        if refine_resume and Path(refine_resume).exists():
            ckpt = torch.load(refine_resume, map_location="cpu")
            refine_state = ckpt.get("refine", ckpt)
            refine_net.load_state_dict(refine_state, strict=False)

        masks_out: List[np.ndarray] = []

        for start_idx in range(0, total_instances, chunk):
            end_idx = min(total_instances, start_idx + chunk)
            boxes_chunk_1024 = boxes_1024_all[start_idx:end_idx]

            # 在 1024 图上裁 ROI
            roi_img_256 = _crop_resize_img(img_bchw_sam, boxes_chunk_1024, 256)

            # prompt mask：sam 的 256 概率 + 256 框
            prompt_mask_256 = None
            if use_prompt:
                prompt_mask_256 = _crop_resize_prob(
                    sam_prob_256[start_idx:end_idx],
                    boxes_256_all[start_idx:end_idx].to(torch.float32),
                    out_hw=256
                )

            logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
            probs_256 = torch.sigmoid(logits_refine_256)

            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                # 粘回 1024 画布
                prob_canvas_1024 = base._paste_roi_prob_back_1024(
                    probs_256[local_idx:local_idx+1], boxes_chunk_1024[local_idx], 1024, 1024
                )

                if fuse_with_sam:
                    sam_roi_256 = _crop_resize_prob(
                        sam_prob_256[global_idx:global_idx+1],
                        boxes_256_all[global_idx:global_idx+1].to(torch.float32),
                        out_hw=256
                    )
                    sam_canvas_1024 = base._paste_roi_prob_back_1024(
                        sam_roi_256, boxes_chunk_1024[local_idx], 1024, 1024
                    )
                    prob_canvas_1024 = 0.5 * prob_canvas_1024 + 0.5 * sam_canvas_1024

                # 使用 cv2.resize 将 1024 画布直接缩放到原图尺寸（不调用 postprocess）
                p1024 = prob_canvas_1024.squeeze(0).squeeze(0).detach().cpu().numpy()
                p_orig = cv2.resize(p1024, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
                mask_np = (p_orig > 0.5).astype(np.uint8) * 255
                masks_out.append(mask_np)

        return masks_out

    else:
        # --- crop_source == "sam"：保持原有“原图坐标系”路径 ---
        # 需要 prob_full_orig（已通过 postprocess 得到）
        core_list = []
        for i in range(prob_full_orig.shape[0]):
            box = _binary_mask_to_box((prob_full_orig[i, 0] > 0.5))
            if box is None:
                # 回退：把 1024 框映射回原图
                box_1024 = boxes_1024_from_prompt[i:i+1]
                box_img = _scale_boxes_to_image(
                    box_1024, orig_H, orig_W, (target_len, target_len)
                )
                x1, y1, x2, y2 = box_img[0].tolist()
            else:
                x1, y1, x2, y2 = box
            core_list.append([x1, y1, x2, y2])
        core_boxes_img = torch.tensor(core_list, device=device_t, dtype=torch.float32)
        boxes_img_all = _expand_boxes_xyxy(core_boxes_img, roi_pad_256, orig_W, orig_H)

        total_instances = boxes_img_all.shape[0]
        chunk = chunk_size if (chunk_size and chunk_size > 0) else total_instances

        refine_net = base.ROIRefineNet(
            variant=convnext_variant,
            convnext_ckpt=convnext_ckpt,
            use_prompt=use_prompt,
            build_prompt=False,
        ).to(device_t).eval()
        if refine_resume and Path(refine_resume).exists():
            ckpt = torch.load(refine_resume, map_location="cpu")
            refine_state = ckpt.get("refine", ckpt)
            refine_net.load_state_dict(refine_state, strict=False)

        masks_out: List[np.ndarray] = []

        for start_idx in range(0, total_instances, chunk):
            end_idx = min(total_instances, start_idx + chunk)
            boxes_chunk_img = boxes_img_all[start_idx:end_idx]                      # 原图坐标
            roi_img_256 = _crop_resize_img(img_bchw_orig, boxes_chunk_img, 256)

            prompt_mask_256 = None
            if use_prompt:
                prompt_mask_256 = _crop_resize_prob(
                    prob_full_orig[start_idx:end_idx],
                    boxes_chunk_img,
                    out_hw=256
                )

            logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
            probs_256 = torch.sigmoid(logits_refine_256)

            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                prob_canvas = _paste_roi_prob_back(
                    probs_256[local_idx:local_idx+1], boxes_chunk_img[local_idx], orig_H, orig_W
                )

                if fuse_with_sam:
                    sam_roi_256 = _crop_resize_prob(
                        prob_full_orig[global_idx:global_idx+1],
                        boxes_chunk_img[local_idx:local_idx+1],
                        out_hw=256
                    )
                    sam_canvas = _paste_roi_prob_back(sam_roi_256, boxes_chunk_img[local_idx], orig_H, orig_W)
                    prob_canvas = 0.5 * prob_canvas + 0.5 * sam_canvas

                mask_np = (prob_canvas.squeeze(0).squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                masks_out.append(_ensure_orig_size(mask_np, orig_H, orig_W))

        return masks_out


# -----------------------------
# CLI helpers
# -----------------------------
def _parse_boxes_arg(boxes_arg: Optional[str]) -> Optional[List[List[float]]]:
    if boxes_arg is None:
        return None
    p = Path(boxes_arg)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return [[float(x) for x in box] for box in data]
    parts = [seg.strip() for seg in boxes_arg.split(";") if seg.strip()]
    boxes: List[List[float]] = []
    for seg in parts:
        vals = [float(x) for x in seg.replace(" ", "").split(",")]
        if len(vals) != 4:
            raise ValueError(f"Invalid box segment: {seg}")
        boxes.append(vals)
    return boxes


def _parse_points_arg(points_arg: Optional[str]) -> Optional[List[Tuple[float, float, int]]]:
    """
    支持两种格式：
    1) 纯文本："x1,y1,lbl1; x2,y2,lbl2; ..."，其中 lbl∈{0,1}
    2) 文件路径：JSON 数组 [[x,y,lbl], ...]
    """
    if points_arg is None:
        return None
    p = Path(points_arg)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        pts: List[Tuple[float, float, int]] = []
        for it in data:
            if len(it) != 3:
                raise ValueError("Invalid point triplet in file")
            x, y, lbl = it
            pts.append((float(x), float(y), int(lbl)))
        return pts
    parts = [seg.strip() for seg in points_arg.split(";") if seg.strip()]
    pts: List[Tuple[float, float, int]] = []
    for seg in parts:
        vals = [t for t in seg.replace(" ", "").split(",")]
        if len(vals) != 3:
            raise ValueError(f"Invalid point segment: {seg}")
        x, y, lbl = float(vals[0]), float(vals[1]), int(float(vals[2]))
        pts.append((x, y, lbl))
    return pts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Refine inference (force 1024 input; returns original-size masks)")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--boxes", type=str, default=None)
    p.add_argument("--points", type=str, default=None, help="x,y,lbl;... or JSON file path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-type", type=str, default="vit_b_lm")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--refine-ckpt", type=str, default=None)
    p.add_argument("--convnext-variant", type=str, default="atto")
    p.add_argument("--convnext-ckpt", type=str, default=None)
    p.add_argument("--fuse-with-sam", action="store_true", default=False)
    p.add_argument("--crop-source", choices=["sam", "prompt"], default="prompt")
    p.add_argument("--roi-pad-256", type=float, default=0.10)   # used when crop_source="sam"
    p.add_argument("--roi-pad", type=float, default=0.10)       # used when crop_source="prompt"
    p.add_argument("--input-preprocess", choices=["orig", "resize1024"], default="resize1024")  # 保留参数但已强制 1024
    p.add_argument("--use-prompt", action="store_true", default=True)
    p.add_argument("--no-use-prompt", dest="use_prompt", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    boxes = _parse_boxes_arg(args.boxes)
    points = _parse_points_arg(args.points)
    root = Path(__file__).resolve().parents[2]
    bpe_path = str(root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    ckpt = args.checkpoint or str(root / "weights" / "sam3.pt")
    model, processor = build_sam3_inst_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt,
        device=args.device,
    )
    sam_masks, refine_masks = infer_one_sam3_with_refine(
        model=model,
        processor=processor,
        image_path=args.image,
        boxes=boxes or [],
        refine_ckpt=args.refine_ckpt,
        device=args.device,
        roi_pad=args.roi_pad,
    )
    if args.boxes:
        # 如果提供了 GT（例如 COCO），这里可以计算 IoU；当前 CLI 先仅保存 refine 输出
        for i, m in enumerate(refine_masks):
            out_path = Path(args.image).with_suffix(f".refine_inst{i+1}.png")
            cv2.imwrite(str(out_path), m)
    else:
        print(f"Predicted {len(refine_masks)} instances")


if __name__ == "__main__":
    main()
