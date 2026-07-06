from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Only depend on sam_convnext
from .sam_convnext import (
    _prepare_prompts,
    _sam_forward_batched_for_one_image,
    _bbox_from_mask_256,
    _batch_crop_resize_img_bchw_1024,
    _batch_crop_resize_mask_256,
    _paste_roi_prob_back_1024,
    build_plain_sam,
    ResizeLongestSide,
    ROIRefineNet,
)


def _to_bchw(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def _boxes_to_masks_1024(
    box_list: Optional[List[List[float]]],
    orig_H: int,
    orig_W: int,
    device_t: torch.device,
    Hs: int = 1024,
    Ws: int = 1024,
) -> torch.Tensor:
    masks_np: List[np.ndarray] = []
    if box_list:
        for b in box_list:
            x1, y1, x2, y2 = map(float, b)
            sx, sy = Ws / float(orig_W), Hs / float(orig_H)
            x1s, y1s, x2s, y2s = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            xi1 = int(max(0, min(Ws - 1, np.floor(x1s))))
            yi1 = int(max(0, min(Hs - 1, np.floor(y1s))))
            xi2 = int(max(0, min(Ws,     np.ceil(x2s))))
            yi2 = int(max(0, min(Hs,     np.ceil(y2s))))
            if xi2 <= xi1:
                xi2 = min(Ws, xi1 + 1)
            if yi2 <= yi1:
                yi2 = min(Hs, yi1 + 1)
            mask = np.zeros((Hs, Ws), dtype=np.float32)
            mask[yi1:yi2, xi1:xi2] = 1.0
            masks_np.append(mask)
    return (
        torch.from_numpy(np.stack(masks_np)).to(device_t)
        if masks_np else torch.zeros((0, Hs, Ws), device=device_t)
    )


def _extract_refine_state(ckpt_obj: dict) -> dict:
    if isinstance(ckpt_obj, dict):
        if 'refine' in ckpt_obj and isinstance(ckpt_obj['refine'], dict):
            return ckpt_obj['refine']
        if 'state_dict' in ckpt_obj and isinstance(ckpt_obj['state_dict'], dict):
            sd = ckpt_obj['state_dict']
            fixed = {k.replace('refine.', ''): v for k, v in sd.items() if isinstance(k, str)}
            return fixed or sd
        if 'model' in ckpt_obj and isinstance(ckpt_obj['model'], dict):
            sd = ckpt_obj['model']
            fixed = {k.replace('refine.', ''): v for k, v in sd.items() if isinstance(k, str)}
            return fixed or sd
    return ckpt_obj


@torch.no_grad()
def infer_one(
    image_path: str,
    boxes: Optional[List[List[float]]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "vit_b_lm",
    sam_checkpoint: Optional[str] = None,
    refine_resume: Optional[str] = None,
    convnext_variant: str = "atto",
    convnext_ckpt: Optional[str] = None,
    fuse_with_sam: bool = False,
    crop_source: str = "prompt",
    roi_pad_256: float = 0.10,
    roi_pad: float = 0.10,
    use_prompt: bool = True,
    input_preprocess: str = "resize1024",
    use_refine_net: bool = True,
    chunk_size: Optional[int] = None,
) -> List[np.ndarray]:
    device_t = torch.device(device)

    # Force convnext_variant to 'atto'
    convnext_variant = "atto"

    # Load image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_H, orig_W = img_rgb.shape[:2]

    # Prepare 1024x1024 input
    img_1024 = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    h_sam, w_sam = img_1024.shape[:2]
    img_bchw_1024 = _to_bchw(img_1024).to(device_t)

    # Build SAM
    sam = build_plain_sam(model_type, sam_checkpoint, device_t)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # Prompts from boxes on 1024 canvas
    if not boxes or len(boxes) == 0:
        raise ValueError("infer_one requires non-empty 'boxes'.")
    masks_for_sam_1024 = _boxes_to_masks_1024(boxes, orig_H, orig_W, device_t, 1024, 1024)
    entries = _prepare_prompts(
        masks=masks_for_sam_1024,
        prompt_type="boxes",
        device=device_t,
        image_size=1024,
        mask_input_size=256,
    )
    if not entries:
        return []

    # SAM forward
    sam_out = _sam_forward_batched_for_one_image(
        sam_model=sam,
        image_bchw=img_bchw_1024,
        entries=entries,
        resize_transform=resize_transform,
        input_hw=(h_sam, w_sam),
        mask_input_size=256,
        device=device_t,
        precomputed_img_emb=None,
        precomputed_image_pe=None,
    )
    low_res_logits_256: torch.Tensor = sam_out["low_res_logits_256"]
    boxes_1024_from_prompt: torch.Tensor = sam_out["boxes_1024"].long()
    sam_prob_256: torch.Tensor = torch.sigmoid(low_res_logits_256)

    # Accurate original-size SAM prob for fallback
    masks_full_logits = sam.postprocess_masks(
        low_res_logits_256, input_size=(h_sam, w_sam), original_size=(orig_H, orig_W)
    )
    prob_full_orig: torch.Tensor = torch.sigmoid(masks_full_logits)

    # If not using refine, return SAM masks
    if not use_refine_net:
        masks_out: List[np.ndarray] = []
        for i in range(prob_full_orig.shape[0]):
            mask_np = (prob_full_orig[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            masks_out.append(mask_np)
        return masks_out

    # Build ROI boxes
    crop_src = str(crop_source).lower()
    if crop_src == "prompt":
        x1y1x2y2 = boxes_1024_from_prompt.clone().to(torch.int64)
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
    elif crop_src == "sam":
        boxes_256_all = _bbox_from_mask_256(low_res_logits_256, pad_ratio=roi_pad_256).to(torch.int64)
        boxes_1024_all = boxes_256_all.to(torch.float32)
        boxes_1024_all[:, 0] *= 4; boxes_1024_all[:, 1] *= 4; boxes_1024_all[:, 2] *= 4; boxes_1024_all[:, 3] *= 4
    else:
        raise ValueError("crop_source must be 'prompt' or 'sam'")

    total_instances = boxes_1024_all.shape[0]
    chunk = chunk_size if (chunk_size and chunk_size > 0) else total_instances

    # Build refine net (atto) and optionally load weights
    refine_net = ROIRefineNet(
        use_convnext=True,
        convnext_variant=convnext_variant,
        convnext_ckpt=convnext_ckpt,
        use_prompt=use_prompt,
        build_prompt=False,
    ).to(device_t).eval()

    refine_loaded = False
    if refine_resume is not None and Path(refine_resume).exists():
        try:
            ckpt = torch.load(refine_resume, map_location="cpu")
            refine_state = _extract_refine_state(ckpt)
            try:
                refine_net.load_state_dict(refine_state, strict=True)
                refine_loaded = True
            except Exception:
                refine_net.load_state_dict(refine_state, strict=False)
                refine_loaded = True
        except Exception:
            refine_loaded = False

    if not refine_loaded:
        # Fallback: SAM-only masks (original size)
        masks_out: List[np.ndarray] = []
        for i in range(prob_full_orig.shape[0]):
            mask_np = (prob_full_orig[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            masks_out.append(mask_np)
        return masks_out

    masks_out: List[np.ndarray] = []

    for start_idx in range(0, total_instances, chunk):
        end_idx = min(total_instances, start_idx + chunk)
        boxes_chunk_1024 = boxes_1024_all[start_idx:end_idx]

        # ROI image crop
        roi_img_256 = _batch_crop_resize_img_bchw_1024(img_bchw_1024, boxes_chunk_1024, out_hw=256)

        # Optional prompt mask for refine
        prompt_mask_256 = None
        if use_prompt:
            prompt_mask_256 = _batch_crop_resize_mask_256(
                sam_prob_256[start_idx:end_idx],
                boxes_256_all[start_idx:end_idx].to(torch.float32),
                out_hw=256,
            )

        logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
        probs_256 = torch.sigmoid(logits_refine_256)

        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            prob_canvas_1024 = _paste_roi_prob_back_1024(
                probs_256[local_idx:local_idx+1], boxes_chunk_1024[local_idx], 1024, 1024
            )

            if fuse_with_sam:
                sam_roi_256 = _batch_crop_resize_mask_256(
                    sam_prob_256[global_idx:global_idx+1],
                    boxes_256_all[global_idx:global_idx+1].to(torch.float32),
                    out_hw=256,
                )
                sam_canvas_1024 = _paste_roi_prob_back_1024(
                    sam_roi_256, boxes_chunk_1024[local_idx], 1024, 1024
                )
                prob_canvas_1024 = 0.5 * prob_canvas_1024 + 0.5 * sam_canvas_1024

            # Degenerate refine fallback: constant-like ROI -> fallback to SAM ROI
            p_local = probs_256[local_idx:local_idx+1].detach()
            m_val = float(p_local.mean().cpu().item())
            s_val = float(p_local.std().cpu().item())
            if s_val < 0.05 or m_val > 0.95 or m_val < 0.05:
                sam_roi_256 = _batch_crop_resize_mask_256(
                    sam_prob_256[global_idx:global_idx+1],
                    boxes_256_all[global_idx:global_idx+1].to(torch.float32),
                    out_hw=256,
                )
                prob_canvas_1024 = _paste_roi_prob_back_1024(
                    sam_roi_256, boxes_chunk_1024[local_idx], 1024, 1024
                )

            # Heuristic: if ROI covers most of canvas and predicted positive covers ~all ROI -> fallback to full-size SAM
            bx = boxes_chunk_1024[local_idx].tolist()
            x1i, y1i, x2i, y2i = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            roi_w = max(1, x2i - x1i)
            roi_h = max(1, y2i - y1i)
            roi_cover = (roi_w * roi_h) / float(1024 * 1024)
            roi_prob = prob_canvas_1024.squeeze(0).squeeze(0).detach().cpu().numpy()
            roi_region = roi_prob[y1i:y2i, x1i:x2i]
            pos_frac = float((roi_region > 0.5).mean()) if roi_region.size > 0 else 0.0
            if roi_cover > 0.50 and pos_frac > 0.98:
                prob_canvas_1024 = _paste_roi_prob_back_1024(
                    _batch_crop_resize_mask_256(
                        sam_prob_256[global_idx:global_idx+1],
                        boxes_256_all[global_idx:global_idx+1].to(torch.float32),
                        out_hw=256,
                    ),
                    boxes_chunk_1024[local_idx], 1024, 1024
                )

            # Map 1024 canvas back to original size
            p1024 = prob_canvas_1024.squeeze(0).squeeze(0).detach().cpu().numpy()
            p_orig = cv2.resize(p1024, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            mask_np = (p_orig > 0.5).astype(np.uint8) * 255
            masks_out.append(mask_np)

    return masks_out


# CLI helpers

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Refine inference (1024 input; original-size masks) [sam_convnext-dep]")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--boxes", type=str, required=True, help="x1,y1,x2,y2;... or JSON path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-type", type=str, default="vit_b_lm")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--refine-ckpt", type=str, default=None)
    p.add_argument("--convnext-variant", type=str, default="atto")
    p.add_argument("--convnext-ckpt", type=str, default=None)
    p.add_argument("--fuse-with-sam", action="store_true", default=False)
    p.add_argument("--crop-source", choices=["sam", "prompt"], default="prompt")
    p.add_argument("--roi-pad-256", type=float, default=0.10)
    p.add_argument("--roi-pad", type=float, default=0.10)
    p.add_argument("--use-prompt", action="store_true", default=True)
    p.add_argument("--no-use-prompt", dest="use_prompt", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    boxes = _parse_boxes_arg(args.boxes)
    masks = infer_one(
        image_path=args.image,
        boxes=boxes,
        device=args.device,
        model_type=args.model_type,
        sam_checkpoint=args.checkpoint,
        refine_resume=args.refine_ckpt,
        convnext_variant="atto",
        convnext_ckpt=args.convnext_ckpt,
        fuse_with_sam=args.fuse_with_sam,
        crop_source=args.crop_source,
        roi_pad_256=args.roi_pad_256,
        roi_pad=args.roi_pad,
        use_prompt=args.use_prompt,
        input_preprocess="resize1024",
        use_refine_net=True,
    )
    for i, m in enumerate(masks):
        out_path = Path(args.image).with_suffix(f".inst{i+1}.png")
        cv2.imwrite(str(out_path), m)


if __name__ == "__main__":
    main()

# Train-like direct path (reference implementation)
@torch.no_grad()
def _tight_boxes_1024_from_masks_tensor(masks_tensor: torch.Tensor) -> torch.Tensor:
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)
    N, H, W = masks_tensor.shape
    assert H == 1024 and W == 1024, 'masks must be on 1024 canvas'
    boxes = []
    for k in range(N):
        m = masks_tensor[k] > 0.5
        ys, xs = torch.where(m)
        if ys.numel() == 0:
            boxes.append(torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=masks_tensor.device))
        else:
            x1 = int(xs.min().item()); x2 = int(xs.max().item()) + 1
            y1 = int(ys.min().item()); y2 = int(ys.max().item()) + 1
            boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.int64, device=masks_tensor.device))
    return torch.stack(boxes, dim=0)

@torch.no_grad()
def run_train_like_masks_direct(
    image_path: str,
    gt_masks: List[np.ndarray],
    sam_model,
    refine_net,
    crop_source: str = 'sam',
    roi_pad_256: float = 0.09,
    roi_pad: float = 0.09,
    use_prompt: bool = True,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    device_t = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    # 1) load + force 1024
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_bchw = torch.from_numpy(rgb_1024).permute(2, 0, 1).unsqueeze(0).float().to(device_t)
    H = W = 1024

    # 2) build GT masks tensor on 1024
    m_list = []
    for m in gt_masks:
        mb = (m.astype(np.uint8) > 0).astype(np.uint8)
        m1024 = cv2.resize(mb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        m_list.append(m1024)
    if not m_list:
        return []
    masks_tensor = torch.from_numpy(np.stack(m_list).astype(np.float32)).to(device_t)  # (G,1024,1024)

    # 3) tight boxes on 1024 as anchors
    gt_boxes_1024 = _tight_boxes_1024_from_masks_tensor(masks_tensor)

    # 4) prompts
    prompts = _prepare_prompts(
        masks=masks_tensor,
        prompt_type='boxes',
        device=device_t,
        image_size=1024,
        mask_input_size=256,
    )
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    out = _sam_forward_batched_for_one_image(
        sam_model,
        img_bchw,
        prompts,
        resize_transform,
        (H, W),
        256,
        device_t,
        precomputed_img_emb=None,
        precomputed_image_pe=None,
    )
    low_res_logits_256 = out['low_res_logits_256']
    boxes_1024_from_prompt = out['boxes_1024'].long()
    sam_prob_256 = torch.sigmoid(low_res_logits_256)

    # 5) ROI boxes
    if crop_source.lower() == 'sam':
        boxes_256 = _bbox_from_mask_256(low_res_logits_256, pad_ratio=roi_pad_256)
        boxes_1024 = boxes_256.clone()
        boxes_1024[:, 0] *= 4; boxes_1024[:, 1] *= 4; boxes_1024[:, 2] *= 4; boxes_1024[:, 3] *= 4
    else:
        x1y1x2y2 = boxes_1024_from_prompt.clone().long()
        w = (x1y1x2y2[:, 2] - x1y1x2y2[:, 0] + 1).clamp(min=1)
        h = (x1y1x2y2[:, 3] - x1y1x2y2[:, 1] + 1).clamp(min=1)
        px = torch.round(w.float() * float(roi_pad)).long()
        py = torch.round(h.float() * float(roi_pad)).long()
        x1y1x2y2[:, 0] = torch.clamp(x1y1x2y2[:, 0] - px, 0, 1024)
        x1y1x2y2[:, 1] = torch.clamp(x1y1x2y2[:, 1] - py, 0, 1024)
        x1y1x2y2[:, 2] = torch.clamp(x1y1x2y2[:, 2] + px + 1, 0, 1024)
        x1y1x2y2[:, 3] = torch.clamp(x1y1x2y2[:, 3] + py + 1, 0, 1024)
        boxes_1024 = x1y1x2y2
        boxes_256 = torch.stack([
            torch.tensor([
                max(0, b[0]//4),
                max(0, b[1]//4),
                min(256, (b[2]+3)//4),
                min(256, (b[3]+3)//4),
            ], device=device_t, dtype=torch.int64)
            for b in boxes_1024
        ], dim=0)

    # 6) Crop ROI and prompt mask
    roi_img_256 = _batch_crop_resize_img_bchw_1024(img_bchw, boxes_1024, out_hw=256)
    prompt_mask_256 = None
    if use_prompt:
        if crop_source.lower() == 'sam':
            boxes_256_prompt = _bbox_from_mask_256(
                torch.log(sam_prob_256/(1 - sam_prob_256 + 1e-6)), pad_ratio=roi_pad_256
            )
        else:
            boxes_256_prompt = boxes_256
        prompt_mask_256 = _batch_crop_resize_mask_256(sam_prob_256, boxes_256_prompt, out_hw=256)

    # 7) Refine + paste back
    logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
    probs_256 = torch.sigmoid(logits_refine_256)
    masks_out: List[np.ndarray] = []
    P = probs_256.shape[0]
    for i in range(P):
        prob_canvas = _paste_roi_prob_back_1024(probs_256[i:i+1], boxes_1024[i], 1024, 1024)
        masks_out.append((prob_canvas.squeeze().detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255)
    return masks_out
