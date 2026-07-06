from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

from orgsam.refine.infer import infer_one
from orgsam.refine import util as refine_util

from sam_convnext_5a import (
    SamCocoDataset,
    _prepare_prompts,
    _sam_forward_batched_for_one_image,
    _bbox_from_mask_256,
    _batch_crop_resize_img_bchw_1024,
    _batch_crop_resize_mask_256,
    _paste_roi_prob_back_1024,
    build_plain_sam,
    ResizeLongestSide,
)

# =========================
# 路径 & 设备
# =========================
DATA_ROOT = Path(r"C:/localtask/digOrg/colon/test")
IMG_DIR = DATA_ROOT / "images"
ANN_PATH = DATA_ROOT / "data.json"
SAM_CKPT = Path(r"C:/localtask/digOrg/models/vit_b_lm.pth")
REFINE_CKPT = Path(r"C:/localtask/digOrg/student_atto_v1_epoch_90.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 数据与 IoU
# =========================
def _first_sample() -> Tuple[int, str, List[dict]]:
    with ANN_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    img_id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    img_to_anns: dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_to_anns.setdefault(ann["image_id"], []).append(ann)
    for img_id, anns in img_to_anns.items():
        if anns:
            file_name = img_id_to_file[img_id]
            img_path = _resolve_image(file_name)
            return img_id, img_path, anns
    raise RuntimeError("No annotated image found.")


def _resolve_image(file_name: str) -> str:
    cand = IMG_DIR / file_name
    if cand.exists():
        return cand.as_posix()
    stem = Path(file_name).stem
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
    for ext in exts:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p.as_posix()
    for ext in exts:
        hits = list(IMG_DIR.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0].as_posix()
    raise FileNotFoundError(f"Cannot resolve image '{file_name}'")


def _ann_masks(anns: List[dict], img_h: int, img_w: int) -> List[np.ndarray]:
    masks = []
    for ann in anns:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, img_h, img_w)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            rle = seg if isinstance(seg.get("counts"), (bytes, bytearray)) else mask_utils.frPyObjects(seg, img_h, img_w)
        else:
            continue
        m = mask_utils.decode(rle)
        masks.append(m.astype(bool))
    return masks


def _resize_to_1024(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)


def _pairwise_iou(preds: List[np.ndarray], gts: List[np.ndarray]) -> float:
    n = min(len(preds), len(gts))
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        pred = _resize_to_1024(preds[i] > 0)
        gt = _resize_to_1024(gts[i])
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        total += (inter / union) if union > 0 else 0.0
    return total / n


# =========================
# 推理侧保序：锚定 + 重排（非匈T利）
# =========================
def _tight_boxes_1024_from_masks_tensor(masks_tensor: torch.Tensor) -> torch.Tensor:
    """
    输入: masks_tensor (N,H,W) 或 (N,1,H,W)，值域 [0,1]/{0,1}，H/W 可非 1024。
    输出: (N,4) [x1,y1,x2,y2) 半开区间坐标，映射到 1024×1024 网格。
    """
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)
    N, H, W = masks_tensor.shape
    if (H, W) != (1024, 1024):
        masks_tensor = F.interpolate(masks_tensor.unsqueeze(1), size=(1024, 1024), mode="nearest").squeeze(1)
    boxes = []
    for k in range(masks_tensor.shape[0]):
        m = (masks_tensor[k] > 0.5)
        ys, xs = torch.where(m)
        if xs.numel() == 0:
            boxes.append(torch.tensor([0, 0, 1024, 1024], device=masks_tensor.device, dtype=torch.int64))
        else:
            x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
            y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
            boxes.append(torch.tensor([x1, y1, x2, y2], device=masks_tensor.device, dtype=torch.int64))
    return torch.stack(boxes, dim=0) if boxes else torch.zeros((0, 4), dtype=torch.int64, device=masks_tensor.device)


def _map_pred_to_gt_order_greedy(boxes_pred_1024: torch.Tensor, boxes_gt_1024: torch.Tensor) -> List[int]:
    """
    用中心点距离贪心匹配，返回 list: pred_idx -> gt_idx。
    """
    def _ctr(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    P = int(boxes_pred_1024.shape[0])
    G = int(boxes_gt_1024.shape[0])
    if P == 0 or G == 0:
        return list(range(P))

    pairs = []
    for i in range(P):
        cxp, cyp = _ctr(boxes_pred_1024[i].tolist())
        for j in range(G):
            cxg, cyg = _ctr(boxes_gt_1024[j].tolist())
            d2 = (cxp - cxg) ** 2 + (cyp - cyg) ** 2
            pairs.append((d2, i, j))
    pairs.sort(key=lambda x: x[0])

    mapping = [-1] * P
    pred_used, gt_used = set(), set()
    for _, i, j in pairs:
        if i in pred_used or j in gt_used:
            continue
        mapping[i] = j
        pred_used.add(i)
        gt_used.add(j)
        if len(pred_used) == min(P, G):
            break

    remainder = [j for j in range(G) if j not in gt_used]
    ridx = 0
    for i in range(P):
        if mapping[i] == -1:
            mapping[i] = remainder[ridx] if ridx < len(remainder) else min(G - 1, i)
            ridx += 1
    return mapping


def _reorder_masks_to_gt(masks_pred_list_1024: List[np.ndarray],
                         mapping_pred_to_gt: List[int],
                         gt_len: int) -> List[np.ndarray]:
    """
    将预测掩码（按 pred 顺序）重排为 GT 顺序（长度 = gt_len）。
    未匹配的 GT 位置填零掩码。
    """
    out = [None] * gt_len
    H = W = 1024
    zero = np.zeros((H, W), dtype=np.uint8)
    for p_idx, g_idx in enumerate(mapping_pred_to_gt):
        if 0 <= g_idx < gt_len:
            out[g_idx] = masks_pred_list_1024[p_idx]
    for i in range(gt_len):
        if out[i] is None:
            out[i] = zero
    return out


# =========================
# 模型构建 & 推理
# =========================
def _build_models():
    sam_model = build_plain_sam("vit_b_lm", str(SAM_CKPT), DEVICE)
    refine_net = refine_util.ROIRefineNet(
        variant="atto",
        convnext_ckpt=None,
        use_prompt=True,
        build_prompt=False,
    ).to(DEVICE).eval()
    ckpt = torch.load(REFINE_CKPT, map_location="cpu")
    refine_state = ckpt.get("refine", ckpt)
    refine_net.load_state_dict(refine_state, strict=False)
    return sam_model, refine_net


def run_simple_infer(img_path: str, boxes: List[List[float]]) -> List[np.ndarray]:
    return infer_one(
        image_path=img_path,
        boxes=boxes,
        device=str(DEVICE),
        model_type="vit_b_lm",
        sam_checkpoint=str(SAM_CKPT),
        refine_resume=str(REFINE_CKPT),
        convnext_variant="atto",
        convnext_ckpt=None,
        fuse_with_sam=False,
        crop_source="prompt",
        roi_pad_256=0.10,
        roi_pad=0.10,
        use_prompt=True,
        input_preprocess="resize1024",
        use_refine_net=True,
    )


def run_train_like_masks(
    image_id: int,
    sam_model,
    refine_net,
    eval_gt_masks: List[np.ndarray],  # ★ 评估用 GT（与 IoU 对齐）
) -> List[np.ndarray]:
    """
    dataset 方式推理；prompts 用 dataset 的 masks，但回排顺序用 eval_gt_masks 做锚，
    确保与评估侧 gt_masks 的实例顺序一致（无需匈牙利）。
    """
    dataset = SamCocoDataset(
        image_root=IMG_DIR,
        annotation_path=ANN_PATH,
        image_size=1024,
        use_augmentation=False,
        prompt_type="boxes",
        allow_empty=False,
    )
    idx = dataset.image_ids.index(image_id)
    sample = dataset[idx]
    image_bchw = sample["image"].unsqueeze(0).to(DEVICE)  # (1,3,1024,1024)
    masks_tensor_ds = sample["masks"].to(DEVICE)          # 仅用于 prompts

    # --- 用“评估用”的 gt_masks 作为锚，映射到 1024 ---
    m_list = []
    for m in eval_gt_masks:
        mb = (m.astype(np.uint8) > 0).astype(np.uint8)
        m_list.append(cv2.resize(mb, (1024, 1024), interpolation=cv2.INTER_NEAREST))
    masks_tensor_eval = torch.from_numpy(np.stack(m_list).astype(np.float32)).to(DEVICE)  # (G,1024,1024)
    gt_boxes_1024 = _tight_boxes_1024_from_masks_tensor(masks_tensor_eval)              # (G,4)

    # prompts（按训练习惯）
    prompts = _prepare_prompts(
        masks=masks_tensor_ds,
        prompt_type="boxes",
        device=DEVICE,
        image_size=1024,
        mask_input_size=256,
    )
    if not prompts:
        return []

    args = SimpleNamespace(
        crop_source="sam",
        roi_pad_256=0.10,
        roi_pad=0.10,
        use_prompt=True,
        fuse_with_sam=False,
        mask_threshold=0.5,
        dice_loss_weight=1.0,
        mask_input_size=256,
        instance_chunk_size=0,
    )
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    masks_out_pred_order: List[np.ndarray] = []
    with torch.no_grad():
        sam_out = _sam_forward_batched_for_one_image(
            sam_model,
            image_bchw,
            prompts,
            resize_transform,
            sample["orig_size"],
            args.mask_input_size,
            DEVICE,
            precomputed_img_emb=None,
            precomputed_image_pe=None,
        )
        low_res_logits_256 = sam_out["low_res_logits_256"]        # (P,1,256,256)
        boxes_1024_from_prompt = sam_out["boxes_1024"].long()      # (P,4)
        sam_prob_256 = torch.sigmoid(low_res_logits_256)

        # ROI（仅用于裁剪/粘贴）
        boxes_256 = _bbox_from_mask_256(low_res_logits_256, pad_ratio=args.roi_pad_256)
        boxes_1024 = boxes_256.clone()
        boxes_1024[:, 0] *= 4; boxes_1024[:, 1] *= 4; boxes_1024[:, 2] *= 4; boxes_1024[:, 3] *= 4

        roi_img_256 = _batch_crop_resize_img_bchw_1024(image_bchw, boxes_1024, out_hw=256)

        boxes_256_prompt = _bbox_from_mask_256(
            torch.log(sam_prob_256 / (1 - sam_prob_256 + 1e-6)),
            pad_ratio=args.roi_pad_256,
        )
        prompt_mask_256 = _batch_crop_resize_mask_256(sam_prob_256, boxes_256_prompt, out_hw=256)

        logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
        probs_256 = torch.sigmoid(logits_refine_256)

        # 先按预测顺序做 1024 掩码
        P = probs_256.shape[0]
        for i in range(P):
            prob_canvas = _paste_roi_prob_back_1024(probs_256[i:i+1], boxes_1024[i], 1024, 1024)
            mask_np = (prob_canvas.squeeze().detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
            masks_out_pred_order.append(mask_np)

        # 用“评估 GT 锚”把预测顺序映射回评估顺序
        mapping = _map_pred_to_gt_order_greedy(boxes_1024_from_prompt, gt_boxes_1024)
        masks_out_gt_order = _reorder_masks_to_gt(masks_out_pred_order, mapping, gt_len=gt_boxes_1024.shape[0])

    return masks_out_gt_order


def run_train_like_masks_direct(
    image_path: str,
    gt_masks: List[np.ndarray],
    refine_net,
    roi_pad: float = 0.3,
) -> List[np.ndarray]:
    """
    ★ 修改版：完全绕过 SAM ★
    - 不运行 SAM forward。
    - 直接使用 GT 掩码计算的 boxes (gt_boxes_1024) 作为裁剪基准。
    - 使用 'roi_pad' 参数对 gt_boxes_1024 进行扩充。
    - 将裁剪后的图像 (roi_img_256) 直接送入 refine_net，不带任何掩码提示。
    - 输出掩码的顺序
    """
    # load image and force 1024 RGB
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_bchw = torch.from_numpy(rgb_1024).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    H = W = 1024

    # resize GT masks to 1024 and build tensor
    m_list = []
    for m in gt_masks:
        mb = (m.astype(np.uint8) > 0).astype(np.uint8)
        m1024 = cv2.resize(mb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        m_list.append(m1024)
    if not m_list:
        return []

    masks_tensor = torch.from_numpy(np.stack(m_list).astype(np.float32)).to(DEVICE)  # (G,1024,1024)

    # GT 1024 紧致 bbox 作为“锚”
    gt_boxes_1024 = _tight_boxes_1024_from_masks_tensor(masks_tensor)  # (G,4)

    # --- 移除 SAM Forward ---
    # 不再运行 _sam_forward_batched_for_one_image
    # 不再有 low_res_logits_256, boxes_1024_from_prompt, sam_prob_256

    # --- 裁剪逻辑：使用带 padding 的 GT 框 ---
    boxes_1024_crop = gt_boxes_1024.clone().long()
    # 注意：[x1, y1, x2, y2) 是半开区间，宽度 = x2 - x1
    w = (boxes_1024_crop[:, 2] - boxes_1024_crop[:, 0]).clamp(min=1)
    h = (boxes_1024_crop[:, 3] - boxes_1024_crop[:, 1]).clamp(min=1)
    px = torch.round(w.float() * float(roi_pad)).long()
    py = torch.round(h.float() * float(roi_pad)).long()
    
    boxes_1024_crop[:, 0] = torch.clamp(boxes_1024_crop[:, 0] - px, 0, 1024)
    boxes_1024_crop[:, 1] = torch.clamp(boxes_1024_crop[:, 1] - py, 0, 1024)
    boxes_1024_crop[:, 2] = torch.clamp(boxes_1024_crop[:, 2] + px, 0, 1024)
    boxes_1024_crop[:, 3] = torch.clamp(boxes_1024_crop[:, 3] + py, 0, 1024)
    
    # 裁剪图像
    roi_img_256 = _batch_crop_resize_img_bchw_1024(img_bchw, boxes_1024_crop, out_hw=256)
    
    # --- 提示逻辑：无提示 ---
    prompt_mask_256 = None

    # --- 运行 RefineNet ---
    with torch.no_grad():
        logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
        probs_256 = torch.sigmoid(logits_refine_256)

    # --- 粘贴回掩码 ---
    # 因为我们是按 GT 顺序裁剪的，所以 refine_net 的输出 (G, ...) 
    # 已经与 GT 顺序对齐。不需要贪心匹配。
    masks_out_gt_order: List[np.ndarray] = []
    for i in range(probs_256.shape[0]):
        prob_canvas = _paste_roi_prob_back_1024(probs_256[i:i+1], boxes_1024_crop[i], 1024, 1024)
        mask_np = (prob_canvas.squeeze().detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
        masks_out_gt_order.append(mask_np)

    # --- 移除 贪心匹配 和 重排序 ---

    return masks_out_gt_order


# =========================
# 主流程
# =========================
def main():
    image_id, image_path, anns = _first_sample()
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    gt_masks = _ann_masks(anns, img_bgr.shape[0], img_bgr.shape[1])

    # 供 simple_infer 使用的 box 提示（来自 GT）
    boxes: List[List[float]] = []
    for mask in gt_masks:
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            boxes.append([0.0, 0.0, float(img_bgr.shape[1]), float(img_bgr.shape[0])])
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes.append([float(x1), float(y1), float(x2 + 1), float(y2 + 1)])

    # --- 路径 1：简单推理 ---
    print("--- 运行路径 1: simple_infer ---")
    simple_masks = run_simple_infer(image_path, boxes)
    simple_iou = _pairwise_iou(simple_masks, gt_masks)
    print(f"[simple_infer_demo]   avg IoU (train-style metric): {simple_iou:.4f}")

    # --- 构建模型 (路径 2 & 3 共用) ---
    sam_model, refine_net = _build_models()

    # --- 路径 2：train-like (SAM+Refine) ---
    print("\n--- 运行路径 2: train_like (SAM+Refine) ---")
    train_like_masks = run_train_like_masks(image_id, sam_model, refine_net, gt_masks)
    train_like_iou = _pairwise_iou(train_like_masks, gt_masks)
    print(f"[sam_convnext_5a]     avg IoU (gt-order, no Hungarian): {train_like_iou:.4f}")

    # --- 路径 3：train-like (Direct GT Crop + Refine)，测试不同 padding ---
    print("\n--- 运行路径 3: Direct GT Crop + Refine (测试不同 padding) ---")
    pad_results = {}
    padding_ratios = [0.1, 0.3, 0.5] # 您想测试的扩框比例
    
    for pad in padding_ratios:
        masks = run_train_like_masks_direct(
            image_path, gt_masks, refine_net, roi_pad=pad
        )
        iou = _pairwise_iou(masks, gt_masks)
        pad_results[pad] = (iou, masks)
        print(f"[Direct GT Crop @ pad={pad:.1f}] avg IoU: {iou:.4f}")


    # --- 可视化 ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _make_overlay(rgb_img: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        overlay = rgb_img.copy().astype(np.float32)
        alpha = 0.5
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (128, 255, 0),
        ]
        for i, m in enumerate(masks):
            color = np.array(palette[i % len(palette)], dtype=np.float32)
            mask_bin = (cv2.resize(m, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
            if mask_bin.any():
                overlay[mask_bin] = overlay[mask_bin] * (1 - alpha) + color * alpha
        return overlay.clip(0, 255).astype(np.uint8)

    # 准备 2x3 布局的绘图
    plt.figure(figsize=(24, 12)) # 增大画布
    
    # 1. Ground Truth
    overlay_gt = _make_overlay(img_rgb, [m.astype(np.uint8) * 255 for m in gt_masks])
    plt.subplot(2, 3, 1)
    plt.imshow(overlay_gt)
    plt.title("Ground Truth Masks")
    plt.axis("off")

    # 2. 路径 1 (Simple Infer)
    overlay_simple = _make_overlay(img_rgb, simple_masks)
    plt.subplot(2, 3, 2)
    plt.imshow(overlay_simple)
    plt.title(f"Path 1 (Simple Infer) IoU={simple_iou:.3f}")
    plt.axis("off")

    # 3. 路径 2 (Train-like SAM+Refine)
    overlay_train_like = _make_overlay(img_rgb, train_like_masks)
    plt.subplot(2, 3, 3)
    plt.imshow(overlay_train_like)
    plt.title(f"Path 2 (Train-like) IoU={train_like_iou:.3f}")
    plt.axis("off")
    
    # 4. 路径 3 (Pad=0.1)
    iou_p01, masks_p01 = pad_results[0.1]
    overlay_p01 = _make_overlay(img_rgb, masks_p01)
    plt.subplot(2, 3, 4)
    plt.imshow(overlay_p01)
    plt.title(f"Path 3 (Direct GT Crop) Pad=0.1\nIoU={iou_p01:.3f}")
    plt.axis("off")
    
    # 5. 路径 3 (Pad=0.3)
    iou_p03, masks_p03 = pad_results[0.3]
    overlay_p03 = _make_overlay(img_rgb, masks_p03)
    plt.subplot(2, 3, 5)
    plt.imshow(overlay_p03)
    plt.title(f"Path 3 (Direct GT Crop) Pad=0.3\nIoU={iou_p03:.3f}")
    plt.axis("off")
    
    # 6. 路径 3 (Pad=0.5)
    iou_p05, masks_p05 = pad_results[0.5]
    overlay_p05 = _make_overlay(img_rgb, masks_p05)
    plt.subplot(2, 3, 6)
    plt.imshow(overlay_p05)
    plt.title(f"Path 3 (Direct GT Crop) Pad=0.5\nIoU={iou_p05:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()