#!/usr/bin/env python
from __future__ import annotations

"""
Video-level text + refine inference utilities.

提供两个推理接口（方案一 / 方案二）：

1) track_and_refine
   - 调用已有的 `scripts/text_to_box_sam2_tracking.py` 做：
       文本检测 -> box 提示 -> SAM2 tracker 视频追踪（coarse 掩码）；
   - 然后对追踪得到的 coarse 掩码逐帧做 ROIRefineNet 细化；
   - 输出：refined 掩码 + 可视化视频。

2) refine_from_index
   - 假定你已经有 coarse 掩码（由 text_to_box_sam2_tracking.py 生成）：
       /.../masks/frame_XXXX_obj_YY.npy + mask_index.json；
   - 仅对这些掩码做 ROIRefineNet 细化，并保存 refined 掩码 + 视频。
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .util import ROIRefineNet, _mask_to_box


@torch.no_grad()
def _build_refine_net(
    refine_ckpt: Optional[Path],
    device: str = "cuda",
) -> ROIRefineNet:
    device_t = torch.device(device)
    refine_net = ROIRefineNet(
        variant="atto", convnext_ckpt=None, use_prompt=True, build_prompt=False
    )
    if refine_ckpt is not None and refine_ckpt.is_file():
        ckpt = torch.load(refine_ckpt, map_location="cpu")
        state_dict = ckpt.get("refine", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        refine_net.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded refine checkpoint: {refine_ckpt}")
    else:
        print("[WARN] No valid refine checkpoint provided; using random-initialized ROIRefineNet.")
    refine_net.to(device_t).eval()
    return refine_net


@torch.no_grad()
def _refine_single_mask_on_frame(
    frame_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    refine_net: ROIRefineNet,
    device: str = "cuda",
    roi_pad: float = 0.1,
) -> np.ndarray:
    """
    对单帧上的一个 coarse_mask 做 ROIRefineNet 细化。

    Args:
        frame_bgr: 原视频帧 (H,W,3) BGR uint8
        coarse_mask: (H,W) 0/1 或 bool

    Returns:
        refine_mask: (H,W) uint8, 0/1
    """
    device_t = torch.device(device)
    H, W = frame_bgr.shape[:2]

    m_bool = coarse_mask.astype(bool)
    # 1) 计算 bbox
    bbox = _mask_to_box((m_bool.astype(np.uint8) * 255))
    if bbox is None:
        return coarse_mask.astype(np.uint8)
    x1, y1, x2, y2 = bbox.tolist()
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    px = w * float(roi_pad)
    py = h * float(roi_pad)
    x1p = max(0.0, float(x1) - px)
    y1p = max(0.0, float(y1) - py)
    x2p = min(float(W), float(x2 + 1) + px)
    y2p = min(float(H), float(y2 + 1) + py)
    if x2p <= x1p or y2p <= y1p:
        return coarse_mask.astype(np.uint8)

    x1i, y1i = int(x1p), int(y1p)
    x2i, y2i = int(x2p), int(y2p)

    # 2) 裁剪 ROI 图像 & coarse mask
    roi_img = frame_bgr[y1i:y2i, x1i:x2i, :]  # H_roi, W_roi, 3
    roi_mask = m_bool[y1i:y2i, x1i:x2i]        # H_roi, W_roi
    if roi_img.size == 0:
        return coarse_mask.astype(np.uint8)

    # 3) resize 到 256x256
    roi_img_256 = cv2.resize(roi_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    roi_mask_256 = cv2.resize(
        roi_mask.astype(np.float32),
        (256, 256),
        interpolation=cv2.INTER_NEAREST,
    )

    # 4) 构建网络输入
    img_bchw = (
        torch.from_numpy(roi_img_256)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device_t)
    )  # [1,3,256,256]
    prompt_mask = (
        torch.from_numpy(roi_mask_256)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device_t)
    )  # [1,1,256,256]

    logits_refine = refine_net(img_bchw, prompt_mask_bchw_256=prompt_mask)
    refine_prob_256 = torch.sigmoid(logits_refine)  # [1,1,256,256]

    # 5) paste back to full image
    # 先 resize 回 ROI 尺寸
    refine_prob_roi = F.interpolate(
        refine_prob_256,
        size=(y2i - y1i, x2i - x1i),
        mode="bilinear",
        align_corners=False,
    )[0, 0]  # (H_roi, W_roi)
    refine_prob_roi_np = refine_prob_roi.detach().cpu().numpy()

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[y1i:y2i, x1i:x2i] = refine_prob_roi_np

    # 最后阈值化
    refine_mask = (canvas > 0.5).astype(np.uint8)
    return refine_mask


@torch.no_grad()
def refine_video_masks_from_index(
    video_path: str,
    mask_index_json: str,
    refine_ckpt: Optional[str] = None,
    device: str = "cuda",
    roi_pad: float = 0.1,
    out_dir: Optional[str] = None,
) -> None:
    """
    方案二：对已有的 SAM2 追踪掩码做离线 refine。

    Args:
        video_path: 原始视频路径
        mask_index_json: text_to_box_sam2_tracking.py 生成的 mask_index.json
        refine_ckpt: ROIRefineNet 的 checkpoint（例如 student_atto_v1_epoch_140latest.pt）
        out_dir: 输出路径（refined 掩码与视频）
    """
    video_path = str(video_path)
    mask_index_json = str(mask_index_json)
    base_dir = Path(out_dir) if out_dir is not None else Path(
        Path(mask_index_json).parent
    ) / "refined"
    base_dir.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    refine_net = _build_refine_net(
        Path(refine_ckpt) if refine_ckpt is not None else None, device=device
    )

    # 读取掩码索引
    index_data = json.loads(Path(mask_index_json).read_text())

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 简单的帧缓存，避免重复 seek
    frame_cache: Dict[int, np.ndarray] = {}

    refined_masks_dir = base_dir / "refined_masks"
    refined_masks_dir.mkdir(parents=True, exist_ok=True)
    refined_index: Dict[str, List[Dict]] = {}

    for obj_id_str, entries in index_data.items():
        for item in entries:
            f_idx = int(item["frame_idx"])
            mask_path = Path(item["path"])
            if not mask_path.is_file():
                print(f"[WARN] mask file not found: {mask_path}")
                continue

            # 读视频帧
            if f_idx not in frame_cache:
                if f_idx < 0 or (total_frames > 0 and f_idx >= total_frames):
                    print(f"[WARN] frame_idx {f_idx} out of range, skip.")
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARN] failed to read frame {f_idx}, skip.")
                    continue
                frame_cache[f_idx] = frame
            frame_bgr = frame_cache[f_idx]

            # 读 coarse mask
            coarse_mask = np.load(mask_path)
            if coarse_mask.ndim == 3:
                coarse_mask = coarse_mask[0]
            Hf, Wf = frame_bgr.shape[:2]
            if coarse_mask.shape != (Hf, Wf):
                coarse_mask = cv2.resize(
                    coarse_mask.astype(np.uint8),
                    (Wf, Hf),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)

            # refine
            refine_mask = _refine_single_mask_on_frame(
                frame_bgr=frame_bgr,
                coarse_mask=coarse_mask,
                refine_net=refine_net,
                device=device,
                roi_pad=roi_pad,
            )

            # 保存 refine mask
            fname = f"frame_{f_idx:04d}_obj_{obj_id_str}.npy"
            out_path = refined_masks_dir / fname
            np.save(out_path, refine_mask.astype(np.uint8))
            refined_index.setdefault(obj_id_str, []).append(
                {"frame_idx": f_idx, "path": str(out_path)}
            )

    cap.release()

    refined_index_path = base_dir / "refined_mask_index.json"
    refined_index_path.write_text(json.dumps(refined_index, indent=2), encoding="utf-8")
    print(f"[INFO] Saved refined masks under: {refined_masks_dir}")
    print(f"[INFO] Refined mask index saved to: {refined_index_path}")

    # 同时保存一个可视化视频
    out_video_path = base_dir / "tracking_overlay_refined.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    ret, sample = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame for writer init")
    H, W = sample.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_video_path}")

    # 读 refine_index 方便快速查找
    refine_map: Dict[int, Dict[int, np.ndarray]] = {}
    for obj_id_str, entries in refined_index.items():
        oid = int(obj_id_str)
        for item in entries:
            f_idx = int(item["frame_idx"])
            m = np.load(item["path"]).astype(bool)
            refine_map.setdefault(f_idx, {})[oid] = m

    frame_idx = 0
    alpha = 0.5
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.astype(np.float32)

        if frame_idx in refine_map:
            frame_objs = refine_map[frame_idx]
            for idx, (oid, mask) in enumerate(frame_objs.items()):
                if mask.shape != (H, W):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                color = np.array(palette[idx % len(palette)], dtype=np.float32)
                m3 = np.repeat(mask[..., None], 3, axis=2)
                color_img = np.zeros_like(overlay, dtype=np.float32)
                color_img[..., 0] = color[0]
                color_img[..., 1] = color[1]
                color_img[..., 2] = color[2]
                blended = overlay * (1.0 - alpha) + color_img * alpha
                overlay = np.where(m3, blended, overlay)

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[INFO] Refined overlay video saved to: {out_video_path}")


def track_and_refine_video_text(
    video_path: str,
    text_prompt: str = "cell",
    frame_index: int = 0,
    det_conf_thr: float = 0.3,
    use_original_weights: bool = True,
    max_objects: int = 10,
    refine_ckpt: Optional[str] = None,
    device: str = "cuda",
    roi_pad: float = 0.1,
    out_dir: Optional[str] = None,
) -> None:
    """
    方案一：在一个接口内完成
      文本检测 -> SAM2 追踪 (coarse masks) -> ROIRefineNet 细化。

    实现上是：
      1) 通过子进程调用 text_to_box_sam2_tracking.py 生成 coarse masks + mask_index；
      2) 调用 refine_video_masks_from_index 做 refine。
    """
    proj_root = Path(__file__).resolve().parents[2]
    video_path = str(video_path)
    out_dir = (
        Path(out_dir)
        if out_dir is not None
        else proj_root / "output" / "text_to_box_sam2_track_and_refine"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tracking_script = proj_root / "scripts" / "text_to_box_sam2_tracking.py"
    if not tracking_script.is_file():
        raise FileNotFoundError(f"Tracking script not found at {tracking_script}")

    # 1) 调用 tracking 脚本
    cmd = [
        "python",
        str(tracking_script),
        "--video-path",
        video_path,
        "--text-prompt",
        text_prompt,
        "--frame-index",
        str(frame_index),
        "--det-conf-thr",
        str(det_conf_thr),
        "--obj-id",
        "1",
        "--max-objects",
        str(max_objects),
        "--out-dir",
        str(out_dir),
    ]
    if use_original_weights:
        cmd.append("--use-original-weights")

    print("[INFO] Running tracking via:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2) refine
    mask_index_json = out_dir / "mask_index.json"
    if not mask_index_json.is_file():
        raise FileNotFoundError(f"mask_index.json not found at {mask_index_json}")
    refine_video_masks_from_index(
        video_path=video_path,
        mask_index_json=str(mask_index_json),
        refine_ckpt=refine_ckpt,
        device=device,
        roi_pad=roi_pad,
        out_dir=str(out_dir / "refine"),
    )


def geo_track_and_refine_video(
    video_path: str,
    geo_json: str,
    refine_ckpt: Optional[str] = None,
    device: str = "cuda",
    roi_pad: float = 0.1,
    out_dir: Optional[str] = None,
) -> None:
    """
    方案三：仅使用几何提示（box）进行 SAM2 追踪 + ROIRefineNet 细化。

    设计目的：
        - 不再依赖文本检测，直接从几何框出发进行追踪；
        - 支持多帧、多次 box 提示（例如 0 帧 / 中间帧 / 末帧都给标注）。

    geo_json 支持两种格式：

    1) 单帧 / 多框（兼容 best_detection_text_to_box.json）::

        {
          "frame_index": 0,
          "boxes_xyxy": [[x1,y1,x2,y2], ...],
          // 可选: "obj_ids": [1,2,...]
        }

    2) 多帧 / 多框::

        {
          "prompts": [
            {
              "frame_index": 0,
              "boxes_xyxy": [[...], [...]],
              "obj_ids": [1,2]   // 可选，不给则自动递增分配
            },
            {
              "frame_index": 10,
              "boxes_xyxy": [[...]],
              "obj_ids": [1]     // 可以对已有 obj_id 做中间帧纠正
            }
          ]
        }
    """
    from sam3.model_builder import build_sam3_video_model

    proj_root = Path(__file__).resolve().parents[2]
    video_path = str(video_path)
    geo_json = str(geo_json)

    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else proj_root / "output" / "geo_track_and_refine"
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) 读取几何提示 JSON
    meta = json.loads(Path(geo_json).read_text())
    if "prompts" in meta:
        prompts = meta["prompts"]
    else:
        # 兼容 best_detection_text_to_box.json 这类单帧多框格式
        prompts = [meta]

    if not prompts:
        raise RuntimeError(f"No prompts found in geo_json: {geo_json}")

    # 2) 打开视频，获取尺寸与帧数
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame for size")
    H, W = frame0.shape[:2]
    cap.release()

    print(
        f"[INFO] Video opened for geo_track_and_refine: {video_path}, "
        f"size={W}x{H}, total_frames={total_frames}"
    )

    # 3) 构建 SAM2-style tracker
    ckpt = proj_root / "weights" / "sam3.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")
    print(f"[INFO] Building SAM3 video model from {ckpt} ...")
    model = build_sam3_video_model(
        checkpoint_path=str(ckpt),
        load_from_HF=False,
        device=device,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print(f"[INFO] Initializing SAM2 tracker state on video: {video_path}")
    inference_state = predictor.init_state(video_path=str(video_path))

    # 4) 依次把所有几何 box 提示加到 tracker 中（支持多帧、多 obj_id）
    all_prompt_frames: List[int] = []
    next_obj_id = 1

    for entry in prompts:
        frame_idx = int(entry.get("frame_index", 0))
        all_prompt_frames.append(frame_idx)

        # 支持两种键名：boxes_xyxy 或 单个 box_xyxy
        boxes_xyxy = entry.get("boxes_xyxy", None)
        if boxes_xyxy is None and "box_xyxy" in entry:
            boxes_xyxy = [entry["box_xyxy"]]
        if boxes_xyxy is None:
            raise RuntimeError(
                f"Entry in geo_json missing 'boxes_xyxy' or 'box_xyxy': {entry}"
            )

        obj_ids_entry = entry.get("obj_ids", None)
        if obj_ids_entry is not None and len(obj_ids_entry) != len(boxes_xyxy):
            raise RuntimeError(
                "Length of obj_ids must match boxes_xyxy when provided."
            )

        rel_boxes = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(float, b)
            rel_boxes.append([x1 / W, y1 / H, x2 / W, y2 / H])

        for i, rel_box in enumerate(rel_boxes):
            if obj_ids_entry is not None:
                ann_obj_id = int(obj_ids_entry[i])
            else:
                ann_obj_id = next_obj_id
                next_obj_id += 1

            print(
                f"[INFO] Adding geo box prompt on frame {frame_idx}, "
                f"obj_id={ann_obj_id}, rel_box={rel_box}"
            )
            _, out_obj_ids, low_res_masks, video_res_masks = (
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    points=None,
                    labels=None,
                    box=np.array([rel_box], dtype=np.float32),
                )
            )

    # 5) 整段视频 propagate，得到 coarse masks
    start_frame_idx = min(all_prompt_frames)
    print(
        f"[INFO] Running SAM2 propagate_in_video from frame {start_frame_idx} "
        f"over the whole video (geo prompts)."
    )
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for (
        f_idx,
        obj_ids,
        low_res_masks,
        video_res_masks,
        obj_scores,
    ) in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=total_frames,
        reverse=False,
        propagate_preflight=True,
    ):
        if video_res_masks is None or len(obj_ids) == 0:
            continue
        masks_bool = (video_res_masks > 0.0).cpu().numpy()
        frame_dict: Dict[int, np.ndarray] = {}
        for i, oid in enumerate(obj_ids):
            m = masks_bool[i]
            if m.ndim == 3:
                m = m[0]
            frame_dict[int(oid)] = m.astype(bool)
        video_segments[f_idx] = frame_dict

    print(
        f"[INFO] SAM2 geo tracking finished. "
        f"Frames with masks: {sorted(video_segments.keys())}"
    )

    # 6) 保存 coarse masks + index（格式与 text_to_box_sam2_tracking.py 保持一致）
    masks_dir = out_dir_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    index_meta: Dict[str, List[Dict]] = {}
    for f_idx, obj_dict in video_segments.items():
        for oid, mask in obj_dict.items():
            fname = f"frame_{f_idx:04d}_obj_{oid}.npy"
            fpath = masks_dir / fname
            np.save(fpath, mask.astype(np.uint8))
            index_meta.setdefault(str(oid), []).append(
                {"frame_idx": int(f_idx), "path": str(fpath)}
            )
    index_path = out_dir_path / "mask_index.json"
    index_path.write_text(json.dumps(index_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Saved geo coarse masks under: {masks_dir}")
    print(f"[INFO] Geo mask index saved to: {index_path}")

    # 7) 粗追踪可视化（多实例调色板）
    out_video_path = out_dir_path / "tracking_overlay_geo_boxes.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    ret, sample = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame for writer init")
    Hh, Ww = sample.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (Ww, Hh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_video_path}")

    frame_idx = 0
    alpha = 0.5
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.astype(np.float32)

        if frame_idx in video_segments:
            obj_dict = video_segments[frame_idx]
            for idx, (oid, mask) in enumerate(obj_dict.items()):
                if mask.shape != (Hh, Ww):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (Ww, Hh),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                color = np.array(palette[idx % len(palette)], dtype=np.float32)
                m3 = np.repeat(mask[..., None], 3, axis=2)
                color_img = np.zeros_like(overlay, dtype=np.float32)
                color_img[..., 0] = color[0]
                color_img[..., 1] = color[1]
                color_img[..., 2] = color[2]
                blended = overlay * (1.0 - alpha) + color_img * alpha
                overlay = np.where(m3, blended, overlay)

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        writer.write(overlay)
        frame_idx += 1
        if total_frames > 0 and frame_idx >= total_frames:
            break

    cap.release()
    writer.release()
    print(f"[INFO] Geo tracking visualization saved to: {out_video_path}")

    # 8) 调用统一的 refine 逻辑，对上述 coarse 掩码做细化
    if refine_ckpt is not None:
        print("[INFO] Running ROIRefineNet on geo masks ...")
        refine_video_masks_from_index(
            video_path=video_path,
            mask_index_json=str(index_path),
            refine_ckpt=refine_ckpt,
            device=device,
            roi_pad=roi_pad,
            out_dir=str(out_dir_path / "refine"),
        )
    else:
        print("[WARN] refine_ckpt is None; skip refine stage for geo_track_and_refine.")


@torch.no_grad()
def text_geo_track_and_refine_video(
    video_path: str,
    text_prompt: str = "cell",
    det_conf_thr: float = 0.3,
    use_original_weights: bool = True,
    max_objects: int = 10,
    max_prompt_frames: int = 3,
    refine_ckpt: Optional[str] = None,
    device: str = "cuda",
    roi_pad: float = 0.1,
    out_dir: Optional[str] = None,
) -> None:
    """
    方案四：自动从视频中选取若干帧做文本检测 -> 几何提示 -> SAM2 追踪 + refine。

    流程：
      1) 用图像版 SAM3 文本检测器，对视频每一帧做检测；
      2) 对每帧保留分数 >= det_conf_thr 的前 max_objects 个框；
      3) 按帧内最高分，从所有帧中选出最多 max_prompt_frames 帧作为几何提示帧；
      4) 将这些帧和框打包成 geo_json，调用 geo_track_and_refine_video 完成追踪 + refine。
    """

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    proj_root = Path(__file__).resolve().parents[2]
    video_path = str(video_path)

    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else proj_root / "output" / "text_geo_track_and_refine"
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) 构建图像级文本检测模型
    bpe_path = proj_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    base_ckpt = proj_root / "weights" / "sam3.pt"
    finetune_ckpt = (
        proj_root
        / "sam3_logs"
        / "example_dataset_text_only"
        / "checkpoints"
        / "checkpoint.pt"
    )

    print("[INFO] Building SAM3 image model for per-frame text detection...")
    image_model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        checkpoint_path=str(base_ckpt),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    if use_original_weights:
        print("[INFO] Using original SAM3 image weights (no ExampleDataset fine-tune).")
    else:
        print(f"[INFO] Loading ExampleDataset fine-tuned checkpoint: {finetune_ckpt}")
        ckpt = torch.load(finetune_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = image_model.load_state_dict(state_dict, strict=False)
        print(
            f"[INFO] Loaded fine-tuned weights. missing={len(missing)} unexpected={len(unexpected)}"
        )

    processor = Sam3Processor(image_model, device=device, confidence_threshold=det_conf_thr)

    # 2) 遍历视频帧，做文本检测
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[INFO] Running per-frame text detection on video: {video_path}, "
        f"total_frames={total_frames}, text_prompt='{text_prompt}'"
    )

    frames_info: List[Dict] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        try:
            processor.set_confidence_threshold(det_conf_thr)
            state = processor.set_image(img, state={})
            state = processor.set_text_prompt(prompt=text_prompt, state=state)

            boxes_t = state.get("boxes", None)
            scores_t = state.get("scores", None)
            if boxes_t is None or scores_t is None or boxes_t.numel() == 0:
                frame_idx += 1
                continue

            boxes = boxes_t.float().detach().cpu().numpy()  # [N,4]
            scores = scores_t.float().detach().cpu().numpy()  # [N]
            keep = scores >= float(det_conf_thr)
            if not np.any(keep):
                frame_idx += 1
                continue

            boxes_kept = boxes[keep]
            scores_kept = scores[keep]

            # 按分数排序，并截断到 max_objects
            order = np.argsort(-scores_kept)
            if max_objects is not None and max_objects > 0:
                order = order[:max_objects]
            boxes_kept = boxes_kept[order]
            scores_kept = scores_kept[order]

            best_score = float(scores_kept[0])
            frames_info.append(
                {
                    "frame_index": frame_idx,
                    "boxes_xyxy": boxes_kept.tolist(),
                    "scores": scores_kept.tolist(),
                    "best_score": best_score,
                }
            )
        except Exception as e:
            print(f"[WARN] Text detection failed on frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()

    if not frames_info:
        raise RuntimeError(
            f"No valid text detections on any frame (text='{text_prompt}', thr={det_conf_thr})."
        )

    # 3) 选出若干帧作为几何提示帧（按 best_score 降序）
    frames_info_sorted = sorted(
        frames_info, key=lambda d: float(d["best_score"]), reverse=True
    )
    if max_prompt_frames is not None and max_prompt_frames > 0:
        frames_selected = frames_info_sorted[:max_prompt_frames]
    else:
        frames_selected = frames_info_sorted

    print(
        "[INFO] Selected frames for geo prompts (frame_idx, best_score): "
        + ", ".join(
            f"({d['frame_index']}, {d['best_score']:.3f})" for d in frames_selected
        )
    )

    # 4) 构建 geo_json，并调用 geo_track_and_refine_video
    geo_meta = {
        "video_path": video_path,
        "text_prompt": text_prompt,
        "det_conf_thr": float(det_conf_thr),
        "max_objects_per_frame": int(max_objects),
        "prompts": [
            {
                "frame_index": int(d["frame_index"]),
                "boxes_xyxy": d["boxes_xyxy"],
                "scores": d["scores"],
            }
            for d in frames_selected
        ],
    }
    geo_json_path = out_dir_path / "auto_text_geo_prompts.json"
    geo_json_path.write_text(json.dumps(geo_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Auto text-geo prompt json saved to: {geo_json_path}")

    geo_track_and_refine_video(
        video_path=video_path,
        geo_json=str(geo_json_path),
        refine_ckpt=refine_ckpt,
        device=device,
        roi_pad=roi_pad,
        out_dir=str(out_dir_path),
    )


@torch.no_grad()
def dense_text_track_and_refine_video(
    video_path: str,
    text_prompt: str = "cell",
    frame_index: int = 0,
    min_prob: float = 0.5,
    refine_ckpt: Optional[str] = None,
    device: str = "cuda",
    roi_pad: float = 0.1,
    out_dir: Optional[str] = None,
) -> None:
    """
    方案五：使用 SAM3 原始 Sam3VideoPredictor（dense tracking）做
        文本提示自动追踪，再对所有实例掩码做 ROIRefineNet 细化。

    流程：
      1) 构建 Sam3VideoPredictor；
      2) start_session(resource_path=video_path)；
      3) 在 frame_index 上 add_prompt(text=...)；
      4) propagate_in_video（forward），得到每帧的 out_binary_masks；
      5) 将所有 obj_id/帧的二值掩码保存为 masks + mask_index.json；
      6) 调用 refine_video_masks_from_index 做 refine + 可视化。
    """

    from sam3.model_builder import build_sam3_video_predictor

    proj_root = Path(__file__).resolve().parents[2]
    video_path = str(video_path)

    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else proj_root / "output" / "dense_text_track_and_refine"
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) 构建 dense tracking 模型
    ckpt = proj_root / "weights" / "sam3.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Local SAM3 checkpoint not found at: {ckpt}")
    print(f"[INFO] Building SAM3 video predictor from {ckpt} ...")
    video_predictor = build_sam3_video_predictor(checkpoint_path=str(ckpt))

    # 2) 启动 session
    print(f"[INFO] Starting dense-text session on video: {video_path}")
    resp = video_predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": video_path,
        }
    )
    session_id = resp["session_id"]
    print(f"[INFO] session_id = {session_id}")

    # 3) 在指定帧上添加“纯文本提示”
    print(
        f"[INFO] Adding dense text prompt='{text_prompt}' at frame {frame_index} "
        "(no external boxes or points)."
    )
    video_predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_index),
            "text": text_prompt,
        }
    )

    # 4) 做视频传播（forward）
    print("[INFO] Propagating in video (dense Sam3VideoPredictor, forward)...")
    stream_req = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": int(frame_index),
        "max_frame_num_to_track": None,
    }

    # 记录所有掩码，转成 mask_index.json 格式
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for out in video_predictor.handle_stream_request(stream_req):
        f_idx = int(out["frame_index"])
        outputs = out["outputs"]
        obj_ids = outputs["out_obj_ids"]  # np.ndarray [K]
        probs = outputs["out_probs"]  # [K]
        masks = outputs["out_binary_masks"]  # [K,H,W] bool

        if obj_ids.size == 0:
            continue

        keep_indices: List[int] = [
            i for i, p in enumerate(probs) if float(p) >= float(min_prob)
        ]
        if not keep_indices:
            continue

        kept_ids = obj_ids[keep_indices].tolist()
        kept_masks = masks[keep_indices]  # [N,H,W]

        frame_dict: Dict[int, np.ndarray] = {}
        for mi, obj_id in enumerate(kept_ids):
            mask = kept_masks[mi].astype(bool)
            frame_dict[int(obj_id)] = mask
        if frame_dict:
            video_segments[f_idx] = frame_dict

    print(
        f"[INFO] Dense text-tracking finished. Frames with masks (prob>={min_prob}): "
        f"{sorted(video_segments.keys())}"
    )

    # 5) 保存 coarse masks + index
    masks_dir = out_dir_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    index_meta: Dict[str, List[Dict]] = {}
    for f_idx, obj_dict in video_segments.items():
        for oid, mask in obj_dict.items():
            fname = f"frame_{f_idx:04d}_obj_{oid}.npy"
            fpath = masks_dir / fname
            np.save(fpath, mask.astype(np.uint8))
            index_meta.setdefault(str(oid), []).append(
                {"frame_idx": int(f_idx), "path": str(fpath)}
            )
    index_path = out_dir_path / "mask_index.json"
    index_path.write_text(json.dumps(index_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Saved dense-text coarse masks under: {masks_dir}")
    print(f"[INFO] Dense-text mask index saved to: {index_path}")

    # 6) 粗追踪可视化
    out_video_path = out_dir_path / "tracking_overlay_dense_text.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    ret, sample = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame for writer init")
    Hh, Ww = sample.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (Ww, Hh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_video_path}")

    frame_idx_vis = 0
    alpha = 0.5
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.astype(np.float32)

        if frame_idx_vis in video_segments:
            obj_dict = video_segments[frame_idx_vis]
            for idx, (oid, mask) in enumerate(obj_dict.items()):
                if mask.shape != (Hh, Ww):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (Ww, Hh),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                color = np.array(palette[idx % len(palette)], dtype=np.float32)
                m3 = np.repeat(mask[..., None], 3, axis=2)
                color_img = np.zeros_like(overlay, dtype=np.float32)
                color_img[..., 0] = color[0]
                color_img[..., 1] = color[1]
                color_img[..., 2] = color[2]
                blended = overlay * (1.0 - alpha) + color_img * alpha
                overlay = np.where(m3, blended, overlay)

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        writer.write(overlay)
        frame_idx_vis += 1

    cap.release()
    writer.release()
    print(f"[INFO] Dense-text tracking visualization saved to: {out_video_path}")

    # 7) refine
    if refine_ckpt is not None:
        print("[INFO] Running ROIRefineNet on dense-text masks ...")
        refine_video_masks_from_index(
            video_path=video_path,
            mask_index_json=str(index_path),
            refine_ckpt=refine_ckpt,
            device=device,
            roi_pad=roi_pad,
            out_dir=str(out_dir_path / "refine"),
        )
    else:
        print(
            "[WARN] refine_ckpt is None; skip refine stage for dense_text_track_and_refine."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video text + refine inference (tracking + ROIRefineNet)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="refine_from_index",
        choices=[
            "track_and_refine",
            "refine_from_index",
            "geo_track_and_refine",
            "text_geo_track_and_refine",
            "dense_text_track_and_refine",
        ],
        help=(
            "track_and_refine: 文本检测 + SAM2 追踪 + refine；"
            "refine_from_index: 仅对已有掩码做 refine；"
            "geo_track_and_refine: 仅几何提示 (box) + SAM2 追踪 + refine；"
            "text_geo_track_and_refine: 视频多帧文本检测 -> 几何提示 + SAM2 追踪 + refine；"
            "dense_text_track_and_refine: Sam3VideoPredictor 原始文本追踪 + refine。"
        ),
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="20210518_U049MIS002_XY010_Z3_C1.mp4",
        help="视频路径（默认：20210518_U049MIS002_XY010_Z3_C1.mp4，相对项目根）。",
    )
    parser.add_argument(
        "--mask-index",
        type=str,
        default="output/text_to_box_sam2_tracking/mask_index.json",
        help="已有 coarse 掩码的 mask_index.json 路径（refine_from_index 模式用）。",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="cell",
        help="track_and_refine 模式下用于检测的文本提示（默认：cell）。",
    )
    parser.add_argument(
        "--geo-json",
        type=str,
        default="",
        help="geo_track_and_refine 模式下，几何 box 提示的 JSON 路径。",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="track_and_refine / dense_text_track_and_refine 模式下，在哪一帧上添加文本提示或做文本检测（默认 0）。",
    )
    parser.add_argument(
        "--det-conf-thr",
        type=float,
        default=0.3,
        help="track_and_refine 模式下，文本检测阈值（默认 0.3）。",
    )
    parser.add_argument(
        "--use-original-weights",
        action="store_true",
        help="track_and_refine 模式下，图像检测阶段是否仅用原始 SAM3 权重。",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=10,
        help="track_and_refine 模式下，最多追踪多少个检测到的目标（默认 10）；text_geo_track_and_refine 模式下，每帧最多保留多少个检测框。",
    )
    parser.add_argument(
        "--max-prompt-frames",
        type=int,
        default=3,
        help="text_geo_track_and_refine 模式下，最多选择多少帧作为几何提示帧（默认 3）。",
    )
    parser.add_argument(
        "--refine-ckpt",
        type=str,
        default="student_atto_v1_epoch_140latest.pt",
        help="ROIRefineNet 的 checkpoint 路径（默认项目根下 student_atto_v1_epoch_140latest.pt）。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录（为空则使用默认）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（默认 cuda 或 cpu）。",
    )
    parser.add_argument(
        "--roi-pad",
        type=float,
        default=0.1,
        help="ROI 边界框扩展比例（默认 0.1）。",
    )
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[2]
    video_path = str((proj_root / args.video_path).resolve())
    refine_ckpt = (
        str((proj_root / args.refine_ckpt).resolve())
        if args.refine_ckpt
        else None
    )
    out_dir = str((proj_root / args.out_dir).resolve()) if args.out_dir else None

    if args.mode == "refine_from_index":
        mask_index = str((proj_root / args.mask_index).resolve())
        refine_video_masks_from_index(
            video_path=video_path,
            mask_index_json=mask_index,
            refine_ckpt=refine_ckpt,
            device=args.device,
            roi_pad=args.roi_pad,
            out_dir=out_dir,
        )
    elif args.mode == "dense_text_track_and_refine":
        dense_text_track_and_refine_video(
            video_path=video_path,
            text_prompt=args.text_prompt,
            frame_index=args.frame_index,
            min_prob=0.5,
            refine_ckpt=refine_ckpt,
            device=args.device,
            roi_pad=args.roi_pad,
            out_dir=out_dir,
        )
    elif args.mode == "track_and_refine":
        track_and_refine_video_text(
            video_path=video_path,
            text_prompt=args.text_prompt,
            frame_index=args.frame_index,
            det_conf_thr=args.det_conf_thr,
            use_original_weights=args.use_original_weights,
            max_objects=args.max_objects,
            refine_ckpt=refine_ckpt,
            device=args.device,
            roi_pad=args.roi_pad,
            out_dir=out_dir,
        )
    elif args.mode == "geo_track_and_refine":
        if not args.geo_json:
            raise ValueError(
                "geo_track_and_refine 模式需要提供 --geo-json (几何 box 提示的 JSON)。"
            )
        geo_json = str((proj_root / args.geo_json).resolve())
        geo_track_and_refine_video(
            video_path=video_path,
            geo_json=geo_json,
            refine_ckpt=refine_ckpt,
            device=args.device,
            roi_pad=args.roi_pad,
            out_dir=out_dir,
        )
    elif args.mode == "text_geo_track_and_refine":
        text_geo_track_and_refine_video(
            video_path=video_path,
            text_prompt=args.text_prompt,
            det_conf_thr=args.det_conf_thr,
            use_original_weights=args.use_original_weights,
            max_objects=args.max_objects,
            max_prompt_frames=args.max_prompt_frames,
            refine_ckpt=refine_ckpt,
            device=args.device,
            roi_pad=args.roi_pad,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
