from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

from . import util as base
from .train import ROIRefineNetV1
from PIL import Image

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except Exception:
    mask_utils = None  # type: ignore[assignment]
    HAS_PYCOCOTOOLS = False


DATA_ROOT = Path("C:/localtask/digOrg")
DEFAULT_OUTPUT_CSV = DATA_ROOT / "eval_full_dataset_metrics.csv"
# 默认 JSON 输出模板（支持 {dataset}/{split}/{kind}）
DEFAULT_OUTPUT_JSON_REFINE = DATA_ROOT / "eval_predictions_refine_{dataset}_{split}.json"
DEFAULT_OUTPUT_JSON_SAM = DATA_ROOT / "eval_predictions_sam_{dataset}_{split}.json"

DEFAULT_SAM_CHECKPOINT = base.DEFAULT_SAM_CHECKPOINT
DEFAULT_DEVICE = base.DEFAULT_DEVICE

SPLIT_CFG = {
    "train": {"images": "train/images", "annotations": "train/data.json"},
    "val": {"images": "test/images", "annotations": "test/data.json"},
}

DATASET_REGISTRY: Dict[str, Path] = {name: DATA_ROOT / name for name in ("pdac", "brain", "colon", "example_dataset")}


def _parse_dataset_names(spec: str) -> List[str]:
    token = spec.strip().lower()
    if token in {"", "none"}:
        raise ValueError("至少指定一个数据集。")
    if token == "all":
        return sorted(DATASET_REGISTRY.keys())
    names = [item.strip().lower() for item in spec.split(",") if item.strip()]
    unknown = [n for n in names if n not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"未知数据集: {', '.join(unknown)}，可选: {', '.join(sorted(DATASET_REGISTRY))}")
    return names


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def _build_loader(dataset_root: Path, split: str, args: argparse.Namespace) -> DataLoader:
    cfg = SPLIT_CFG[split]
    images = dataset_root / cfg["images"]
    ann = dataset_root / cfg["annotations"]
    ds = base.SamCocoDataset(
        image_root=images,
        annotation_path=ann,
        image_size=args.image_size,
        max_instances=None,
        use_augmentation=False,
        prompt_type=args.prompt_type,
        allow_empty=True,
    )
    # pin_memory 只在 CUDA 可用 & 未强制 CPU eval 时才可能开启；默认 False（避免大量小 tensor 递归 pin 触发 OOM）
    pin_memory = (
        bool(getattr(args, "pin_memory", False))
        and torch.cuda.is_available()
        and not getattr(args, "_force_cpu_eval", False)
    )

    loader_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=base.plain_collate,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    loader = DataLoader(**loader_kwargs)
    return loader


def _bbox_from_bool_mask(mask: np.ndarray) -> List[float]:
    if not mask.any():
        return [0.0, 0.0, 0.0, 0.0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min = int(y_indices[0])
    y_max = int(y_indices[-1])
    x_min = int(x_indices[0])
    x_max = int(x_indices[-1])
    width = float(x_max - x_min + 1)
    height = float(y_max - y_min + 1)
    return [float(x_min), float(y_min), width, height]


def _prepare_mask_for_json(mask: np.ndarray, meta: Optional[Dict[str, object]]) -> np.ndarray:
    """
    将预测二值掩码 resize 回原图尺寸（NEAREST），用于 COCO 评估对齐 GT。
    输入 mask: 0/1 (H, W) np.uint8；输出同 dtype
    """
    mask_uint8 = mask.astype(np.uint8, copy=False)
    if meta is None:
        return mask_uint8

    orig_h = meta.get("height")
    orig_w = meta.get("width")
    if orig_h is None or orig_w is None:
        return mask_uint8

    orig_h = int(orig_h)
    orig_w = int(orig_w)
    if mask_uint8.shape == (orig_h, orig_w):
        return mask_uint8

    resized = Image.fromarray(mask_uint8 * 255).resize((orig_w, orig_h), resample=Image.NEAREST)
    resized_np = np.array(resized, dtype=np.uint8)
    return (resized_np > 0).astype(np.uint8)


def _torch_binary_dice(pred_bool: torch.Tensor, tgt_bool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    在 GPU 上计算二值 Dice。
    pred_bool, tgt_bool: (..., H, W) 的 bool Tensor
    返回：按最后两个维度聚合后的 Dice（与前面批次维度对齐）
    """
    pred_bool = pred_bool.bool()
    tgt_bool = tgt_bool.bool()
    inter = (pred_bool & tgt_bool).sum(dim=(-2, -1))
    denom = pred_bool.sum(dim=(-2, -1)) + tgt_bool.sum(dim=(-2, -1))
    return (2.0 * inter + eps) / (denom + eps)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super().default(obj)


def _resolve_json_path(path_or_template: Optional[str], dataset: str, split: str, kind: str) -> Optional[Path]:
    """把传入路径/模板解析为 (dataset, split) 专属 JSON 路径。
    kind: 'refine' 或 'sam'
    支持占位符 {dataset} / {split} / {kind}。
    - 若传目录或缺少 .json 后缀：落在该目录下，文件名为默认命名。
    - 若传固定文件名（.json 且不含占位符）：自动在文件名注入 _{dataset}_{split}。
    """
    if not path_or_template:
        return None
    raw = path_or_template
    is_template = any(tok in raw for tok in ("{dataset}", "{split}", "{kind}"))
    s = raw.format(dataset=dataset, split=split, kind=kind) if is_template else raw
    p = Path(s).expanduser()

    if p.suffix.lower() != ".json":
        p = p / f"eval_predictions_{kind}_{dataset}_{split}.json"
    elif not is_template:
        p = p.with_name(f"{p.stem}_{dataset}_{split}{p.suffix}")

    return p.resolve()


def _evaluate_image(
    sam_model: Optional[torch.nn.Module],
    refine_net: torch.nn.Module,
    pred_fuse: Optional[torch.nn.Module],
    image_bchw: torch.Tensor,
    prompts: Sequence[base.PromptEntry],  # 这里的 prompts 可以来自 CPU（entry.target_mask 在循环内再搬到 GPU）
    resize_transform: Optional[ResizeLongestSide],
    input_hw: Tuple[int, int],
    args: argparse.Namespace,
    img_emb: Optional[torch.Tensor],
    dense_pe: Optional[torch.Tensor],
    refine_json_buffer: Optional[List[Dict[str, object]]] = None,
    refine_json_meta: Optional[Dict[str, object]] = None,
    sam_json_buffer: Optional[List[Dict[str, object]]] = None,
    sam_json_meta: Optional[Dict[str, object]] = None,
) -> Tuple[int, float, float, float, float]:
    if not prompts:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")

    need_sam_ops = (args.crop_source == "sam") or args.use_prompt or args.fuse_with_sam
    report_sam_metrics = getattr(args, "report_sam_metrics", True)
    request_sam_json = sam_json_buffer is not None
    run_sam = (need_sam_ops or report_sam_metrics or request_sam_json) and sam_model is not None

    device = image_bchw.device

    def process_prompts(prompts_subset: Sequence[base.PromptEntry]) -> Tuple[int, float, float, float, float]:
        if not prompts_subset:
            return 0, float("nan"), float("nan"), float("nan"), float("nan")

        sam_prob_256: Optional[torch.Tensor] = None
        sam_boxes_1024: Optional[torch.Tensor] = None
        boxes_1024: Optional[torch.Tensor] = None
        boxes_256: Optional[torch.Tensor] = None

        # 仅在 CUDA 上开启 AMP
        use_amp = (device.type == "cuda") and bool(getattr(args, "amp", False))
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
            if run_sam:
                if resize_transform is None:
                    raise ValueError("resize_transform must be provided when running SAM.")
                sam = _unwrap(sam_model)  # type: ignore[arg-type]
                sam_out = base._sam_forward_batched_for_one_image(  # type: ignore[attr-defined]
                    sam,
                    image_bchw,
                    list(prompts_subset),
                    resize_transform,
                    input_hw,
                    args.mask_input_size,
                    device,
                    precomputed_img_emb=img_emb,
                    precomputed_image_pe=dense_pe,
                )
                low_res_logits_256 = sam_out["low_res_logits_256"]
                sam_prob_256 = torch.sigmoid(low_res_logits_256)
                sam_boxes_1024 = sam_out["boxes_1024"]

            if need_sam_ops:
                if sam_boxes_1024 is None:
                    raise RuntimeError("需要 SAM 输出以生成 ROI 但未获得有效的 boxes。")
                boxes_1024 = sam_boxes_1024

            if boxes_1024 is None:
                boxes_1024 = base._boxes_from_entries_direct(
                    list(prompts_subset), device=device, pad_ratio_1024=args.roi_pad  # type: ignore[attr-defined]
                )

            roi_img_256 = base._batch_crop_resize_img_bchw_1024(image_bchw, boxes_1024, out_hw=256)  # type: ignore[attr-defined]
            prompt_mask_256 = None

            if args.use_prompt and sam_prob_256 is not None:
                if args.crop_source == "sam":
                    boxes_256 = base._bbox_from_mask_256(
                        torch.log(sam_prob_256 / (1 - sam_prob_256 + 1e-6)), pad_ratio=args.roi_pad_256  # type: ignore[attr-defined]
                    )
                else:
                    coords = []
                    for box in boxes_1024:
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        coords.append(
                            torch.tensor(
                                [
                                    max(0, x1 // 4),
                                    max(0, y1 // 4),
                                    min(256, (x2 + 3) // 4),
                                    min(256, (y2 + 3) // 4),
                                ],
                                dtype=torch.int64,
                                device=device,
                            )
                        )
                    boxes_256 = torch.stack(coords, dim=0)
                prompt_mask_256 = base._batch_crop_resize_mask_256(sam_prob_256, boxes_256, out_hw=256)  # type: ignore[attr-defined]

            logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
            prob_refine_256 = torch.sigmoid(logits_refine_256)

            if args.fuse_with_sam and sam_prob_256 is not None and pred_fuse is not None and boxes_256 is not None:
                fused = base._batch_crop_resize_mask_256(sam_prob_256, boxes_256, out_hw=256)  # type: ignore[attr-defined]
                final_prob_256 = torch.sigmoid(pred_fuse(fused, prob_refine_256))
            else:
                final_prob_256 = prob_refine_256

            dices: List[float] = []
            ious: List[float] = []
            sam_dices: List[float] = []
            sam_ious: List[float] = []

            need_sam_outputs = report_sam_metrics or request_sam_json
            full_box: Optional[torch.Tensor] = None
            if sam_prob_256 is not None and need_sam_outputs:
                img_h = int(image_bchw.shape[-2])
                img_w = int(image_bchw.shape[-1])
                full_box = torch.tensor([0, 0, img_w, img_h], dtype=torch.int64, device=device)

            for idx, entry in enumerate(prompts_subset):
                # 仅把当前子批次的 target_mask 搬到 GPU（避免一次性把整张图的所有实例上 GPU）
                tm = entry.target_mask.to(device, non_blocking=True)

                # IoU（refine）：保持在 GPU 上
                iou_tensor = base._hard_iou_full_image(  # type: ignore[attr-defined]
                    final_prob_256[idx: idx + 1],
                    tm,
                    boxes_1024[idx],
                    args.mask_threshold,
                )
                ious.append(float(iou_tensor.item()))

                # paste 回整图并阈值（仍在 GPU）
                pasted = base._paste_roi_prob_back_1024(  # type: ignore[attr-defined]
                    final_prob_256[idx: idx + 1],
                    boxes_1024[idx],
                    args.image_size,
                    args.image_size,
                )
                pred_mask_bool = (pasted.squeeze(0).squeeze(0) > args.mask_threshold)
                target_mask_bool = (tm.squeeze(0).squeeze(0) > 0.5)
                dice_tensor = _torch_binary_dice(pred_mask_bool, target_mask_bool)
                dices.append(float(dice_tensor.item()))

                # 按需写 refine JSON（仅在 --save-json 时）
                if refine_json_buffer is not None and refine_json_meta is not None and HAS_PYCOCOTOOLS:
                    bool_mask_np = pred_mask_bool.detach().cpu().numpy().astype(np.uint8)
                    if bool_mask_np.any() or getattr(args, "json_keep_empty", False):
                        mask_uint8 = _prepare_mask_for_json(bool_mask_np, refine_json_meta)
                        rle = mask_utils.encode(np.asfortranarray(mask_uint8))  # type: ignore[arg-type]
                        rle["counts"] = rle["counts"].decode("ascii")  # type: ignore[index]
                        bbox = _bbox_from_bool_mask(mask_uint8.astype(bool))
                        score = float(final_prob_256[idx].mean().detach().float().item())
                        record = {
                            "image_id": refine_json_meta["image_id"],
                            "category_id": refine_json_meta.get("category_id", 1),
                            "bbox": bbox,
                            "score": score,
                            "segmentation": rle,
                        }
                        if "file_name" in refine_json_meta:
                            record["file_name"] = refine_json_meta["file_name"]
                        if "dataset" in refine_json_meta:
                            record["dataset"] = refine_json_meta["dataset"]
                        if "split" in refine_json_meta:
                            record["split"] = refine_json_meta["split"]
                        refine_json_buffer.append(record)

                # SAM 基线的 IoU/Dice（按需）
                if sam_prob_256 is not None and need_sam_outputs and full_box is not None:
                    sam_iou_tensor = base._hard_iou_full_image(  # type: ignore[attr-defined]
                        sam_prob_256[idx: idx + 1],
                        tm,
                        full_box,
                        args.mask_threshold,
                    )
                    sam_ious.append(float(sam_iou_tensor.item()))

                    sam_pasted = base._paste_roi_prob_back_1024(  # type: ignore[attr-defined]
                        sam_prob_256[idx: idx + 1],
                        full_box,
                        args.image_size,
                        args.image_size,
                    )
                    sam_pred_mask_bool = (sam_pasted.squeeze(0).squeeze(0) > args.mask_threshold)
                    sam_dice_tensor = _torch_binary_dice(sam_pred_mask_bool, target_mask_bool)
                    sam_dices.append(float(sam_dice_tensor.item()))

                    if sam_json_buffer is not None and sam_json_meta is not None and HAS_PYCOCOTOOLS:
                        bool_mask_np = sam_pred_mask_bool.detach().cpu().numpy().astype(np.uint8)
                        if bool_mask_np.any() or getattr(args, "json_keep_empty", False):
                            mask_uint8 = _prepare_mask_for_json(bool_mask_np, sam_json_meta)
                            rle = mask_utils.encode(np.asfortranarray(mask_uint8))  # type: ignore[arg-type]
                            rle["counts"] = rle["counts"].decode("ascii")  # type: ignore[index]
                            bbox = _bbox_from_bool_mask(mask_uint8.astype(bool))
                            score = float(sam_prob_256[idx].mean().detach().float().item())
                            record = {
                                "image_id": sam_json_meta["image_id"],
                                "category_id": sam_json_meta.get("category_id", 1),
                                "bbox": bbox,
                                "score": score,
                                "segmentation": rle,
                            }
                            if "file_name" in sam_json_meta:
                                record["file_name"] = sam_json_meta["file_name"]
                            if "dataset" in sam_json_meta:
                                record["dataset"] = sam_json_meta["dataset"]
                            if "split" in sam_json_meta:
                                record["split"] = sam_json_meta["split"]
                            sam_json_buffer.append(record)

        mean_dice = float(np.nanmean(dices)) if dices else float("nan")
        mean_iou = float(np.nanmean(ious)) if ious else float("nan")
        mean_sam_dice = float(np.nanmean(sam_dices)) if sam_dices else float("nan")
        mean_sam_iou = float(np.nanmean(sam_ious)) if sam_ious else float("nan")
        return len(prompts_subset), mean_iou, mean_dice, mean_sam_iou, mean_sam_dice

    max_prompts = getattr(args, "max_prompts_per_batch", None)
    if max_prompts is None or max_prompts <= 0 or len(prompts) <= max_prompts:
        return process_prompts(prompts)

    total_instances = 0
    iou_sum = 0.0
    dice_sum = 0.0
    sam_iou_sum = 0.0
    sam_dice_sum = 0.0
    iou_count = 0
    dice_count = 0
    sam_iou_count = 0
    sam_dice_count = 0

    prompts_list = list(prompts)
    for start in range(0, len(prompts_list), max_prompts):
        chunk = prompts_list[start: start + max_prompts]
        instances, mean_iou, mean_dice, mean_sam_iou, mean_sam_dice = process_prompts(chunk)
        total_instances += instances
        if instances <= 0:
            continue
        if not np.isnan(mean_iou):
            iou_sum += mean_iou * instances
            iou_count += instances
        if not np.isnan(mean_dice):
            dice_sum += mean_dice * instances
            dice_count += instances
        if not np.isnan(mean_sam_iou):
            sam_iou_sum += mean_sam_iou * instances
            sam_iou_count += instances
        if not np.isnan(mean_sam_dice):
            sam_dice_sum += mean_sam_dice * instances
            sam_dice_count += instances

    mean_iou = iou_sum / iou_count if iou_count > 0 else float("nan")
    mean_dice = dice_sum / dice_count if dice_count > 0 else float("nan")
    mean_sam_iou = sam_iou_sum / sam_iou_count if sam_iou_count > 0 else float("nan")
    mean_sam_dice = sam_dice_sum / sam_dice_count if sam_dice_count > 0 else float("nan")
    return total_instances, mean_iou, mean_dice, mean_sam_iou, mean_sam_dice


def evaluate_split(
    args: argparse.Namespace,
    device: torch.device,
    sam_model: torch.nn.Module,
    refine_net: torch.nn.Module,
    pred_fuse: Optional[torch.nn.Module],
    dataset_name: str,
    split: str,
    refine_json_predictions: Optional[List[Dict[str, object]]] = None,
    sam_json_predictions: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    try:
        loader = _build_loader(DATASET_REGISTRY[dataset_name], split, args)
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "out of memory" in msg.lower()
            or "cuda error" in msg.lower()
            or "cuda runtime error" in msg.lower()
        ) and torch.cuda.is_available():
            if not getattr(args, "_force_cpu_eval_msg", False):
                base._maybe_print("Out of GPU memory detected; switching to CPU for the remaining evaluation.")
                args._force_cpu_eval_msg = True
            args._force_cpu_eval = True
            args.num_workers = 0
            args.pin_memory = False
            loader = _build_loader(DATASET_REGISTRY[dataset_name], split, args)
        else:
            raise

    dataset: base.SamCocoDataset = loader.dataset  # type: ignore[assignment]
    records: List[Dict[str, object]] = []

    refine_net.eval()
    if pred_fuse is not None:
        pred_fuse.eval()
    sam_model.eval()

    report_sam_metrics = getattr(args, "report_sam_metrics", True)
    need_sam = (args.crop_source == "sam") or args.use_prompt or args.fuse_with_sam or report_sam_metrics
    resize_transform = ResizeLongestSide(_unwrap(sam_model).image_encoder.img_size) if need_sam else None

    if not hasattr(args, "_force_cpu_eval"):
        args._force_cpu_eval = False
    current_device = torch.device("cpu") if args._force_cpu_eval else device

    def move_models(target_device: torch.device) -> None:
        nonlocal current_device
        if current_device == target_device:
            return
        refine_net.to(target_device)
        if pred_fuse is not None:
            pred_fuse.to(target_device)
        if need_sam and sam_model is not None:
            sam_model.to(target_device)
        refine_net.eval()
        if pred_fuse is not None:
            pred_fuse.eval()
        if need_sam:
            sam_model.eval()
        current_device = target_device

    move_models(current_device)

    img_idx = 0
    for batch in loader:
        filenames: List[str] = batch["filenames"]
        orig_sizes = batch["orig_sizes"]
        masks_list: List[torch.Tensor] = batch["masks"]
        images = batch["images"]

        for b in range(images.shape[0]):
            image_id = int(dataset.image_ids[img_idx])

            while True:
                runtime_device = torch.device("cpu") if getattr(args, "_force_cpu_eval", False) else device
                if runtime_device.type == "cuda" and not torch.cuda.is_available():
                    args._force_cpu_eval = True
                    runtime_device = torch.device("cpu")
                move_models(runtime_device)

                # 非阻塞拷贝（仅图像）
                image_bchw = images[b: b + 1].to(runtime_device, non_blocking=True)

                # === 改动点：在 CPU 上构造 prompts，避免一次性把所有实例 mask 搬上 GPU ===
                prompts = base._prepare_prompts(  # type: ignore[attr-defined]
                    masks=masks_list[b].cpu(),               # 保持在 CPU
                    prompt_type=args.prompt_type,
                    device=torch.device("cpu"),              # 明确在 CPU 构造
                    image_size=args.image_size,
                    mask_input_size=args.mask_input_size,
                )

                img_emb_single: Optional[torch.Tensor] = None
                dense_pe_single: Optional[torch.Tensor] = None
                if need_sam and sam_model is not None and prompts:
                    sam = _unwrap(sam_model)
                    # 仅 CUDA 上启 AMP
                    use_amp = (runtime_device.type == "cuda") and bool(getattr(args, "amp", False))
                    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                        inputs_single = sam.preprocess(image_bchw)
                        img_emb_single = sam.image_encoder(inputs_single)
                        dense_pe_single = sam.prompt_encoder.get_dense_pe().to(runtime_device)

                # 取原图尺寸，便于 JSON RLE 回写
                image_info = dataset.coco.loadImgs([image_id])[0]
                orig_height = int(image_info.get("height", args.image_size))
                orig_width = int(image_info.get("width", args.image_size))

                refine_json_meta = None
                sam_json_meta = None
                if refine_json_predictions is not None:
                    refine_json_meta = {
                        "image_id": image_id,
                        "category_id": getattr(args, "json_category_id", 1),
                        "file_name": filenames[b],
                        "dataset": dataset_name,
                        "split": split,
                        "height": orig_height,
                        "width": orig_width,
                    }
                if sam_json_predictions is not None:
                    sam_json_meta = {
                        "image_id": image_id,
                        "category_id": getattr(args, "json_category_id", 1),
                        "file_name": filenames[b],
                        "dataset": dataset_name,
                        "split": split,
                        "height": orig_height,
                        "width": orig_width,
                    }
                try:
                    instances, mean_iou, mean_dice, sam_mean_iou, sam_mean_dice = _evaluate_image(
                        sam_model if need_sam else None,
                        refine_net,
                        pred_fuse,
                        image_bchw,
                        prompts,                    # 注意：prompts 的 target_mask 仍在 CPU，_evaluate_image 内部分块搬运
                        resize_transform,
                        orig_sizes[b],
                        args,
                        img_emb_single,
                        dense_pe_single,
                        refine_json_predictions,
                        refine_json_meta,
                        sam_json_predictions,
                        sam_json_meta,
                    )
                    # 成功路径：记录并前进
                    records.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "image": filenames[b],
                            "image_id": image_id,
                            "instances": instances,
                            "mean_iou": mean_iou,
                            "mean_dice": mean_dice,
                            "sam_mean_iou": sam_mean_iou,
                            "sam_mean_dice": sam_mean_dice,
                        }
                    )
                    img_idx += 1
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    msg = str(exc)
                    is_oom = (
                        "CUDA out of memory" in msg
                        or "CUDA error: out of memory" in msg
                        or "CUDNN_STATUS_ALLOC_FAILED" in msg
                    )
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        is_oom = True
                    if runtime_device.type == "cuda" and torch.cuda.is_available() and is_oom:
                        torch.cuda.empty_cache()
                        args._force_cpu_eval = True
                        if not getattr(args, "_force_cpu_eval_msg", False):
                            base._maybe_print("Out of GPU memory detected; switching to CPU for the remaining evaluation.")
                            args._force_cpu_eval_msg = True
                        continue
                    raise

    return records


def load_models(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module]]:
    # 性能开关
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    sam_ckpt = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if sam_ckpt is not None and not sam_ckpt.exists():
        fallback = base._find_default_sam_checkpoint(args.model_type)
        if fallback is not None:
            base._maybe_print(f"SAM checkpoint '{sam_ckpt}' 未找到，改用 '{fallback}'。")
            sam_ckpt = fallback
        else:
            base._maybe_print(f"SAM checkpoint '{sam_ckpt}' 未找到，将随机初始化。")
            sam_ckpt = None

    sam_model = base.build_plain_sam(args.model_type, str(sam_ckpt) if sam_ckpt is not None else None, device)
    for p in sam_model.parameters():
        p.requires_grad_(False)

    refine_net = ROIRefineNetV1(
        variant=args.student_variant,
        ckpt=args.student_convnext_ckpt,
        use_prompt=args.use_prompt,
        build_prompt=args.build_prompt,
    ).to(device)

    pred_fuse = base.PredFuse(kernel=args.fuse_kernel).to(device) if args.fuse_with_sam else None

    if args.refine_ckpt is not None:
        ckpt_path = Path(args.refine_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Refine checkpoint 未找到: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        payload = ckpt.get("refine", ckpt)
        refine_net.load_state_dict(payload, strict=False)
        if pred_fuse is not None and "fuse" in ckpt:
            pred_fuse.load_state_dict(ckpt["fuse"], strict=False)

    refine_net.eval()
    if pred_fuse is not None:
        pred_fuse.eval()

    return sam_model, refine_net, pred_fuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM+Refine full-dataset evaluation (train+val)")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))

    # 模型 & 运行
    parser.add_argument("--model-type", type=str, default="vit_b_lm")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_SAM_CHECKPOINT))
    parser.add_argument("--refine-ckpt", type=str, default=r'C:/localtask/digOrg/student_atto_v1_epoch_90.pt')
    parser.add_argument("--student-variant", type=str, default="atto")
    parser.add_argument("--student-convnext-ckpt", type=str, default=None)
    parser.add_argument("--use-prompt", action="store_true", default=False)
    parser.add_argument("--build-prompt", action="store_true", default=False)
    parser.add_argument("--fuse-with-sam", action="store_true", default=False)
    parser.add_argument("--fuse-kernel", type=int, choices=[1, 3], default=1)
    parser.add_argument("--crop-source", choices=["sam", "prompt"], default="prompt")
    parser.add_argument("--roi-pad-256", type=float, default=0.10)
    parser.add_argument("--roi-pad", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--mask-input-size", type=int, default=256)
    parser.add_argument("--prompt-type", choices=["points", "boxes", "points_boxes", "dense"], default="boxes")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    # AMP / 设备 / 内存
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))
    parser.add_argument("--no-sam-metrics", action="store_true", help="Disable baseline SAM metric computation.")
    parser.add_argument("--pin-memory", action="store_true", default=False, help="Enable CUDA pinned-memory DataLoader buffers.")
    parser.add_argument("--max-prompts-per-batch", type=int, default=32, help="Limit prompts per SAM forward pass to control memory; 0 disables the limit.")

    # JSON 输出控制：支持模板占位符
    parser.add_argument("--save-json", action="store_true", default=True, help="是否保存 COCO JSON 输出。")
    parser.add_argument("--output-json-refine", type=str, default=str(DEFAULT_OUTPUT_JSON_REFINE),
                        help="Refine 预测 JSON 路径/目录/模板，支持 {dataset}/{split}/{kind}。")
    parser.add_argument("--output-json-sam", type=str, default=str(DEFAULT_OUTPUT_JSON_SAM),
                        help="SAM 基线预测 JSON 路径/目录/模板，支持 {dataset}/{split}/{kind}。")
    parser.add_argument("--output-json", type=str, default=None, help="(Deprecated) same as --output-json-refine.")

    parser.add_argument("--json-category-id", type=int, default=1, help="Category ID to use in COCO predictions.")
    parser.add_argument("--json-keep-empty", action="store_true", help="Keep empty masks in JSON output instead of skipping them.")
    args = parser.parse_args()

    args.report_sam_metrics = not getattr(args, "no_sam_metrics", False)

    # Windows：为避免共享内存/锁页内存问题，强制 worker=0
    if os.name == "nt" and args.num_workers > 0:
        base._maybe_print("Detected Windows environment; forcing num_workers=0 to avoid shared-memory limits.")
        args.num_workers = 0

    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 统一 JSON 开关逻辑
    if not args.save_json:
        args.output_json_refine = None
        args.output_json_sam = None
    else:
        if args.output_json and not args.output_json_refine:
            args.output_json_refine = args.output_json
        if not HAS_PYCOCOTOOLS:
            raise RuntimeError("--save-json 需要安装 pycocotools。")
        if args.output_json_refine:
            args.output_json_refine = str(Path(args.output_json_refine).expanduser())
        if args.output_json_sam:
            args.output_json_sam = str(Path(args.output_json_sam).expanduser())

    # pin_memory 仅在 CUDA 可用时有效；默认 False
    args.pin_memory = bool(getattr(args, "pin_memory", False)) and torch.cuda.is_available()
    return args


def main() -> None:
    args = parse_args()
    datasets = _parse_dataset_names(args.datasets)
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or DEFAULT_DEVICE)
    sam_model, refine_net, pred_fuse = load_models(args, device)

    all_records: List[Dict[str, object]] = []
    for dataset in datasets:
        for split in ("train", "val"):
            # 每个 (dataset, split) 各自的 JSON 缓冲
            refine_json_predictions: Optional[List[Dict[str, object]]] = [] if args.output_json_refine else None
            sam_json_predictions: Optional[List[Dict[str, object]]] = [] if args.output_json_sam else None

            records = evaluate_split(
                args,
                device,
                sam_model,
                refine_net,
                pred_fuse,
                dataset,
                split,
                refine_json_predictions=refine_json_predictions,
                sam_json_predictions=sam_json_predictions,
            )
            all_records.extend(records)

            # 打印当前 split 汇总
            mean_iou_split = (
                float(np.nanmean([r["mean_iou"] for r in records if not np.isnan(r["mean_iou"])]))
                if records
                else float("nan")
            )
            mean_dice_split = (
                float(np.nanmean([r["mean_dice"] for r in records if not np.isnan(r["mean_dice"])]))
                if records
                else float("nan")
            )
            log_msg = f"[DONE] {dataset} / {split}: {len(records)} images | IoU={mean_iou_split:.4f} | Dice={mean_dice_split:.4f}"
            if getattr(args, "report_sam_metrics", True):
                mean_sam_iou_split = (
                    float(np.nanmean([r["sam_mean_iou"] for r in records if not np.isnan(r["sam_mean_iou"])]))
                    if records
                    else float("nan")
                )
                mean_sam_dice_split = (
                    float(np.nanmean([r["sam_mean_dice"] for r in records if not np.isnan(r["sam_mean_dice"])]))
                    if records
                    else float("nan")
                )
                log_msg += f" | SAM IoU={mean_sam_iou_split:.4f} | SAM Dice={mean_sam_dice_split:.4f}"
            base._maybe_print(log_msg)

            # 立刻按 dataset/split 落盘各自的 JSON（路径支持模板/目录/固定名）
            if refine_json_predictions is not None and args.output_json_refine:
                out_refine = _resolve_json_path(args.output_json_refine, dataset, split, kind="refine")
                assert out_refine is not None
                out_refine.parent.mkdir(parents=True, exist_ok=True)
                with out_refine.open("w", encoding="utf-8") as f_json:
                    json.dump(refine_json_predictions, f_json, cls=NumpyEncoder, ensure_ascii=False)
                base._maybe_print(f"Saved refine predictions JSON to {out_refine}")

            if sam_json_predictions is not None and args.output_json_sam:
                out_sam = _resolve_json_path(args.output_json_sam, dataset, split, kind="sam")
                assert out_sam is not None
                out_sam.parent.mkdir(parents=True, exist_ok=True)
                with out_sam.open("w", encoding="utf-8") as f_json:
                    json.dump(sam_json_predictions, f_json, cls=NumpyEncoder, ensure_ascii=False)
                base._maybe_print(f"Saved SAM predictions JSON to {out_sam}")

    # 写 CSV（全量汇总）
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "image_id",
                "instances",
                "mean_iou",
                "mean_dice",
                "sam_mean_iou",
                "sam_mean_dice",
            ]
        )
        for row in all_records:
            writer.writerow(
                [
                    row["image"],
                    row["image_id"],
                    row["instances"],
                    f"{row['mean_iou']:.6f}" if not np.isnan(row["mean_iou"]) else "nan",
                    f"{row['mean_dice']:.6f}" if not np.isnan(row["mean_dice"]) else "nan",
                    f"{row['sam_mean_iou']:.6f}" if not np.isnan(row["sam_mean_iou"]) else "nan",
                    f"{row['sam_mean_dice']:.6f}" if not np.isnan(row["sam_mean_dice"]) else "nan",
                ]
            )

    # 全量打印
    base._maybe_print(f"总图像数: {len(all_records)}")
    base._maybe_print(
        f"平均 IoU: {np.nanmean([r['mean_iou'] for r in all_records]) if all_records else float('nan'):.4f}"
    )
    base._maybe_print(
        f"平均 Dice: {np.nanmean([r['mean_dice'] for r in all_records]) if all_records else float('nan'):.4f}"
    )
    if getattr(args, "report_sam_metrics", True):
        base._maybe_print(
            f"平均 SAM IoU: {np.nanmean([r['sam_mean_iou'] for r in all_records]) if all_records else float('nan'):.4f}"
        )
        base._maybe_print(
            f"平均 SAM Dice: {np.nanmean([r['sam_mean_dice'] for r in all_records]) if all_records else float('nan'):.4f}"
        )


if __name__ == "__main__":
    main()
