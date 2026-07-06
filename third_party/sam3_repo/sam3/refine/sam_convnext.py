from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
import inspect
# -------- Optional deps --------
try:
    import albumentations as A  # type: ignore
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    from pycocotools.coco import COCO  # type: ignore
except Exception as exc:
    COCO = None
    _COCO_IMPORT_ERROR: Optional[Exception] = exc
else:
    _COCO_IMPORT_ERROR = None
# SAM (optional; not required for SAM3 refine path)
try:
    from segment_anything.utils.transforms import ResizeLongestSide
except Exception:  # pragma: no cover - allow SAM3-only usage without SAM1 deps
    ResizeLongestSide = None  # type: ignore
try:
    from orgsam.micro_sam.models.build_sam import sam_model_registry  # type: ignore
except Exception:
    sam_model_registry = None  # type: ignore
try:
    from orgsam.micro_sam import util as microsam_util  # type: ignore[attr-defined]
except Exception:
    microsam_util = None  # type: ignore[assignment]
# ConvNeXt（可选）
HAS_CONVNEXT = False
try:
    from convnext import convnextv2_backbone  # 你的 ConvNeXtV2 实现
    HAS_CONVNEXT = True
except Exception:
    HAS_CONVNEXT = False
# ---------------- Paths & defaults ----------------
DEFAULT_TRAIN_IMAGES = Path('ExampleDataset/train/images')
DEFAULT_TRAIN_ANN = Path('ExampleDataset/train/data.json')
DEFAULT_VAL_IMAGES = Path('ExampleDataset/test/images')
DEFAULT_VAL_ANN = Path('ExampleDataset/test/data.json')
DEFAULT_OUTPUT_DIR = Path('plain_sam_checkpoints')
DEFAULT_MODELS_DIR = Path('models')
DEFAULT_SAM_CHECKPOINT = DEFAULT_MODELS_DIR / 'vit_b_lm.pth'
DEFAULT_SAM_FALLBACKS: Sequence[str] = (
    '{model_type}.pth',
    '{model_type}.pt',
    '{model_type}',
    'sam_{model_type}.pth',
    'sam_{model_type}',
    '{model_type}_encoder.pth',
    '{model_type}_encoder.pt',
)
DEFAULT_RESUME_CHECKPOINT = DEFAULT_MODELS_DIR / 'plain_sam_resume.pt'
DEFAULT_AUG_CONFIG = Path('C:/localtask/Point2Org/conf/hybrid_aug.json')
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------- Utils ----------------
def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
def _maybe_print(*args, **kwargs) -> None:
    if _is_main_process():
        print(*args, **kwargs)
def _normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    if value_str == '':
        return None
    if value_str.lower() in {'none', 'null'}:
        return None
    return value_str
def _find_default_sam_checkpoint(model_type: str) -> Optional[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()
    for template in DEFAULT_SAM_FALLBACKS:
        candidate = (DEFAULT_MODELS_DIR / template.format(model_type=model_type)).expanduser()
        if candidate.exists():
            resolved = candidate.resolve()
            key = resolved.as_posix()
            if key not in seen:
                candidates.append(resolved); seen.add(key)
    if DEFAULT_SAM_CHECKPOINT.exists():
        resolved = DEFAULT_SAM_CHECKPOINT.resolve()
        key = resolved.as_posix()
        if key not in seen:
            candidates.append(resolved)
    return candidates[0] if candidates else None
def _apply_argument_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.checkpoint = _normalize_optional_path(getattr(args, 'checkpoint', None))
    args.init_weights = _normalize_optional_path(getattr(args, 'init_weights', None))
    args.resume = _normalize_optional_path(getattr(args, 'resume', None))
    if args.checkpoint is None:
        default_ckpt = _find_default_sam_checkpoint(args.model_type)
        if default_ckpt is not None:
            args.checkpoint = str(default_ckpt)
    else:
        args.checkpoint = str(Path(args.checkpoint).expanduser())
    if args.init_weights is None and DEFAULT_SAM_CHECKPOINT.exists():
        args.init_weights = str(DEFAULT_SAM_CHECKPOINT.resolve())
    elif args.init_weights is not None:
        args.init_weights = str(Path(args.init_weights).expanduser())
    if args.resume is None and DEFAULT_RESUME_CHECKPOINT.exists():
        args.resume = str(DEFAULT_RESUME_CHECKPOINT.resolve())
    elif args.resume is not None:
        args.resume = str(Path(args.resume).expanduser())
    return args
def _interp(x: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    else:
        return F.interpolate(x, size=size, mode=mode)
def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
def _apply_basic_augmentation(image: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=1).copy()
        masks = [np.flip(m, axis=1).copy() for m in masks]
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=0).copy()
        masks = [np.flip(m, axis=0).copy() for m in masks]
    k = np.random.randint(0, 4)
    if k:
        image = np.rot90(image, k, axes=(0, 1)).copy()
        masks = [np.rot90(m, k, axes=(0, 1)).copy() for m in masks]
    return image, masks
# ---------------- Augmentation config ----------------
@dataclass
class AugmentationConfig:
    random_resized_crop_scale: Tuple[float, float] = (0.7, 1.0)
    random_resized_crop_ratio: Tuple[float, float] = (0.75, 1.33)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    rotate90_prob: float = 0.5
    color_jitter_prob: float = 0.0
    blur_prob: float = 0.0
    @staticmethod
    def from_json(path: Optional[str]) -> 'AugmentationConfig':
        if path is None:
            return AugmentationConfig()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return AugmentationConfig(**data)
# ---------------- Dataset ----------------
class SamCocoDataset(Dataset):
    def __init__(
        self,
        image_root: Path | str,
        annotation_path: Path | str,
        image_size: int = 1024,
        max_instances: Optional[int] = None,
        use_augmentation: bool = False,
        augmentation_config: Optional[AugmentationConfig] = None,
        prompt_type: str = 'boxes',
        allow_empty: bool = False,
    ) -> None:
        if COCO is None:
            raise ImportError(
                'pycocotools is required to use SamCocoDataset.'
                f' Original import error: {_COCO_IMPORT_ERROR}'
            )
        self.image_root = Path(image_root)
        self.annotation_path = Path(annotation_path)
        self.image_size = int(image_size)
        self.max_instances = max_instances
        self.prompt_type = prompt_type
        self.allow_empty = allow_empty
        self._exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        self._name_lookup: Dict[str, Path] = {}
        self._stem_lookup: Dict[str, List[Path]] = defaultdict(list)
        self.coco = COCO(str(self.annotation_path))
        self.image_ids = sorted(self.coco.getImgIds())
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self._albumentations = None
        if HAS_ALBUMENTATIONS:
            rrc_kwargs = dict(
                scale=self.augmentation_config.random_resized_crop_scale,
                ratio=self.augmentation_config.random_resized_crop_ratio,
            )
            if "size" in inspect.signature(A.RandomResizedCrop.__init__).parameters:
                size_hw = (int(self.image_size), int(self.image_size))
                rrc = A.RandomResizedCrop(size=size_hw, **rrc_kwargs)  # v2
            else:
                rrc = A.RandomResizedCrop(height=int(self.image_size), width=int(self.image_size), **rrc_kwargs)  # v1
            self._albumentations = A.Compose(
                [
                    rrc,
                    A.HorizontalFlip(p=self.augmentation_config.horizontal_flip_prob),
                    A.VerticalFlip(p=self.augmentation_config.vertical_flip_prob),
                    A.RandomRotate90(p=self.augmentation_config.rotate90_prob),
                    A.ColorJitter(p=self.augmentation_config.color_jitter_prob),
                    A.GaussianBlur(p=self.augmentation_config.blur_prob),
                ],
            )
        elif self.use_augmentation:
            _maybe_print('Albumentations not available; using basic geometric augmentation.')
        self._build_image_index()
    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_image_index(self) -> None:
        if not self.image_root.exists():
            return
        try:
            iterator = self.image_root.rglob('*')
        except Exception:
            return
        for path in iterator:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in self._exts:
                continue
            name_key = path.name.lower()
            if name_key not in self._name_lookup:
                self._name_lookup[name_key] = path
            stem_key = path.stem.lower()
            self._stem_lookup[stem_key].append(path)

    def _resolve_image_path(self, file_name: str) -> Path:
        raw = file_name.strip()
        normalized_raw = raw.replace('\\', '/')
        candidate = Path(normalized_raw)

        def try_variants(base: Path) -> Optional[Path]:
            if base.exists():
                return base
            stem_local = base.stem or base.name
            parent = base.parent
            for ext in self._exts:
                if base.suffix.lower() == ext:
                    continue
                alt = parent / f'{stem_local}{ext}'
                if alt.exists():
                    return alt
            return None

        direct = candidate if candidate.is_absolute() else self.image_root / candidate
        resolved = try_variants(direct)
        if resolved is not None:
            return resolved

        name_only = self.image_root / candidate.name
        resolved = try_variants(name_only)
        if resolved is not None:
            return resolved

        parts = [p for p in candidate.parts if p not in (candidate.anchor,)]
        for start in range(len(parts)):
            tail = Path(*parts[start:])
            resolved = try_variants(self.image_root / tail)
            if resolved is not None:
                return resolved

        name_key = candidate.name.lower()
        if name_key and name_key in self._name_lookup:
            return self._name_lookup[name_key]

        stem_key_original = candidate.stem or candidate.name
        stem_key = stem_key_original.lower()
        if stem_key and stem_key in self._stem_lookup:
            return self._stem_lookup[stem_key][0]

        if stem_key_original:
            for path in self.image_root.rglob(f'{stem_key_original}.*'):
                if path.suffix.lower() in self._exts and path.exists():
                    return path
        raise FileNotFoundError(f"Could not resolve image path for '{file_name}'.")
    def _load_image(self, image_info: Dict) -> np.ndarray:
        path = self._resolve_image_path(image_info['file_name'])
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))
    def _load_masks(self, image_id: int, height: int, width: int) -> Tuple[List[np.ndarray], List[int]]:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)
        masks: List[np.ndarray] = []
        labels: List[int] = []
        for ann in anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue
            masks.append(mask.astype(np.uint8))
            labels.append(int(ann['category_id']))
        return masks, labels
    def _resize(self, image: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        if cv2 is None:
            image = np.array(Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR))
            resized_masks = [
                np.array(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))
                for mask in masks
            ]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            resized_masks = [cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST) for mask in masks]
        return image, resized_masks
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Tuple[int, int]]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image = self._load_image(image_info)
        masks_np, labels = self._load_masks(image_id, image_info['height'], image_info['width'])
        if not masks_np and not self.allow_empty:
            raise RuntimeError(f"No instances in {image_info['file_name']}. Set allow_empty=True to keep such samples.")
        if self.max_instances is not None and len(masks_np) > self.max_instances:
            keep = np.random.choice(len(masks_np), self.max_instances, replace=False)
            masks_np = [masks_np[i] for i in keep]
            labels = [labels[i] for i in keep]
        if self.use_augmentation:
            if HAS_ALBUMENTATIONS and self._albumentations is not None:
                augmented = self._albumentations(image=image, masks=masks_np)
                image = augmented['image']; masks_np = augmented['masks']
            else:
                image, masks_np = _apply_basic_augmentation(image, masks_np)
        image, masks_np = self._resize(image, masks_np)
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image *= 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.long)
        if masks_np:
            masks_stack = np.stack(masks_np).astype(np.float32)
            masks_tensor = torch.from_numpy(masks_stack)
            boxes = []
            for m in masks_np:
                box = _mask_to_box(m)
                boxes.append(box if box is not None else np.zeros(4, dtype=np.float32))
            boxes_tensor = torch.from_numpy(np.stack(boxes))
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        sample = {
            'image': image_tensor,                    # (3, 1024, 1024)
            'masks': masks_tensor,                    # (N, 1024, 1024)
            'boxes': boxes_tensor,                    # (N, 4)  xyxy
            'labels': labels_tensor,                  # (N,)
            'filename': str(image_info['file_name']),
            'orig_size': (self.image_size, self.image_size),
        }
        return sample
def plain_collate(batch: List[Dict]) -> Dict[str, torch.Tensor | List[torch.Tensor] | List[str] | List[Tuple[int, int]]]:
    images = torch.stack([item['image'] for item in batch], dim=0)
    masks = [item['masks'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    filenames = [item['filename'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    return {
        'images': images,
        'masks': masks,
        'boxes': boxes,
        'labels': labels,
        'filenames': filenames,
        'orig_sizes': orig_sizes,
    }
# ---------------- Prompt assembly ----------------
@dataclass
class PromptEntry:
    points: Optional[torch.Tensor]
    point_labels: Optional[torch.Tensor]
    boxes: Optional[torch.Tensor]
    mask_input: Optional[torch.Tensor]
    target_mask: torch.Tensor
def _prepare_prompts(
    masks: torch.Tensor,
    prompt_type: str,
    device: torch.device,
    image_size: int,
    mask_input_size: int,
) -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    if masks.numel() == 0:
        return entries
    for idx in range(masks.shape[0]):
        mask = masks[idx]
        if mask.sum() <= 0:
            continue
        mask_bin = mask > 0.5
        ys, xs = torch.where(mask_bin)
        if ys.numel() == 0 or xs.numel() == 0:
            continue
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        points = None
        point_labels = None
        boxes = None
        mask_input = None
        if prompt_type in {'points', 'points_boxes'}:
            cx = (xmin + xmax).float() / 2.0
            cy = (ymin + ymax).float() / 2.0
            points = torch.tensor([[cx.item(), cy.item()]], dtype=torch.float32, device=device)
            point_labels = torch.ones((1,), dtype=torch.int64, device=device)
        if prompt_type in {'boxes', 'points_boxes'}:
            boxes = torch.tensor([[xmin.item(), ymin.item(), xmax.item(), ymax.item()]], dtype=torch.float32, device=device)
        if prompt_type == 'dense':
            mask_low_res = _interp(mask.unsqueeze(0).unsqueeze(0), size=(mask_input_size, mask_input_size), mode='bilinear')
            mask_low_res = mask_low_res.clamp(1e-4, 1 - 1e-4)
            mask_input = torch.log(mask_low_res / (1 - mask_low_res))
        target_mask = mask.unsqueeze(0).unsqueeze(0).to(device)
        entries.append(PromptEntry(points, point_labels, boxes, mask_input, target_mask))
    return entries
def _transform_prompts_batched(
    entries: List[PromptEntry],
    transform: ResizeLongestSide,
    input_size: Tuple[int, int],
    device: torch.device,
):
    N = len(entries)
    pts_list, lb_list, bx_list, mi_list, tg_list = [], [], [], [], []
    for e in entries:
        if e.points is not None:
            coords = e.points.detach().cpu().numpy()[None, :, :]
            coords = transform.apply_coords(coords, input_size)
            pts = torch.from_numpy(coords).to(device)
            lbs = e.point_labels.unsqueeze(0)
            pts_list.append(pts); lb_list.append(lbs)
        else:
            pts_list.append(None); lb_list.append(None)
        if e.boxes is not None:
            box_np = e.boxes.detach().cpu().numpy()
            box_np = transform.apply_boxes(box_np, input_size)
            bx = torch.from_numpy(box_np).to(device)
            bx_list.append(bx)
        else:
            bx_list.append(None)
        if e.mask_input is not None:
            mi_list.append(e.mask_input.to(device))
        else:
            mi_list.append(None)
        tg_list.append(e.target_mask)
    if all(p is None for p in pts_list):
        pts = None; plabels = None
    else:
        pts = torch.cat([p if p is not None else torch.zeros((1,1,2), device=device) for p in pts_list], dim=0)
        plabels = torch.cat([l if l is not None else torch.zeros((1,1), dtype=torch.int64, device=device) for l in lb_list], dim=0)
    if all(b is None for b in bx_list):
        boxes_1024 = None
    else:
        boxes_1024 = torch.cat([b if b is not None else torch.zeros((1,4), device=device) for b in bx_list], dim=0)
    if all(m is None for m in mi_list):
        mask_inputs = None
    else:
        mask_inputs = torch.cat([m if m is not None else torch.zeros((1,1,256,256), device=device) for m in mi_list], dim=0)
    targets_1024 = torch.cat(tg_list, dim=0)
    return pts, plabels, boxes_1024, mask_inputs, targets_1024
# ---------------- ROI helpers ----------------
def _clamp_box_int(x1, y1, x2, y2, W, H) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W,     x2)))
    y2 = int(max(0, min(H,     y2)))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2
def _bbox_from_mask_256(logits_or_prob_256: torch.Tensor, pad_ratio: float = 0.10) -> torch.Tensor:
    """
    与原函数等价，但 pad 由像素改为“按外框宽高的比例”（每边分别按 w*pad_ratio / h*pad_ratio 外扩）。
    """
    if logits_or_prob_256.dim() == 4:
        m = logits_or_prob_256[:, 0]
    elif logits_or_prob_256.dim() == 3:
        m = logits_or_prob_256
    else:
        raise ValueError("Invalid mask tensor shape for bbox.")
    prob = torch.sigmoid(m) if not (m.min() >= 0 and m.max() <= 1) else m
    boxes = []
    for i in range(prob.shape[0]):
        ys, xs = torch.where(prob[i] > 0.5)
        if ys.numel() == 0 or xs.numel() == 0:
            boxes.append(torch.tensor([64, 64, 192, 192], device=prob.device, dtype=torch.int64))
            continue
        xmin = xs.min().item(); xmax = xs.max().item()
        ymin = ys.min().item(); ymax = ys.max().item()
        w = int(xmax - xmin + 1); h = int(ymax - ymin + 1)
        px = int(round(w * float(pad_ratio))); py = int(round(h * float(pad_ratio)))
        x1 = int(max(0, xmin - px))
        y1 = int(max(0, ymin - py))
        x2 = int(min(256, xmax + 1 + px))
        y2 = int(min(256, ymax + 1 + py))
        boxes.append(torch.tensor([x1, y1, x2, y2], device=prob.device, dtype=torch.int64))
    return torch.stack(boxes, dim=0)
def _batch_crop_resize_img_bchw_1024(img_bchw: torch.Tensor, boxes_xyxy_1024: torch.Tensor, out_hw: int = 256) -> torch.Tensor:
    B, C, H, W = img_bchw.shape
    assert B == 1 and H == 1024 and W == 1024
    crops = []
    for i in range(boxes_xyxy_1024.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy_1024[i].tolist()]
        x1, y1, x2, y2 = _clamp_box_int(x1, y1, x2, y2, W, H)
        crop = img_bchw[..., y1:y2, x1:x2]
        roi = _interp(crop, size=(out_hw, out_hw), mode='bilinear')
        crops.append(roi)
    return torch.cat(crops, dim=0)
def _batch_crop_resize_mask_256(mask_bchw_256: torch.Tensor, boxes_xyxy_256: torch.Tensor, out_hw: int = 256) -> torch.Tensor:
    rois = []
    for i in range(mask_bchw_256.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy_256[i].tolist()]
        x1, y1, x2, y2 = _clamp_box_int(x1, y1, x2, y2, 256, 256)
        crop = mask_bchw_256[i:i+1, ...][..., y1:y2, x1:x2]
        roi = _interp(crop, size=(out_hw, out_hw), mode='bilinear')
        rois.append(roi)
    return torch.cat(rois, dim=0)
def _paste_roi_prob_back_1024(prob_roi_256: torch.Tensor, box_xyxy_1024: torch.Tensor, H: int = 1024, W: int = 1024) -> torch.Tensor:
    canvas = prob_roi_256.new_zeros((1, 1, H, W))
    x1, y1, x2, y2 = [int(v) for v in box_xyxy_1024.tolist()]
    x1, y1, x2, y2 = _clamp_box_int(x1, y1, x2, y2, W, H)
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return canvas
    roi_resized = _interp(prob_roi_256, size=(y2 - y1, x2 - x1), mode='bilinear')
    canvas[..., y1:y2, x1:x2] = roi_resized
    return canvas
# ---------------- Models ----------------
class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, groups=1, act=nn.GELU):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False)
        self.bn = LayerNorm2d(c_out)
        self.act = act()
    def forward(self, x): return self.act(self.bn(self.conv(x)))
class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out, act=nn.GELU):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            ConvBNAct(c_out + c_skip, c_out, k=3, act=act),
            ConvBNAct(c_out, c_out, k=3, act=act),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = _interp(skip, size=x.shape[-2:], mode='bilinear')
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)
class LiteUNetEncoder(nn.Module):
    def __init__(self, in_ch=3, base=64, act=nn.GELU):
        super().__init__()
        self.enc0 = nn.Sequential(ConvBNAct(in_ch, base, 3, act=act), ConvBNAct(base, base, 3, act=act))      # 256
        self.down1 = nn.Conv2d(base, base*2, 3, 2, 1)                                                         # 128
        self.enc1 = nn.Sequential(ConvBNAct(base*2, base*2, 3, act=act))
        self.down2 = nn.Conv2d(base*2, base*4, 3, 2, 1)                                                       # 64
        self.enc2 = nn.Sequential(ConvBNAct(base*4, base*4, 3, act=act))
        self.down3 = nn.Conv2d(base*4, base*8, 3, 2, 1)                                                       # 32
        self.enc3 = nn.Sequential(ConvBNAct(base*8, base*8, 3, act=act))
        self.down4 = nn.Conv2d(base*8, base*8, 3, 2, 1)                                                       # 16
        self.enc4 = nn.Sequential(ConvBNAct(base*8, base*8, 3, act=act))
        self.out_dims = (base, base*2, base*4, base*8, base*8)
    def forward(self, x):
        x0 = self.enc0(x)               # 256x256
        x1 = self.enc1(self.down1(x0))  # 128x128
        x2 = self.enc2(self.down2(x1))  # 64x64   <-- 融合点
        x3 = self.enc3(self.down3(x2))  # 32x32
        x4 = self.enc4(self.down4(x3))  # 16x16
        return x0, x1, x2, x3, x4
# -------- ConvNeXt GRN 兼容加载 --------
def safe_load_convnextv2_state_dict(backbone: nn.Module, raw_state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str], List[str]]:
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
            fixed_state[k] = v.to(dtype=dst.dtype); continue
        if (".grn.gamma" in k) or (".grn.beta" in k):
            if v.numel() == dst.numel():
                vv = v.reshape(dst.shape).to(dtype=dst.dtype)
                fixed_state[k] = vv; converted.append(f"{k}: {tuple(v.shape)} -> {tuple(dst.shape)}"); continue
            if v.shape[-1] == dst.shape[-1]:
                leading_ones = [1] * (len(dst.shape) - 1)
                vv = v.reshape(*leading_ones, dst.shape[-1]).to(dtype=dst.dtype)
                if vv.shape == dst.shape:
                    fixed_state[k] = vv; converted.append(f"{k}: {tuple(v.shape)} -> {tuple(dst.shape)}"); continue
        skipped.append(f"{k}: {tuple(v.shape)} != {tuple(dst.shape)}")
    missing, unexpected = backbone.load_state_dict(fixed_state, strict=False)
    if converted:
        _maybe_print(f"[INFO] ConvNeXt GRN-compat loaded ({len(converted)} keys).")
    if skipped:
        _maybe_print(f"[WARN] skipped {len(skipped)} mismatched keys (kept default init).")
    return ([str(m) for m in missing], [str(u) for u in unexpected], converted)
# -------- 提示掩码编码器 --------
class PromptMaskEncoder(nn.Module):
    def __init__(self, out_channels: int, img_in_ch: int = 3, patch: int = 8,
                 n_heads: int = 4, depth: int = 8, n_mem_tokens: int = 8):
        super().__init__()
        self.out_c = out_channels
        self.patch = patch
        d = out_channels
        self.mask_proj_spatial = nn.Sequential(
            ConvBNAct(1, max(16, d // 4), k=3),
            ConvBNAct(max(16, d // 4), d, k=3),
        )
        self.img_proj_spatial = nn.Sequential(
            ConvBNAct(img_in_ch, max(16, d // 4), k=3),
            ConvBNAct(max(16, d // 4), d, k=3),
        )
        self.mask_to_tok = nn.Conv2d(d, d, kernel_size=patch, stride=patch, bias=True)
        self.img_to_tok  = nn.Conv2d(d, d, kernel_size=patch, stride=patch, bias=True)
        self.pos_tok     = nn.Parameter(torch.zeros(1, (64 // patch) * (64 // patch), d))
        self.mem_tokens  = nn.Parameter(torch.randn(1, n_mem_tokens, d))
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.tok_to_map = nn.ConvTranspose2d(d, d, kernel_size=patch, stride=patch)
        self.refine_out = ConvBNAct(d, d, k=3)
        self.to_out     = nn.Conv2d(d, out_channels, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(1.0))
    def forward(self,
                mask_bchw_256: Optional[torch.Tensor],
                roi_rgb_bchw:  Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask_bchw_256 is None or roi_rgb_bchw is None:
            return None
        m = mask_bchw_256
        if not (m.min() >= 0 and m.max() <= 1):
            m = torch.sigmoid(m)
        m64 = _interp(m, size=(64, 64), mode='bilinear')
        m64 = self.mask_proj_spatial(m64)
        if roi_rgb_bchw.dtype != torch.float32:
            roi_rgb_bchw = roi_rgb_bchw.float()
        r64 = _interp(roi_rgb_bchw / 255.0, size=(64, 64), mode='bilinear')
        i64 = self.img_proj_spatial(r64)
        mtok = self.mask_to_tok(m64).flatten(2).transpose(1, 2)
        itok = self.img_to_tok(i64).flatten(2).transpose(1, 2)
        B, T, d = mtok.shape
        pos = self.pos_tok
        if pos.shape[1] != T:
            pos = F.interpolate(self.pos_tok.transpose(1, 2).reshape(1, d, 8, 8),
                                size=(int(64 / self.patch), int(64 / self.patch)),
                                mode='bilinear', align_corners=False).reshape(1, d, -1).transpose(1, 2)
        tok = mtok + itok + pos
        mem = self.mem_tokens.expand(B, -1, -1)
        seq = torch.cat([mem, tok], dim=1)
        seq = self.encoder(seq)
        tok_out = seq[:, mem.size(1):]
        tok_map = tok_out.transpose(1, 2).reshape(B, d, 64 // self.patch, 64 // self.patch)
        feat64  = self.tok_to_map(tok_map)
        feat64  = self.refine_out(feat64)
        emb     = self.to_out(feat64) * self.gate
        return emb
class ROIRefineNet(nn.Module):
    def __init__(self, use_convnext: bool, convnext_variant: str = 'tiny', convnext_ckpt: Optional[str] = None,
                 use_prompt: bool = True, build_prompt: bool = False):
        super().__init__()
        self.use_convnext = use_convnext and HAS_CONVNEXT
        self.use_prompt = use_prompt
        self.build_prompt = build_prompt
        self.register_buffer("imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("imnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)
        if self.use_convnext:
            self.backbone = convnextv2_backbone(convnext_variant, in_chans=3)
            if convnext_ckpt and Path(convnext_ckpt).exists():
                try:
                    ckpt = torch.load(convnext_ckpt, map_location="cpu")
                    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
                    safe_load_convnextv2_state_dict(self.backbone, state)
                except Exception as e:
                    _maybe_print(f"[WARN] load ConvNeXt ckpt failed (compat path): {e}")
            self.variant_dims = {
                'atto':  (40,  80, 160, 320),
                'femto': (48,  96, 192, 384),
                'pico':  (64, 128, 256, 512),
                'nano':  (80, 160, 320, 640),
                'tiny':  (96, 192, 384, 768),
                'small': (96, 192, 384, 768),
                'base':  (128,256,512,1024),
                'large': (192,384,768,1536),
                'huge':  (352,704,1408,2816),
            }
            c0, c1, c2, c3 = self.variant_dims.get(convnext_variant, self.variant_dims['tiny'])
            self.prompt_encoder = PromptMaskEncoder(out_channels=c0) if (self.build_prompt or self.use_prompt) else None
            self.up3 = UpBlock(c_in=c3,  c_skip=c2, c_out=256)   # 8→16
            self.up2 = UpBlock(c_in=256, c_skip=c1, c_out=128)   # 16→32
            self.up1 = UpBlock(c_in=128, c_skip=c0, c_out=64)    # 32→64
            self.head64 = nn.Sequential(ConvBNAct(64, 32, 3), nn.Conv2d(32, 1, kernel_size=1))
        else:
            base = 64
            self.backbone = LiteUNetEncoder(in_ch=3, base=base)
            self.prompt_encoder = PromptMaskEncoder(out_channels=base*4) if (self.build_prompt or self.use_prompt) else None
            self.up3 = UpBlock(self.backbone.out_dims[4], self.backbone.out_dims[3], 256)  # 16→32
            self.up2 = UpBlock(256, self.backbone.out_dims[2], 256)                         # 32→64
            self.head64 = nn.Sequential(ConvBNAct(256, 128, 3), nn.Conv2d(128, 1, kernel_size=1))
    def forward(self, roi_img_bchw: torch.Tensor, prompt_mask_bchw_256: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_im = roi_img_bchw / 255.0
        x_im = (x_im - self.imnet_mean) / self.imnet_std
        if self.use_convnext:
            x0, x1, x2, x3 = self.backbone(x_im)      # 64,32,16,8
            if self.use_prompt and self.prompt_encoder is not None and prompt_mask_bchw_256 is not None:
                pe = self.prompt_encoder(prompt_mask_bchw_256, roi_img_bchw)   # (B,c0,64,64)
                x0 = x0 + pe
            y = self.up3(x3, x2)
            y = self.up2(y, x1)
            y = self.up1(y, x0)
            logits64 = self.head64(y)
        else:
            x0, x1, x2, x3b, x4 = self.backbone(x_im)            # 256,128,64,32,16
            if self.use_prompt and self.prompt_encoder is not None and prompt_mask_bchw_256 is not None:
                pe = self.prompt_encoder(prompt_mask_bchw_256, roi_img_bchw)   # (B,base*4,64,64)
                x2 = x2 + pe
            dec32 = self.up3(x4, x3b)
            dec64 = self.up2(dec32, x2)
            logits64 = self.head64(dec64)
        logits256 = _interp(logits64, size=(256, 256), mode='bilinear')
        return logits256
class PredFuse(nn.Module):
    def __init__(self, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel, padding=pad, bias=True)
    def forward(self, sam_prob: torch.Tensor, refine_prob: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sam_prob, refine_prob], dim=1)
        logits = self.conv(x)
        return logits
# ---------------- SAM builder ----------------
def build_plain_sam(model_type: str, checkpoint: Optional[str], device: torch.device) -> nn.Module:
    base_predictor = None
    sam_model = None
    if microsam_util is not None:
        base_predictor, sam_model = microsam_util.get_sam_model(
            model_type=model_type, device=device, checkpoint_path=checkpoint, return_sam=True,
        )
    else:
        base_type = model_type.split('_')[0] if model_type.startswith('vit_') else model_type
        if base_type not in sam_model_registry:
            available = ', '.join(sorted(sam_model_registry.keys()))
            raise ValueError(f"Unknown model_type '{model_type}'. Available models: {available}")
        sam_model = sam_model_registry[base_type](checkpoint=checkpoint)
        sam_model.to(device)
    sam_model.to(device)
    sam_model.eval()
    if base_predictor is not None:
        del base_predictor
    return sam_model
# ---------------- Trainable helpers & losses ----------------
def _summarize_trainable_parameters(mods: List[nn.Module]) -> None:
    if not _is_main_process():
        return
    params = []
    for m in mods:
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    total = sum(p.numel() for p in params)
    _maybe_print(f"Trainable params: {total:,}")
def _bce_with_logits_mean(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction='mean', weight=weight, pos_weight=pos_weight)
def _dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum() + eps
    dice = (2.0 * inter + eps) / denom
    return 1.0 - dice
def _hard_iou_full_image(prob_roi_256: torch.Tensor, entry_target_mask_1024: torch.Tensor, box_xyxy_1024: torch.Tensor, thr: float) -> torch.Tensor:
    pred_canvas = _paste_roi_prob_back_1024(prob_roi_256, box_xyxy_1024, 1024, 1024)
    pred_bin = (pred_canvas > thr).float()
    tgt_bin = (entry_target_mask_1024 > 0.5).float()
    inter = (pred_bin * tgt_bin).sum()
    union = pred_bin.sum() + tgt_bin.sum() - inter + 1e-6
    return (inter + 1e-6) / union
# ---------------- (New) Freeze plan helpers ----------------
def _set_requires_grad(mod: Optional[nn.Module], flag: bool) -> None:
    if mod is None: return
    for p in mod.parameters():
        p.requires_grad = flag
def _count_params(mod: Optional[nn.Module], trainable_only: bool = False) -> int:
    if mod is None: return 0
    if trainable_only:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return sum(p.numel() for p in mod.parameters())
def _parse_items_csv(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(',') if x.strip()]
def apply_freeze_plan_and_collect(
    refine_net: nn.Module,
    pred_fuse: Optional[nn.Module],
    args: argparse.Namespace,
) -> Tuple[List[torch.nn.Parameter], List[Tuple[str, torch.nn.Parameter]]]:
    _set_requires_grad(refine_net, True)
    if pred_fuse is not None:
        _set_requires_grad(pred_fuse, True)
    freeze_set = set(map(str.lower, _parse_items_csv(getattr(args, 'freeze', ''))))
    if 'decoder' in freeze_set:
        freeze_set |= {'up1', 'up2', 'up3', 'head'}
    if 'all' in freeze_set:
        _set_requires_grad(refine_net, False)
    named_parts = {
        'backbone': getattr(refine_net, 'backbone', None),
        'prompt_encoder': getattr(refine_net, 'prompt_encoder', None),
        'up1': getattr(refine_net, 'up1', None),
        'up2': getattr(refine_net, 'up2', None),
        'up3': getattr(refine_net, 'up3', None),
        'head': getattr(refine_net, 'head64', None),
    }
    for key, mod in named_parts.items():
        if key in freeze_set:
            _set_requires_grad(mod, False)
    if 'fuse' in freeze_set and pred_fuse is not None:
        _set_requires_grad(pred_fuse, False)
    stages_spec = _parse_items_csv(getattr(args, 'backbone_train_stages', ''))
    if stages_spec and hasattr(refine_net, 'backbone') and hasattr(refine_net.backbone, 'stages'):
        try:
            idxs = [int(s) for s in stages_spec]
        except Exception:
            idxs = []
        bb = refine_net.backbone
        if 'backbone' in freeze_set:
            _set_requires_grad(bb, False)
        for i in idxs:
            if 0 <= i < len(bb.stages):
                _set_requires_grad(bb.stages[i], True)
    patterns = _parse_items_csv(getattr(args, 'unfreeze_pattern', ''))
    if patterns:
        for n, p in refine_net.named_parameters():
            if any(pat in n for pat in patterns):
                p.requires_grad = True
        if pred_fuse is not None:
            for n, p in pred_fuse.named_parameters():
                fulln = f"fuse.{n}"
                if any(pat in fulln for pat in patterns):
                    p.requires_grad = True
    train_params: List[torch.nn.Parameter] = []
    named_train_params: List[Tuple[str, torch.nn.Parameter]] = []
    for n, p in refine_net.named_parameters():
        if p.requires_grad:
            train_params.append(p)
            named_train_params.append((f"refine.{n}", p))
    if pred_fuse is not None:
        for n, p in pred_fuse.named_parameters():
            if p.requires_grad:
                train_params.append(p)
                named_train_params.append((f"fuse.{n}", p))
    if _is_main_process():
        msg = ["[Trainable summary]"]
        for key, mod in named_parts.items():
            msg.append(f"  {key:<15} total={_count_params(mod):>10,}  trainable={_count_params(mod, True):>10,}")
        if pred_fuse is not None:
            msg.append(f"  {'fuse':<15} total={_count_params(pred_fuse):>10,}  trainable={_count_params(pred_fuse, True):>10,}")
        _maybe_print("\n".join(msg))
    return train_params, named_train_params
# ---------------- Batched instance forward ----------------
@torch.no_grad()
def _sam_forward_batched_for_one_image(
    sam_model: nn.Module,
    image_bchw: torch.Tensor,
    entries: List[PromptEntry],
    resize_transform: ResizeLongestSide,
    input_hw: Tuple[int, int],
    mask_input_size: int,
    device: torch.device,
    precomputed_img_emb: Optional[torch.Tensor] = None,
    precomputed_image_pe: Optional[torch.Tensor] = None,
):
    N = len(entries)
    if N == 0:
        return None
    if precomputed_img_emb is None:
        inputs = sam_model.preprocess(image_bchw)
        img_emb_1 = sam_model.image_encoder(inputs)
    else:
        img_emb_1 = precomputed_img_emb
    if precomputed_image_pe is None:
        image_pe_1 = sam_model.prompt_encoder.get_dense_pe().to(device)
    else:
        image_pe_1 = precomputed_image_pe
    B1, C, H, W = img_emb_1.shape
    pts, plabels, boxes_1024, mask_inputs, targets_1024 = _transform_prompts_batched(
        entries, resize_transform, input_hw, device
    )
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=(pts, plabels) if pts is not None else None,
        boxes=boxes_1024,
        masks=mask_inputs,
    )
    if dense_embeddings is None:
        dense_prompt_img = img_emb_1.new_zeros((N, C, H, W))
    elif dense_embeddings.dim() == 4:
        dense_prompt_img = dense_embeddings
    else:
        dense_prompt_img = img_emb_1.new_zeros((N, C, H, W))
    low_res_masks, iou_preds = sam_model.mask_decoder(
        image_embeddings=img_emb_1,
        image_pe=image_pe_1,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_prompt_img,
        multimask_output=False,
    )
    return {
        "low_res_logits_256": low_res_masks,
        "iou_preds": iou_preds,
        "boxes_1024": boxes_1024,
        "targets_1024": targets_1024,
    }
def _boxes_from_entries_direct(entries: List[PromptEntry], device: torch.device, pad_ratio_1024: float = 0.10) -> torch.Tensor:
    """
    直接从 entries 的 1024 坐标系 boxes 生成裁剪框；外扩由“比例”控制（每边按 w*pad_ratio / h*pad_ratio）。
    """
    boxes = []
    for e in entries:
        if e.boxes is None:
            m = (e.target_mask[0,0] > 0.5).nonzero(as_tuple=False)
            if m.numel() == 0:
                boxes.append(torch.tensor([0,0,1024,1024], device=device, dtype=torch.int64)); continue
            ymin = int(m[:,0].min().item()); ymax = int(m[:,0].max().item())
            xmin = int(m[:,1].min().item()); xmax = int(m[:,1].max().item())
            w = xmax - xmin + 1; h = ymax - ymin + 1
            px = int(round(w * float(pad_ratio_1024))); py = int(round(h * float(pad_ratio_1024)))
            x1 = max(0, xmin - px); y1 = max(0, ymin - py)
            x2 = min(1024, xmax + 1 + px); y2 = min(1024, ymax + 1 + py)
            boxes.append(torch.tensor([x1,y1,x2,y2], device=device, dtype=torch.int64))
        else:
            x1,y1,x2,y2 = [int(v) for v in e.boxes[0].tolist()]
            w = x2 - x1 + 1; h = y2 - y1 + 1
            px = int(round(w * float(pad_ratio_1024))); py = int(round(h * float(pad_ratio_1024)))
            x1 = max(0, x1 - px); y1 = max(0, y1 - py)
            x2 = min(1024, x2 + px + 1); y2 = min(1024, y2 + py + 1)
            boxes.append(torch.tensor([x1,y1,x2,y2], device=device, dtype=torch.int64))
    return torch.stack(boxes, dim=0)
def _batched_instances_forward_for_one_image(
    sam_model: Optional[nn.Module],
    refine_net: ROIRefineNet,
    pred_fuse: Optional[PredFuse],
    image_bchw: torch.Tensor,
    entries: List[PromptEntry],
    resize_transform: Optional[ResizeLongestSide],
    input_hw: Tuple[int, int],
    args: argparse.Namespace,
    precomputed_img_emb: Optional[torch.Tensor] = None,
    precomputed_image_pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    device = image_bchw.device
    N = len(entries)
    if N == 0:
        return image_bchw.new_tensor(0.0), image_bchw.new_tensor(0.0), {}
    need_sam = (args.crop_source == 'sam') or args.use_prompt or args.fuse_with_sam
    chunk_size = int(getattr(args, 'instance_chunk_size', 0) or 0)
    if chunk_size <= 0:
        chunk_size = N
    loss_sum: Optional[torch.Tensor] = None
    total_instances = 0
    total_iou = 0.0
    total_bce = 0.0
    total_dice = 0.0
    for start in range(0, N, chunk_size):
        chunk_entries = entries[start:start + chunk_size]
        if not chunk_entries:
            continue
        chunk_N = len(chunk_entries)
        sam_prob_256: Optional[torch.Tensor] = None
        boxes_1024: torch.Tensor
        boxes_256: Optional[torch.Tensor] = None
        if need_sam:
            assert sam_model is not None and resize_transform is not None, "SAM path needs sam_model & transform."
            with torch.no_grad():
                sam_out = _sam_forward_batched_for_one_image(
                    sam_model, image_bchw, chunk_entries, resize_transform, input_hw, args.mask_input_size, device,
                    precomputed_img_emb=precomputed_img_emb, precomputed_image_pe=precomputed_image_pe,
                )
                low_res_logits_256 = sam_out["low_res_logits_256"]
                boxes_1024_from_prompt = sam_out["boxes_1024"]
                sam_prob_256 = torch.sigmoid(low_res_logits_256)
                if args.crop_source == 'sam':
                    boxes_256 = _bbox_from_mask_256(low_res_logits_256, pad_ratio=args.roi_pad_256)
                    boxes_1024 = boxes_256.clone()
                    boxes_1024[:, 0] *= 4; boxes_1024[:, 1] *= 4; boxes_1024[:, 2] *= 4; boxes_1024[:, 3] *= 4
                else:
                    assert boxes_1024_from_prompt is not None, "prompt boxes are required when crop_source=prompt"
                    x1y1x2y2 = boxes_1024_from_prompt.clone().long()
                    w = (x1y1x2y2[:,2] - x1y1x2y2[:,0] + 1).clamp(min=1)
                    h = (x1y1x2y2[:,3] - x1y1x2y2[:,1] + 1).clamp(min=1)
                    px = torch.round(w.float() * float(args.roi_pad)).long()
                    py = torch.round(h.float() * float(args.roi_pad)).long()
                    x1y1x2y2[:, 0] = torch.clamp(x1y1x2y2[:, 0] - px, 0, 1024)
                    x1y1x2y2[:, 1] = torch.clamp(x1y1x2y2[:, 1] - py, 0, 1024)
                    x1y1x2y2[:, 2] = torch.clamp(x1y1x2y2[:, 2] + px + 1, 0, 1024)
                    x1y1x2y2[:, 3] = torch.clamp(x1y1x2y2[:, 3] + py + 1, 0, 1024)
                    boxes_1024 = x1y1x2y2
                    boxes_256 = torch.stack([
                        torch.tensor([max(0, b[0]//4), max(0, b[1]//4), min(256, (b[2]+3)//4), min(256, (b[3]+3)//4)],
                                     device=device, dtype=torch.int64)
                        for b in boxes_1024
                    ], dim=0)
        else:
            boxes_1024 = _boxes_from_entries_direct(chunk_entries, device=device, pad_ratio_1024=args.roi_pad)
        roi_img_256 = _batch_crop_resize_img_bchw_1024(image_bchw, boxes_1024, out_hw=256)
        prompt_mask_256 = None
        if args.use_prompt and need_sam:
            assert sam_prob_256 is not None
            if args.crop_source == 'sam':
                boxes_256 = _bbox_from_mask_256(torch.log(sam_prob_256/(1 - sam_prob_256 + 1e-6)), pad_ratio=args.roi_pad_256)
            assert boxes_256 is not None
            prompt_mask_256 = _batch_crop_resize_mask_256(sam_prob_256, boxes_256, out_hw=256)
        logits_refine_256 = refine_net(roi_img_256, prompt_mask_bchw_256=prompt_mask_256)
        prob_refine_256 = torch.sigmoid(logits_refine_256)
        if args.fuse_with_sam:
            assert need_sam and sam_prob_256 is not None and pred_fuse is not None, \
                "fuse_with_sam=True requires SAM path and PredFuse."
            assert boxes_256 is not None
            fused_mask_256 = _batch_crop_resize_mask_256(sam_prob_256, boxes_256, out_hw=256)
            final_logits_256 = pred_fuse(fused_mask_256, prob_refine_256)
        else:
            final_logits_256 = logits_refine_256
        target_roi_256_list = []
        for idx_entry, entry in enumerate(chunk_entries):
            tgt1024 = _interp(entry.target_mask, size=(1024, 1024), mode='bilinear')
            roi = _batch_crop_resize_img_bchw_1024(tgt1024, boxes_1024[idx_entry:idx_entry+1], out_hw=256)
            target_roi_256_list.append(roi)
        target_roi_256 = torch.cat(target_roi_256_list, dim=0)
        loss_bce = _bce_with_logits_mean(final_logits_256, target_roi_256)
        loss_dice = _dice_loss(torch.sigmoid(final_logits_256), target_roi_256)
        loss = loss_bce + args.dice_loss_weight * loss_dice
        with torch.no_grad():
            probs_256 = torch.sigmoid(final_logits_256)
            ious = []
            for idx_entry, entry in enumerate(chunk_entries):
                iou = _hard_iou_full_image(
                    probs_256[idx_entry:idx_entry+1], entry.target_mask, boxes_1024[idx_entry], args.mask_threshold,
                )
                ious.append(iou)
            hard_iou_mean = torch.stack(ious).mean()
        scaled_loss = loss * chunk_N
        loss_sum = scaled_loss if loss_sum is None else loss_sum + scaled_loss
        total_instances += chunk_N
        total_iou += float(hard_iou_mean.detach()) * chunk_N
        total_bce += float(loss_bce.detach()) * chunk_N
        total_dice += float(loss_dice.detach()) * chunk_N
    if loss_sum is None or total_instances == 0:
        zero = image_bchw.new_tensor(0.0)
        return zero, zero, {}
    mean_loss = loss_sum / total_instances
    mean_iou_tensor = image_bchw.new_tensor(total_iou / total_instances)
    comps = {'bce': total_bce / total_instances, 'dice': total_dice / total_instances}
    return mean_loss, mean_iou_tensor, comps
# ---------------- Train / Eval ----------------
def train_one_epoch(
    sam_model: Optional[nn.Module],
    refine_net: ROIRefineNet,
    pred_fuse: Optional[PredFuse],
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    args: argparse.Namespace,
    clip_params: List[nn.Parameter],
) -> Tuple[float, float]:
    refine_net.train()
    if pred_fuse is not None:
        pred_fuse.train()
    total_loss = 0.0
    total_iou = 0.0
    total_instances = 0
    need_sam_global = (args.crop_source == 'sam') or args.use_prompt or args.fuse_with_sam
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size) if need_sam_global and sam_model is not None else None
    steps_per_epoch = max(1, len(data_loader))
    warmup_steps = args.warmup_steps if args.warmup_epochs <= 0 else args.warmup_epochs * steps_per_epoch
    for batch_idx, batch in enumerate(data_loader):
        global_step = (args.current_epoch - 1) * steps_per_epoch + batch_idx
        scale = min(1.0, float(global_step + 1) / float(max(1, warmup_steps))) if warmup_steps > 0 else 1.0
        for pg in optimizer.param_groups:
            pg['lr'] = pg.get('lr', args.lr) * 0 + (args.lr_backbone if 'backbone' in pg.get('name','') and args.lr_backbone is not None else args.lr) * scale  # 保底
        images = batch['images'].to(device)
        masks_list: List[torch.Tensor] = batch['masks']
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
            if need_sam_global and sam_model is not None:
                with torch.no_grad():
                    inputs_b = sam_model.preprocess(images)
                    img_emb_B = sam_model.image_encoder(inputs_b)
                    dense_pe_1 = sam_model.prompt_encoder.get_dense_pe().to(device)
            else:
                img_emb_B = None
                dense_pe_1 = None
            batch_loss = 0.0
            batch_iou = 0.0
            batch_instances = 0
            for b in range(images.shape[0]):
                image_bchw = images[b:b+1]
                prompts = _prepare_prompts(
                    masks=masks_list[b].to(device),
                    prompt_type=args.prompt_type,
                    device=device,
                    image_size=args.image_size,
                    mask_input_size=args.mask_input_size,
                )
                if not prompts:
                    continue
                loss_i, iou_i, _ = _batched_instances_forward_for_one_image(
                    sam_model if need_sam_global else None, refine_net, pred_fuse,
                    image_bchw, prompts, resize_transform, batch['orig_sizes'][b], args,
                    precomputed_img_emb=(img_emb_B[b:b+1] if img_emb_B is not None else None),
                    precomputed_image_pe=dense_pe_1,
                )
                batch_loss += loss_i
                batch_iou += iou_i.detach()
                batch_instances += 1
            if batch_instances == 0:
                continue
            loss = batch_loss / batch_instances
        if scaler is not None and args.amp:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                clip_grad_norm_(clip_params, max_norm=args.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                clip_grad_norm_(clip_params, max_norm=args.clip_max_norm)
            optimizer.step()
        total_loss += loss.item() * batch_instances
        total_iou += batch_iou.item()
        total_instances += batch_instances
        if _is_main_process() and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / max(1, total_instances)
            avg_iou = total_iou / max(1, total_instances)
            _maybe_print(f'[Epoch {args.current_epoch}] Step {batch_idx + 1}/{len(data_loader)} '
                         f'lr:{optimizer.param_groups[0]["lr"]:.2e} | loss:{avg_loss:.4f} | IoU(full):{avg_iou:.4f}')
    mean_loss = total_loss / max(1, total_instances)
    mean_iou = total_iou / max(1, total_instances)
    return mean_loss, mean_iou
@torch.no_grad()
def evaluate(
    sam_model: Optional[nn.Module],
    refine_net: ROIRefineNet,
    pred_fuse: Optional[PredFuse],
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    refine_net.eval()
    if pred_fuse is not None:
        pred_fuse.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_instances = 0
    need_sam_global = (args.crop_source == 'sam') or args.use_prompt or args.fuse_with_sam
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size) if need_sam_global and sam_model is not None else None
    for batch in data_loader:
        images = batch['images'].to(device)
        masks_list: List[torch.Tensor] = batch['masks']
        with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
            if need_sam_global and sam_model is not None:
                with torch.no_grad():
                    inputs_b = sam_model.preprocess(images)
                    img_emb_B = sam_model.image_encoder(inputs_b)
                    dense_pe_1 = sam_model.prompt_encoder.get_dense_pe().to(device)
            else:
                img_emb_B = None
                dense_pe_1 = None
            for b in range(images.shape[0]):
                image_bchw = images[b:b+1]
                prompts = _prepare_prompts(
                    masks=masks_list[b].to(device),
                    prompt_type=args.prompt_type,
                    device=device,
                    image_size=args.image_size,
                    mask_input_size=args.mask_input_size,
                )
                if not prompts:
                    continue
                loss_i, iou_i, _ = _batched_instances_forward_for_one_image(
                    sam_model if need_sam_global else None, refine_net, pred_fuse,
                    image_bchw, prompts, resize_transform, batch['orig_sizes'][b], args,
                    precomputed_img_emb=(img_emb_B[b:b+1] if img_emb_B is not None else None),
                    precomputed_image_pe=dense_pe_1,
                )
                total_loss += float(loss_i.item())
                total_iou += float(iou_i.item())
                total_instances += 1
    mean_loss = total_loss / max(1, total_instances)
    mean_iou = total_iou / max(1, total_instances)
    return mean_loss, mean_iou
# ---------------- Dist / runtime ----------------
def setup_device_and_distributed(args: argparse.Namespace) -> torch.device:
    if args.distributed:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        backend = 'nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        return device
    device_str = args.device or DEFAULT_DEVICE
    return torch.device(device_str)
def cleanup_distributed(args: argparse.Namespace) -> None:
    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
# ---------------- Args ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SAM + Prompted ROI refine (batched, SAM usage on-demand)')
    parser.add_argument('--train-images', type=str, default=str(DEFAULT_TRAIN_IMAGES))
    parser.add_argument('--train-annotations', type=str, default=str(DEFAULT_TRAIN_ANN))
    parser.add_argument('--val-images', type=str, default=str(DEFAULT_VAL_IMAGES))
    parser.add_argument('--val-annotations', type=str, default=str(DEFAULT_VAL_ANN))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--model-type', type=str, default='vit_b_lm')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--init-weights', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--mask-input-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--opt', type=str, choices=['adamw', 'adam', 'sgd'], default='adamw')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--prompt-type', choices=['points', 'boxes', 'points_boxes', 'dense'], default='boxes')
    parser.add_argument('--max-instances', type=int, default=32)
    parser.add_argument('--instance-chunk-size', type=int, default=0,
                        help='Max instances per chunk when running SAM/ROI forward; 0 processes all instances at once.')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Run validation every N epochs (0 disables mid-epoch validation; final epoch still runs).')
    parser.add_argument('--skip-initial-eval', action='store_true', default=False,
                        help='Skip the epoch-0 warmup evaluation to reduce startup overhead.')
    # 损失
    parser.add_argument('--dice-loss-weight', type=float, default=1.0)
    # AMP / 分布式 / 设备
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    # 数据增强
    parser.add_argument('--use-augmentation', action='store_true', default=True)
    parser.add_argument('--augmentation-config', type=str, default=(str(DEFAULT_AUG_CONFIG) if DEFAULT_AUG_CONFIG.exists() else None))
    parser.add_argument('--aug-until-epoch', type=int, default=90,
                        help='>0 时仅在前 N 个 epoch 开启数据增强；0 表示按 use-augmentation 全程控制')
    # 细化网络 backbone
    parser.add_argument('--refine-backbone', choices=['convnext', 'lite'], default='convnext')
    parser.add_argument('--convnext-variant', type=str, default='tiny')
    parser.add_argument('--convnext-ckpt', type=str, default=os.environ.get('CONVNEXT_CKPT'))
    # 是否采用 SAM 掩码作为提示
    parser.add_argument('--use-prompt', dest='use_prompt', action='store_true', default=True)
    parser.add_argument('--no-prompt',  dest='use_prompt', action='store_false')
    # 是否总是构建 PromptMaskEncoder（即使不用）
    parser.add_argument('--build-prompt', action='store_true', default=False,
                        help='始终实例化 PromptMaskEncoder 以便进入 state_dict；是否实际使用仍由 --use-prompt 决定（不会因此触发 SAM 前向）')
    # 融合
    parser.add_argument('--fuse-with-sam', action='store_true', default=False)
    parser.add_argument('--fuse-kernel', type=int, choices=[1, 3], default=1)
    # ROI 裁剪来源与“比例外扩”参数（修改点：改为 float 比例）
    parser.add_argument('--crop-source', choices=['sam', 'prompt'], default='sam')
    parser.add_argument('--roi-pad-256', type=float, default=0.10, help='sam 裁剪时在 256 尺度上的外扩比例（每边），如 0.10=10%')
    parser.add_argument('--roi-pad', type=float, default=0.10, help='prompt 裁剪时在 1024 尺度上的外扩比例（每边），如 0.10=10%')
    # IoU / 阈值
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    # Warmup
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    # 梯度裁剪
    parser.add_argument('--clip-grad', dest='clip_grad', action='store_true', default=True)
    parser.add_argument('--no-clip-grad', dest='clip_grad', action='store_false')
    parser.add_argument('--clip-max-norm', type=float, default=1.0)
    # 训练/冻结开关
    parser.add_argument('--freeze', type=str, default='backbone,up1,up2,up3,head,decoder,fuse',
                        help='逗号分隔模块：backbone,prompt_encoder,up1,up2,up3,head,decoder,all,fuse')
    parser.add_argument('--backbone-train-stages', type=str, default='',
                        help='仅 ConvNeXt：指定需解冻的 stages，如 "2,3"')
    parser.add_argument('--unfreeze-pattern', type=str, default='',
                        help='按参数名子串强制解冻，逗号分隔，如 "backbone.stages.3,prompt_encoder.gate"')
    parser.add_argument('--lr-backbone', type=float, default=None, help='给 backbone 的更小学习率（可选）')
    # 保存/恢复
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=r'C:\localtask\Point2Org\roi_refine_epoch_75.pt')
    parser.add_argument('--seed', type=int, default=42)
    # 冷启动只评估
    parser.add_argument('--eval-only', action='store_true', default=False)
    return _apply_argument_defaults(parser.parse_args())
# ---------------- Main ----------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device_and_distributed(args)
    # dataset / loader
    aug_cfg = AugmentationConfig.from_json(args.augmentation_config) if args.use_augmentation else None
    train_dataset = SamCocoDataset(
        image_root=args.train_images,
        annotation_path=args.train_annotations,
        image_size=args.image_size,
        max_instances=args.max_instances,
        use_augmentation=args.use_augmentation,
        augmentation_config=aug_cfg,
        prompt_type=args.prompt_type,
    )
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True); shuffle = False
    else:
        train_sampler = None; shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=plain_collate,
    )
    val_loader = None
    if args.val_images and args.val_annotations:
        val_dataset = SamCocoDataset(
            image_root=args.val_images,
            annotation_path=args.val_annotations,
            image_size=args.image_size,
            max_instances=None,  # no cap during validation; evaluate on all instances
            use_augmentation=False,
            prompt_type=args.prompt_type,
            allow_empty=True,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=plain_collate,
        )
    # ---- SAM ----
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is not None and not ckpt.exists():
        fallback = _find_default_sam_checkpoint(args.model_type)
        if fallback is not None:
            _maybe_print(f"SAM checkpoint '{ckpt}' not found; using '{fallback}' instead.")
            ckpt = fallback
        else:
            _maybe_print(f"SAM checkpoint '{ckpt}' not found; using random initialization.")
            ckpt = None
    sam_model = build_plain_sam(args.model_type, str(ckpt) if ckpt is not None else None, device)
    # ---- ROI refine net + fuse head ----
    use_cx = (args.refine_backbone == 'convnext') and HAS_CONVNEXT
    refine_net = ROIRefineNet(
        use_convnext=use_cx,
        convnext_variant=args.convnext_variant,
        convnext_ckpt=args.convnext_ckpt,
        use_prompt=args.use_prompt,
        build_prompt=args.build_prompt,
    ).to(device)
    pred_fuse = PredFuse(kernel=args.fuse_kernel).to(device) if args.fuse_with_sam else None
    # 冻结 SAM
    for p in sam_model.parameters():
        p.requires_grad = False
    # 冻结计划
    train_params, named_train_params = apply_freeze_plan_and_collect(refine_net, pred_fuse, args)
    # 分组 LR
    if args.lr_backbone is not None:
        group_backbone = [p for (n, p) in named_train_params if n.startswith("refine.backbone.")]
        group_default  = [p for (n, p) in named_train_params if not n.startswith("refine.backbone.")]
        param_groups = []
        if group_default:
            param_groups.append({'params': group_default, 'lr': args.lr, 'weight_decay': args.weight_decay, 'name': 'default'})
        if group_backbone:
            param_groups.append({'params': group_backbone, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay, 'name': 'backbone'})
    else:
        param_groups = [{'params': train_params, 'lr': args.lr, 'weight_decay': args.weight_decay, 'name': 'default'}]
    # optimizer
    if args.opt == 'adamw':
        optimizer = AdamW(param_groups)
    elif args.opt == 'adam':
        optimizer = Adam(param_groups)
    else:
        optimizer = SGD(param_groups, momentum=args.momentum, nesterov=args.nesterov)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    # resume（仅 refine/fuse）
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location='cpu')
            if 'refine' in ckpt:
                refine_net.load_state_dict(ckpt['refine'], strict=False)
            if pred_fuse is not None and 'fuse' in ckpt:
                pred_fuse.load_state_dict(ckpt['fuse'], strict=False)
            _maybe_print(f"Resumed refine/fuse weights from '{resume_path}'")
        else:
            _maybe_print(f"Resume checkpoint '{resume_path}' not found, skipping.")
    if args.distributed:
        refine_net = DDP(refine_net, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=False)
        if pred_fuse is not None:
            pred_fuse = DDP(pred_fuse, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=False)
    # ---- 冷启动评估（Epoch 0）----
    train0_loss, train0_iou = float('nan'), float('nan')
    val0_loss, val0_iou = float('nan'), float('-inf')
    if not args.skip_initial_eval:
        train0_loss, train0_iou = evaluate(sam_model, refine_net, pred_fuse, train_loader, device, args)
        _maybe_print(f"Epoch 0: train loss {train0_loss:.4f}, IoU(full) {train0_iou:.4f}")
        if val_loader is not None:
            val0_loss, val0_iou = evaluate(sam_model, refine_net, pred_fuse, val_loader, device, args)
            _maybe_print(f"Epoch 0:  val  loss {val0_loss:.4f}, IoU(full) {val0_iou:.4f}")
    else:
        if _is_main_process():
            _maybe_print("Skip initial evaluation (train/val) per --skip-initial-eval.")
    # ---- 初始化并保存一次 best（以 val IoU 为准）----
    best_iou = float('-inf')
    best_epoch = -1
    if val_loader is not None and not args.skip_initial_eval:
        current_iou = val0_iou
        if current_iou > best_iou and _is_main_process():
            best_iou = float(current_iou)
            best_epoch = 0
            out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
            payload = {
                'refine': (refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict()),
                'meta': {'epoch': best_epoch, 'best_iou': float(best_iou)}
            }
            if pred_fuse is not None:
                payload['fuse'] = (pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict())
            torch.save(payload, str(out / 'best.pt'))
            _maybe_print(f"[BEST] Epoch {best_epoch}: IoU={best_iou:.4f} -> saved to best.pt")
    if args.eval_only:
        cleanup_distributed(args)
        return
    # ---- Train ----
    turned_off_aug = False
    for epoch in range(args.epochs):
        args.current_epoch = epoch + 1
        if args.distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        if args.aug_until_epoch > 0:
            use_aug_now = args.use_augmentation and ((epoch + 1) <= args.aug_until_epoch)
        else:
            use_aug_now = args.use_augmentation
        if train_dataset.use_augmentation != use_aug_now:
            train_dataset.use_augmentation = use_aug_now
            if _is_main_process():
                _maybe_print(f"[AUG] Epoch {epoch + 1}: use_augmentation = {use_aug_now}")
            if (not use_aug_now) and (not turned_off_aug):
                turned_off_aug = True
        clip_params = train_params
        train_loss, train_iou = train_one_epoch(
            sam_model, refine_net, pred_fuse, train_loader, optimizer, scaler, device, args, clip_params
        )
        _maybe_print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, IoU(full) {train_iou:.4f}")
        val_loss, val_iou = (None, None)
        run_validation = False
        if val_loader is not None:
            if args.eval_interval == 0:
                run_validation = (epoch + 1) == args.epochs
            else:
                run_validation = ((epoch + 1) % args.eval_interval == 0) or ((epoch + 1) == args.epochs)
        if run_validation:
            val_loss, val_iou = evaluate(sam_model, refine_net, pred_fuse, val_loader, device, args)
            _maybe_print(f"Epoch {epoch + 1}:  val  loss {val_loss:.4f}, IoU(full) {val_iou:.4f}")
            if _is_main_process() and val_iou is not None and val_iou > best_iou:
                best_iou = float(val_iou)
                best_epoch = epoch + 1
                out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
                payload = {
                    'refine': (refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict()),
                    'meta': {'epoch': best_epoch, 'best_iou': float(best_iou)}
                }
                if pred_fuse is not None:
                    payload['fuse'] = (pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict())
                torch.save(payload, str(out / 'best.pt'))
                _maybe_print(f"[BEST] Epoch {best_epoch}: IoU={best_iou:.4f} -> saved to best.pt")
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs):
            if _is_main_process():
                out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
                payload = {
                    'refine': (refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict())
                }
                if pred_fuse is not None:
                    payload['fuse'] = (pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict())
                torch.save(payload, str(out / f'roi_refine_epoch_{epoch + 1}.pt'))
                _maybe_print(f"Saved checkpoint at epoch {epoch + 1}")
    cleanup_distributed(args)
if __name__ == '__main__':
    main()
