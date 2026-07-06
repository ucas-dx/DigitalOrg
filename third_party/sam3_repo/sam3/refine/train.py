from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import util as base
from ultralytics.nn.modules.convnextv1 import ConvNeXtV1

DATA_ROOT = Path("C:/localtask/digOrg")
DEFAULT_OUTPUT_BASE = DATA_ROOT / "plain_sam_checkpoints_convnext_v1"
DEFAULT_SAM_CHECKPOINT = base.DEFAULT_SAM_CHECKPOINT
DEFAULT_STUDENT_WEIGHTS = DATA_ROOT / "student_atto_v1_epoch_90.pt"
SUPPORTED_FRACTIONS: Tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
STUDENT_VARIANTS: Tuple[str, ...] = (
    "atto",
    "femto",
    "pico",
    "nano",
    "tiny",
    "small",
    "base",
    "large",
    "huge",
)

DATASET_REGISTRY: Dict[str, Dict[str, Path]] = {
    name: {
        "root": DATA_ROOT / name,
        "train_images": DATA_ROOT / name / "train" / "images",
        "train_annotations": DATA_ROOT / name / "train" / "data.json",
        "val_images": DATA_ROOT / name / "test" / "images",
        "val_annotations": DATA_ROOT / name / "test" / "data.json",
    }
    for name in ("brain", "colon", "example_dataset", "pdac")
}


class TrainSamCocoDataset(base.SamCocoDataset):
    def __init__(
        self,
        *args,
        subset_fraction: Optional[float] = None,
        subset_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
            rng = np.random.default_rng(subset_seed)
            total = len(self.image_ids)
            keep = max(1, int(round(total * subset_fraction)))
            if keep < total:
                indices = rng.choice(total, keep, replace=False)
                indices.sort()
                self.image_ids = [self.image_ids[i] for i in indices]


def _parse_dataset_names(spec: str) -> List[str]:
    token = spec.strip().lower()
    if token in {"", "none"}:
        raise ValueError("At least one dataset must be specified.")
    if token == "all":
        return sorted(DATASET_REGISTRY.keys())
    names = [part.strip().lower() for part in spec.split(",") if part.strip()]
    invalid = [name for name in names if name not in DATASET_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(invalid)}. Available: {', '.join(sorted(DATASET_REGISTRY))}"
        )
    return names


def _parse_fraction_list(spec: str) -> List[float]:
    token = spec.strip().lower()
    if token in {"", "none"}:
        raise ValueError("At least one train fraction must be provided.")
    if token == "all":
        return list(SUPPORTED_FRACTIONS)
    fractions: List[float] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw.endswith("%"):
            value = float(raw[:-1]) / 100.0
        else:
            value = float(raw)
            if value > 1.0:
                value = value / 100.0
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid train fraction '{raw}'. Must be in (0, 1].")
        fractions.append(round(value, 4))
    if not fractions:
        raise ValueError("No valid fractions parsed.")
    return fractions


def _fraction_tag(value: float) -> str:
    pct = int(round(value * 100))
    return f"p{pct:02d}"


def _module_label(args: argparse.Namespace) -> str:
    parts: List[str] = [f"convnextv1_{args.student_variant}"]
    parts.append("prompt" if args.use_prompt else "noprompt")
    parts.append(f"fuse{args.fuse_kernel}" if args.fuse_with_sam else "nofuse")
    return "_".join(parts)


def _ensure_output(base_dir: Path, dataset: str, fraction_tag: str, module: str) -> Path:
    out = (base_dir / dataset / fraction_tag / module).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


class ROIRefineNetV1(nn.Module):
    """ConvNeXt-V1 based refine head mirroring refine2 distillation student."""

    def __init__(
        self,
        variant: str = "atto",
        ckpt: Optional[str] = None,
        use_prompt: bool = False,
        build_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.use_prompt = use_prompt
        self.build_prompt = build_prompt

        self.register_buffer(
            "imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

        self.backbone = ConvNeXtV1(
            variant=variant,
            in_chans=3,
            ckpt=ckpt if ckpt else None,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.0,
            layer_scale=1e-6,
        )
        dims = ConvNeXtV1.VARIANT_DIMS.get(variant, ConvNeXtV1.VARIANT_DIMS["tiny"])
        c0, c1, c2, c3 = dims

        self.prompt_encoder = (
            base.PromptMaskEncoder(out_channels=c0) if (self.build_prompt or self.use_prompt) else None
        )

        self.up3 = base.UpBlock(c_in=c3, c_skip=c2, c_out=256)
        self.up2 = base.UpBlock(c_in=256, c_skip=c1, c_out=128)
        self.up1 = base.UpBlock(c_in=128, c_skip=c0, c_out=64)
        self.head64 = nn.Sequential(base.ConvBNAct(64, 32, 3), nn.Conv2d(32, 1, kernel_size=1))

    def forward(
        self, roi_img_bchw: torch.Tensor, prompt_mask_bchw_256: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_im = (roi_img_bchw / 255.0 - self.imnet_mean) / self.imnet_std
        x0, x1, x2, x3 = self.backbone(x_im)

        if self.use_prompt and self.prompt_encoder is not None and prompt_mask_bchw_256 is not None:
            x0 = x0 + self.prompt_encoder(prompt_mask_bchw_256, roi_img_bchw)

        y = self.up3(x3, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)
        logits64 = self.head64(y)
        return base._interp(logits64, size=(256, 256), mode="bilinear")


def _load_student_checkpoint(refine_net: torch.nn.Module, path: Path) -> None:
    if not path.exists():
        return
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("refine", checkpoint)
    target = refine_net.module if isinstance(refine_net, DDP) else refine_net
    missing, unexpected = target.load_state_dict(state, strict=False)
    if base._is_main_process():
        base._maybe_print(
            f"[Init] Loaded student weights from '{path}'. missing={len(missing)} unexpected={len(unexpected)}"
        )


def run_single_experiment(
    cfg: argparse.Namespace,
    device: torch.device,
    dataset_name: str,
    fraction: float,
) -> Dict[str, object]:
    dataset_cfg = DATASET_REGISTRY[dataset_name]
    dataset_tag = dataset_name.lower()
    fraction_tag = _fraction_tag(fraction)
    module_label = _module_label(cfg)
    output_dir = _ensure_output(Path(cfg.output_dir), dataset_tag, fraction_tag, module_label)
    best_path = output_dir / "best.pt"

    run_args = copy.copy(cfg)
    run_args.output_dir = str(output_dir)
    run_args.dataset_name = dataset_name
    run_args.current_dataset = dataset_name
    run_args.current_fraction = fraction
    run_args.current_module = module_label
    run_args.train_images = str(dataset_cfg["train_images"])
    run_args.train_annotations = str(dataset_cfg["train_annotations"])
    run_args.val_images = str(dataset_cfg["val_images"])
    run_args.val_annotations = str(dataset_cfg["val_annotations"])

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    aug_cfg = (
        base.AugmentationConfig.from_json(run_args.augmentation_config)
        if run_args.use_augmentation
        else None
    )
    train_dataset = TrainSamCocoDataset(
        image_root=run_args.train_images,
        annotation_path=run_args.train_annotations,
        image_size=run_args.image_size,
        max_instances=run_args.max_instances,
        use_augmentation=run_args.use_augmentation,
        augmentation_config=aug_cfg,
        prompt_type=run_args.prompt_type,
        subset_fraction=fraction if fraction < 0.9999 else None,
        subset_seed=run_args.seed,
    )
    if run_args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=run_args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=run_args.num_workers,
        pin_memory=True,
        collate_fn=base.plain_collate,
    )

    val_dataset = base.SamCocoDataset(
        image_root=run_args.val_images,
        annotation_path=run_args.val_annotations,
        image_size=run_args.image_size,
        max_instances=None,
        use_augmentation=False,
        prompt_type=run_args.prompt_type,
        allow_empty=True,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if run_args.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=run_args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=run_args.num_workers,
        pin_memory=True,
        collate_fn=base.plain_collate,
    )

    sam_ckpt = Path(run_args.checkpoint).expanduser() if getattr(run_args, "checkpoint", None) else None
    if sam_ckpt is not None and not sam_ckpt.exists():
        fallback = base._find_default_sam_checkpoint(run_args.model_type)
        if fallback is not None:
            base._maybe_print(f"SAM checkpoint '{sam_ckpt}' not found; using '{fallback}' instead.")
            sam_ckpt = fallback
        else:
            base._maybe_print(f"SAM checkpoint '{sam_ckpt}' not found; using random initialization.")
            sam_ckpt = None
    sam_model = base.build_plain_sam(
        run_args.model_type,
        str(sam_ckpt) if sam_ckpt is not None else None,
        device,
    )
    for param in sam_model.parameters():
        param.requires_grad_(False)

    refine_net = ROIRefineNetV1(
        variant=run_args.student_variant,
        ckpt=run_args.student_convnext_ckpt,
        use_prompt=run_args.use_prompt,
        build_prompt=run_args.build_prompt,
    ).to(device)
    if run_args.student_init is not None:
        init_path = Path(run_args.student_init).expanduser()
        if init_path.exists():
            _load_student_checkpoint(refine_net, init_path)
        elif base._is_main_process():
            base._maybe_print(f"[Init] Student init checkpoint '{init_path}' not found, skipping.")

    pred_fuse = base.PredFuse(kernel=run_args.fuse_kernel).to(device) if run_args.fuse_with_sam else None

    train_params, named_train_params = base.apply_freeze_plan_and_collect(refine_net, pred_fuse, run_args)

    if run_args.lr_backbone is not None:
        backbone_params = [p for (n, p) in named_train_params if n.startswith("refine.backbone.")]
        default_params = [p for (n, p) in named_train_params if not n.startswith("refine.backbone.")]
        param_groups = []
        if default_params:
            param_groups.append(
                {"params": default_params, "lr": run_args.lr, "weight_decay": run_args.weight_decay, "name": "default"}
            )
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": run_args.lr_backbone,
                    "weight_decay": run_args.weight_decay,
                    "name": "backbone",
                }
            )
    else:
        param_groups = [
            {"params": train_params, "lr": run_args.lr, "weight_decay": run_args.weight_decay, "name": "default"}
        ]

    if run_args.opt == "adamw":
        optimizer = AdamW(param_groups)
    elif run_args.opt == "adam":
        optimizer = Adam(param_groups)
    else:
        optimizer = SGD(param_groups, momentum=run_args.momentum, nesterov=run_args.nesterov)

    scaler = GradScaler(enabled=run_args.amp and torch.cuda.is_available())

    resume_path = Path(run_args.resume).expanduser() if getattr(run_args, "resume", None) else None
    if resume_path and resume_path.exists():
        ckpt = torch.load(str(resume_path), map_location="cpu")
        target_refine = refine_net.module if isinstance(refine_net, DDP) else refine_net
        target_fuse = pred_fuse.module if isinstance(pred_fuse, DDP) else pred_fuse
        if "refine" in ckpt:
            target_refine.load_state_dict(ckpt["refine"], strict=False)
        else:
            target_refine.load_state_dict(ckpt, strict=False)
        if target_fuse is not None and "fuse" in ckpt:
            target_fuse.load_state_dict(ckpt["fuse"], strict=False)
        if base._is_main_process():
            base._maybe_print(f"[Resume] Loaded refine checkpoint from {resume_path}")
    elif resume_path and base._is_main_process():
        base._maybe_print(f"[Resume] Checkpoint {resume_path} not found; skipping.")

    if run_args.distributed:
        refine_net = DDP(
            refine_net,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        if pred_fuse is not None:
            pred_fuse = DDP(
                pred_fuse,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

    best_iou = float("-inf")
    best_epoch = -1

    if not run_args.skip_initial_eval:
        train0_loss, train0_iou, train0_sam_iou = base.evaluate(
            sam_model, refine_net, pred_fuse, train_loader, device, run_args
        )
        if base._is_main_process():
            base._maybe_print(
                f"[{dataset_tag}][{fraction_tag}][{module_label}] Epoch 0: "
                f"train loss {train0_loss:.4f}, IoU(full) {train0_iou:.4f}, SAM IoU {train0_sam_iou:.4f}"
            )
        val0_loss, val0_iou, val0_sam_iou = base.evaluate(
            sam_model, refine_net, pred_fuse, val_loader, device, run_args
        )
        if base._is_main_process():
            base._maybe_print(
                f"[{dataset_tag}][{fraction_tag}][{module_label}] Epoch 0:  "
                f"val  loss {val0_loss:.4f}, IoU(full) {val0_iou:.4f}, SAM IoU {val0_sam_iou:.4f}"
            )
        best_iou = float(val0_iou)
        best_epoch = 0
        payload = {
            "refine": (refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict()),
            "meta": {"epoch": best_epoch, "best_iou": float(best_iou)},
        }
        if pred_fuse is not None:
            payload["fuse"] = (
                pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict()
            )
        torch.save(payload, best_path)
    else:
        if base._is_main_process():
            base._maybe_print(f"[{dataset_tag}][{fraction_tag}][{module_label}] Skip initial evaluation per flag.")

    if getattr(run_args, "eval_only", False):
        if resume_path is None and not best_path.exists():
            raise FileNotFoundError("Eval-only mode requires --resume or an existing best.pt checkpoint.")
        if base._is_main_process():
            base._maybe_print(f"[{dataset_tag}][{fraction_tag}][{module_label}] Eval-only mode enabled.")
        if best_path.exists():
            ckpt = torch.load(str(best_path), map_location="cpu")
            target_refine = refine_net.module if isinstance(refine_net, DDP) else refine_net
            target_refine.load_state_dict(ckpt.get("refine", ckpt), strict=False)
            if pred_fuse is not None and "fuse" in ckpt:
                target_fuse = pred_fuse.module if isinstance(pred_fuse, DDP) else pred_fuse
                target_fuse.load_state_dict(ckpt["fuse"], strict=False)
            best_epoch = ckpt.get("meta", {}).get("epoch", best_epoch)
            best_iou = ckpt.get("meta", {}).get("best_iou", best_iou)
        final_loss, final_iou, final_sam_iou = base.evaluate(
            sam_model, refine_net, pred_fuse, val_loader, device, run_args
        )
        torch.cuda.empty_cache()
        return {
            "train_dataset": dataset_name,
            "train_fraction": fraction,
            "train_samples": len(train_dataset),
            "eval_dataset": dataset_name,
            "eval_samples": len(val_dataset),
            "module": module_label,
            "best_epoch": best_epoch,
            "val_loss": final_loss,
            "val_iou": final_iou,
            "sam_iou": final_sam_iou,
            "checkpoint": str(best_path.resolve() if best_path.exists() else resume_path.resolve()),
        }

    for epoch in range(run_args.epochs):
        run_args.current_epoch = epoch + 1
        if run_args.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if run_args.aug_until_epoch > 0:
            use_aug_now = run_args.use_augmentation and ((epoch + 1) <= run_args.aug_until_epoch)
        else:
            use_aug_now = run_args.use_augmentation
        if hasattr(train_dataset, "use_augmentation") and train_dataset.use_augmentation != use_aug_now:
            train_dataset.use_augmentation = use_aug_now
            if base._is_main_process():
                base._maybe_print(
                    f"[{dataset_tag}][{fraction_tag}][{module_label}] Epoch {epoch + 1}: use_augmentation = {use_aug_now}"
                )

        clip_params = train_params
        train_loss, train_iou = base.train_one_epoch(
            sam_model, refine_net, pred_fuse, train_loader, optimizer, scaler, device, run_args, clip_params
        )
        if base._is_main_process():
            base._maybe_print(
                f"[{dataset_tag}][{fraction_tag}][{module_label}] Epoch {epoch + 1}: train loss {train_loss:.4f}, IoU(full) {train_iou:.4f}"
            )

        run_validation = False
        if run_args.eval_interval == 0:
            run_validation = (epoch + 1) == run_args.epochs
        else:
            run_validation = ((epoch + 1) % run_args.eval_interval == 0) or ((epoch + 1) == run_args.epochs)

        if run_validation:
            val_loss, val_iou, val_sam_iou = base.evaluate(
                sam_model, refine_net, pred_fuse, val_loader, device, run_args
            )
            if base._is_main_process():
                base._maybe_print(
                    f"[{dataset_tag}][{fraction_tag}][{module_label}] Epoch {epoch + 1}:  "
                    f"val  loss {val_loss:.4f}, IoU(full) {val_iou:.4f}, SAM IoU {val_sam_iou:.4f}"
                )
            if val_iou is not None and val_iou > best_iou:
                best_iou = float(val_iou)
                best_epoch = epoch + 1
                payload = {
                    "refine": (
                        refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict()
                    ),
                    "meta": {"epoch": best_epoch, "best_iou": float(best_iou)},
                }
                if pred_fuse is not None:
                    payload["fuse"] = (
                        pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict()
                    )
                torch.save(payload, best_path)
                if base._is_main_process():
                    base._maybe_print(
                        f"[{dataset_tag}][{fraction_tag}][{module_label}] [BEST] epoch={best_epoch} IoU={best_iou:.4f}"
                    )

        if run_args.save_every > 0 and ((epoch + 1) % run_args.save_every == 0 or (epoch + 1) == run_args.epochs):
            if base._is_main_process():
                payload = {
                    "refine": (refine_net.module.state_dict() if isinstance(refine_net, DDP) else refine_net.state_dict())
                }
                if pred_fuse is not None:
                    payload["fuse"] = (
                        pred_fuse.module.state_dict() if isinstance(pred_fuse, DDP) else pred_fuse.state_dict()
                    )
                torch.save(payload, output_dir / f"roi_refine_epoch_{epoch + 1}.pt")
                base._maybe_print(
                    f"[{dataset_tag}][{fraction_tag}][{module_label}] Saved checkpoint at epoch {epoch + 1}"
                )

    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location="cpu")
        target_refine = refine_net.module if isinstance(refine_net, DDP) else refine_net
        target_refine.load_state_dict(ckpt.get("refine", ckpt), strict=False)
        if pred_fuse is not None and "fuse" in ckpt:
            target_fuse = pred_fuse.module if isinstance(pred_fuse, DDP) else pred_fuse
            target_fuse.load_state_dict(ckpt["fuse"], strict=False)
        best_epoch = ckpt.get("meta", {}).get("epoch", best_epoch)
        best_iou = ckpt.get("meta", {}).get("best_iou", best_iou)

    final_loss, final_iou, final_sam_iou = base.evaluate(
        sam_model, refine_net, pred_fuse, val_loader, device, run_args
    )
    torch.cuda.empty_cache()
    return {
        "train_dataset": dataset_name,
        "train_fraction": fraction,
        "train_samples": len(train_dataset),
        "eval_dataset": dataset_name,
        "eval_samples": len(val_dataset),
        "module": module_label,
        "best_epoch": best_epoch,
        "val_loss": final_loss,
        "val_iou": final_iou,
        "sam_iou": final_sam_iou,
        "checkpoint": str(best_path.resolve()) if best_path.exists() else str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM + ConvNeXt-V1 ROI refine training (atto student)")
    parser.add_argument("--datasets", type=str, default="example_dataset,brain,colon,pdac")
    parser.add_argument("--train-fractions", type=str, default="all")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)

    parser.add_argument("--model-type", type=str, default="vit_b_lm")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_SAM_CHECKPOINT))
    parser.add_argument("--resume", type=str, default=r'C:\localtask\digOrg\student_atto_v1_epoch_90.pt')

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--mask-input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--opt", type=str, choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)
    parser.add_argument("--lr-backbone", type=float, default=None)

    parser.add_argument("--prompt-type", choices=["points", "boxes", "points_boxes", "dense"], default="boxes")
    parser.add_argument("--max-instances", type=int, default=64)
    parser.add_argument("--instance-chunk-size", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--skip-initial-eval", action="store_true", default=False)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=base.DEFAULT_DEVICE)

    parser.add_argument("--use-augmentation", action="store_true", default=True)
    parser.add_argument(
        "--augmentation-config",
        type=str,
        default=(str(base.DEFAULT_AUG_CONFIG) if base.DEFAULT_AUG_CONFIG.exists() else None),
    )
    parser.add_argument("--aug-until-epoch", type=int, default=5)

    parser.add_argument("--student-variant", type=str, choices=STUDENT_VARIANTS, default="atto")
    parser.add_argument("--student-convnext-ckpt", type=str, default=None)
    parser.add_argument(
        "--student-init",
        type=str,
        default=(str(DEFAULT_STUDENT_WEIGHTS) if DEFAULT_STUDENT_WEIGHTS.exists() else None),
    )
    parser.add_argument("--use-prompt", action="store_true", default=True)
    parser.add_argument("--no-prompt", dest="use_prompt", action="store_false")
    parser.add_argument("--build-prompt", action="store_true", default=False)
    parser.add_argument("--fuse-with-sam", action="store_true", default=False)
    parser.add_argument("--fuse-kernel", type=int, choices=[1, 3], default=1)
    parser.add_argument("--crop-source", choices=["sam", "prompt"], default="prompt")
    parser.add_argument("--roi-pad-256", type=float, default=0.10)
    parser.add_argument("--roi-pad", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.5)

    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--clip-grad", action="store_true", default=True)
    parser.add_argument("--no-clip-grad", dest="clip_grad", action="store_false")
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--freeze", type=str, default="backbone,up1,up2,up3,head,decoder,fuse")
    parser.add_argument("--backbone-train-stages", type=str, default="")
    parser.add_argument("--unfreeze-pattern", type=str, default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.student_variant = args.student_variant.lower()
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    for attr in ("checkpoint", "resume", "student_init", "student_convnext_ckpt"):
        value = base._normalize_optional_path(getattr(args, attr, None))
        if value is not None:
            value = str(Path(value).expanduser())
        setattr(args, attr, value)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_names = _parse_dataset_names(args.datasets)
    fractions = _parse_fraction_list(args.train_fractions)

    device = base.setup_device_and_distributed(args)

    summary_rows: List[Dict[str, object]] = []

    for dataset in dataset_names:
        for fraction in fractions:
            row = run_single_experiment(args, device, dataset, fraction)
            summary_rows.append(row)

    if base._is_main_process():
        print("\nSummary:")
        for row in summary_rows:
            print(
                f"{row['train_dataset']} fraction={row['train_fraction']:.2f} module={row['module']} "
                f"best_epoch={row['best_epoch']} val_loss={row['val_loss']:.4f} "
                f"val_iou={row['val_iou']:.4f} sam_iou={row.get('sam_iou', float('nan')):.4f}"
            )

    if args.summary_json and base._is_main_process():
        out_json = Path(args.summary_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    if args.summary_csv and base._is_main_process():
        out_csv = Path(args.summary_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "train_dataset",
            "train_fraction",
            "train_samples",
            "eval_dataset",
            "eval_samples",
            "module",
            "best_epoch",
            "val_loss",
            "val_iou",
            "sam_iou",
            "checkpoint",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    base.cleanup_distributed(args)


if __name__ == "__main__":
    main()
