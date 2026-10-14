"""Opt-in box-only SAM3 neck and exact Refine dependency scheduling.

This adapter is restricted to 1024-square image-to-mask inference. It is not
an alternative processor for SAM3 text prompts or video tracking.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from .fast_refine import paste_binary
from .stream_refine import StreamingSam3Refiner


class BoxOnlyProcessor:
    """Compute only the interactive neck, preserving all retained operators.

    This adapter deliberately exposes no text-prompt API. It cannot substitute
    for Sam3Processor in SAM3 semantic detection or video tracking.
    """
    def __init__(self, processor):
        self.model = processor.model
        self.device = processor.device
        self.transform = processor.transform
        self.resolution = processor.resolution
        backbone = self.model.backbone
        neck = backbone.vision_backbone
        if self.model.training or neck.sam2_convs is None:
            raise ValueError("Requires an eval-mode interactive SAM3 model")
        self.retained_scales = len(neck.sam2_convs) - backbone.scalp
        if self.retained_scales != 3:
            raise ValueError("This experiment validates exactly three retained scales")

    @torch.inference_mode()
    def set_image(self, image):
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            raise TypeError("Expected a PIL image or CHW tensor")
        if (width, height) != (1024, 1024):
            raise ValueError("This experiment is restricted to 1024-square inputs")
        image = self.transform(v2.functional.to_image(image).to(self.device)).unsqueeze(0)
        neck = self.model.backbone.vision_backbone
        x = neck.trunk(image)[-1]
        features, positions = [], []
        for conv in neck.sam2_convs[:self.retained_scales]:
            feature = conv(x)
            features.append(feature)
            positions.append(neck.position_encoding(feature).to(feature.dtype))
        interactive = dict(vision_features=features[-1], vision_pos_enc=positions, backbone_fpn=features)
        decoder = self.model.inst_interactive_predictor.model.sam_mask_decoder
        features[0] = decoder.conv_s0(features[0])
        features[1] = decoder.conv_s1(features[1])
        return dict(original_height=height, original_width=width,
                    backbone_out=dict(sam2_backbone_out=interactive))


class SplitGraphRunner:
    def __init__(self, model, max_graphs=1, use_graphs=True):
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.model = model
        self.max_graphs = max_graphs
        self.use_graphs = use_graphs
        self.states = OrderedDict()

    def image_branch(self, rgb):
        m = self.model
        x0, x1, x2, x3 = m.backbone((rgb / 255.0 - m.imnet_mean) / m.imnet_std)
        return x0, m.up2(m.up3(x3, x2), x1)

    def finish_branch(self, rgb, prompt, x0, y):
        m = self.model
        x0 = x0 + m.prompt_encoder(prompt, rgb)
        logits = F.interpolate(m.head64(m.up1(y, x0)), (256, 256),
                               mode="bilinear", align_corners=False)
        return logits.float().sigmoid()

    def begin(self, rgb):
        if not self.use_graphs:
            return (rgb, *self.image_branch(rgb))
        key = (len(rgb), rgb.dtype, rgb.device)
        if key not in self.states:
            x = torch.empty_like(rgb)
            x.copy_(rgb)
            p = torch.full((len(rgb), 1, 256, 256), 0.5, device=rgb.device)
            warm = torch.cuda.Stream(device=rgb.device)
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    a, b = self.image_branch(x)
                    self.finish_branch(x, p, a, b)
            torch.cuda.current_stream().wait_stream(warm)
            image_graph, finish_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(image_graph):
                a, b = self.image_branch(x)
            with torch.cuda.graph(finish_graph):
                out = self.finish_branch(x, p, a, b)
            self.states[key] = (x, p, image_graph, finish_graph, out)
            while len(self.states) > self.max_graphs:
                self.states.popitem(last=False)
        self.states.move_to_end(key)
        state = self.states[key]
        state[0].copy_(rgb)
        state[2].replay()
        return state

    def finish(self, state, prompt):
        if not self.use_graphs:
            return self.finish_branch(state[0], prompt, *state[1:])
        state[1].copy_(prompt)
        state[3].replay()
        return state[4]


class StructuralSam3Refiner(StreamingSam3Refiner):
    """Skip unused neck branches; optionally overlap mask-independent work."""
    def __init__(self, *args, overlap=False, **kwargs):
        kwargs.setdefault("exact", True)
        kwargs.setdefault("shared_image_kv", False)
        if not kwargs["exact"]:
            raise ValueError("The structural backend requires exact mode")
        super().__init__(*args, **kwargs)
        if not isinstance(self.processor, BoxOnlyProcessor):
            self.processor = BoxOnlyProcessor(self.processor)
        self.overlap = overlap
        self.image_branch_stream = torch.cuda.Stream(device=self.device)
        self.image_branch_stream.wait_stream(torch.cuda.current_stream())
        self.split = SplitGraphRunner(self.runner.model, use_graphs=self.runner.use_graphs)
        self.runner.max_graphs = 1

    def _masks_gpu(self, frame, boxes, big, small, sam_only=False, return_coarse=False):
        if not self.overlap or sam_only or return_coarse:
            return super()._masks_gpu(frame, boxes, big, small, sam_only, return_coarse)
        if frame.owner is not self._owner:
            raise ValueError("Foreign frame")
        if len(boxes) > self.max_instances:
            raise ValueError("No implicit re-batching in structural experiment")
        self.compute_stream.wait_event(frame.ready)
        if not len(boxes):
            empty = torch.empty((0, 1024, 1024), dtype=torch.uint8, device=self.device)
            return empty, empty

        def start_image_branch():
            self.image_branch_stream.wait_stream(self.compute_stream)
            with torch.cuda.stream(self.image_branch_stream):
                image_rois = torch.cat([F.interpolate(frame.rgb[..., y1:y2, x1:x2],
                    (256, 256), mode="bilinear", align_corners=False) for x1, y1, x2, y2 in big])
                state = self.split.begin(image_rois)
                ready = torch.cuda.Event()
                ready.record()
            return state, ready

        state, ready = start_image_branch()
        try:
            logits = self._decode_boxes(frame, boxes)
        except Exception:
            # Join even on an exception so temporary frame storage stays alive.
            self.compute_stream.wait_event(ready)
            raise
        probs = F.interpolate(logits.sigmoid(), (256, 256), mode="bilinear", align_corners=False)
        prompts = torch.cat([F.interpolate(probs[i:i+1, :, y1:y2, x1:x2],
            (256, 256), mode="bilinear", align_corners=False) for i, (x1, y1, x2, y2) in enumerate(small)])
        self.compute_stream.wait_event(ready)
        refined = self.split.finish(state, prompts)
        masks = paste_binary(refined, big, self.threshold)
        return masks, torch.empty((0, 1024, 1024), dtype=torch.uint8, device=self.device)
