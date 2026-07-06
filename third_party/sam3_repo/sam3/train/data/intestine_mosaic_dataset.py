import random
from typing import List, Tuple

import numpy as np
from PIL import Image as PILImage

from sam3.train.data.sam3_image_dataset import Sam3ImageDataset, Datapoint, Image, Object
from sam3.model.box_ops import box_xywh_to_xyxy


def _build_mosaic_from_datapoints(
    datapoints: List[Datapoint],
    mosaic_size: int = 1008,
) -> Datapoint:
    """
    Build a 4-image mosaic Datapoint from 4 Sam3ImageDataset Datapoints.

    Assumptions:
      - Each input Datapoint has exactly one Image in `images`.
      - Bboxes are in absolute XYXY coordinates in `obj.bbox` and match the current
        PIL image size.
      - We only support merging annotations from images[0] in each datapoint.
    """
    assert len(datapoints) == 4, "Need exactly 4 datapoints for mosaic"
    S = mosaic_size

    # Random mosaic center
    min_off = int(S * 0.3)
    max_off = int(S * 0.7)
    cx = random.randint(min_off, max_off)
    cy = random.randint(min_off, max_off)

    # Create output canvas
    mosaic_img = PILImage.new("RGB", (S, S), (114, 114, 114))
    merged_objects: List[Object] = []

    for idx, dp in enumerate(datapoints):
        assert len(dp.images) == 1, "ExampleDataset mosaic expects single-image datapoints"
        img = dp.images[0].data
        assert isinstance(img, PILImage.Image), "Mosaic expects PIL images"
        W, H = img.size
        if W <= 0 or H <= 0:
            continue

        # scale + letterbox to SxS
        scale = S / max(W, H)
        new_w, new_h = int(W * scale), int(H * scale)
        img_resized = img.resize((new_w, new_h), PILImage.BILINEAR)

        canvas = PILImage.new("RGB", (S, S), (114, 114, 114))
        pad_x = (S - new_w) // 2
        pad_y = (S - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        # scale + translate bboxes from this image
        scaled_boxes = []
        for obj in dp.images[0].objects:
            b = obj.bbox.view(1, 4).clone().float().numpy()
            x1, y1, x2, y2 = b[0]
            x1 = x1 * scale + pad_x
            y1 = y1 * scale + pad_y
            x2 = x2 * scale + pad_x
            y2 = y2 * scale + pad_y
            scaled_boxes.append((x1, y1, x2, y2, obj))

        # choose crop region for this quadrant
        if idx == 0:  # top-left
            crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, cx, cy
            dest_x0, dest_y0 = 0, 0
        elif idx == 1:  # top-right
            crop_x0, crop_y0, crop_x1, crop_y1 = S - cx, 0, S, cy
            dest_x0, dest_y0 = cx, 0
        elif idx == 2:  # bottom-left
            crop_x0, crop_y0, crop_x1, crop_y1 = 0, S - cy, cx, S
            dest_x0, dest_y0 = 0, cy
        else:  # idx == 3, bottom-right
            crop_x0, crop_y0, crop_x1, crop_y1 = S - cx, S - cy, S, S
            dest_x0, dest_y0 = cx, cy

        # crop and paste image patch
        patch = canvas.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        mosaic_img.paste(patch, (dest_x0, dest_y0))

        # crop + shift boxes into mosaic coordinates
        for (x1, y1, x2, y2, obj) in scaled_boxes:
            x1_c = np.clip(x1, crop_x0, crop_x1)
            y1_c = np.clip(y1, crop_y0, crop_y1)
            x2_c = np.clip(x2, crop_x0, crop_x1)
            y2_c = np.clip(y2, crop_y0, crop_y1)
            if x2_c <= x1_c or y2_c <= y1_c:
                continue
            # shift into mosaic coordinates
            x1_m = x1_c - crop_x0 + dest_x0
            x2_m = x2_c - crop_x0 + dest_x0
            y1_m = y1_c - crop_y0 + dest_y0
            y2_m = y2_c - crop_y0 + dest_y0
            new_bbox = np.array([x1_m, y1_m, x2_m, y2_m], dtype=np.float32)

            merged_objects.append(
                Object(
                    bbox=torch.from_numpy(new_bbox),
                    area=float((x2_m - x1_m) * (y2_m - y1_m)),
                    object_id=obj.object_id,
                    frame_index=obj.frame_index,
                    segment=None,
                    is_crowd=obj.is_crowd,
                    source=obj.source,
                )
            )

    # Build a new Datapoint
    mosaic_image = Image(
        data=mosaic_img,
        objects=merged_objects,
        size=(S, S),
    )
    # We reuse the find_queries from the first datapoint (text prompt),
    # since ExampleDataset is text-only detector with same prompt.
    dp0 = datapoints[0]
    return Datapoint(find_queries=dp0.find_queries, images=[mosaic_image], raw_images=None)


class ExampleDatasetMosaicImageDataset(Sam3ImageDataset):
    """
    Dataset wrapper that introduces YOLO-style 4-image mosaic augmentation
    for the ExampleDataset text-only detector.

    It randomly replaces a datapoint with a mosaic of 4 datapoints
    with probability `mosaic_prob`.
    """

    def __init__(self, *args, mosaic_prob: float = 0.5, mosaic_size: int = 1008, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic_prob = mosaic_prob
        self.mosaic_size = mosaic_size

    def __getitem__(self, idx):
        # fall back to original behavior
        if not self.training or random.random() > self.mosaic_prob:
            return super().__getitem__(idx)

        # sample 4 indices (including current) and build mosaic
        idxs = [idx]
        if len(self.ids) >= 4:
            # ensure distinct indices when possible
            candidates = list(range(len(self.ids)))
            random.shuffle(candidates)
            for i in candidates:
                if i == idx:
                    continue
                idxs.append(i)
                if len(idxs) == 4:
                    break
        while len(idxs) < 4:
            idxs.append(random.randint(0, len(self.ids) - 1))

        dps = [super(Sam3ImageDataset, self).__getitem__(i) for i in idxs]
        mosaic_dp = _build_mosaic_from_datapoints(dps, mosaic_size=self.mosaic_size)

        # Apply the same transforms chain that Sam3ImageDataset would apply
        for transform in self._transforms:
            mosaic_dp = transform(mosaic_dp, epoch=self.curr_epoch)

        return mosaic_dp

