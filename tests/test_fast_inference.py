import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digitalorg.config import DigitalOrgConfig
from digitalorg.fast_refine import boxes_on_cpu, crop_batch, paste_binary
from digitalorg.refiner import Sam3BoxRefiner


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_legacy_stays_default(self):
        cfg = DigitalOrgConfig()
        self.assertEqual(cfg.refine_backend, "legacy")
        self.assertFalse(cfg.fast_refine_box_only)
        self.assertFalse(cfg.fast_refine_overlap)

    def test_opt_in_presets(self):
        for name in ("exact_1024.yaml", "structural_1024.yaml"):
            cfg = DigitalOrgConfig.from_yaml(ROOT / "configs" / name)
            self.assertTrue(cfg.fast_refine_exact)
            self.assertEqual(cfg.fast_refine_precision, "fp32")
            self.assertFalse(cfg.fast_refine_shared_image_kv)
        self.assertTrue(cfg.fast_refine_box_only and cfg.fast_refine_overlap)

    def test_invalid_configuration(self):
        for values in (dict(refine_backend="unknown"), dict(fast_refine_max_instances=0),
                       dict(fast_refine_prefetch_depth=0), dict(fast_refine_decode_workers=0),
                       dict(fast_refine_box_only=True),
                       dict(refine_backend="lowres_stream", fast_refine_overlap=True),
                       dict(refine_backend="lowres_stream", fast_refine_box_only=True, fast_refine_exact=False)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DigitalOrgConfig(**values)

    def test_exact_rejects_numeric_changes(self):
        for values in (dict(fast_refine_precision="fp16"), dict(fast_refine_fused_roi=True),
                       dict(fast_refine_channels_last=True), dict(fast_refine_shared_image_kv=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DigitalOrgConfig(refine_backend="lowres_stream", **values)

    def test_empty_detections_do_not_load(self):
        refiner = Sam3BoxRefiner(DigitalOrgConfig(refine_backend="lowres_fast"))
        self.assertEqual(refiner.segment("not-opened.png", []), ([], []))
        self.assertIsNone(refiner.model)


class GeometryTests(unittest.TestCase):
    def test_empty_boxes(self):
        big, small = boxes_on_cpu([])
        self.assertEqual(big.shape, (0, 4))
        self.assertEqual(small.shape, (0, 4))

    def test_invalid_boxes(self):
        for boxes in ([[0, 0, 1]], [[0, 0, 0, 1]], [[-1, 0, 1, 1]],
                      [[0, 0, 1025, 1]], [[0, 0, float("nan"), 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                boxes_on_cpu(boxes)
        with self.assertRaises(ValueError):
            boxes_on_cpu([[0, 0, 1, 1]], roi_pad=-1)

    def test_boundary_geometry(self):
        big, small = boxes_on_cpu([[0, 0, 1024, 1024], [1023.2, 1023.1, 1024, 1024]])
        np.testing.assert_array_equal(big[0], [0, 0, 1024, 1024])
        np.testing.assert_array_equal(small[0], [0, 0, 256, 256])
        self.assertTrue(np.all(big[:, 2:] > big[:, :2]))
        self.assertTrue(np.all(small[:, 2:] > small[:, :2]))

    def test_soft_prior_crop_preserves_interpolation(self):
        rgb = torch.arange(3 * 1024**2, dtype=torch.float32).reshape(1, 3, 1024, 1024)
        prob = torch.linspace(0, 1, 256**2).reshape(1, 1, 256, 256)
        big, small = boxes_on_cpu([[100, 200, 250, 330]])
        x, p = crop_batch(rgb, prob, big, small)
        x1, y1, x2, y2 = small[0]
        expected = F.interpolate(prob[..., y1:y2, x1:x2], (256, 256), mode="bilinear", align_corners=False)
        self.assertEqual(x.shape, (1, 3, 256, 256))
        self.assertTrue(torch.equal(p, expected))

    def test_paste_matches_float_canvas(self):
        p = torch.linspace(0, 1, 2 * 256**2).reshape(2, 1, 256, 256)
        boxes = np.array([[0, 0, 83, 111], [900, 912, 1024, 1024]])
        actual = paste_binary(p, boxes)
        expected = torch.zeros((2, 1024, 1024))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            expected[i, y1:y2, x1:x2] = F.interpolate(p[i:i+1], (y2-y1, x2-x1), mode="bilinear", align_corners=False)[0, 0]
        self.assertTrue(torch.equal(actual, (expected > .5).byte() * 255))


if __name__ == "__main__":
    unittest.main()
