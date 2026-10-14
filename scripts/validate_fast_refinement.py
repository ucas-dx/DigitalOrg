"""Fail-closed comparison against eager low-resolution soft-prior refinement.

Supply private weights through YAML/environment variables and a JSON list of
{"image_path": "...", "boxes": [[x1, y1, x2, y2], ...]} records. Nothing is
uploaded. This is not a comparison to legacy full-resolution mask processing.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from digitalorg.config import DigitalOrgConfig
from digitalorg.fast_refine import boxes_on_cpu, raw_sam_logits
from digitalorg.refiner import Sam3BoxRefiner
from digitalorg.schemas import Detection


def checksum(masks):
    digest = hashlib.sha256()
    for mask in masks:
        digest.update(mask.tobytes())
    return digest.hexdigest()


class Capture:
    def __init__(self, target):
        self.target = target
        self.probability = None

    def __getattr__(self, key):
        return getattr(self.target, key)

    def __call__(self, *args):
        result = self.target(*args)
        self.probability = result.clone()
        return result

    def finish(self, *args):
        result = self.target.finish(*args)
        self.probability = result.clone()
        return result


def detections(case):
    return [Detection(bbox_xyxy=b, score=1., prompt="organoid") for b in case['boxes']]


@torch.no_grad()
def validate(cfg, cases):
    if cfg.refine_backend == 'legacy' or not cfg.fast_refine_exact or cfg.save_sam_masks:
        raise ValueError('Use an exact fast preset with save_sam_masks: false')
    wrapper = Sam3BoxRefiner(cfg)
    wrapper.load()
    engine = wrapper._fast
    from sam3.refine import util
    reference = util.ROIRefineNet(variant='atto', use_prompt=True, build_prompt=False)
    checkpoint = torch.load(cfg.refine_checkpoint, map_location='cpu', weights_only=True)
    weights = checkpoint.get('refine', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
    reference.load_state_dict(weights, strict=True)
    reference.to(cfg.device).eval()
    if getattr(engine, 'overlap', False):
        capture = Capture(engine.split)
        engine.split = capture
    else:
        capture = Capture(engine.runner)
        engine.runner = capture
    results = []
    for index, case in enumerate(cases):
        with Image.open(case['image_path']) as src:
            image = src.convert('RGB')
        if image.size != (1024, 1024):
            raise ValueError('Images must be 1024-square; boxes must use the same coordinates')
        if not case['boxes'] or len(case['boxes']) > cfg.fast_refine_max_instances:
            raise ValueError('Validation cases need 1 to max_instances boxes')
        with torch.autocast('cuda', enabled=False):
            state = wrapper.processor.set_image(image)
            logits = raw_sam_logits(wrapper.model, state, case['boxes'])
            prob = F.interpolate(logits.sigmoid(), (256, 256), mode='bilinear', align_corners=False)
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(cfg.device).float()
            big, small = boxes_on_cpu(case['boxes'], cfg.roi_pad)
            bt, st = torch.as_tensor(big, device=cfg.device), torch.as_tensor(small, device=cfg.device)
            x = util._batch_crop_resize_img_bchw_1024(rgb, bt, out_hw=256)
            p = util._batch_crop_resize_mask_256(prob, st, out_hw=256)
            expected_prob = reference(x, prompt_mask_bchw_256=p).sigmoid()
            full = torch.cat([util._paste_roi_prob_back_1024(expected_prob[i:i+1], b) for i, b in enumerate(bt)])
            expected = ((full[:, 0] > cfg.mask_threshold).byte() * 255).cpu().numpy()
            _, actual = wrapper.segment(case['image_path'], detections(case))
        torch.cuda.synchronize()
        equal_prob = torch.equal(expected_prob, capture.probability)
        changed = sum(int(np.count_nonzero(a != b)) for a, b in zip(expected, actual))
        if len(expected) != len(actual) or not equal_prob or changed:
            raise AssertionError(f'Exact-output regression failed at case {index}')
        results.append(dict(case_index=index, instances=len(actual), probability_equal=equal_prob,
                            changed_mask_pixels=changed, sha256=checksum(actual)))
        print(f'EXACT {index + 1}/{len(cases)}', flush=True)
        capture.probability = None
        del expected_prob, expected, actual, full, rgb, x, p, prob, logits, state
    # Restore normal execution so the ordered stream check does not retain tensors.
    if getattr(engine, 'overlap', False):
        engine.split = capture.target
    else:
        engine.runner = capture.target
    count = 0
    for _, masks in wrapper.segment_many((c['image_path'], detections(c)) for c in cases):
        if checksum(masks) != results[count]['sha256']:
            raise AssertionError(f'Ordered-stream regression failed at case {count}')
        count += 1
    if count != len(cases):
        raise AssertionError('Missing outputs')
    return dict(status='passed', cases=len(cases), instances=sum(r['instances'] for r in results),
                ordered_stream_cases=count, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', default='outputs/strict_validation.json')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--tf32', choices=('on', 'off'), default='on',
                        help='Use the same policy as the reference, on for the reported benchmark')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == 'on'
    torch.backends.cudnn.allow_tf32 = args.tf32 == 'on'
    torch.backends.cudnn.benchmark = False
    cases_path = Path(args.cases).resolve()
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    if not cases:
        raise ValueError('At least one case is required')
    for case in cases:
        image_path = Path(case['image_path'])
        if not image_path.is_absolute():
            case['image_path'] = str(cases_path.parent / image_path)
    report = validate(DigitalOrgConfig.from_yaml(args.config), cases)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS: {report["cases"]} cases, {report["instances"]} instance checks', flush=True)


if __name__ == '__main__':
    main()
