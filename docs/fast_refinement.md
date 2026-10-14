# Strict-output inference at 1024 x 1024

Updated 2026-10-14. This is the current latest usage guide. The current requirement
is exact agreement, not a small or statistically negligible accuracy change.
The earlier FP16 throughput presets are experimental historical configurations
and are NOT the recommended deployment path.

## Current preset

The latest structural preset is `configs/structural_1024.yaml`. It selects
`lowres_stream` with `fast_refine_box_only: true` and
`fast_refine_overlap: true`. Set the latter to `false` to retain only the simpler
unused-neck removal when overlap does not benefit the deployment workload.
Both switches default to `false`; `configs/exact_1024.yaml` retains the earlier
streaming path without either structural change.

Use the exact preset with the same checkpoint paths as the working reference:

```yaml
refine_backend: lowres_stream
fast_refine_exact: true
fast_refine_precision: fp32
fast_refine_graphs: true
fast_refine_fused_roi: false
fast_refine_channels_last: false
fast_refine_shared_image_kv: false
fast_refine_max_instances: 100
fast_refine_prefetch_depth: 2
fast_refine_decode_workers: 2
save_sam_masks: false
```

A complete environment-variable-based template is provided in
`configs/exact_1024.yaml`. Resolve the model path variables before running.
The existing default configuration remains `legacy`; it is not silently
switched to a different segmentation algorithm.

The config defaults to strict numerical options for either optional fast
backend. It rejects FP16, fused ROI interpolation, channels-last and shared SAM
projection batching when exact mode is enabled. Direct engine construction
must also pass `exact=True`, with `shared_image_kv=False` for streaming.

The mode retains original PyTorch interpolation, tensor precision and layout,
model weights, ROI dimensions, decision thresholds and per-image target batches.
It does not change TF32 or cuDNN policy; these must match the established baseline
in the running process. The benchmark uses the same pre-existing TF32-enabled
policy in reference and candidate, not TF32 enabled only in the optimized path.

## Low-resolution priors and structural scheduling

The current SAM3 processor retains its internal 1008-square image transform.
The interactive decoder produces 288-square logits. Their sigmoid probabilities
are resized to a 256-square soft-prior map; box coordinates are mapped into that
map and the corresponding ROI is cropped and resized to 256-square. RGB ROIs
are independently cropped from the original 1024-square image. Refine restores
each output to its ROI extent before final full-image uint8 mask assembly.
The pipeline does not first build 1024-square floating-point SAM masks for
Refine. This is soft-mask prior cropping, not ROIAlign on SAM encoder features.

The box-only adapter computes the shared image trunk and three required
interactive neck scales. It skips the semantic neck and discarded fourth
interactive scale, without changing retained operators. This does not remove
the separate DigitalOrg detector, and is not valid for SAM3 text-prompt or video
APIs. Skipped weights remain loaded; this is not parameter compression.

Optional overlap runs ConvNeXt and the first two Refine decoder blocks while
the SAM mask decoder runs. The prompt encoder and final refinement stages wait
for the soft prior. CUDA graphs cache execution plans, not image-dependent
predictions. No cross-frame KV reuse or shared projection re-batching is used
in the exact presets. Setting `fast_refine_graphs: false` uses the same eager
operators, including in the split schedule.

Structural graph caches retain one ROI-count configuration per branch. New
counts require recapture, which can reduce throughput for irregular workloads.
Compare overlap on and off with the actual target-count distribution.

## What remains optimized

- Resident strictly loaded model weights.
- CPU box preparation without per-box GPU scalar synchronization.
- A prompt-encoder specialization for already validated probability inputs.
- CUDA graph replay of the same FP32 Refine operations.
- Direct uint8 final-mask reconstruction using the original interpolation.
- One decoded GPU image shared by SAM and Refine.
- Bounded CPU prefetch and pinned-memory transfers with completion events.
- Independent returned arrays and ordered output, without frame skipping.

Shared first-layer KV projections are disabled even though earlier samples
matched exactly: projection re-batching changes matrix-operation shapes. The
strict path does not rely on a claim that mathematically equivalent operations
must always be bitwise equivalent across devices and library versions.

More than `fast_refine_max_instances` raises an error in exact mode instead of
silently splitting the batch. If a larger reference batch is required, validate
that same batch size and configure its actual size explicitly. Do not truncate
or skip targets to meet the throughput target.

## Continuous inference

```python
from digitalorg.refiner import Sam3BoxRefiner

refiner = Sam3BoxRefiner(cfg)
# Each entry contains a 1024-square image path and existing Detection objects.
outputs = refiner.segment_many(zip(image_paths, detections_per_image))
try:
    for coarse_masks, refined_masks in outputs:
        consume_masks(refined_masks)
finally:
    outputs.close()
```

Every new image receives fresh SAM image encoding. Decode, computation and
completed CPU mask export are overlapped; model execution is serialized on an
owned compute stream. The queue is bounded. Ordinary single-image calls cannot
hide I/O behind later images and should not be assigned batch-throughput timings.
The detector, output-file writing and overlays are outside this iterator.

`save_sam_masks: true` falls back to sequential wrapper calls returning both
coarse and refined masks. Results are full 1024-square uint8 arrays, zero or 255.
Do not concurrently mutate or train the shared model. Close an iterator when
stopping early. Holding every returned mask increases host memory consumption.

## Same-image feature reuse

```python
from digitalorg.stream_refine import StreamingSam3Refiner

engine = StreamingSam3Refiner(model, processor, refine,
                             exact=True, shared_image_kv=False)
frame = engine.encode_image(image_path)
masks = engine.segment_encoded(frame, boxes_xyxy)
```

Only reuse this explicit image handle for the identical image and unchanged
model. It is not a cache for a video sequence. New frames require new encoding.
Do not modify handle tensors; a handle from another engine is rejected.

## Acceptance criterion and scope

A speed change is accepted only when the tested original FP32 probability tensors
and final masks are exactly equal. Mean Dice similarity, high IoU or a small
number of altered pixels is not an acceptance criterion. The regression script
aborts immediately on any difference; it does not round values or use tolerance.

The reference is the original low-resolution SAM3 soft-prior path used in the
local throughput experiment. It is NOT the released legacy full-mask wrapper,
which uses different prompt handling and postprocessing, nor the manuscript's
SAM2 backend. A different reference requires a separate equivalence check.

Tests are finite: matching on tested images and this GPU is not a universal
proof for new hardware, weights or PyTorch versions. Re-run the gate on the
intended full evaluation set before replacing any published evaluation pipeline.
When exactness is uncertain, retain the original reference rather than accept an
approximation. No manuscript outputs or checkpoints are modified by these tests.

### Reproducible regression check

Use a private JSON manifest containing 1024-square image paths and XYXY boxes.
Relative image paths are resolved against the manifest's directory:

```json
[
  {"image_path": "images/sample.png", "boxes": [[10, 20, 90, 110]]}
]
```

Run from the repository root after setting the checkpoint environment variables:

```bash
python -m unittest discover -s tests -v
python scripts/validate_fast_refinement.py --config configs/structural_1024.yaml --cases private_cases.json --output outputs/strict_validation.json --tf32 on
```

The validator compares probability tensors and masks against eager refinement
using the original ROI helpers and floating-point canvas, then checks ordered
streaming outputs. It exits on any difference. Select the same TF32 policy as
your reference. Private weights, images and local benchmark artifacts are not
part of the repository.

## Development benchmark

The structural prototype was validated on 56 configurations: 48 source-image
cases and eight controlled-density cases, totaling 2,926 instance checks
(including repeated objects in density subsets). Probability tensors and final
mask pixels matched the original low-resolution reference exactly. This is
finite same-device evidence, not a guarantee on all inputs or environments.

RTX 5090, PyTorch 2.11.0+cu128, 1024-square inputs. Median of three 12-image
batches; loading and cold graph capture excluded. Image decoding, SAM encoding,
box decoding, Refine and completed CPU masks included. Detector and file output
excluded. These development measurements are not timings of the single-image
CLI including output files.

| Targets | Original SAM3 (ms) | Optimized SAM3 alone (ms) | Optimized SAM3 + Refine (ms) |
|---:|---:|---:|---:|
| 1 | 116.84 | 96.99 | 97.38 |
| 10 | 121.64 | 99.47 | 104.36 |
| 50 | 148.95 | 126.74 | 140.12 |
| 100 | 181.61 | 154.23 | 184.69 |

At 100 targets, Refine still adds 19.75% latency compared with equally optimized
SAM3 alone. Do not claim parity with that baseline. Overlap gains are density-
and environment-dependent; a separate preliminary high-density run did not
outperform neck removal alone.

RefineNet has 7,638,122 parameters and 4.606755 GFLOPs per 256-square ROI.
The loaded SAM3 object has 860,055,224 parameters; adding Refine totals
867,693,346. Skipping neck branches saves 217.092981 GFLOPs per image but does
not remove their resident weights. Dense-operator forward counts use 2 FLOPs
per multiply-accumulate and include attention QK/AV products, but exclude bias,
normalization, activation, softmax, positional elementwise arithmetic,
interpolation and I/O. At 100 targets the original SAM3 path is 6,291.601
GFLOPs, equally optimized SAM3 is 6,074.508 GFLOPs, and optimized SAM3 + Refine
is 6,535.184 GFLOPs. These counts exclude the separate detector.
