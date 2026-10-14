"""Optional fused CUDA ROI operations; requires Triton, not used on CPU."""
import torch
import triton
import triton.language as tl


@triton.jit
def _crop_kernel(X, B, Y, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
                 SHARED: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(1)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < C * 256 * 256
    c, y, x = idx // (256 * 256), (idx // 256) % 256, idx % 256
    x1 = tl.load(B + n * 4)
    y1 = tl.load(B + n * 4 + 1)
    x2 = tl.load(B + n * 4 + 2)
    y2 = tl.load(B + n * 4 + 3)
    sx = tl.maximum(((x2 - x1).to(tl.float32) / 256) * (x.to(tl.float32) + .5) - .5, 0)
    sy = tl.maximum(((y2 - y1).to(tl.float32) / 256) * (y.to(tl.float32) + .5) - .5, 0)
    xl, yl = sx.to(tl.int32), sy.to(tl.int32)
    xh = tl.minimum(xl + 1, x2 - x1 - 1)
    yh = tl.minimum(yl + 1, y2 - y1 - 1)
    wx, wy = sx - xl, sy - yl
    batch = 0 if SHARED else n
    base = (batch * C + c) * H * W
    a = tl.load(X + base + (y1 + yl) * W + x1 + xl, valid, 0)
    b = tl.load(X + base + (y1 + yl) * W + x1 + xh, valid, 0)
    d = tl.load(X + base + (y1 + yh) * W + x1 + xl, valid, 0)
    e = tl.load(X + base + (y1 + yh) * W + x1 + xh, valid, 0)
    val = (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * d + wx * e)
    tl.store(Y + n * C * 256 * 256 + idx, val, valid)


@triton.jit
def _paste_kernel(X, B, Y, THRESHOLD: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(1)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x, y = idx % 1024, idx // 1024
    x1, y1 = tl.load(B + n * 4), tl.load(B + n * 4 + 1)
    x2, y2 = tl.load(B + n * 4 + 2), tl.load(B + n * 4 + 3)
    inside = (idx < 1024 * 1024) & (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
    sx = tl.maximum(tl.div_rn(256., (x2 - x1).to(tl.float32)) * ((x-x1).to(tl.float32) + .5) - .5, 0)
    sy = tl.maximum(tl.div_rn(256., (y2 - y1).to(tl.float32)) * ((y-y1).to(tl.float32) + .5) - .5, 0)
    xl, yl = sx.to(tl.int32), sy.to(tl.int32)
    xh, yh = tl.minimum(xl + 1, 255), tl.minimum(yl + 1, 255)
    wx, wy = sx - xl, sy - yl
    base = n * 256 * 256
    a = tl.load(X + base + yl * 256 + xl, inside, 0)
    b = tl.load(X + base + yl * 256 + xh, inside, 0)
    d = tl.load(X + base + yh * 256 + xl, inside, 0)
    e = tl.load(X + base + yh * 256 + xh, inside, 0)
    val = (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * d + wx * e)
    tl.store(Y + n * 1024 * 1024 + idx, tl.where(inside & (val > THRESHOLD), 255, 0).to(tl.uint8), idx < 1024 * 1024)


def crop_cuda(rgb, prob, boxes, small_boxes):
    n = boxes.shape[0]
    images = torch.empty((n, 3, 256, 256), device=rgb.device, dtype=torch.float32)
    prompts = torch.empty((n, 1, 256, 256), device=prob.device, dtype=torch.float32)
    _crop_kernel[(triton.cdiv(3 * 256 * 256, 256), n)](rgb.contiguous(), boxes, images, 1024, 1024, 3, True, 256, enable_fp_fusion=False)
    _crop_kernel[(triton.cdiv(256 * 256, 256), n)](prob.contiguous(), small_boxes, prompts, 256, 256, 1, False, 256, enable_fp_fusion=False)
    return images, prompts


def paste_cuda(prob, boxes, threshold=.5):
    out = torch.empty((prob.shape[0], 1024, 1024), device=prob.device, dtype=torch.uint8)
    _paste_kernel[(triton.cdiv(1024 * 1024, 256), prob.shape[0])](prob.contiguous(), boxes, out, threshold, 256, enable_fp_fusion=False)
    return out
