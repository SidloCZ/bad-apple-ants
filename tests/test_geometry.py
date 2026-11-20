"""Verify GPU grid sampling matches Pillow affine rendering across rotations and mirroring."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from badapple_ants.geometry import build_torch_grid, max_fit_crop_w, pose_in_bounds, render_crop


def _make_synthetic_image(w: int = 192, h: int = 140) -> Image.Image:
    """Create an asymmetric smooth-edged test image with an F shape and corner markers."""
    img = Image.new("L", (w, h), color=28)
    draw = ImageDraw.Draw(img)
    # Asymmetric F shape
    draw.rectangle((48, 28, 68, 112), fill=230)
    draw.rectangle((68, 28, 128, 46), fill=230)
    draw.rectangle((68, 62, 112, 78), fill=210)
    # Distinct corner markers so 90/180/mirror mistakes show up immediately
    draw.ellipse((14, 14, 34, 34), fill=160)
    draw.polygon([(150, 100), (178, 100), (178, 128)], fill=245)
    # Subtle gradient so bilinear/bicubic shifts are meaningful everywhere
    arr = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.clip(arr + 0.15 * xx + 0.12 * yy, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("deg", [0.0, 22.5, 112.5, 247.5])
def test_torch_vs_pillow_alignment(deg: float, mirrored: bool) -> None:
    canvas_size = 192
    w, h = 192, 140
    pil_img = _make_synthetic_image(w, h)
    arr = np.asarray(pil_img, dtype=np.uint8)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[:h, :w] = arr.astype(np.float32)
    tensor_in = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

    theta = math.radians(deg)
    cw = max_fit_crop_w(w, h, theta, margin=0.5) * 0.72
    cx_frac = 0.47
    cy_frac = 0.53
    w_frac = cw / float(w)
    assert pose_in_bounds(w, h, cx_frac, cy_frac, w_frac, theta, margin=0.5)

    poses_t = torch.tensor(
        [[cx_frac, cy_frac, w_frac, theta, 1.0 if mirrored else 0.0]],
        dtype=torch.float32,
    )
    grid = build_torch_grid(poses_t, float(w), float(h), canvas_size, grid_w=64, grid_h=48)
    sampled = F.grid_sample(tensor_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    torch_32 = F.avg_pool2d(sampled, kernel_size=2, stride=2).squeeze().numpy()

    pil_crop = render_crop(pil_img, 32, 24, cx_frac, cy_frac, w_frac, theta, mirrored)
    pil_32 = np.asarray(pil_crop, dtype=np.float32)

    t_norm = (torch_32 - torch_32.mean()) / (np.linalg.norm(torch_32 - torch_32.mean()) + 1e-6)
    p_norm = (pil_32 - pil_32.mean()) / (np.linalg.norm(pil_32 - pil_32.mean()) + 1e-6)
    ncc = float((t_norm * p_norm).sum())
    mad = float(np.abs(torch_32 - pil_32).mean())

    assert ncc > 0.99, f"deg={deg}, mirrored={mirrored}: NCC={ncc}"
    assert mad < 4.0, f"deg={deg}, mirrored={mirrored}: MAD={mad}"


def test_pose_in_bounds() -> None:
    for w, h in ((192, 144), (192, 192), (108, 192)):
        for i in range(16):
            theta = i * (2.0 * math.pi / 16.0)
            cw_max = max_fit_crop_w(w, h, theta, margin=0.5)
            assert cw_max > 40.0
            assert pose_in_bounds(w, h, 0.5, 0.5, (cw_max * 0.99) / w, theta, margin=0.5)
            assert not pose_in_bounds(w, h, 0.5, 0.5, (cw_max * 1.05) / w, theta, margin=0.5)


def test_coarse_mirror_identity() -> None:
    """Coarse unmirrored patch vs flipped target equals mirrored patch vs unflipped target."""
    w, h, canvas_size = 192, 140, 192
    pil_img = _make_synthetic_image(w, h)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[:h, :w] = np.asarray(pil_img, dtype=np.float32)
    tensor_in = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

    theta = math.radians(67.5)
    w_frac = (max_fit_crop_w(w, h, theta, margin=0.5) * 0.65) / float(w)
    pose_unmir = torch.tensor([[0.46, 0.52, w_frac, theta, 0.0]], dtype=torch.float32)
    pose_mir = torch.tensor([[0.46, 0.52, w_frac, theta, 1.0]], dtype=torch.float32)

    g_unmir = build_torch_grid(pose_unmir, float(w), float(h), canvas_size, 32, 24)
    g_mir = build_torch_grid(pose_mir, float(w), float(h), canvas_size, 32, 24)
    p_unmir = F.grid_sample(tensor_in, g_unmir, mode="bilinear", align_corners=False).squeeze()
    p_mir = F.grid_sample(tensor_in, g_mir, mode="bilinear", align_corners=False).squeeze()

    rng = np.random.default_rng(42)
    target = torch.from_numpy(rng.normal(size=(24, 32)).astype(np.float32))
    target_flip = torch.flip(target, dims=[-1])

    dot_coarse = float((p_unmir * target_flip).sum())
    dot_refined = float((p_mir * target).sum())
    assert abs(dot_coarse - dot_refined) < 1e-2
