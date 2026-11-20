"""Shared crop coordinate math for GPU sampling and Pillow rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass(frozen=True)
class Pose:
    """Normalized crop pose independent of image resolution."""

    image_idx: int
    cx_frac: float
    cy_frac: float
    w_frac: float
    theta: float
    mirrored: bool


def max_fit_crop_w(img_w: float, img_h: float, theta: float, margin: float = 0.0) -> float:
    """Return the maximum 4:3 crop width centered on the image that fits in bounds."""
    cos_a = abs(math.cos(theta))
    sin_a = abs(math.sin(theta))
    avail_w = max(0.0, float(img_w) - 2.0 * margin)
    avail_h = max(0.0, float(img_h) - 2.0 * margin)
    denom_x = cos_a + 0.75 * sin_a
    denom_y = sin_a + 0.75 * cos_a
    return min(avail_w / denom_x, avail_h / denom_y)


def crop_half_extents(cw: float, theta: float) -> tuple[float, float]:
    """Axis-aligned half-extents (hx, hy) of a rotated 4:3 box of width cw."""
    cos_a = abs(math.cos(theta))
    sin_a = abs(math.sin(theta))
    ch = 0.75 * cw
    hx = 0.5 * cw * cos_a + 0.5 * ch * sin_a
    hy = 0.5 * cw * sin_a + 0.5 * ch * cos_a
    return hx, hy


def pose_in_bounds(
    img_w: float,
    img_h: float,
    cx_frac: float,
    cy_frac: float,
    w_frac: float,
    theta: float,
    margin: float = 0.5,
) -> bool:
    """Check that all four rotated corners lie inside [margin, W - margin] x [margin, H - margin]."""
    cw = w_frac * img_w
    if cw <= 0.0:
        return False
    px = cx_frac * img_w
    py = cy_frac * img_h
    hx, hy = crop_half_extents(cw, theta)
    eps = 1e-4
    return (
        px - hx >= margin - eps
        and px + hx <= img_w - margin + eps
        and py - hy >= margin - eps
        and py + hy <= img_h - margin + eps
    )


def clamp_pose(
    img_w: float,
    img_h: float,
    cx_frac: float,
    cy_frac: float,
    w_frac: float,
    theta: float,
    min_w_px: float = 44.0,
    min_fit_ratio: float = 0.50,
    margin: float = 0.5,
) -> tuple[float, float, float, float]:
    """Clamp pose parameters to valid bounds and minimum ant-first crop scale."""
    cw_max = max_fit_crop_w(img_w, img_h, theta, margin=margin)
    if cw_max <= 0.0:
        return 0.5, 0.5, 0.0, theta
    cw_min = min(cw_max, max(min_w_px, min_fit_ratio * cw_max))
    cw = max(cw_min, min(cw_max, w_frac * img_w))
    hx, hy = crop_half_extents(cw, theta)
    px_lo = margin + hx
    px_hi = img_w - margin - hx
    py_lo = margin + hy
    py_hi = img_h - margin - hy
    if px_hi < px_lo:
        px = 0.5 * img_w
    else:
        px = max(px_lo, min(px_hi, cx_frac * img_w))
    if py_hi < py_lo:
        py = 0.5 * img_h
    else:
        py = max(py_lo, min(py_hi, cy_frac * img_h))
    return px / img_w, py / img_h, cw / img_w, theta


def build_torch_grid(
    poses_tensor: torch.Tensor,
    img_w: float | torch.Tensor,
    img_h: float | torch.Tensor,
    canvas_size: int,
    grid_w: int,
    grid_h: int,
) -> torch.Tensor:
    """Build (B, grid_h, grid_w, 2) grid for F.grid_sample(..., align_corners=False).

    poses_tensor columns: [cx_frac, cy_frac, w_frac, theta, mirrored_flag]
    where mirrored_flag is 0.0 or 1.0.
    """
    device = poses_tensor.device
    w = torch.as_tensor(img_w, dtype=torch.float32, device=device).view(-1, 1, 1)
    h = torch.as_tensor(img_h, dtype=torch.float32, device=device).view(-1, 1, 1)

    px = poses_tensor[:, 0].view(-1, 1, 1) * w
    py = poses_tensor[:, 1].view(-1, 1, 1) * h
    cw = poses_tensor[:, 2].view(-1, 1, 1) * w
    ch = 0.75 * cw
    theta = poses_tensor[:, 3].view(-1, 1, 1)
    sgn = 1.0 - 2.0 * poses_tensor[:, 4].view(-1, 1, 1)

    u = ((torch.arange(grid_w, dtype=torch.float32, device=device) + 0.5) / grid_w - 0.5).view(
        1, 1, grid_w
    )
    v = ((torch.arange(grid_h, dtype=torch.float32, device=device) + 0.5) / grid_h - 0.5).view(
        1, grid_h, 1
    )

    du = sgn * cw * u
    dv = ch * v
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    src_x = px + du * cos_t - dv * sin_t
    src_y = py + du * sin_t + dv * cos_t

    inv_canvas = 2.0 / float(canvas_size)
    x_norm = src_x * inv_canvas - 1.0
    y_norm = src_y * inv_canvas - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def pillow_affine_coeffs(
    img_w: float,
    img_h: float,
    out_w: int,
    out_h: int,
    cx_frac: float,
    cy_frac: float,
    w_frac: float,
    theta: float,
    mirrored: bool,
) -> tuple[float, float, float, float, float, float]:
    """Return exact 6-tuple (a, b, c, d, e, f) for Pillow Image.transform(AFFINE)."""
    px = cx_frac * img_w
    py = cy_frac * img_h
    cw = w_frac * img_w
    ch = 0.75 * cw
    sgn = -1.0 if mirrored else 1.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    sx = sgn * cw / float(out_w)
    sy = ch / float(out_h)

    a = sx * cos_t
    b = -sy * sin_t
    c = px - 0.5 * sgn * cw * cos_t + 0.5 * ch * sin_t

    d = sx * sin_t
    e = sy * cos_t
    f = py - 0.5 * sgn * cw * sin_t - 0.5 * ch * cos_t
    return (a, b, c, d, e, f)


def render_crop(
    pil_img: Image.Image,
    out_w: int,
    out_h: int,
    cx_frac: float,
    cy_frac: float,
    w_frac: float,
    theta: float,
    mirrored: bool,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    """Render an affine crop matching build_torch_grid coordinates."""
    img_w, img_h = pil_img.size
    cw = w_frac * img_w
    # Supersample downsampling when source pixels per output pixel > 2.2 avoids aliasing.
    if cw > 2.2 * out_w:
        max_factor = 4 if out_w < 480 else 2
        factor = min(max_factor, max(2, int(math.ceil(cw / (1.5 * out_w)))))
        work_w, work_h = out_w * factor, out_h * factor
        coeffs = pillow_affine_coeffs(
            img_w, img_h, work_w, work_h, cx_frac, cy_frac, w_frac, theta, mirrored
        )
        hi = pil_img.transform((work_w, work_h), Image.Transform.AFFINE, coeffs, resample=resample)
        return hi.resize((out_w, out_h), Image.Resampling.BOX)

    coeffs = pillow_affine_coeffs(
        img_w, img_h, out_w, out_h, cx_frac, cy_frac, w_frac, theta, mirrored
    )
    return pil_img.transform((out_w, out_h), Image.Transform.AFFINE, coeffs, resample=resample)
