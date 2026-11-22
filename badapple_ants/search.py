"""GPU-accelerated 32x24 coarse NCC search, 64x48 refinement, and timeline selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .ant_gate import MIN_ANT_SCORE, AntGate
from .catalog import has_genus_or_species
from .geometry import build_torch_grid, crop_half_extents, max_fit_crop_w, pose_in_bounds
from .media import extract_clip_frames

ANGLES = [i * (2.0 * math.pi / 16.0) for i in range(16)]
MIN_FIT_RATIO = 0.65
MIN_SHARPNESS_RATIO = 0.25
MAX_CENTER_OFFSET = 0.14
MIN_PATCH_STD = 10.0
MIN_FRAME_STD = 2.0
TOPK_COARSE = 48
REFINED_CANDIDATES = 32
VRAM_CHUNK_IMGS = 4096
QUERY_TILE = 2048
DEFAULT_DEDUP_L1 = 1.2


def _coarse_scales(min_scale: float) -> tuple[float, ...]:
    """Build 3-level coarse crop scale tuple (or 1 level when >= 0.90) clamped to [0.50, 0.98]."""
    clamped = min(0.98, max(0.50, float(min_scale)))
    if clamped >= 0.90:
        return (clamped,)
    mid = round(0.5 * (0.94 + clamped), 3)
    return (0.94, mid, clamped)


COARSE_SCALES = _coarse_scales(MIN_FIT_RATIO)


def _normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Center each row to zero mean and unit L2 norm; return (normed, std)."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    std = centered.std(dim=-1, unbiased=False)
    norm = centered.norm(dim=-1, keepdim=True).clamp_min(eps)
    return centered / norm, std


def _generate_image_poses(
    img_idx: int,
    w: int,
    h: int,
    scales: tuple[float, ...] = COARSE_SCALES,
    max_center_offset: float = MAX_CENTER_OFFSET,
) -> np.ndarray:
    """Generate ant-first candidate poses [img_idx, cx_frac, cy_frac, w_frac, theta, 0.0]."""
    poses: list[tuple[float, float, float, float, float, float]] = []
    wf = float(w)
    hf = float(h)
    for theta in ANGLES:
        cw_max = max_fit_crop_w(wf, hf, theta, margin=0.5)
        if cw_max < 48.0:
            continue
        for scale in scales:
            cw = cw_max * scale
            if cw < 48.0:
                continue
            w_frac = cw / wf
            if scale >= 0.78:
                if pose_in_bounds(wf, hf, 0.5, 0.5, w_frac, theta, margin=0.5):
                    poses.append((float(img_idx), 0.5, 0.5, w_frac, theta, 0.0))
                continue
            hx, hy = crop_half_extents(cw, theta)
            px_lo = 0.5 + hx
            px_hi = wf - 0.5 - hx
            py_lo = 0.5 + hy
            py_hi = hf - 0.5 - hy
            if px_hi < px_lo or py_hi < py_lo:
                continue
            if 0.0 < max_center_offset < 0.5:
                px_lo = max(px_lo, wf * (0.5 - max_center_offset))
                px_hi = min(px_hi, wf * (0.5 + max_center_offset))
                py_lo = max(py_lo, hf * (0.5 - max_center_offset))
                py_hi = min(py_hi, hf * (0.5 + max_center_offset))
            xs = (
                [0.5 * wf]
                if px_hi < px_lo
                else (
                    [0.5 * (px_lo + px_hi)]
                    if (px_hi - px_lo) < 4.0
                    else [px_lo, 0.5 * (px_lo + px_hi), px_hi]
                )
            )
            ys = (
                [0.5 * hf]
                if py_hi < py_lo
                else (
                    [0.5 * (py_lo + py_hi)]
                    if (py_hi - py_lo) < 4.0
                    else [py_lo, 0.5 * (py_lo + py_hi), py_hi]
                )
            )
            for py in ys:
                cy_frac = py / hf
                for px in xs:
                    cx_frac = px / wf
                    if pose_in_bounds(wf, hf, cx_frac, cy_frac, w_frac, theta, margin=0.5):
                        poses.append((float(img_idx), cx_frac, cy_frac, w_frac, theta, 0.0))
    if not poses:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(poses, dtype=np.float32)


def _laplacian_kernel(device: torch.device) -> torch.Tensor:
    """Return the 3x3 4-neighbor discrete Laplacian convolution kernel on `device`."""
    return torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
        device=device,
    )


def _generate_poses_batch_gpu(
    img_ids_gpu: torch.Tensor,
    dims_batch_gpu: torch.Tensor,
    scales: tuple[float, ...],
    max_center_offset: float = MAX_CENTER_OFFSET,
    eligible_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GPU equivalent of _generate_image_poses returning (poses_bp, valid_bp)."""
    device = dims_batch_gpu.device
    B = dims_batch_gpu.shape[0]
    wf = dims_batch_gpu[:, 0].float().unsqueeze(1)  # (B, 1)
    hf = dims_batch_gpu[:, 1].float().unsqueeze(1)  # (B, 1)

    angles = torch.tensor(ANGLES, dtype=torch.float32, device=device)  # (16,)
    theta = angles.unsqueeze(0).expand(B, 16)  # (B, 16)
    cos_a = torch.abs(torch.cos(theta))
    sin_a = torch.abs(torch.sin(theta))

    avail_w = torch.clamp(wf - 1.0, min=0.0)
    avail_h = torch.clamp(hf - 1.0, min=0.0)
    cw_max = torch.minimum(avail_w / (cos_a + 0.75 * sin_a), avail_h / (sin_a + 0.75 * cos_a))

    img_col = img_ids_gpu.float().view(B, 1)
    eps = 1e-4

    pose_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []

    for scale in scales:
        cw = cw_max * float(scale)
        ok_size = (cw_max >= 48.0) & (cw >= 48.0)
        w_frac = cw / wf
        ch = 0.75 * cw
        hx = 0.5 * cw * cos_a + 0.5 * ch * sin_a
        hy = 0.5 * cw * sin_a + 0.5 * ch * cos_a

        if scale >= 0.78:
            px = 0.5 * wf
            py = 0.5 * hf
            in_bounds = (
                (px - hx >= 0.5 - eps)
                & (px + hx <= wf - 0.5 + eps)
                & (py - hy >= 0.5 - eps)
                & (py + hy <= hf - 0.5 + eps)
            )
            poses_s = torch.stack(
                [
                    img_col.expand(B, 16),
                    torch.full((B, 16), 0.5, dtype=torch.float32, device=device),
                    torch.full((B, 16), 0.5, dtype=torch.float32, device=device),
                    w_frac,
                    theta,
                    torch.zeros((B, 16), dtype=torch.float32, device=device),
                ],
                dim=-1,
            )
            pose_parts.append(poses_s)
            mask_parts.append(ok_size & in_bounds)
            continue

        px_lo = 0.5 + hx
        px_hi = wf - 0.5 - hx
        py_lo = 0.5 + hy
        py_hi = hf - 0.5 - hy
        base_ok = ok_size & (px_hi >= px_lo) & (py_hi >= py_lo)

        if 0.0 < max_center_offset < 0.5:
            px_lo = torch.maximum(px_lo, wf * (0.5 - max_center_offset))
            px_hi = torch.minimum(px_hi, wf * (0.5 + max_center_offset))
            py_lo = torch.maximum(py_lo, hf * (0.5 - max_center_offset))
            py_hi = torch.minimum(py_hi, hf * (0.5 + max_center_offset))

        x_inv = px_hi < px_lo
        px_mid = torch.where(x_inv, 0.5 * wf, 0.5 * (px_lo + px_hi))
        x_use_ends = (~x_inv) & ((px_hi - px_lo) >= 4.0)
        xs_vals = torch.stack((px_lo, px_mid, px_hi), dim=-1)  # (B, 16, 3)
        ones_x = torch.ones_like(x_use_ends)
        xs_mask = torch.stack((x_use_ends, ones_x, x_use_ends), dim=-1)  # (B, 16, 3)

        y_inv = py_hi < py_lo
        py_mid = torch.where(y_inv, 0.5 * hf, 0.5 * (py_lo + py_hi))
        y_use_ends = (~y_inv) & ((py_hi - py_lo) >= 4.0)
        ys_vals = torch.stack((py_lo, py_mid, py_hi), dim=-1)  # (B, 16, 3)
        ys_mask = torch.stack((y_use_ends, ones_x, y_use_ends), dim=-1)  # (B, 16, 3)

        px_grid = xs_vals.unsqueeze(2).expand(B, 16, 3, 3)
        py_grid = ys_vals.unsqueeze(3).expand(B, 16, 3, 3)
        hx_g = hx[:, :, None, None]
        hy_g = hy[:, :, None, None]
        wf_g = wf[:, :, None, None]
        hf_g = hf[:, :, None, None]

        in_bounds = (
            (px_grid - hx_g >= 0.5 - eps)
            & (px_grid + hx_g <= wf_g - 0.5 + eps)
            & (py_grid - hy_g >= 0.5 - eps)
            & (py_grid + hy_g <= hf_g - 0.5 + eps)
        )
        grid_mask = (
            base_ok[:, :, None, None]
            & xs_mask.unsqueeze(2)
            & ys_mask.unsqueeze(3)
            & in_bounds
        )

        cx_flat = (px_grid / wf_g).reshape(B, 144)
        cy_flat = (py_grid / hf_g).reshape(B, 144)
        wf_flat = w_frac.unsqueeze(-1).expand(B, 16, 9).reshape(B, 144)
        th_flat = theta.unsqueeze(-1).expand(B, 16, 9).reshape(B, 144)
        zeros_flat = torch.zeros((B, 144), dtype=torch.float32, device=device)

        poses_s = torch.stack(
            [img_col.expand(B, 144), cx_flat, cy_flat, wf_flat, th_flat, zeros_flat],
            dim=-1,
        )
        pose_parts.append(poses_s)
        mask_parts.append(grid_mask.reshape(B, 144))

    poses_bp = torch.cat(pose_parts, dim=1)
    valid_bp = torch.cat(mask_parts, dim=1)
    if eligible_batch is not None:
        valid_bp = valid_bp & eligible_batch.bool().unsqueeze(1)
    return poses_bp, valid_bp


def _sample_patches(
    gray_pack: torch.Tensor,
    dims_gpu: torch.Tensor,
    poses_gpu: torch.Tensor,
    canvas_size: int,
    grid_w: int,
    grid_h: int,
    supersample_2x: bool,
    pose_batch: int = 2048,
    img_id_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample poses in mini-batches and return (normalized_patches, std_per_pose, sharpness_ratio)."""
    total = poses_gpu.shape[0]
    dim = grid_w * grid_h
    device = poses_gpu.device
    out_patches = torch.empty((total, dim), dtype=torch.float32, device=device)
    out_stds = torch.empty((total,), dtype=torch.float32, device=device)
    out_sharpness = torch.empty((total,), dtype=torch.float32, device=device)

    lap_kernel = _laplacian_kernel(device)

    sw = grid_w * 2 if supersample_2x else grid_w
    sh = grid_h * 2 if supersample_2x else grid_h

    for start in range(0, total, pose_batch):
        end = min(total, start + pose_batch)
        sub = poses_gpu[start:end]
        local_ids = sub[:, 0].long() - img_id_offset
        w_vec = dims_gpu[local_ids, 0].float()
        h_vec = dims_gpu[local_ids, 1].float()
        grid = build_torch_grid(sub[:, 1:6], w_vec, h_vec, canvas_size, sw, sh)
        imgs = gray_pack[local_ids].float()
        sampled = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        pre_std = sampled.flatten(1).std(dim=-1, unbiased=False)
        lap_std = F.conv2d(sampled, lap_kernel).flatten(1).std(dim=-1, unbiased=False)
        out_sharpness[start:end] = lap_std / pre_std.clamp_min(1e-6)

        if supersample_2x:
            sampled = F.avg_pool2d(sampled, kernel_size=2, stride=2)
        flat = sampled.view(end - start, dim)
        normed, stds = _normalize_rows(flat)
        out_patches[start:end] = normed
        out_stds[start:end] = stds
    return out_patches, out_stds, out_sharpness


def _sample_patches_grouped(
    imgs_batch_f32: torch.Tensor,
    dims_batch_f32: torch.Tensor,
    poses_bp: torch.Tensor,
    canvas_size: int,
    grid_w: int,
    grid_h: int,
    supersample_2x: bool,
    lap_kernel: torch.Tensor,
    pose_chunk: int = 160,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-copy grouped sampling: sample P poses per image without duplicating image tensors."""
    B, P, _ = poses_bp.shape
    dim = grid_w * grid_h
    device = poses_bp.device

    out_patches = torch.empty((B, P, dim), dtype=torch.float32, device=device)
    out_stds = torch.empty((B, P), dtype=torch.float32, device=device)
    out_sharpness = torch.empty((B, P), dtype=torch.float32, device=device)

    sw = grid_w * 2 if supersample_2x else grid_w
    sh = grid_h * 2 if supersample_2x else grid_h

    w_base = dims_batch_f32[:, 0].view(B, 1)
    h_base = dims_batch_f32[:, 1].view(B, 1)

    for p0 in range(0, P, pose_chunk):
        p1 = min(P, p0 + pose_chunk)
        p_len = p1 - p0
        sub = poses_bp[:, p0:p1, 1:6].reshape(B * p_len, 5)
        w_exp = w_base.expand(B, p_len).reshape(B * p_len)
        h_exp = h_base.expand(B, p_len).reshape(B * p_len)
        grid = build_torch_grid(sub, w_exp, h_exp, canvas_size, sw, sh).view(B, p_len * sh, sw, 2)

        sampled = F.grid_sample(
            imgs_batch_f32,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(B * p_len, 1, sh, sw)

        pre_std = sampled.flatten(1).std(dim=-1, unbiased=False)
        lap_std = F.conv2d(sampled, lap_kernel).flatten(1).std(dim=-1, unbiased=False)
        out_sharpness[:, p0:p1] = (lap_std / pre_std.clamp_min(1e-6)).view(B, p_len)

        if supersample_2x:
            sampled = F.avg_pool2d(sampled, kernel_size=2, stride=2)
        flat = sampled.view(B * p_len, dim)
        normed, stds = _normalize_rows(flat)
        out_patches[:, p0:p1] = normed.view(B, p_len, dim)
        out_stds[:, p0:p1] = stds.view(B, p_len)

    return out_patches, out_stds, out_sharpness


def _clamp_poses_batch(
    poses: torch.Tensor,
    dims_gpu: torch.Tensor,
    min_w_px: float = 44.0,
    min_fit_ratio: float = MIN_FIT_RATIO,
    max_center_offset: float = MAX_CENTER_OFFSET,
    margin: float = 0.5,
) -> torch.Tensor:
    """Vectorized bounds + ant-first minimum scale and center-offset clamping on GPU.

    poses shape: (..., 6) -> [img_idx, cx_frac, cy_frac, w_frac, theta, mirrored]
    """
    img_ids = poses[..., 0].long()
    w = dims_gpu[img_ids, 0].float()
    h = dims_gpu[img_ids, 1].float()
    theta = poses[..., 4]
    cos_a = torch.abs(torch.cos(theta))
    sin_a = torch.abs(torch.sin(theta))

    avail_w = torch.clamp(w - 2.0 * margin, min=1.0)
    avail_h = torch.clamp(h - 2.0 * margin, min=1.0)
    cw_max = torch.minimum(avail_w / (cos_a + 0.75 * sin_a), avail_h / (sin_a + 0.75 * cos_a))
    cw_min = torch.minimum(cw_max, torch.clamp(cw_max * min_fit_ratio, min=min_w_px))
    cw = torch.clamp(poses[..., 3] * w, min=cw_min, max=cw_max)

    ch = 0.75 * cw
    hx = 0.5 * cw * cos_a + 0.5 * ch * sin_a
    hy = 0.5 * cw * sin_a + 0.5 * ch * cos_a

    px_lo = margin + hx
    px_hi = w - margin - hx
    py_lo = margin + hy
    py_hi = h - margin - hy
    if 0.0 < max_center_offset < 0.5:
        cx_min = w * (0.5 - max_center_offset)
        cx_max = w * (0.5 + max_center_offset)
        cy_min = h * (0.5 - max_center_offset)
        cy_max = h * (0.5 + max_center_offset)
        px_lo = torch.minimum(torch.maximum(px_lo, cx_min), 0.5 * w)
        px_hi = torch.maximum(torch.minimum(px_hi, cx_max), 0.5 * w)
        py_lo = torch.minimum(torch.maximum(py_lo, cy_min), 0.5 * h)
        py_hi = torch.maximum(torch.minimum(py_hi, cy_max), 0.5 * h)

    px = torch.clamp(poses[..., 1] * w, min=px_lo, max=px_hi)
    py = torch.clamp(poses[..., 2] * h, min=py_lo, max=py_hi)

    out = poses.clone()
    out[..., 1] = px / w
    out[..., 2] = py / h
    out[..., 3] = cw / w
    return out


def _refine_candidates(
    gray_pack: torch.Tensor,
    dims_gpu: torch.Tensor,
    q64: torch.Tensor,
    cand_poses: torch.Tensor,
    cand_q_idx: torch.Tensor,
    canvas_size: int,
    rounds: int = 5,
    pose_batch: int = 1024,
    min_fit_ratio: float = MIN_FIT_RATIO,
    min_sharpness: float = MIN_SHARPNESS_RATIO,
    max_center_offset: float = MAX_CENTER_OFFSET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 3^4 coordinate-neighborhood refinement at 64x48 on GPU."""
    device = cand_poses.device
    num_cands = cand_poses.shape[0]

    grid_offsets = torch.cartesian_prod(
        torch.tensor([-1.0, 0.0, 1.0], device=device),
        torch.tensor([-1.0, 0.0, 1.0], device=device),
        torch.tensor([-1.0, 0.0, 1.0], device=device),
        torch.tensor([-1.0, 0.0, 1.0], device=device),
    )  # (81, 4)

    current = cand_poses.clone()
    best_scores = torch.full((num_cands,), -2.0, dtype=torch.float32, device=device)

    for r in range(rounds):
        scale = 0.5**r
        step_xy = 0.04 * scale
        step_w = 0.04 * scale
        step_th = (math.pi / 16.0) * scale

        deltas = torch.zeros((81, 6), dtype=torch.float32, device=device)
        deltas[:, 1] = grid_offsets[:, 0] * step_xy
        deltas[:, 2] = grid_offsets[:, 1] * step_xy
        deltas[:, 3] = grid_offsets[:, 2] * step_w
        deltas[:, 4] = grid_offsets[:, 3] * step_th

        expanded = current.unsqueeze(1) + deltas.unsqueeze(0)  # (K, 81, 6)
        clamped = _clamp_poses_batch(
            expanded,
            dims_gpu,
            min_fit_ratio=min_fit_ratio,
            max_center_offset=max_center_offset,
        )
        flat_poses = clamped.view(num_cands * 81, 6)
        flat_q_idx = cand_q_idx.unsqueeze(1).expand(num_cands, 81).reshape(-1)

        scores_flat = torch.full((num_cands * 81,), -2.0, dtype=torch.float32, device=device)
        for start in range(0, num_cands * 81, pose_batch):
            end = min(num_cands * 81, start + pose_batch)
            sub_p = flat_poses[start:end]
            sub_q = flat_q_idx[start:end]
            patches, stds, sharpness = _sample_patches(
                gray_pack,
                dims_gpu,
                sub_p,
                canvas_size,
                grid_w=64,
                grid_h=48,
                supersample_2x=False,
                pose_batch=pose_batch,
            )
            dots = (patches * q64[sub_q]).sum(dim=-1)
            valid = (stds >= MIN_PATCH_STD) & (sharpness >= min_sharpness)
            dots = torch.where(valid, dots, torch.full_like(dots, -2.0))
            scores_flat[start:end] = dots

        scores_grid = scores_flat.view(num_cands, 81)
        round_best, round_idx = scores_grid.max(dim=1)
        current = clamped[torch.arange(num_cands, device=device), round_idx]
        best_scores = round_best

    return current, best_scores


def _refine_chunked(
    gray_np: np.ndarray,
    dims_gpu: torch.Tensor,
    q64: torch.Tensor,
    q32: torch.Tensor,
    cand_poses_gpu: torch.Tensor,
    cand_qidx_gpu: torch.Tensor,
    canvas_size: int,
    chunk_imgs: int = VRAM_CHUNK_IMGS,
    min_fit_ratio: float = MIN_FIT_RATIO,
    min_sharpness: float = MIN_SHARPNESS_RATIO,
    max_center_offset: float = MAX_CENTER_OFFSET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine top candidates in VRAM-bounded image chunks and evaluate final 32x24 scores."""
    device = cand_poses_gpu.device
    total_cands = cand_poses_gpu.shape[0]
    out_poses = torch.empty_like(cand_poses_gpu)
    out_s64 = torch.full((total_cands,), -2.0, dtype=torch.float32, device=device)
    out_s32 = torch.full((total_cands,), -2.0, dtype=torch.float32, device=device)

    unique_imgs, inv = torch.unique(cand_poses_gpu[:, 0].long(), sorted=True, return_inverse=True)
    for c_start in range(0, len(unique_imgs), chunk_imgs):
        c_end = min(len(unique_imgs), c_start + chunk_imgs)
        mask = (inv >= c_start) & (inv < c_end)
        cand_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        chunk_global_ids = unique_imgs[c_start:c_end]

        sub_pack = (
            torch.from_numpy(np.array(gray_np[chunk_global_ids.cpu().numpy()], copy=True))
            .unsqueeze(1)
            .to(device=device, dtype=torch.uint8)
        )
        sub_dims = dims_gpu[chunk_global_ids]
        sub_poses = cand_poses_gpu[cand_idx].clone()
        orig_ids = sub_poses[:, 0].clone()
        sub_poses[:, 0] = (inv[cand_idx] - c_start).float()
        sub_q = cand_qidx_gpu[cand_idx]

        ref_sub, s64_sub = _refine_candidates(
            sub_pack,
            sub_dims,
            q64,
            sub_poses,
            sub_q,
            canvas_size,
            rounds=5,
            pose_batch=2048,
            min_fit_ratio=min_fit_ratio,
            min_sharpness=min_sharpness,
            max_center_offset=max_center_offset,
        )
        p32_sub, std32_sub, sharp32_sub = _sample_patches(
            sub_pack,
            sub_dims,
            ref_sub,
            canvas_size,
            grid_w=32,
            grid_h=24,
            supersample_2x=True,
            pose_batch=2048,
        )
        s32_sub = (p32_sub * q32[sub_q]).sum(dim=-1)
        valid32 = (std32_sub >= MIN_PATCH_STD) & (sharp32_sub >= min_sharpness)
        s32_sub = torch.where(valid32, s32_sub, torch.full_like(s32_sub, -2.0))

        ref_sub[:, 0] = orig_ids
        out_poses[cand_idx] = ref_sub
        out_s64[cand_idx] = s64_sub
        out_s32[cand_idx] = s32_sub

    return out_poses, out_s64, out_s32


def run_search(
    video_path: Path,
    cache_dir: Path,
    output_dir: Path,
    start_s: float = 25.0,
    duration_s: float = 25.0,
    cooldown: int = 15,
    max_reuse: int = 6,
    min_scale: float = MIN_FIT_RATIO,
    min_sharpness: float = MIN_SHARPNESS_RATIO,
    max_center_offset: float = MAX_CENTER_OFFSET,
    min_ant_score: float = MIN_ANT_SCORE,
    dedup_l1: float = DEFAULT_DEDUP_L1,
    max_images: int = 0,
) -> dict:
    """Run coarse 32x24 GPU NCC search, 64x48 refinement, CLIP ant-presence gate, and timeline assignment."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for search")
    device = torch.device("cuda")

    pack_dir = cache_dir / "pack"
    gray_np = np.load(pack_dir / "gray.npy", mmap_mode="r")
    dims_np = np.load(pack_dir / "dims.npy")
    items = json.loads((pack_dir / "items.json").read_text(encoding="utf-8"))
    pack_meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    canvas_size = int(pack_meta["canvas_size"])
    num_images = len(items)
    if max_images > 0 and max_images < num_images:
        num_images = max_images
        dims_np = dims_np[:num_images]
        items = items[:num_images]

    eligible_mask = np.asarray([has_genus_or_species(it) for it in items], dtype=bool)
    skipped_family_only = int((~eligible_mask).sum())

    scales = _coarse_scales(min_scale)

    frames_u8, clip_meta = extract_clip_frames(video_path, start_s, duration_s, cache_dir)
    n_frames = frames_u8.shape[0]

    # Deduplicate identical or near-identical consecutive frames while preserving exact order mapping.
    hash_to_distinct: dict[str, int] = {}
    distinct_indices: list[int] = []
    frame_to_distinct: list[int] = []
    for i in range(n_frames):
        if i > 0 and dedup_l1 > 0.0:
            prev_d = frame_to_distinct[i - 1]
            prev_rep = frames_u8[distinct_indices[prev_d]]
            diff = float(np.abs(frames_u8[i].astype(np.int16) - prev_rep.astype(np.int16)).mean())
            if diff <= dedup_l1:
                frame_to_distinct.append(prev_d)
                continue
        digest = hashlib.sha256(frames_u8[i].tobytes()).hexdigest()
        d_idx = hash_to_distinct.get(digest)
        if d_idx is None:
            d_idx = len(distinct_indices)
            hash_to_distinct[digest] = d_idx
            distinct_indices.append(i)
        frame_to_distinct.append(d_idx)

    distinct_u8 = frames_u8[distinct_indices]
    m_distinct = len(distinct_indices)
    distinct_f32 = torch.from_numpy(distinct_u8).float()

    frame_stds = distinct_f32.view(m_distinct, -1).std(dim=-1, unbiased=False)
    frame_means = distinct_f32.view(m_distinct, -1).mean(dim=-1)
    is_solid = (frame_stds < MIN_FRAME_STD).numpy()
    textured_distinct = [i for i in range(m_distinct) if not is_solid[i]]
    distinct_to_tex = {d_i: t_i for t_i, d_i in enumerate(textured_distinct)}
    m_tex = len(textured_distinct)

    print(
        f"search: {n_frames} clip frames -> {m_distinct} distinct "
        f"({m_tex} textured, {m_distinct - m_tex} solid, dedup_l1={dedup_l1}) against "
        f"{num_images - skipped_family_only} eligible images ({skipped_family_only} family-only skipped) "
        f"on {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    need_solid_stats = bool(is_solid.any())
    img_means = np.empty((num_images,), dtype=np.float32) if need_solid_stats else None
    img_stds = np.empty((num_images,), dtype=np.float32) if need_solid_stats else None

    # Load dims and queries onto GPU; stream gray_np in VRAM-bounded chunks.
    dims_gpu = torch.from_numpy(dims_np).to(device=device, dtype=torch.int32)

    refined_per_tex: list[list[dict]] = [[] for _ in range(m_tex)]

    if m_tex > 0:
        with torch.no_grad():
            tex_f32 = distinct_f32[textured_distinct].unsqueeze(1).to(device=device)
            t32 = F.interpolate(tex_f32, size=(24, 32), mode="area").squeeze(1)
            t64 = F.interpolate(tex_f32, size=(48, 64), mode="area").squeeze(1)

            t32_mir = torch.flip(t32, dims=[-1])
            t64_mir = torch.flip(t64, dims=[-1])

            q32_pairs = torch.stack((t32, t32_mir), dim=1).view(2 * m_tex, 768)
            q64_pairs = torch.stack((t64, t64_mir), dim=1).view(2 * m_tex, 3072)
            q32, _ = _normalize_rows(q32_pairs)
            q64, _ = _normalize_rows(q64_pairs)

            num_q = 2 * m_tex
            top_scores = torch.full((TOPK_COARSE, num_q), -2.0, dtype=torch.float32, device=device)
            top_poses = torch.zeros((TOPK_COARSE, num_q, 6), dtype=torch.float32, device=device)

            acc_scores: list[torch.Tensor] = []
            acc_poses: list[torch.Tensor] = []

            batch_imgs = 32
            flush_every = 4 if num_q > 4096 else 8
            lap_kernel = _laplacian_kernel(device)
            if need_solid_stats:
                pix_ys = torch.arange(canvas_size, device=device, dtype=torch.int32).view(1, canvas_size, 1)
                pix_xs = torch.arange(canvas_size, device=device, dtype=torch.int32).view(1, 1, canvas_size)

            def _load_gray_chunk(s: int, e: int) -> np.ndarray:
                return np.array(gray_np[s:e], copy=True)

            chunk_bounds = [
                (cs, min(num_images, cs + VRAM_CHUNK_IMGS))
                for cs in range(0, num_images, VRAM_CHUNK_IMGS)
            ]
            pbar = tqdm(total=num_images, desc="coarse-32x24", unit="img")
            with ThreadPoolExecutor(max_workers=1) as io_pool:
                next_fut = io_pool.submit(_load_gray_chunk, *chunk_bounds[0]) if chunk_bounds else None
                for b_i, (chunk_start, chunk_end) in enumerate(chunk_bounds):
                    assert next_fut is not None
                    chunk_np = next_fut.result()
                    if b_i + 1 < len(chunk_bounds):
                        next_fut = io_pool.submit(_load_gray_chunk, *chunk_bounds[b_i + 1])
                    else:
                        next_fut = None

                    chunk_pack = torch.from_numpy(chunk_np).unsqueeze(1).to(device=device, dtype=torch.uint8)
                    chunk_dims = dims_gpu[chunk_start:chunk_end]
                    if need_solid_stats:
                        assert img_means is not None and img_stds is not None
                        c_len = chunk_end - chunk_start
                        pix_mask = (pix_ys < chunk_dims[:, 1].view(c_len, 1, 1)) & (
                            pix_xs < chunk_dims[:, 0].view(c_len, 1, 1)
                        )
                        vals = chunk_pack.squeeze(1).float() * pix_mask
                        area = (chunk_dims[:, 0] * chunk_dims[:, 1]).float().clamp_min(1.0)
                        sum1 = vals.sum(dim=(1, 2))
                        sum2 = (vals * vals).sum(dim=(1, 2))
                        mean_g = sum1 / area
                        std_g = torch.sqrt((sum2 / area - mean_g * mean_g).clamp_min(0.0))
                        img_means[chunk_start:chunk_end] = mean_g.cpu().numpy()
                        img_stds[chunk_start:chunk_end] = std_g.cpu().numpy()

                    chunk_ids = torch.arange(chunk_start, chunk_end, device=device, dtype=torch.float32)
                    chunk_elig = torch.from_numpy(eligible_mask[chunk_start:chunk_end]).to(device=device)
                    chunk_poses_bp, chunk_valid_bp = _generate_poses_batch_gpu(
                        chunk_ids,
                        chunk_dims,
                        scales=scales,
                        max_center_offset=max_center_offset,
                        eligible_batch=chunk_elig,
                    )

                    for img_start in range(chunk_start, chunk_end, batch_imgs):
                        img_end = min(chunk_end, img_start + batch_imgs)
                        B = img_end - img_start
                        local_s = img_start - chunk_start
                        local_e = img_end - chunk_start
                        elig_slice = eligible_mask[img_start:img_end]
                        pbar.update(B)

                        if bool(elig_slice.any()) and chunk_poses_bp.shape[1] > 0:
                            dims_batch = chunk_dims[local_s:local_e]
                            poses_bp = chunk_poses_bp[local_s:local_e]
                            valid_bp = chunk_valid_bp[local_s:local_e]
                            imgs_half = F.avg_pool2d(
                                chunk_pack[local_s:local_e].float(), kernel_size=2, stride=2
                            )
                            patches_bp, stds_bp, sharp_bp = _sample_patches_grouped(
                                imgs_half,
                                dims_batch.float(),
                                poses_bp,
                                canvas_size,
                                grid_w=32,
                                grid_h=24,
                                supersample_2x=False,
                                lap_kernel=lap_kernel,
                                pose_chunk=304,
                            )
                            P = poses_bp.shape[1]
                            patches_flat = patches_bp.view(B * P, 768)
                            invalid_mask = (
                                (~valid_bp)
                                | (stds_bp < MIN_PATCH_STD)
                                | (sharp_bp < (min_sharpness * 0.8))
                            ).unsqueeze(-1)
                            has_invalid = bool(invalid_mask.any().item())
                            batch_best_scores = torch.empty((B, num_q), dtype=torch.float32, device=device)
                            batch_best_poses = torch.empty((B, num_q, 6), dtype=torch.float32, device=device)
                            for q0 in range(0, num_q, QUERY_TILE):
                                q1 = min(num_q, q0 + QUERY_TILE)
                                tile_scores = (patches_flat @ q32[q0:q1].T).view(B, P, q1 - q0)
                                if has_invalid:
                                    tile_scores.masked_fill_(invalid_mask, -2.0)
                                tile_best, tile_loc = tile_scores.max(dim=1)
                                batch_best_scores[:, q0:q1] = tile_best
                                batch_best_poses[:, q0:q1, :] = poses_bp.gather(
                                    1, tile_loc.unsqueeze(-1).expand(B, q1 - q0, 6)
                                )
                            acc_scores.append(batch_best_scores)
                            acc_poses.append(batch_best_poses)

                        if acc_scores and (len(acc_scores) >= flush_every or img_end == num_images):
                            cand_s = torch.cat([top_scores, *acc_scores], dim=0)
                            cand_p = torch.cat([top_poses, *acc_poses], dim=0)
                            new_scores, new_idx = torch.topk(cand_s, k=TOPK_COARSE, dim=0)
                            q_arange = torch.arange(num_q, device=device).unsqueeze(0).expand(TOPK_COARSE, num_q)
                            top_scores = new_scores
                            top_poses = cand_p[new_idx, q_arange]
                            acc_scores.clear()
                            acc_poses.clear()
            pbar.close()

            # Merge as-is and mirrored top-48 lists per textured frame, keeping top-32 distinct images.
            top_scores_cpu = top_scores.cpu().numpy()
            top_poses_cpu = top_poses.cpu().numpy()

            cand_list_poses: list[np.ndarray] = []
            cand_list_qidx: list[int] = []
            cand_list_s32: list[float] = []
            cand_slices: list[tuple[int, int]] = []

            for t_i in range(m_tex):
                q_even = 2 * t_i
                q_odd = 2 * t_i + 1
                best_by_img: dict[int, tuple[float, np.ndarray, int]] = {}
                for q_id, mir_flag in ((q_even, 0.0), (q_odd, 1.0)):
                    for rank in range(TOPK_COARSE):
                        sc = float(top_scores_cpu[rank, q_id])
                        if sc <= -1.5:
                            continue
                        pose_row = top_poses_cpu[rank, q_id].copy()
                        pose_row[5] = mir_flag
                        im_id = int(pose_row[0])
                        prev = best_by_img.get(im_id)
                        if prev is None or sc > prev[0]:
                            best_by_img[im_id] = (sc, pose_row, q_id)

                sorted_cands = sorted(best_by_img.values(), key=lambda x: x[0], reverse=True)[
                    :REFINED_CANDIDATES
                ]
                s_idx = len(cand_list_poses)
                for sc32, pose_row, _ in sorted_cands:
                    cand_list_poses.append(pose_row)
                    # pose_row[5] already encodes mir_flag, so refine against unflipped query q_even.
                    cand_list_qidx.append(q_even)
                    cand_list_s32.append(sc32)
                cand_slices.append((s_idx, len(cand_list_poses)))

            if cand_list_poses:
                cand_poses_gpu = torch.from_numpy(np.stack(cand_list_poses, axis=0)).to(device=device)
                cand_qidx_gpu = torch.tensor(cand_list_qidx, dtype=torch.long, device=device)
                print(f"refining {cand_poses_gpu.shape[0]} candidates at 64x48...", flush=True)
                ref_poses_gpu, ref_scores_gpu, ref_s32 = _refine_chunked(
                    gray_np,
                    dims_gpu,
                    q64,
                    q32,
                    cand_poses_gpu,
                    cand_qidx_gpu,
                    canvas_size,
                    min_fit_ratio=min_scale,
                    min_sharpness=min_sharpness,
                    max_center_offset=max_center_offset,
                )

                ref_poses_cpu = ref_poses_gpu.cpu().numpy()
                ref_s64_cpu = ref_scores_gpu.cpu().numpy()
                ref_s32_cpu = ref_s32.cpu().numpy()

                for t_i, (s_idx, e_idx) in enumerate(cand_slices):
                    entries = []
                    for idx in range(s_idx, e_idx):
                        row = ref_poses_cpu[idx]
                        entries.append(
                            {
                                "image_idx": int(row[0]),
                                "cx_frac": float(row[1]),
                                "cy_frac": float(row[2]),
                                "w_frac": float(row[3]),
                                "theta": float(row[4]),
                                "mirrored": bool(row[5] > 0.5),
                                "score_64": float(ref_s64_cpu[idx]),
                                "score_32": float(max(float(ref_s32_cpu[idx]), cand_list_s32[idx])),
                            }
                        )
                    entries.sort(key=lambda e: e["score_64"], reverse=True)
                    refined_per_tex[t_i] = entries

    dark_pool: list[int] = []
    bright_pool: list[int] = []
    if need_solid_stats:
        assert img_means is not None and img_stds is not None
        if m_tex == 0:
            for s in range(0, num_images, VRAM_CHUNK_IMGS):
                e = min(num_images, s + VRAM_CHUNK_IMGS)
                block = np.asarray(gray_np[s:e])
                for offset_i, i in enumerate(range(s, e)):
                    w, h = int(dims_np[i, 0]), int(dims_np[i, 1])
                    patch = block[offset_i, :h, :w]
                    img_means[i] = float(patch.mean())
                    img_stds[i] = float(patch.std())
        eligible_ids = [i for i in range(num_images) if eligible_mask[i]] or list(range(num_images))
        dark_pool = sorted(
            [i for i in eligible_ids if img_means[i] < 100.0],
            key=lambda i: img_means[i] + 1.5 * img_stds[i],
        ) or sorted(eligible_ids, key=lambda i: img_means[i])
        bright_pool = sorted(
            [i for i in eligible_ids if img_means[i] > 155.0],
            key=lambda i: (255.0 - img_means[i]) + 1.5 * img_stds[i],
        ) or sorted(eligible_ids, key=lambda i: -img_means[i])

    ant_gate: AntGate | None = None
    if min_ant_score > 0.0 and m_tex > 0:
        ant_gate = AntGate(device=device)
        pre_cands: list[dict] = []
        for entries in refined_per_tex:
            pre_cands.extend(entries[:8])
        if pre_cands:
            print(f"ant-gate: scoring top {len(pre_cands)} candidate crops with CLIP...", flush=True)
            pre_scores = ant_gate.score_candidates(pre_cands, items, cache_dir)
            for c, sc in zip(pre_cands, pre_scores):
                c["ant_score"] = sc

    def _get_ant_score(cand: dict) -> float:
        if ant_gate is None:
            return 1.0
        cached_sc = cand.get("ant_score")
        if cached_sc is not None:
            return float(cached_sc)
        sc = ant_gate.score_candidates([cand], items, cache_dir)[0]
        cand["ant_score"] = sc
        return sc

    # Sequential timeline assignment with anti-repetition cooldown, max reuse, and CLIP ant-presence gate.
    last_used: dict[int, int] = {}
    use_count: dict[int, int] = {}
    frame_records: list[dict] = []

    solid_dark_cursor = 0
    solid_bright_cursor = 0
    clip_rejected_candidates = 0
    ant_fallback_frames = 0

    for t in range(n_frames):
        d_i = frame_to_distinct[t]
        if t > 0 and d_i == frame_to_distinct[t - 1]:
            prev = dict(frame_records[-1])
            prev["frame_idx"] = t
            prev["reused_from_prev"] = True
            frame_records.append(prev)
            continue

        chosen_ant_score: float | None = None
        if is_solid[d_i]:
            is_dark = float(frame_means[d_i]) < 128.0
            pool = dark_pool if is_dark else bright_pool
            if is_dark:
                im_idx = pool[solid_dark_cursor % len(pool)]
                solid_dark_cursor += 1
            else:
                im_idx = pool[solid_bright_cursor % len(pool)]
                solid_bright_cursor += 1
            w, h = int(dims_np[im_idx, 0]), int(dims_np[im_idx, 1])
            cw = max_fit_crop_w(float(w), float(h), 0.0, margin=0.5) * 0.94
            chosen = {
                "image_idx": im_idx,
                "cx_frac": 0.5,
                "cy_frac": 0.5,
                "w_frac": cw / float(w),
                "theta": 0.0,
                "mirrored": False,
                "score_64": 0.0,
                "score_32": 0.0,
            }
            is_solid_flag = True
        else:
            t_i = distinct_to_tex[d_i]
            cands = refined_per_tex[t_i]
            chosen = None
            if ant_gate is not None:
                for c in cands:
                    im_idx = c["image_idx"]
                    if (t - last_used.get(im_idx, -100000)) >= cooldown and use_count.get(
                        im_idx, 0
                    ) < max_reuse:
                        if _get_ant_score(c) >= min_ant_score:
                            chosen = c
                            break
                        clip_rejected_candidates += 1
                if chosen is None:
                    for c in cands:
                        im_idx = c["image_idx"]
                        if (t - last_used.get(im_idx, -100000)) >= cooldown:
                            if _get_ant_score(c) >= min_ant_score:
                                chosen = c
                                break
                if chosen is None:
                    passing = [c for c in cands if _get_ant_score(c) >= min_ant_score]
                    if passing:
                        chosen = max(
                            passing,
                            key=lambda c: (t - last_used.get(c["image_idx"], -100000)) * 0.01
                            + c["score_64"],
                        )
                if chosen is None:
                    ant_fallback_frames += 1
                    chosen = max(cands, key=lambda c: _get_ant_score(c) + 0.25 * c["score_64"])
                chosen_ant_score = _get_ant_score(chosen)
            else:
                for c in cands:
                    im_idx = c["image_idx"]
                    if (t - last_used.get(im_idx, -100000)) >= cooldown and use_count.get(
                        im_idx, 0
                    ) < max_reuse:
                        chosen = c
                        break
                if chosen is None:
                    for c in cands:
                        im_idx = c["image_idx"]
                        if (t - last_used.get(im_idx, -100000)) >= cooldown:
                            chosen = c
                            break
                if chosen is None:
                    chosen = max(
                        cands,
                        key=lambda c: (t - last_used.get(c["image_idx"], -100000)) * 0.01 + c["score_64"],
                    )
            is_solid_flag = False

        im_idx = chosen["image_idx"]
        last_used[im_idx] = t
        use_count[im_idx] = use_count.get(im_idx, 0) + 1
        item = items[im_idx]

        frame_records.append(
            {
                "frame_idx": t,
                "distinct_idx": d_i,
                "is_solid": is_solid_flag,
                "reused_from_prev": False,
                "pose": {
                    "cx_frac": chosen["cx_frac"],
                    "cy_frac": chosen["cy_frac"],
                    "w_frac": chosen["w_frac"],
                    "theta": chosen["theta"],
                    "mirrored": chosen["mirrored"],
                },
                "score_64": chosen["score_64"],
                "score_32": chosen["score_32"],
                "ant_score": chosen_ant_score,
                "item": item,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    matches_payload = {
        "clip": clip_meta,
        "search_params": {
            "start_s": start_s,
            "duration_s": duration_s,
            "cooldown": cooldown,
            "max_reuse": max_reuse,
            "min_scale": min_scale,
            "min_sharpness": min_sharpness,
            "max_center_offset": max_center_offset,
            "min_ant_score": min_ant_score,
            "dedup_l1": dedup_l1,
            "indexed_images": num_images,
        },
        "frames": frame_records,
    }
    matches_path = output_dir / "matches.json"
    matches_path.write_text(json.dumps(matches_payload, indent=2), encoding="utf-8")

    tex_frames = [fr for fr in frame_records if not fr["is_solid"]]
    s64_vals = np.asarray([fr["score_64"] for fr in tex_frames], dtype=np.float32)
    s32_vals = np.asarray([fr["score_32"] for fr in tex_frames], dtype=np.float32)
    ant_vals = np.asarray(
        [fr["ant_score"] for fr in tex_frames if fr.get("ant_score") is not None],
        dtype=np.float32,
    )
    source_counts = Counter(fr["item"]["source"] for fr in frame_records)
    unique_imgs = {fr["item"]["id"] for fr in frame_records}
    unique_species = {fr["item"]["species"] for fr in frame_records if fr["item"].get("species")}

    summary = {
        "indexed_thumbnails": num_images,
        "skipped_family_only": skipped_family_only,
        "clip_frames": n_frames,
        "distinct_frames": m_distinct,
        "textured_distinct_frames": m_tex,
        "textured_timeline_frames": len(tex_frames),
        "solid_timeline_frames": n_frames - len(tex_frames),
        "unique_images_used": len(unique_imgs),
        "unique_species_used": len(unique_species),
        "source_frame_counts": dict(source_counts),
        "ncc_64x48": {
            "p10": float(np.percentile(s64_vals, 10)) if len(s64_vals) else 0.0,
            "median": float(np.median(s64_vals)) if len(s64_vals) else 0.0,
            "p90": float(np.percentile(s64_vals, 90)) if len(s64_vals) else 0.0,
            "mean": float(s64_vals.mean()) if len(s64_vals) else 0.0,
            "min": float(s64_vals.min()) if len(s64_vals) else 0.0,
            "max": float(s64_vals.max()) if len(s64_vals) else 0.0,
        },
        "ncc_32x24": {
            "p10": float(np.percentile(s32_vals, 10)) if len(s32_vals) else 0.0,
            "median": float(np.median(s32_vals)) if len(s32_vals) else 0.0,
            "p90": float(np.percentile(s32_vals, 90)) if len(s32_vals) else 0.0,
            "mean": float(s32_vals.mean()) if len(s32_vals) else 0.0,
        },
        "ant_gate": {
            "enabled": ant_gate is not None,
            "min_ant_score": min_ant_score,
            "min": float(ant_vals.min()) if len(ant_vals) else None,
            "p10": float(np.percentile(ant_vals, 10)) if len(ant_vals) else None,
            "median": float(np.median(ant_vals)) if len(ant_vals) else None,
            "rejected_candidates": clip_rejected_candidates,
            "fallback_frames": ant_fallback_frames,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ant_msg = (
        f", ant_gate median={summary['ant_gate']['median']:.3f} "
        f"(rejected={clip_rejected_candidates}, fallbacks={ant_fallback_frames})"
        if len(ant_vals)
        else ""
    )
    print(
        f"search summary: median NCC(64x48)={summary['ncc_64x48']['median']:.3f}, "
        f"p10={summary['ncc_64x48']['p10']:.3f}, p90={summary['ncc_64x48']['p90']:.3f}, "
        f"sources={dict(source_counts)}, unique_imgs={len(unique_imgs)}{ant_msg}",
        flush=True,
    )
    return summary
