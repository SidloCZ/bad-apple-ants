"""Tests for ant-first minimum crop bounds and anti-bokeh sharpness filtering."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFilter

from badapple_ants.ant_gate import ant_probs_from_logits
from badapple_ants.catalog import has_genus_or_species
from badapple_ants.geometry import max_fit_crop_w
from badapple_ants.search import (
    MAX_CENTER_OFFSET,
    MIN_FIT_RATIO,
    MIN_SHARPNESS_RATIO,
    _clamp_poses_batch,
    _coarse_scales,
    _generate_image_poses,
    _refine_chunked,
    _sample_patches,
)


def _make_sharp_and_blurry_pair(canvas_size: int = 192) -> np.ndarray:
    sharp_img = Image.new("L", (canvas_size, canvas_size), color=235)
    draw = ImageDraw.Draw(sharp_img)
    draw.ellipse((44, 52, 92, 96), fill=25)
    draw.ellipse((88, 66, 126, 94), fill=25)
    draw.ellipse((120, 54, 168, 102), fill=25)
    draw.line((76, 86, 62, 134), fill=25, width=4)
    draw.line((104, 86, 108, 136), fill=25, width=4)
    draw.line((132, 86, 152, 132), fill=25, width=4)

    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=16.0))
    return np.stack(
        [np.asarray(sharp_img, dtype=np.uint8), np.asarray(blurry_img, dtype=np.uint8)],
        axis=0,
    )


def test_coarse_scales_helper() -> None:
    assert _coarse_scales(0.65) == (0.94, 0.795, 0.65)
    assert _coarse_scales(0.92) == (0.92,)
    assert _coarse_scales(0.20) == (0.94, 0.72, 0.50)


def test_clamp_poses_enforces_min_fit_ratio() -> None:
    dims = torch.tensor([[192, 144]], dtype=torch.int32)
    for deg in (0.0, 22.5, 67.5, 112.5):
        theta = math.radians(deg)
        poses = torch.tensor([[[0.0, 0.5, 0.5, 0.15, theta, 0.0]]], dtype=torch.float32)
        clamped = _clamp_poses_batch(poses, dims, min_fit_ratio=MIN_FIT_RATIO)
        cw = float(clamped[0, 0, 3]) * 192.0
        cw_max = max_fit_crop_w(192.0, 144.0, theta, margin=0.5)
        assert cw >= cw_max * MIN_FIT_RATIO - 1e-4


def test_sample_patches_sharpness_distinguishes_bokeh() -> None:
    gray_np = _make_sharp_and_blurry_pair()
    gray_pack = torch.from_numpy(gray_np).unsqueeze(1)
    dims = torch.tensor([[192, 192], [192, 192]], dtype=torch.int32)
    poses = torch.tensor(
        [
            [0.0, 0.5, 0.5, 0.90, 0.0, 0.0],
            [1.0, 0.5, 0.5, 0.90, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    _, stds, sharpness = _sample_patches(
        gray_pack,
        dims,
        poses,
        canvas_size=192,
        grid_w=64,
        grid_h=48,
        supersample_2x=False,
    )
    assert float(stds[0]) > 25.0
    assert float(stds[1]) > 25.0
    assert float(sharpness[0]) > 0.35
    assert float(sharpness[1]) < 0.15


def test_refine_chunked_rejects_blurry_bokeh() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for _refine_chunked test")

    device = torch.device("cuda")
    gray_np = _make_sharp_and_blurry_pair()
    dims_gpu = torch.tensor([[192, 192], [192, 192]], dtype=torch.int32, device=device)

    q64_raw = torch.from_numpy(gray_np[0:1].astype(np.float32)).unsqueeze(1).to(device=device)
    q64_resized = torch.nn.functional.interpolate(q64_raw, size=(48, 64), mode="area").view(1, 3072)
    q64 = (q64_resized - q64_resized.mean(dim=-1, keepdim=True))
    q64 = q64 / q64.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    q32_resized = torch.nn.functional.interpolate(q64_raw, size=(24, 32), mode="area").view(1, 768)
    q32 = (q32_resized - q32_resized.mean(dim=-1, keepdim=True))
    q32 = q32 / q32.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cands = torch.tensor(
        [
            [0.0, 0.5, 0.5, 0.90, 0.0, 0.0],
            [1.0, 0.5, 0.5, 0.90, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    qidx = torch.tensor([0, 0], dtype=torch.long, device=device)
    _, s64, _ = _refine_chunked(
        gray_np,
        dims_gpu,
        q64,
        q32,
        cands,
        qidx,
        canvas_size=192,
        min_sharpness=MIN_SHARPNESS_RATIO,
    )
    assert float(s64[0]) > 0.5
    assert float(s64[1]) == pytest.approx(-2.0)


def test_generate_image_poses_respects_center_offset() -> None:
    poses = _generate_image_poses(
        0,
        w=148,
        h=192,
        scales=(0.65,),
        max_center_offset=MAX_CENTER_OFFSET,
    )
    assert poses.shape[0] > 0
    lo = 0.5 - MAX_CENTER_OFFSET - 1e-5
    hi = 0.5 + MAX_CENTER_OFFSET + 1e-5
    assert bool(np.all((poses[:, 1] >= lo) & (poses[:, 1] <= hi)))
    assert bool(np.all((poses[:, 2] >= lo) & (poses[:, 2] <= hi)))


def test_clamp_poses_batch_enforces_center_offset() -> None:
    dims = torch.tensor([[148, 192]], dtype=torch.int32)
    poses = torch.tensor(
        [[[0.0, 0.85, 0.85, 0.70, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    clamped = _clamp_poses_batch(poses, dims, min_fit_ratio=0.65, max_center_offset=0.14)
    assert float(clamped[0, 0, 1]) <= 0.64 + 1e-5
    assert float(clamped[0, 0, 2]) <= 0.64 + 1e-5


def test_ant_probs_from_logits() -> None:
    pos_logits = torch.tensor([[4.0, 3.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0, -3.0]])
    neg_logits = torch.tensor([[-3.0, -3.0, -3.0, -3.0, 4.0, 2.0, 1.0, 1.0, 0.0]])
    pos_prob = float(ant_probs_from_logits(pos_logits, n_pos=4).item())
    neg_prob = float(ant_probs_from_logits(neg_logits, n_pos=4).item())
    assert pos_prob > 0.9
    assert neg_prob < 0.1
    assert 0.0 <= pos_prob <= 1.0
    assert 0.0 <= neg_prob <= 1.0


def test_has_genus_or_species_filter() -> None:
    assert has_genus_or_species({"genus": "", "species": ""}) is False
    assert has_genus_or_species({"genus": "  ", "species": None}) is False
    assert has_genus_or_species({"genus": "Camponotus", "species": ""}) is True
    assert has_genus_or_species({"genus": "", "species": "Lasius niger"}) is True
