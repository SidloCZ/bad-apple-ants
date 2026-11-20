"""GPU-accelerated 32x24 coarse NCC search, 64x48 refinement, and timeline selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .geometry import build_torch_grid, crop_half_extents, max_fit_crop_w, pose_in_bounds
from .media import extract_clip_frames

ANGLES = [i * (2.0 * math.pi / 16.0) for i in range(16)]
COARSE_SCALES = (0.94, 0.74, 0.55)
MIN_PATCH_STD = 10.0
MIN_FRAME_STD = 2.0
TOPK_COARSE = 32
REFINED_CANDIDATES = 16


def _normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Center each row to zero mean and unit L2 norm; return (normed, std)."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    std = centered.std(dim=-1, unbiased=False)
    norm = centered.norm(dim=-1, keepdim=True).clamp_min(eps)
    return centered / norm, std


def _generate_image_poses(img_idx: int, w: int, h: int) -> np.ndarray:
    """Generate ant-first candidate poses [img_idx, cx_frac, cy_frac, w_frac, theta, 0.0]."""
    poses: list[tuple[float, float, float, float, float, float]] = []
    wf = float(w)
    hf = float(h)
    for theta in ANGLES:
        cw_max = max_fit_crop_w(wf, hf, theta, margin=0.5)
        if cw_max < 48.0:
            continue
        for scale in COARSE_SCALES:
            cw = cw_max * scale
            if cw < 48.0:
                continue
            w_frac = cw / wf
            if scale >= 0.90:
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
            xs = (
                [0.5 * (px_lo + px_hi)]
                if (px_hi - px_lo) < 4.0
                else [px_lo, 0.5 * (px_lo + px_hi), px_hi]
            )
            ys = (
                [0.5 * (py_lo + py_hi)]
                if (py_hi - py_lo) < 4.0
                else [py_lo, 0.5 * (py_lo + py_hi), py_hi]
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


def _sample_patches(
    gray_pack: torch.Tensor,
    dims_gpu: torch.Tensor,
    poses_gpu: torch.Tensor,
    canvas_size: int,
    grid_w: int,
    grid_h: int,
    supersample_2x: bool,
    pose_batch: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample poses in mini-batches and return (normalized_patches, std_per_pose)."""
    total = poses_gpu.shape[0]
    dim = grid_w * grid_h
    device = poses_gpu.device
    out_patches = torch.empty((total, dim), dtype=torch.float32, device=device)
    out_stds = torch.empty((total,), dtype=torch.float32, device=device)

    sw = grid_w * 2 if supersample_2x else grid_w
    sh = grid_h * 2 if supersample_2x else grid_h

    for start in range(0, total, pose_batch):
        end = min(total, start + pose_batch)
        sub = poses_gpu[start:end]
        img_ids = sub[:, 0].long()
        w_vec = dims_gpu[img_ids, 0].float()
        h_vec = dims_gpu[img_ids, 1].float()
        grid = build_torch_grid(sub[:, 1:6], w_vec, h_vec, canvas_size, sw, sh)
        imgs = gray_pack[img_ids].float()
        sampled = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if supersample_2x:
            sampled = F.avg_pool2d(sampled, kernel_size=2, stride=2)
        flat = sampled.view(end - start, dim)
        normed, stds = _normalize_rows(flat)
        out_patches[start:end] = normed
        out_stds[start:end] = stds
    return out_patches, out_stds


def _clamp_poses_batch(
    poses: torch.Tensor,
    dims_gpu: torch.Tensor,
    min_w_px: float = 44.0,
    min_fit_ratio: float = 0.50,
    margin: float = 0.5,
) -> torch.Tensor:
    """Vectorized bounds + ant-first minimum scale clamping on GPU.

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
        clamped = _clamp_poses_batch(expanded, dims_gpu)
        flat_poses = clamped.view(num_cands * 81, 6)
        flat_q_idx = cand_q_idx.unsqueeze(1).expand(num_cands, 81).reshape(-1)

        scores_flat = torch.full((num_cands * 81,), -2.0, dtype=torch.float32, device=device)
        for start in range(0, num_cands * 81, pose_batch):
            end = min(num_cands * 81, start + pose_batch)
            sub_p = flat_poses[start:end]
            sub_q = flat_q_idx[start:end]
            patches, stds = _sample_patches(
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
            dots = torch.where(stds >= MIN_PATCH_STD, dots, torch.full_like(dots, -2.0))
            scores_flat[start:end] = dots

        scores_grid = scores_flat.view(num_cands, 81)
        round_best, round_idx = scores_grid.max(dim=1)
        current = clamped[torch.arange(num_cands, device=device), round_idx]
        best_scores = round_best

    return current, best_scores


def run_search(
    video_path: Path,
    cache_dir: Path,
    output_dir: Path,
    start_s: float = 25.0,
    duration_s: float = 25.0,
    cooldown: int = 15,
    max_reuse: int = 4,
) -> dict:
    """Run coarse 32x24 GPU NCC search, 64x48 refinement, and timeline assignment."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for search")
    device = torch.device("cuda")

    pack_dir = cache_dir / "pack"
    gray_np = np.load(pack_dir / "gray.npy")
    dims_np = np.load(pack_dir / "dims.npy")
    items = json.loads((pack_dir / "items.json").read_text(encoding="utf-8"))
    pack_meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    canvas_size = int(pack_meta["canvas_size"])
    num_images = len(items)

    frames_u8, clip_meta = extract_clip_frames(video_path, start_s, duration_s, cache_dir)
    n_frames = frames_u8.shape[0]

    # Deduplicate identical source frames while keeping frame order mapping.
    hash_to_distinct: dict[str, int] = {}
    distinct_indices: list[int] = []
    frame_to_distinct: list[int] = []
    for i in range(n_frames):
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
        f"({m_tex} textured, {m_distinct - m_tex} solid) against {num_images} images on {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    # Compute image-level stats on real pixels [0:h, 0:w] for solid fallback frames.
    img_means = np.empty((num_images,), dtype=np.float32)
    img_stds = np.empty((num_images,), dtype=np.float32)
    for i in range(num_images):
        w, h = int(dims_np[i, 0]), int(dims_np[i, 1])
        patch = gray_np[i, :h, :w]
        img_means[i] = float(patch.mean())
        img_stds[i] = float(patch.std())

    dark_pool = sorted(
        [i for i in range(num_images) if img_means[i] < 100.0],
        key=lambda i: img_means[i] + 1.5 * img_stds[i],
    ) or sorted(range(num_images), key=lambda i: img_means[i])
    bright_pool = sorted(
        [i for i in range(num_images) if img_means[i] > 155.0],
        key=lambda i: (255.0 - img_means[i]) + 1.5 * img_stds[i],
    ) or sorted(range(num_images), key=lambda i: -img_means[i])

    # Load packed thumbnails and queries onto Quadro T1000.
    gray_pack = torch.from_numpy(gray_np).unsqueeze(1).to(device=device, dtype=torch.uint8)
    dims_gpu = torch.from_numpy(dims_np).to(device=device, dtype=torch.int32)

    refined_per_tex: list[list[dict]] = [[] for _ in range(m_tex)]

    if m_tex > 0:
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

        batch_imgs = 16
        for img_start in tqdm(range(0, num_images, batch_imgs), desc="coarse-32x24", unit="batch"):
            img_end = min(num_images, img_start + batch_imgs)
            pose_chunks: list[np.ndarray] = []
            ranges: list[tuple[int, int]] = []
            cursor = 0
            for idx in range(img_start, img_end):
                arr = _generate_image_poses(idx, int(dims_np[idx, 0]), int(dims_np[idx, 1]))
                if arr.shape[0] > 0:
                    pose_chunks.append(arr)
                    ranges.append((cursor, cursor + arr.shape[0]))
                    cursor += arr.shape[0]
            if not pose_chunks:
                continue

            poses_batch_gpu = torch.from_numpy(np.concatenate(pose_chunks, axis=0)).to(device=device)
            patches_32, stds = _sample_patches(
                gray_pack,
                dims_gpu,
                poses_batch_gpu,
                canvas_size,
                grid_w=32,
                grid_h=24,
                supersample_2x=True,
                pose_batch=1024,
            )
            scores = patches_32 @ q32.T  # (P, 2*M_tex)
            invalid = stds < MIN_PATCH_STD
            if invalid.any():
                scores[invalid] = -2.0

            batch_best_scores = torch.empty((len(ranges), num_q), dtype=torch.float32, device=device)
            batch_best_poses = torch.empty((len(ranges), num_q, 6), dtype=torch.float32, device=device)
            for r_i, (s, e) in enumerate(ranges):
                best_v, best_loc = scores[s:e].max(dim=0)
                batch_best_scores[r_i] = best_v
                batch_best_poses[r_i] = poses_batch_gpu[s + best_loc]

            acc_scores.append(batch_best_scores)
            acc_poses.append(batch_best_poses)

            if len(acc_scores) >= 8 or img_end == num_images:
                cand_s = torch.cat([top_scores, *acc_scores], dim=0)
                cand_p = torch.cat([top_poses, *acc_poses], dim=0)
                new_scores, new_idx = torch.topk(cand_s, k=TOPK_COARSE, dim=0)
                q_arange = torch.arange(num_q, device=device).unsqueeze(0).expand(TOPK_COARSE, num_q)
                top_scores = new_scores
                top_poses = cand_p[new_idx, q_arange]
                acc_scores.clear()
                acc_poses.clear()

        # Merge as-is and mirrored top-32 lists per textured frame, keeping top-16 distinct images.
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
            ref_poses_gpu, ref_scores_gpu = _refine_candidates(
                gray_pack,
                dims_gpu,
                q64,
                cand_poses_gpu,
                cand_qidx_gpu,
                canvas_size,
                rounds=5,
                pose_batch=1024,
            )
            # Also evaluate final 32x24 score of refined poses for transparency.
            patches_ref_32, stds_ref_32 = _sample_patches(
                gray_pack,
                dims_gpu,
                ref_poses_gpu,
                canvas_size,
                grid_w=32,
                grid_h=24,
                supersample_2x=True,
                pose_batch=1024,
            )
            ref_s32 = (patches_ref_32 * q32[cand_qidx_gpu]).sum(dim=-1)
            ref_s32 = torch.where(stds_ref_32 >= MIN_PATCH_STD, ref_s32, torch.full_like(ref_s32, -2.0))

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

    # Sequential timeline assignment with anti-repetition cooldown and max reuse.
    last_used: dict[int, int] = {}
    use_count: dict[int, int] = {}
    frame_records: list[dict] = []

    solid_dark_cursor = 0
    solid_bright_cursor = 0

    for t in range(n_frames):
        d_i = frame_to_distinct[t]
        if t > 0 and d_i == frame_to_distinct[t - 1]:
            prev = dict(frame_records[-1])
            prev["frame_idx"] = t
            prev["reused_from_prev"] = True
            frame_records.append(prev)
            continue

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
            "indexed_images": num_images,
        },
        "frames": frame_records,
    }
    matches_path = output_dir / "matches.json"
    matches_path.write_text(json.dumps(matches_payload, indent=2), encoding="utf-8")

    tex_frames = [fr for fr in frame_records if not fr["is_solid"]]
    s64_vals = np.asarray([fr["score_64"] for fr in tex_frames], dtype=np.float32)
    s32_vals = np.asarray([fr["score_32"] for fr in tex_frames], dtype=np.float32)
    source_counts = Counter(fr["item"]["source"] for fr in frame_records)
    unique_imgs = {fr["item"]["id"] for fr in frame_records}
    unique_species = {fr["item"]["species"] for fr in frame_records if fr["item"].get("species")}

    summary = {
        "indexed_thumbnails": num_images,
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
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"search summary: median NCC(64x48)={summary['ncc_64x48']['median']:.3f}, "
        f"p10={summary['ncc_64x48']['p10']:.3f}, p90={summary['ncc_64x48']['p90']:.3f}, "
        f"sources={dict(source_counts)}, unique_imgs={len(unique_imgs)}",
        flush=True,
    )
    return summary
