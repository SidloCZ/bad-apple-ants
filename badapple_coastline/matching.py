"""Sampled global shape search followed by local FFT translation refinement.

This is a coarse-to-fine search, not a claim of a continuous global optimum.
Every winning window is a single rigid geographic view, with no compositing.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .atlas import Atlas, corners, view_transform
from .data import digest


def feature(mask: np.ndarray, size=(16, 12)) -> np.ndarray:
    mask = gaussian_filter(mask.astype(np.float32), 0.6)
    return np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1)


def build_dictionary(
    atlas: Atlas,
    destination: Path,
    centers: int = 4000,
    widths=(24.0, 48.0, 96.0),
    angles=tuple(range(0, 360, 30)),
    seed: int = 42,
    bounds=None,
) -> dict:
    if centers < 1 or not widths or not angles or min(widths) <= 0:
        raise ValueError("Need positive centers, widths, and at least one angle")
    destination.mkdir(parents=True, exist_ok=True)
    candidates, vectors = [], []
    points = atlas.coastal_centers(centers, seed, bounds)
    for i, (lon, lat) in enumerate(points):
        for width in widths:
            for angle in angles:
                mask, valid = atlas.sample(lon, lat, width, angle)
                if not valid.all():
                    continue
                vectors.append(feature(mask))
                candidates.append([float(lon), float(lat), float(width), float(angle)])
        if (i + 1) % 200 == 0:
            print(f"dictionary: {i+1}/{len(points)} centers, {len(vectors)} windows", flush=True)
    if not vectors:
        raise ValueError("No valid geographic windows")
    np.save(destination / "features.npy", np.stack(vectors), allow_pickle=False)
    np.save(destination / "candidates.npy", np.array(candidates, dtype=np.float64), allow_pickle=False)
    manifest = {
        "schema": 1,
        "atlas": str(atlas.path.resolve()),
        "atlas_sha256": digest(atlas.path / "atlas.json"),
        "global_coverage": atlas.metadata["global_coverage"] and bounds is None,
        "centers": len(points),
        "seed": seed,
        "widths_km": list(widths),
        "angles_clockwise": list(angles),
        "count": len(vectors),
        "feature_size": [16, 12],
        "query_size": [32, 24],
        "search_bounds": bounds,
        "search": "sampled coastal windows, foreground cosine/L2 candidates and local FFT IoU/MSE; approximate spatial search",
        "latitude_range": [-54, 80],
        "white": "land",
        "black": "water",
        "allow_mirror": False,
        "allow_inversion": False,
    }
    (destination / "dictionary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fft_best_translation(scene: np.ndarray, query: np.ndarray, valid: np.ndarray | None = None, metric="mse"):
    """MSE or foreground IoU at every valid translation, including constant frames."""
    scene, query = scene.astype(np.float32), query.astype(np.float32)
    if scene.ndim != 2 or query.ndim != 2 or any(a < b for a, b in zip(scene.shape, query.shape)):
        raise ValueError("Scene must contain the query")
    kernel = np.ones(query.shape, dtype=np.float32)
    correlation = fftconvolve(scene, query[::-1, ::-1], mode="valid")
    if metric == "mse":
        squares = fftconvolve(scene**2, kernel, mode="valid")
        scores = np.maximum(0, (squares + np.sum(query**2) - 2 * correlation) / query.size)
    elif metric == "iou":
        mass = fftconvolve(scene, kernel, mode="valid")
        union = np.maximum(0, mass + query.sum() - correlation)
        similarity = np.divide(np.maximum(0, correlation), union, out=np.ones_like(union), where=union > 1e-5)
        scores = np.clip(1 - similarity, 0, 1)
    else:
        raise ValueError("Metric must be mse or iou")
    if valid is not None:
        covered = fftconvolve(valid.astype(np.float32), kernel, mode="valid")
        scores[covered < query.size - 0.1] = np.inf
    y, x = np.unravel_index(np.argmin(scores), scores.shape)
    return int(x), int(y), float(scores[y, x])


def refine(
    atlas: Atlas, query: np.ndarray, candidate, rotations=(-7.5, 0.0, 7.5), scales=(0.85, 1.0, 1.18), metric="iou"
) -> dict:
    lon, lat, width, angle = map(float, candidate)
    h, w = query.shape
    best = None
    for scale in scales:
        for delta in rotations:
            effective_width, a = width * scale, (angle + delta) % 360
            scene, valid = atlas.sample(lon, lat, effective_width * 3, a, (w * 3, h * 3))
            x, y, score = fft_best_translation(scene, query, valid, metric)
            if not np.isfinite(score) or (best is not None and score >= best["distance"]):
                continue
            transform = view_transform(lon, lat, effective_width * 3, a, (w * 3, h * 3))
            center_lon, center_lat = transform @ (x + w / 2, y + h / 2)
            # Use the same local metric at the shifted center when recording/rendering.
            shifted, shifted_valid = atlas.sample(center_lon, center_lat, effective_width, a, (w, h))
            if not shifted_valid.all():
                continue
            actual_score = float(np.mean((shifted - query) ** 2))
            intersection = float(np.sum(shifted * query))
            union = float(shifted.sum() + query.sum() - intersection)
            iou = intersection / union if union > 1e-5 else 1.0
            distance = 1 - iou if metric == "iou" else actual_score
            if best is None or distance < best["distance"]:
                best = {
                    "longitude": center_lon,
                    "latitude": center_lat,
                    "width_km": effective_width,
                    "height_km": effective_width * h / w,
                    "rotation_clockwise": a,
                    "mse": actual_score,
                    "fft_distance": score,
                    "distance": distance,
                    "metric": metric,
                    "iou": iou,
                }
    if best is None:
        raise ValueError("No covered window during refinement")
    transform = view_transform(
        best["longitude"], best["latitude"], best["width_km"], best["rotation_clockwise"], (w, h)
    )
    best["footprint_lonlat"] = corners(transform, (w, h))
    return best


def _diverse_indices(nearest, candidates, count):
    """Keep different coastal views instead of spending slots on adjacent angles."""
    chosen = []
    for index in nearest:
        lon, lat, width, angle = candidates[index]
        duplicate = False
        for previous in chosen:
            plon, plat, pwidth, pangle = candidates[previous]
            separation = np.hypot((lon - plon) * 111.32 * np.cos(np.deg2rad(lat)), (lat - plat) * 111.32)
            turn = abs((angle - pangle + 180) % 360 - 180)
            if abs(width - pwidth) < 1e-5 and separation < width * 0.2 and turn < 45:
                duplicate = True
                break
        if not duplicate:
            chosen.append(int(index))
        if len(chosen) == count:
            break
    return chosen


def match_queries(
    queries: np.ndarray, directory: Path, top_k: int = 2, progress=None, metric="iou", workers=4, per_scale: int = 0
) -> list[dict]:
    if top_k < 1 or per_scale < 0:
        raise ValueError("top_k must be positive and per_scale nonnegative")
    manifest = json.loads((directory / "dictionary.json").read_text(encoding="utf-8"))
    atlas = Atlas(Path(manifest["atlas"]))
    if digest(atlas.path / "atlas.json") != manifest["atlas_sha256"]:
        raise ValueError("Atlas changed after dictionary construction")
    vectors = np.load(directory / "features.npy", allow_pickle=False)
    candidates = np.load(directory / "candidates.npy", allow_pickle=False)
    # Repeated silhouettes need one database query, even before local refinement.
    unique, inverse = np.unique(queries, axis=0, return_inverse=True)
    features = np.stack([feature(q) for q in unique])
    if metric == "iou":
        # Foreground cosine similarity supplies candidates for FFT IoU refinement.
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    elif metric != "mse":
        raise ValueError("Metric must be mse or iou")
    try:
        import faiss

        faiss.omp_set_num_threads(min(8, __import__("os").cpu_count() or 1))
    except ImportError:
        faiss = None
    groups = (
        [np.flatnonzero(candidates[:, 2] == width) for width in np.unique(candidates[:, 2])]
        if per_scale
        else [np.arange(len(vectors))]
    )
    indices = [[] for _ in unique]
    for group in groups:
        k = min(per_scale * 8 if per_scale else top_k, len(group))
        subset = np.ascontiguousarray(vectors[group])
        if faiss is not None:
            index = faiss.IndexFlatL2(subset.shape[1])
            index.add(subset)
            _, nearest_rows = index.search(features, k)
            del index
        else:
            nearest_rows = []
            norms = np.sum(subset**2, axis=1)
            for q in features:
                distances = norms + np.sum(q**2) - 2 * subset @ q
                nearest = np.argpartition(distances, k - 1)[:k]
                nearest_rows.append(nearest[np.argsort(distances[nearest])])
        for i, row in enumerate(nearest_rows):
            nearest = group[row]
            indices[i].extend(_diverse_indices(nearest, candidates, per_scale) if per_scale else nearest.tolist())
        print(f"search: {len(group)} windows, {len(unique)} unique silhouettes", flush=True)
        del subset
    del vectors
    records, cache, locks, guard = [], {}, {}, Lock()

    def locate(task):
        i, query, nearest = task
        key = query.tobytes()
        with guard:
            lock = locks.setdefault(key, Lock())
        with lock:
            if key not in cache:
                matches = [refine(atlas, query, candidates[j], metric=metric) for j in nearest]
                ordered = sorted(matches, key=lambda r: r["distance"])
                cache[key] = dict(ordered[0], alternatives=ordered[1:])
            return dict(cache[key], frame=i, search_global_coverage=manifest["global_coverage"])

    tasks = ((i, query, indices[inverse[i]]) for i, query in enumerate(queries))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        for i, record in enumerate(pool.map(locate, tasks)):
            records.append(record)
            if progress is not None:
                progress(record)
            if (i + 1) % 100 == 0:
                print(f"match: {i+1}/{len(queries)}, unique masks {len(cache)}", flush=True)
    return records
