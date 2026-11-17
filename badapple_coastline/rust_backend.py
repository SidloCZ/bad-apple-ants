"""Rust/Rayon matching; Python retains GDAL I/O and artifact composition."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .atlas import corners, view_transform
from .matching import feature
from .data import digest

ROOT = Path(__file__).resolve().parents[1] / "rust" / "coastline-matcher"
COARSE_ROTATIONS = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
COARSE_SCALES = [2**-0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 2**0.5]
FINE_ROTATIONS = [-3.75, -2.5, -1.25, 0.0, 1.25, 2.5, 3.75]
FINE_SCALES = [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]


def artifacts():
    binary = ROOT / "target" / "release" / ("coastline-search.exe" if os.name == "nt" else "coastline-search")
    library = ROOT / "target" / "release" / ("coastline_matcher.dll" if os.name == "nt" else "libcoastline_matcher.so")
    if not binary.is_file() or not library.is_file():
        raise FileNotFoundError(
            f'Build Rust matching first: cargo build --release --locked --manifest-path {ROOT / "Cargo.toml"}'
        )
    return binary, library


@lru_cache(maxsize=1)
def native_library():
    library = ctypes.CDLL(str(artifacts()[1]))
    function = library.coastline_refine_batch
    size = ctypes.c_size_t
    double = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        size,
        size,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
        size,
        size,
        double,
        size,
        double,
        size,
        double,
        size,
        double,
    ]
    function.restype = ctypes.c_int
    return library


def refine_batch(atlas, query, candidates, rotations=FINE_ROTATIONS, scales=FINE_SCALES):
    query = np.ascontiguousarray(query, dtype=np.float32)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64).reshape(-1, 4)
    rotations = np.ascontiguousarray(rotations, dtype=np.float64)
    scales = np.ascontiguousarray(scales, dtype=np.float64)
    if query.ndim != 2 or min(query.shape) < 1 or not np.isfinite(query).all() or not np.isfinite(candidates).all():
        raise ValueError("Invalid Rust query or candidates")
    if (candidates[:, 2] <= 0).any() or len(rotations) == 0 or len(scales) == 0 or (scales <= 0).any():
        raise ValueError("Invalid scale or rotation search")
    if atlas.land.dtype != np.uint8 or not atlas.land.flags.c_contiguous:
        raise ValueError("Rust atlas must be contiguous uint8")
    output = np.empty((len(candidates), 10), np.float64)
    ptr = lambda array, ctype: array.ctypes.data_as(ctypes.POINTER(ctype))
    status = native_library().coastline_refine_batch(
        ptr(atlas.land, ctypes.c_uint8),
        *atlas.land.shape,
        atlas.resolution,
        ptr(query, ctypes.c_float),
        query.shape[1],
        query.shape[0],
        ptr(candidates, ctypes.c_double),
        len(candidates),
        ptr(rotations, ctypes.c_double),
        len(rotations),
        ptr(scales, ctypes.c_double),
        len(scales),
        ptr(output, ctypes.c_double),
    )
    if status:
        raise RuntimeError(f"Rust refinement failed with status {status}")
    results = []
    keys = (
        "longitude",
        "latitude",
        "width_km",
        "height_km",
        "rotation_clockwise",
        "mse",
        "fft_distance",
        "distance",
        "iou",
    )
    for row in output:
        if row[9] != 1:
            continue
        result = dict(zip(keys, row[:9].tolist()), metric="iou")
        result["footprint_lonlat"] = corners(
            view_transform(
                result["longitude"],
                result["latitude"],
                result["width_km"],
                result["rotation_clockwise"],
                (query.shape[1], query.shape[0]),
            ),
            (query.shape[1], query.shape[0]),
        )
        results.append(result)
    return results


def match_queries(
    queries, directory, output, per_scale=4, progress=None, rotations=COARSE_ROTATIONS, scales=COARSE_SCALES, threads=24
):
    binary, _ = artifacts()
    manifest = json.loads((directory / "dictionary.json").read_text(encoding="utf-8"))
    atlas = Path(manifest["atlas"])
    atlas_metadata = json.loads((atlas / "atlas.json").read_text(encoding="utf-8"))
    if digest(atlas / "atlas.json") != manifest["atlas_sha256"]:
        raise ValueError("Atlas changed after dictionary construction")
    unique, inverse = np.unique(queries, axis=0, return_inverse=True)
    identity = dict(
        dictionary_sha256=digest(directory / "dictionary.json"),
        binary_sha256=digest(binary),
        query_sha256=hashlib.sha256(unique.tobytes()).hexdigest(),
        per_scale=per_scale,
        rotations=list(rotations),
        scales=list(scales),
        feature_schema="gaussian06-bilinear16x12-v1",
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    stage = output / "rust-input" / key
    stage.mkdir(parents=True, exist_ok=True)
    np.save(stage / "queries.npy", unique, allow_pickle=False)
    np.save(stage / "query-features.npy", np.stack([feature(q) for q in unique]), allow_pickle=False)
    unique_path = stage / "unique-matches.jsonl"
    if unique_path.is_file():
        raw = unique_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            unique_path.write_bytes(raw[: raw.rfind(b"\n") + 1])
    config = dict(
        atlas=str((atlas / "land.npy").resolve()),
        resolution=atlas_metadata["resolution_degrees"],
        candidates=str((directory / "candidates.npy").resolve()),
        features=str((directory / "features.npy").resolve()),
        queries=str((stage / "queries.npy").resolve()),
        query_features=str((stage / "query-features.npy").resolve()),
        output=str(unique_path.resolve()),
        metrics=str((output / "rust-metrics.json").resolve()),
        per_scale=per_scale,
        rotations=list(rotations),
        scales=list(scales),
        threads=min(threads, os.cpu_count() or 1),
    )
    config_path = stage / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    subprocess.run([str(binary), str(config_path)], check=True)
    found = {r["frame"]: r for r in (json.loads(line) for line in unique_path.read_text(encoding="utf-8").splitlines())}
    if set(found) != set(range(len(unique))):
        raise RuntimeError("Rust matching did not return every unique silhouette")
    records = []
    for i, index in enumerate(inverse):
        record = dict(found[int(index)], frame=i, search_global_coverage=manifest["global_coverage"])
        records.append(record)
        if progress is not None:
            progress(record)
    return records
