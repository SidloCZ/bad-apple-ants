"""Resumable original-mask CUDA dictionary and exact INT8 descriptor search."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from .cuda_backend import CudaAtlas
from .data import digest
from .pipeline import write_json

WIDTHS = [4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288, 384, 576, 768]
ANGLES = np.arange(0, 360, 15, dtype=np.float64)
from .precision_config import full_frames

FRAMES = full_frames()
DICTIONARY = Path("cache/coastline-native-dictionary")


def geometry(indices, centers):
    indices = np.asarray(indices, dtype=np.int64)
    angles = indices % len(ANGLES)
    points = (indices // len(ANGLES)) % len(centers)
    scales = indices // (len(ANGLES) * len(centers))
    return np.column_stack([centers[points], np.asarray(WIDTHS)[scales], ANGLES[angles]])


def build(destination=DICTIONARY, chunk=8192):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    centers = np.fromfile(destination / "centers.f64", dtype="<f8").reshape(-1, 2)
    regions = np.fromfile(destination / "regions.i32", dtype="<i4")
    queries = np.load(FRAMES / "query-features.npy", allow_pickle=False)
    total = len(centers) * len(ANGLES) * len(WIDTHS)
    manifest = dict(
        schema="native-cuda-descriptors-v1",
        count=total,
        centers=len(centers),
        widths_km=WIDTHS,
        angles_clockwise=ANGLES.tolist(),
        feature_size=[64, 48],
        source_samples_per_descriptor=[256, 192],
        source="original 0.3 arcsecond RLE; no global downsampled atlas",
        feature_format="int8 land counts 0..16 from 4x4 exact native samples; index representation only",
        source_aspect_ratio=1080 / 1444,
        source_manifest_sha256=digest(Path("cache/coastline-native-rle/manifest.json")),
        centers_sha256=digest(destination / "centers.f64"),
        queries_sha256=digest(FRAMES / "query-features.npy"),
        rank="cuBLAS exact INT8 dots / INT32 accumulators; normalized two-color correlation, top four per scale/native tile",
        final_evaluation_size=[1444, 1080],
        finite_search=True,
    )
    path = destination / "dictionary.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Dictionary parameters changed")
    write_json(path, manifest)
    if (destination / "complete.json").is_file():
        return
    # Stream descriptor chunks through VRAM; only the regional winners need storage.
    # This also avoids a 75 GB scratch file on hosts with per-file allocation limits.
    results = destination / "regional"
    results.mkdir(exist_ok=True)
    started = time.perf_counter()
    finished = 0
    with CudaAtlas() as cuda:
        cuda.queries(queries)
        for scale, width in enumerate(WIDTHS):
            for region in np.unique(regions):
                result = results / f"{scale:02d}-{region:02d}.npz"
                point_indices = np.flatnonzero(regions == region)
                start = (scale * len(centers) + point_indices[0]) * len(ANGLES)
                stop = (scale * len(centers) + point_indices[-1] + 1) * len(ANGLES)
                if result.is_file():
                    finished += stop - start
                    continue
                cuda.start_region(4)
                for offset in range(start, stop, chunk):
                    end = min(stop, offset + chunk)
                    geo = geometry(np.arange(offset, end), centers)
                    features, ok = cuda.features(geo)
                    cuda.rank(features, ok, offset)
                scores, indices = cuda.region_results()
                temporary = result.with_suffix(".partial.npz")
                np.savez_compressed(temporary, scores=scores, indices=indices)
                temporary.replace(result)
                finished += stop - start
                elapsed = time.perf_counter() - started
                print(
                    f"native CUDA dictionary: {finished:,}/{total:,} views; width {width} km, region {region}; {elapsed:.1f}s",
                    flush=True,
                )
        write_json(
            destination / "complete.json",
            dict(manifest, elapsed_seconds=time.perf_counter() - started, cuda_build=cuda.build),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=8192)
    build(chunk=parser.parse_args().chunk)
