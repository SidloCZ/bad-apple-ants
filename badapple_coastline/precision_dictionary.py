"""Stream 1.18 billion original-mask geographic views through CUDA."""

from __future__ import annotations
import argparse, json, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from .cuda_backend import CudaAtlas
from .data import digest
from .pipeline import write_json
from .precision_config import *
from .target_content import extended


def prepare(destination=DICTIONARY):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "centers.json").is_file() and (LEGACY_DICTIONARY / "centers.json").is_file():
        for name in ["centers.f64", "regions.i32", "centers.json"]:
            shutil.copy2(LEGACY_DICTIONARY / name, destination / name)
    if not (destination / "centers.json").is_file():
        subprocess.run(
            [
                "rust/coastline-native-index/target/release/coastline-native-index.exe",
                "cache/coastline-native-rle/manifest.json",
                str(destination),
                str(CENTERS),
                "375",
            ],
            check=True,
        )
    query_path = destination / "queries.npy"
    if not query_path.is_file():
        meta = json.loads((FRAMES / "frames.json").read_text())
        raw = np.memmap(FRAMES / "frames.u8", dtype=np.uint8, mode="r", shape=(meta["count"], 1080, 1444))
        queries = np.empty((len(raw), np.prod(FEATURE_SIZE)), np.int8)
        for i, frame in enumerate(raw):
            image = Image.fromarray(extended(np.uint8(frame >= 128) * 255)).resize(FEATURE_SIZE, Image.Resampling.BOX)
            queries[i] = np.rint(np.asarray(image, dtype=np.float32).ravel() * 16 / 255).astype(np.int8)
        np.save(query_path, queries)
    if not (destination / "queries-mirror.npy").is_file():
        original = np.load(query_path)
        mirrored = original.reshape(-1, FEATURE_SIZE[1], FEATURE_SIZE[0])[:, :, ::-1].reshape(original.shape)
        np.save(destination / "queries-mirror.npy", mirrored)


def build(destination=DICTIONARY, chunk=8192, limit_regions=None, prepare_only=False):
    destination = Path(destination)
    prepare(destination)
    centers = np.fromfile(destination / "centers.f64", dtype="<f8").reshape(-1, 2)
    regions = np.fromfile(destination / "regions.i32", dtype="<i4")
    normal_queries = np.load(destination / "queries.npy")
    query_count = len(normal_queries)
    queries = np.concatenate([normal_queries, np.load(destination / "queries-mirror.npy")])
    total = len(centers) * len(ANGLES) * len(WIDTHS)
    if total >= np.iinfo(np.int32).max:
        raise ValueError("Dictionary exceeds int32 view identifiers")
    manifest = dict(
        schema="precision-native-dictionary-content-v4",
        count=total,
        centers=len(centers),
        widths_km=WIDTHS,
        angles_clockwise=ANGLES.tolist(),
        feature_size=list(FEATURE_SIZE),
        source_samples_per_descriptor=[384, 288],
        source_aspect_ratio=ASPECT,
        source="original 0.3 arcsecond lossless RLE; finite sampled native boundary centers",
        source_manifest_sha256=digest(Path("cache/coastline-native-rle/manifest.json")),
        centers_sha256=digest(destination / "centers.f64"),
        queries_sha256=digest(destination / "queries.npy"),
        regional_top=REGIONAL_TOP,
        final_evaluation_size=[1444, 1080],
        profiles=["area", "centered area plus signed x/y contour gradients"],
        allow_mirror=True,
        count_including_mirrors=total * 2,
        mirror_queries_sha256=digest(destination / "queries-mirror.npy"),
        query_row_layout="normal then horizontal-mirror query",
    )
    manifest["source_video_border_handling"] = (
        "Fixed top/bottom2black rows extended from nearest content row for descriptors only; no source resize or source-data modification"
    )
    path = destination / "dictionary.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous != manifest and (
            any(manifest.get(k) != v for k, v in previous.items()) or (destination / "complete.json").is_file()
        ):
            raise ValueError("Dictionary parameters changed")
    write_json(path, manifest)
    if prepare_only:
        return
    if (destination / "complete.json").is_file():
        return
    results = destination / "regional"
    results.mkdir(exist_ok=True)
    started = time.perf_counter()
    finished = 0
    new = 0
    with CudaAtlas() as cuda, CudaAtlas(load_globe=False) as contours:
        cuda.queries(queries)
        contours.queries(contour_features(queries))
        for scale, width in enumerate(WIDTHS):
            for region in np.unique(regions):
                result = results / f"{scale:02d}-{region:02d}.npz"
                contour_result = results / f"{scale:02d}-{region:02d}-contour.npz"
                mirror_result = results / f"{scale:02d}-{region:02d}-mirror.npz"
                mirror_contour_result = results / f"{scale:02d}-{region:02d}-mirror-contour.npz"
                point_indices = np.flatnonzero(regions == region)
                start = int((scale * len(centers) + point_indices[0]) * len(ANGLES))
                stop = int((scale * len(centers) + point_indices[-1] + 1) * len(ANGLES))
                if all(p.is_file() for p in [result, contour_result, mirror_result, mirror_contour_result]):
                    finished += stop - start
                    continue
                need_area = not (result.is_file() and mirror_result.is_file())
                need_contours = not (contour_result.is_file() and mirror_contour_result.is_file())
                if need_area:
                    cuda.start_region(REGIONAL_TOP)
                if need_contours:
                    contours.start_region(REGIONAL_TOP)
                for offset in range(start, stop, chunk):
                    end = min(stop, offset + chunk)
                    cuda.rank_native_profiles(
                        contours,
                        geometry(np.arange(offset, end), centers),
                        size=FEATURE_SIZE,
                        offset=offset,
                        area=need_area,
                        contours=need_contours,
                    )
                for context, target, mirrored, needed in [
                    (cuda, result, False, need_area),
                    (contours, contour_result, False, need_contours),
                    (cuda, mirror_result, True, need_area),
                    (contours, mirror_contour_result, True, need_contours),
                ]:
                    if needed and not target.is_file():
                        scores, indices = context.region_results()
                        temporary = target.with_suffix(".partial.npz")
                        selected = slice(query_count, None) if mirrored else slice(0, query_count)
                        np.savez_compressed(temporary, scores=scores[selected], indices=indices[selected])
                        temporary.replace(target)
                finished += stop - start
                new += 1
                elapsed = time.perf_counter() - started
                status = dict(
                    finished_views=finished * 2,
                    total_views=total * 2,
                    completed_profile_shards=len(list(results.glob("*.npz"))),
                    scale=scale,
                    width_km=width,
                    region=int(region),
                    elapsed_this_run_seconds=elapsed,
                )
                write_json(destination / "status.json", status)
                print(
                    f"precision dictionary including mirrors: {finished*2:,}/{total*2:,}; width {width:.2f} km; tile {region}; {elapsed:.1f}s",
                    flush=True,
                )
                if limit_regions and new >= limit_regions:
                    return
        write_json(
            destination / "complete.json",
            dict(manifest, elapsed_seconds=time.perf_counter() - started, cuda_build=cuda.build),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--limit-regions", type=int)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    build(chunk=a.chunk, limit_regions=a.limit_regions, prepare_only=a.prepare_only)
