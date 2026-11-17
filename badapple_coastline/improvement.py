"""Wider scale search with a native-mask baseline that can never be discarded."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from badapple_media import Frames
from .atlas import Atlas
from .data import digest
from .matching import match_queries, refine
from .pipeline import read_records, write_json

GEOMETRY = (
    "longitude",
    "latitude",
    "width_km",
    "height_km",
    "rotation_clockwise",
    "mse",
    "fft_distance",
    "distance",
    "metric",
    "iou",
    "footprint_lonlat",
)


def silhouette(thumbnail, size):
    image = Image.fromarray(np.uint8(thumbnail >= 128) * 255)
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255


def mask_iou(mask, target):
    intersection = float(np.sum(mask * target))
    union = float(mask.sum() + target.sum() - intersection)
    return intersection / union if union > 1e-5 else 1.0


def geographic_key(record):
    return tuple(record[k] for k in ("longitude", "latitude", "width_km", "rotation_clockwise"))


def retain_improvements(baseline, candidates, tolerance=1e-7):
    """Every offered satellite fallback must preserve the measured native IoU."""
    seen = {geographic_key(baseline)}
    better = []
    for candidate in sorted(candidates, key=lambda r: -r["native_iou"]):
        key = geographic_key(candidate)
        if candidate["native_iou"] > baseline["native_iou"] + tolerance and key not in seen:
            better.append(candidate)
            seen.add(key)
    return better + [baseline]


def improve(
    baseline: Path,
    dictionary: Path,
    output: Path,
    per_scale=4,
    native_candidates=12,
    refine_finalists=4,
    workers=8,
    backend="rust",
):
    if baseline.resolve() == output.resolve():
        raise ValueError("Improved output must be separate from the baseline")
    if per_scale < 1 or native_candidates < 1 or refine_finalists < 0:
        raise ValueError("Invalid improvement candidate counts")
    output.mkdir(parents=True, exist_ok=True)
    old_settings = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    old = read_records(baseline / "frames.jsonl")
    if [r["frame"] for r in old] != list(range(old_settings["frames"])):
        raise ValueError("Baseline frame sequence incomplete")
    if digest(Path(old_settings["video_source"])) != old_settings["video_source_sha256"]:
        raise ValueError("Baseline source has changed")
    manifest = json.loads((dictionary / "dictionary.json").read_text(encoding="utf-8"))
    atlas = Atlas(Path(manifest["atlas"]))
    if digest(atlas.path / "atlas.json") != manifest["atlas_sha256"]:
        raise ValueError("Atlas changed after dictionary construction")
    directory = Path(old_settings["frames_directory"])
    frames = Frames(directory, json.loads((directory / "frames.json").read_text(encoding="utf-8")))
    thumbs = frames.thumbnails()
    queries = np.stack([silhouette(t, (32, 24)) for t in thumbs])
    settings = dict(
        old_settings,
        dictionary=str(dictionary.resolve()),
        dictionary_sha256=digest(dictionary / "dictionary.json"),
        baseline=str(baseline.resolve()),
        baseline_frames_sha256=digest(baseline / "frames.jsonl"),
        matcher_schema="wide-scale-native-baseline-v2",
        per_scale=per_scale,
        backend=backend,
        native_candidates=native_candidates,
        refine_finalists=refine_finalists,
        native_evaluation_size=[260, 195],
        global_search_coverage=manifest["global_coverage"],
        search=manifest["search"],
        top_k=per_scale * len(manifest["widths_km"]),
    )
    if backend == "rust":
        from . import rust_backend

        binary, library = rust_backend.artifacts()
        settings.update(
            rust_binary_sha256=digest(binary),
            rust_library_sha256=digest(library),
            coarse_rotations=rust_backend.COARSE_ROTATIONS,
            coarse_scales=rust_backend.COARSE_SCALES,
            fine_rotations=rust_backend.FINE_ROTATIONS,
            fine_scales=rust_backend.FINE_SCALES,
        )
    elif backend != "python":
        raise ValueError("Backend must be rust or python")
    settings.pop("match_run_id", None)
    settings["match_run_id"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    run_path = output / "run.json"
    same_run = (
        run_path.is_file()
        and json.loads(run_path.read_text(encoding="utf-8")).get("match_run_id") == settings["match_run_id"]
    )
    if not same_run and (output / "matches.jsonl").exists():
        raise ValueError("Output already contains a different improvement run")
    write_json(run_path, settings)
    coarse_path = output / "coarse-matches.jsonl"
    if same_run and coarse_path.is_file():
        coarse = read_records(coarse_path)
    else:
        partial = output / "coarse-matches.partial.jsonl"
        with partial.open("w", encoding="utf-8") as sink:

            def progress(record):
                sink.write(json.dumps(record) + "\n")
                sink.flush()

            if backend == "rust":
                coarse = rust_backend.match_queries(queries, dictionary, output, per_scale=per_scale, progress=progress)
            else:
                coarse = match_queries(
                    queries, dictionary, settings["top_k"], progress, metric="iou", workers=workers, per_scale=per_scale
                )
        partial.replace(coarse_path)
    if len(coarse) != frames.count:
        raise ValueError("Coarse match count differs from baseline")
    native_cache, cache_lock, query_cache, query_locks, native_locks = {}, Lock(), {}, {}, {}

    def score_native(record, target):
        result = {k: record[k] for k in GEOMETRY if k in record}
        key = geographic_key(record)
        with cache_lock:
            lock = native_locks.setdefault(key, Lock())
        with lock:
            with cache_lock:
                cached = native_cache.get(key)
            if cached is None:
                existing = Path(record.get("native_mask_path", "__missing__"))
                if existing.is_file() and digest(existing) == record.get("native_mask_sha256"):
                    path = existing
                    with Image.open(path) as image:
                        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255
                    checksum = digest(path)
                else:
                    from .native_rle import sample as exact_sample

                    exact = exact_sample(atlas.metadata, *key, (260, 195))
                    if exact is not None:
                        mask = exact.astype(np.float32)
                        path = checksum = None
                    else:
                        path = atlas.native_mask(*key)
                        with Image.open(path) as image:
                            mask = np.asarray(image.convert("L"), dtype=np.float32) / 255
                        checksum = digest(path)
                cached = (mask, str(path.resolve()) if path is not None else None, checksum)
                with cache_lock:
                    native_cache[key] = cached
        mask, path, checksum = cached
        result.update(native_iou=mask_iou(mask, target), native_mse=float(np.mean((mask - target) ** 2)))
        if path is not None:
            result.update(native_mask_path=path, native_mask_sha256=checksum)
        return result

    def persist_winner(record):
        if "native_mask_path" in record:
            return
        key = geographic_key(record)
        with cache_lock:
            lock = native_locks.setdefault(key, Lock())
        with lock:
            mask, path, checksum = native_cache[key]
            if path is None:
                path = str(atlas.native_mask(*key).resolve())
                checksum = digest(Path(path))
                with cache_lock:
                    native_cache[key] = (mask, path, checksum)
            record.update(native_mask_path=path, native_mask_sha256=checksum)

    def improve_frame(index):
        target = silhouette(thumbs[index], (260, 195))
        # Full displayed target and baseline geometry define equivalence, rather
        # than only the smaller retrieval descriptor.
        identity = (target.tobytes(), geographic_key(old[index]))
        with cache_lock:
            lock = query_locks.setdefault(identity, Lock())
        with lock:
            if identity not in query_cache:
                base = score_native(old[index], target)
                candidates = []
                if base["native_iou"] < 1 - 1e-7:
                    views = [coarse[index]] + coarse[index].get("alternatives", [])
                    for candidate in views[:native_candidates]:
                        try:
                            candidates.append(score_native(candidate, target))
                        except ValueError:
                            continue
                    # Refine the original baseline too, so a good existing
                    # location benefits from finer scale and angle sampling.
                    starts = sorted(candidates + [base], key=lambda r: -r["native_iou"])[:refine_finalists]
                    query = silhouette(thumbs[index], (64, 48))
                    if backend == "rust":
                        detailed = rust_backend.refine_batch(atlas, query, [geographic_key(c) for c in starts])
                        for candidate in detailed:
                            try:
                                candidates.append(score_native(candidate, target))
                            except ValueError:
                                continue
                    else:
                        for candidate in starts:
                            seed = geographic_key(candidate)
                            try:
                                detailed = refine(
                                    atlas,
                                    query,
                                    seed,
                                    rotations=(-3.75, 0.0, 3.75),
                                    scales=(0.94, 1.0, 1.06),
                                    metric="iou",
                                )
                                candidates.append(score_native(detailed, target))
                            except ValueError:
                                continue
                selected = retain_improvements(base, candidates)
                query_cache[identity] = (selected, base["native_iou"])
            selected, baseline_iou = query_cache[identity]
        primary = dict(
            selected[0],
            alternatives=selected[1:],
            baseline_native_iou=baseline_iou,
            native_iou_gain=selected[0]["native_iou"] - baseline_iou,
            frame=index,
            time=old[index]["time"],
            original_time=old[index]["original_time"],
            search_global_coverage=manifest["global_coverage"],
            match_run_id=settings["match_run_id"],
        )
        persist_winner(primary)
        return primary

    existing = {}
    for name in ["matches.jsonl", "matches.partial.jsonl"]:
        if same_run and (output / name).is_file():
            for record in read_records(output / name):
                if record.get("match_run_id") == settings["match_run_id"]:
                    existing[record["frame"]] = record

    def resumed(index):
        return existing[index] if index in existing else improve_frame(index)

    partial = output / "matches.partial.jsonl"
    records = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool, partial.open("w", encoding="utf-8") as sink:
        for i, record in enumerate(pool.map(resumed, range(frames.count))):
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if (i + 1) % 100 == 0:
                print(
                    f"native comparison: {i+1}/{frames.count}, mean gain {np.mean([r['native_iou_gain'] for r in records]):.4f}",
                    flush=True,
                )
    partial.replace(output / "matches.jsonl")
    write_json(
        output / "match-summary.json",
        {
            "frames": len(records),
            "metric": "native-mask-iou",
            "global_search_coverage": manifest["global_coverage"],
            "baseline_mean_native_iou": float(np.mean([r["baseline_native_iou"] for r in records])),
            "mean_native_iou": float(np.mean([r["native_iou"] for r in records])),
            "improved_frames": sum(r["native_iou_gain"] > 1e-7 for r in records),
            "regressed_frames": sum(r["native_iou_gain"] < -1e-7 for r in records),
            "mean_iou": float(np.mean([r["iou"] for r in records])),
            "mean_mse": float(np.mean([r["mse"] for r in records])),
        },
    )
    return settings
