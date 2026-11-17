"""Resumable stages; failed image retrieval never turns into invented terrain."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.error import URLError

import numpy as np
from PIL import Image

from badapple_media import FeatureConfig
from badapple_media import Frames, extract_frames, ffmpeg_binary
from .atlas import Atlas
from .data import MASK_CREDIT, digest
from .matching import match_queries
from .render import render
from .satellite import STAC, fetch_places, fetch_window, nearest_place, resolve_interval, window_key


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_records(path: Path):
    raw = path.read_bytes()
    if path.name.endswith(".partial.jsonl"):
        raw = raw[: raw.rfind(b"\n") + 1]
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def match(
    source: Path,
    dictionary: Path,
    output: Path,
    fps=None,
    max_frames=None,
    start_seconds=0.0,
    duration=None,
    top_k=3,
    metric="iou",
    backend="rust",
    per_scale=4,
):
    output.mkdir(parents=True, exist_ok=True)
    if start_seconds < 0 or (duration is not None and duration <= 0):
        raise ValueError("Start must be nonnegative and duration positive")
    original = source.resolve()
    if start_seconds or duration is not None:
        clip_id = __import__("hashlib").sha256(f"{digest(source)}:{start_seconds}:{duration}".encode()).hexdigest()[:12]
        source = output / f"source-{clip_id}.mp4"
        if not source.exists():
            command = [
                ffmpeg_binary("ffmpeg"),
                "-y",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                str(start_seconds),
                "-i",
                str(original),
            ]
            if duration is not None:
                command += ["-t", str(duration)]
            command += ["-c:v", "libx264", "-crf", "15", "-c:a", "aac", str(source)]
            subprocess.run(command, check=True)
    config = FeatureConfig(feature_width=16, feature_height=12)
    frames = extract_frames(source, Path("cache/coastline-frames"), config, fps, max_frames)
    queries = np.stack(
        [
            np.asarray(
                Image.fromarray(np.uint8(t >= 128) * 255).resize((32, 24), Image.Resampling.LANCZOS), dtype=np.float32
            )
            / 255
            for t in frames.thumbnails()
        ]
    )
    manifest = json.loads((dictionary / "dictionary.json").read_text(encoding="utf-8"))
    settings = {
        "source": str(original),
        "source_sha256": digest(original),
        "video_source": str(source.resolve()),
        "silhouette_threshold": 128,
        "frames_directory": str(frames.directory.resolve()),
        "dictionary": str(dictionary.resolve()),
        "dictionary_sha256": digest(dictionary / "dictionary.json"),
        "fps": str(frames.fps),
        "frames": frames.count,
        "start_seconds": start_seconds,
        "duration": duration,
        "video_source_sha256": digest(source),
        "metric": metric,
        "backend": backend,
        "per_scale": per_scale,
        "top_k": top_k,
        "matcher_schema": "binary128-fft-v1",
        "global_search_coverage": manifest["global_coverage"],
        "search": manifest["search"],
    }
    if backend == "rust":
        from . import rust_backend

        if metric != "iou":
            raise ValueError("Rust matching uses IoU; use --backend python for the MSE reference")
        binary, library = rust_backend.artifacts()
        settings.update(
            matcher_schema="rust-dense-fft-v1",
            rust_binary_sha256=digest(binary),
            rust_library_sha256=digest(library),
            coarse_rotations=rust_backend.COARSE_ROTATIONS,
            coarse_scales=rust_backend.COARSE_SCALES,
        )
    elif backend != "python":
        raise ValueError("Backend must be rust or python")
    settings["match_run_id"] = __import__("hashlib").sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    write_json(output / "run.json", settings)
    partial = output / "matches.partial.jsonl"
    with partial.open("w", encoding="utf-8") as sink:

        def progress(record):
            record["match_run_id"] = settings["match_run_id"]
            record["time"] = record["frame"] / float(frames.fps)
            record["original_time"] = start_seconds + record["time"]
            sink.write(json.dumps(record) + "\n")
            sink.flush()

        if backend == "rust":
            records = rust_backend.match_queries(queries, dictionary, output, per_scale=per_scale, progress=progress)
        else:
            records = match_queries(queries, dictionary, top_k, progress, metric)
    partial.replace(output / "matches.jsonl")
    write_json(
        output / "match-summary.json",
        {
            "frames": len(records),
            "global_search_coverage": manifest["global_coverage"],
            "mean_mse": float(np.mean([r["mse"] for r in records])),
            "max_mse": max(r["mse"] for r in records),
            "metric": metric,
            "mean_iou": float(np.mean([r["iou"] for r in records])),
        },
    )
    return settings


def imagery(
    output: Path,
    cache: Path = Path("cache/coastline-satellite"),
    size=(960, 720),
    interval="auto",
    max_scenes=12,
    workers=4,
):
    if os.name == "nt" and workers > 1 and not os.environ.get("COASTLINE_IMAGE_CHILD"):
        from .process_images import retrieve

        return retrieve(output, cache, size, interval, max_scenes, workers)
    settings = json.loads((output / "run.json").read_text(encoding="utf-8"))
    quality_policy = settings.get("image_quality_policy")
    allow_black = settings.get("black_on_image_failure", False)
    places = Path("assets/coastline/places-10m.geojson")
    fetch_places(places)
    records = read_records(output / "matches.jsonl")
    if any(r.get("match_run_id") != settings["match_run_id"] for r in records):
        raise ValueError("Matches belong to a different source/run")
    failures, complete = [], []
    existing = {}
    for name in ["frames.jsonl", "frames.partial.jsonl"]:
        if (output / name).exists():
            for previous in read_records(output / name):
                satellite = previous.get("satellite", {})
                photo = Path(satellite.get("path", "__missing__"))
                if (
                    previous.get("match_run_id") == settings["match_run_id"]
                    and (satellite.get("stac_api") == STAC or (allow_black and satellite.get("placeholder")))
                    and tuple(satellite.get("size", [])) == tuple(size)
                    and satellite.get("interval") == resolve_interval(previous, interval)
                    and photo.is_file()
                    and digest(photo) == satellite.get("image_sha256")
                    and (satellite.get("coverage") == 1 or (allow_black and satellite.get("placeholder")))
                    and (not quality_policy or satellite.get("quality_policy") == quality_policy)
                ):
                    existing[previous["frame"]] = previous
    locks, guard, unavailable = {}, Lock(), set()

    def retrieve(record):
        if record["frame"] in existing:
            previous = existing[record["frame"]]
            previous["place"] = nearest_place(places, previous["longitude"], previous["latitude"])
            return previous
        alternatives = [{k: v for k, v in record.items() if k != "alternatives"}] + record.get("alternatives", [])
        for rank, candidate in enumerate(alternatives):
            try:
                key = window_key(candidate, size, interval, 25.0)
                with guard:
                    lock = locks.setdefault(key, Lock())
                with lock:
                    if key in unavailable:
                        continue
                    try:
                        extra = (
                            dict(
                                quality_policy=quality_policy,
                                exclude_scene_ids=record.get("repair_exclude_scene_ids", []),
                            )
                            if quality_policy
                            else {}
                        )
                        satellite = fetch_window(candidate, cache, size, interval, max_scenes=max_scenes, **extra)
                    except (RuntimeError, ValueError, URLError, TimeoutError):
                        unavailable.add(key)
                        raise
                success = dict(record, **candidate)
                if rank and "native_mask_path" not in candidate:
                    success.pop("native_mask_path", None)
                    success.pop("native_mask_sha256", None)
                success.pop("alternatives", None)
                success.update(
                    satellite=satellite,
                    imagery_candidate_rank=rank,
                    place=nearest_place(places, candidate["longitude"], candidate["latitude"]),
                )
                if "baseline_native_iou" in success:
                    success["native_iou_gain"] = success["native_iou"] - success["baseline_native_iou"]
                return success
            except (RuntimeError, ValueError, URLError, TimeoutError):
                print(f"imagery: frame {record['frame']}, candidate {rank} unavailable", flush=True)
        if allow_black:
            from .quality import black_placeholder

            satellite = black_placeholder(
                cache,
                size,
                resolve_interval(record, interval),
                "No acceptable image after retrying the recorded candidate views",
            )
            success = {k: v for k, v in record.items() if k != "alternatives"}
            success.update(
                satellite=satellite,
                imagery_candidate_rank=-1,
                place=nearest_place(places, record["longitude"], record["latitude"]),
            )
            return success
        return None

    partial = output / "frames.partial.jsonl"
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool, partial.open("w", encoding="utf-8") as sink:
        for i, success in enumerate(pool.map(retrieve, records)):
            if success is None:
                failures.append(
                    {
                        "frame": i,
                        "reason": "All recorded candidate windows unavailable; see satellite cache failure JSON",
                    }
                )
            else:
                complete.append(success)
                sink.write(json.dumps(success, ensure_ascii=False) + "\n")
                sink.flush()
            if (i + 1) % 10 == 0:
                print(f"imagery: {i+1}/{len(records)}, complete {len(complete)}, failed {len(failures)}", flush=True)
    write_json(
        output / "imagery-status.json",
        {
            "requested": len(records),
            "complete": len(complete),
            "failures": failures,
            "alternative_frames": sum(r["imagery_candidate_rank"] > 0 for r in complete),
            "black_placeholder_frames": sum(bool(r["satellite"].get("placeholder")) for r in complete),
            "actual_mean_mse": float(np.mean([r["mse"] for r in complete])) if complete else None,
            "actual_mean_native_iou": (
                float(np.mean([r["native_iou"] for r in complete]))
                if complete and "native_iou" in complete[0]
                else None
            ),
            "actual_regressed_frames": sum(r.get("native_iou_gain", 0) < -1e-7 for r in complete),
        },
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} frames have no complete real imagery. Rerun imagery to retry; partial frames retained"
        )
    partial.replace(output / "frames.jsonl")
    years = sorted({s["datetime"][:4] for r in complete for s in r["satellite"]["scenes"]})
    credits = (
        MASK_CREDIT + "\nhttps://creativecommons.org/licenses/by/4.0/\n"
        "Mask source: https://doi.org/10.5281/zenodo.14899246\n"
        "Underlying sources acknowledged: ESA WorldCover, OpenStreetMap contributors, Copernicus DEM.\n"
        "© OpenStreetMap contributors, https://www.openstreetmap.org/copyright (ODbL source data).\n"
        "© ESA WorldCover project 2020/2021 / Contains modified Copernicus Sentinel data (2020/2021) processed by ESA WorldCover consortium.\n"
        "https://esa-worldcover.org/en/data-access (CC BY 4.0).\n"
        "Contains modified Copernicus Sentinel data " + ", ".join(years) + "\n"
        "https://cds.climate.copernicus.eu/licences/ec-sentinel\n"
        "Via Earth Search / AWS Open Data. Sentinel-2 Cloud-Optimized GeoTIFFs, https://registry.opendata.aws/sentinel-2-l2a-cogs/\n"
        "Cropped, resampled and rotated; cloud-screened same-location temporal mosaics. Remaining real cloud pixels are quantified per frame.\n"
        "Exact scene IDs, dates, footprints and asset URLs are recorded per frame in frames.jsonl.\n"
        "Places: Natural Earth (public domain), https://www.naturalearthdata.com/about/terms-of-use/\n"
        "Place labels identify the nearest dataset settlement and distance, not an asserted exact locality.\n"
        "Video/audio: original Bad Apple material; respective rights holders retain their rights.\n"
        f"Search coverage global: {settings['global_search_coverage']}. Land/water includes lakes and rivers.\n"
    )
    (output / "CREDITS.txt").write_text(credits, encoding="utf-8")
    return {"frames": len(complete), "years": years}


def render_existing(output: Path, encoder="libx264"):
    settings = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if digest(Path(settings["dictionary"]) / "dictionary.json") != settings["dictionary_sha256"]:
        raise ValueError("Dictionary changed since matching")
    manifest = json.loads((Path(settings["dictionary"]) / "dictionary.json").read_text(encoding="utf-8"))
    frame_dir = Path(settings["frames_directory"])
    frames = Frames(frame_dir, json.loads((frame_dir / "frames.json").read_text(encoding="utf-8")))
    records = read_records(output / "frames.jsonl")
    if any(r.get("match_run_id") != settings["match_run_id"] for r in records):
        raise ValueError("Imagery belongs to an earlier matching run; rerun imagery")
    if digest(Path(settings["video_source"])) != settings["video_source_sha256"]:
        raise ValueError("Video source changed since matching")
    if [r["frame"] for r in records] != list(range(frames.count)):
        raise ValueError("Frame sequence missing, duplicated or reordered")
    atlas = Atlas(Path(manifest["atlas"]))
    geometry = lambda r: (r["longitude"], r["latitude"], r["width_km"], r["rotation_clockwise"])
    windows = list(dict.fromkeys(geometry(r) for r in records))
    ready = {}
    for record in records:
        path = Path(record.get("native_mask_path", "__missing__"))
        if (
            path.is_file()
            and path.stem == atlas.native_mask_key(*geometry(record))
            and digest(path) == record.get("native_mask_sha256")
        ):
            ready[geometry(record)] = path

    def native(window):
        return ready[window] if window in ready else atlas.native_mask(*window)

    masks = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (window, mask) in enumerate(zip(windows, pool.map(native, windows))):
            masks[window] = mask
            if (i + 1) % 100 == 0:
                print(f"native masks: {i+1}/{len(windows)}", flush=True)
    for r in records:
        mask = masks[geometry(r)]
        r["native_mask_path"] = str(mask.resolve())
        r["native_mask_sha256"] = digest(mask)
    if records and "native_iou" in records[0]:
        from .improvement import silhouette, mask_iou

        originals = read_records(Path(settings["baseline"]) / "frames.jsonl")
        thumbs = frames.thumbnails()
        for index, record in enumerate(records):
            target = silhouette(thumbs[index], (260, 195))
            with Image.open(record["native_mask_path"]) as image:
                actual = np.asarray(image.convert("L"), dtype=np.float32) / 255
            iou = mask_iou(actual, target)
            if iou < record["baseline_native_iou"] - 1e-7:
                original = originals[index]
                original_mask = Path(original["native_mask_path"])
                if digest(original_mask) != original["native_mask_sha256"]:
                    raise ValueError("Baseline native mask changed during final verification")
                with Image.open(original_mask) as image:
                    actual = np.asarray(image.convert("L"), dtype=np.float32) / 255
                iou = mask_iou(actual, target)
                record = dict(
                    original,
                    match_run_id=settings["match_run_id"],
                    baseline_native_iou=iou,
                    native_validation_fallback=True,
                    imagery_candidate_rank=0,
                )
                records[index] = record
            record.update(
                native_iou=iou,
                native_mse=float(np.mean((actual - target) ** 2)),
                native_iou_gain=iou - record["baseline_native_iou"],
                native_pixels_verified=True,
            )
            if (index + 1) % 1000 == 0:
                print(f"native pixel verification: {index+1}/{len(records)}", flush=True)
        status_path = output / "imagery-status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            actual_mean_native_iou=float(np.mean([r["native_iou"] for r in records])),
            actual_regressed_frames=sum(r["native_iou_gain"] < -1e-7 for r in records),
            native_pixels_verified=len(records),
        )
        write_json(status_path, status)
    (output / "frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return render(
        Path(settings["video_source"]),
        frames,
        records,
        atlas,
        output / "badapple-coastline.mp4",
        encoder,
        settings.get("image_quality_policy"),
    )
