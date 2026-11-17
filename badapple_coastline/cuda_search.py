"""Native-source CUDA search with original-frame final measurements."""

from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from .atlas import view_transform, corners
from .cuda_backend import CudaAtlas
from .cuda_dictionary import DICTIONARY, FRAMES, WIDTHS, geometry
from .data import digest
from .pipeline import read_records, write_json

OUTPUT = Path("output/coastline-fullres-cuda-v2")
BASELINE = Path("output/coastline-wide-rust-landsat")
ASPECT = 1080 / 1444
KEYS = ["longitude", "latitude", "width_km", "rotation_clockwise"]


def diverse(geos, scores, count, overlap=0.25):
    selected = []
    for i in np.argsort(-scores, kind="stable"):
        if not np.isfinite(scores[i]) or scores[i] < 0:
            continue
        lon, lat, width, angle = geos[i]
        unique = True
        for j in selected:
            other = geos[j]
            km = np.hypot((lon - other[0]) * 111.32 * np.cos(np.deg2rad(lat)), (lat - other[1]) * 111.32)
            angle_distance = abs((angle - other[3] + 180) % 360 - 180)
            if km < min(width, other[2]) * overlap and abs(np.log(width / other[2])) < 0.15 and angle_distance < 12:
                unique = False
                break
        if unique:
            selected.append(int(i))
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.int64)


def variations(geos, angles, scales):
    result = np.repeat(geos, len(angles) * len(scales), axis=0)
    result[:, 2] *= np.tile(np.repeat(scales, len(angles)), len(geos))
    result[:, 3] = (result[:, 3] + np.tile(np.tile(angles, len(scales)), len(geos))) % 360
    return result


def refine_batches(cuda, geos, query, margin, batch=128):
    shifted = []
    scores = []
    # Keep batch shapes fixed to reuse FFT plans, including the final short chunk.
    for start in range(0, len(geos), batch):
        current = geos[start : start + batch]
        length = len(current)
        padded = (
            np.concatenate([current, np.repeat(current[-1:], batch - length, axis=0)]) if length < batch else current
        )
        result, values = cuda.refine(padded, query, margin)
        shifted.append(result[:length])
        scores.append(values[:length])
    return np.concatenate(shifted), np.concatenate(scores)


def candidates():
    centers = np.fromfile(DICTIONARY / "centers.f64", dtype="<f8").reshape(-1, 2)
    regions = sorted((DICTIONARY / "regional").glob("*.npz"))
    by_scale = []
    for scale in range(len(WIDTHS)):
        pool = [p for p in regions if p.name.startswith(f"{scale:02d}-")]
        arrays = [np.load(p, allow_pickle=False) for p in pool]
        by_scale.append(
            (
                np.concatenate([a["indices"] for a in arrays], axis=1),
                np.concatenate([a["scores"] for a in arrays], axis=1),
            )
        )
        for a in arrays:
            a.close()
    return centers, by_scale


def select_seeds(frame, centers, by_scale):
    seeds = []
    for indices, scores in by_scale:
        ids = indices[frame]
        values = scores[frame]
        ok = ids >= 0
        if not ok.any():
            continue
        geos = geometry(ids[ok], centers)
        values = values[ok]
        seeds.extend(geos[diverse(geos, values, 4)])
    return np.array(seeds, dtype=np.float64).reshape(-1, 4)


def score_record(geo, metrics, index):
    lon, lat, width, angle = map(float, geo)
    result = dict(zip(KEYS, [lon, lat, width, angle]))
    result.update(
        height_km=width * ASPECT,
        footprint_lonlat=corners(view_transform(*geo, (1444, 1080)), (1444, 1080)),
        metric="0.85 balanced mean IoU + 0.15 boundary F1 (3 original pixels)",
        native_evaluation_size=[1444, 1080],
        native_iou=float(metrics["land_iou"][index]),
        iou=float(metrics["land_iou"][index]),
        native_water_iou=float(metrics["water_iou"][index]),
        native_miou=float(metrics["miou"][index]),
        native_boundary_f1=float(metrics["boundary_f1"][index]),
        native_objective=float(metrics["objective"][index]),
        native_mse=float(metrics["mse"][index]),
        mse=float(metrics["mse"][index]),
        distance=1 - float(metrics["objective"][index]),
        native_counts=metrics["counts"][index].tolist(),
    )
    return result


def run(output=OUTPUT, limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (DICTIONARY / "complete.json").is_file():
        raise ValueError("Native CUDA dictionary is incomplete")
    old = read_records(BASELINE / "frames.jsonl")
    metadata = json.loads((FRAMES / "frames.json").read_text())
    raw = np.memmap(FRAMES / "frames.u8", dtype=np.uint8, mode="r", shape=(metadata["count"], 1080, 1444))
    settings = dict(
        schema="native-fullres-cuda-v2",
        matcher_schema="native-fullres-cuda-v2",
        backend="cuda+rust-rayon",
        source=metadata["source"],
        source_sha256=metadata["source_sha256"],
        video_source=metadata["source"],
        video_source_sha256=metadata["source_sha256"],
        frames=metadata["count"],
        fps=metadata["fps"],
        frames_directory=str(FRAMES.resolve()),
        source_spatial_resize=False,
        native_evaluation_size=[1444, 1080],
        source_aspect_ratio=ASPECT,
        global_search_coverage=True,
        dictionary=str(DICTIONARY.resolve()),
        dictionary_sha256=digest(DICTIONARY / "dictionary.json"),
        baseline=str(BASELINE.resolve()),
        baseline_frames_sha256=digest(BASELINE / "frames.jsonl"),
        objective="0.85 balanced mean IoU + 0.15 boundary F1",
        boundary_tolerance_original_pixels=3,
        per_scale=4,
        index_size=[64, 48],
        coarse_size=[128, 96],
        middle_size=[256, 192],
        full_refinement_size=[1444, 1080],
        search="finite native coastal seeds; INT8 NCC; CUDA FFT translation/rotation/scale refinement; exact original-mask final integer measurements",
        image_quality_policy="no-whiteout-v1",
        silhouette_threshold=128,
        allow_mirror=False,
        allow_inversion=False,
        constant_frame_search="exact native subwindows inside the existing real photograph",
    )
    settings["match_run_id"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    runpath = output / "run.json"
    if runpath.is_file() and json.loads(runpath.read_text())["match_run_id"] != settings["match_run_id"]:
        raise ValueError("Output contains another run")
    write_json(runpath, settings)
    if (output / "matches.jsonl").is_file():
        if len(read_records(output / "matches.jsonl")) != len(old):
            raise ValueError("Incomplete completed matches")
        return
    partial = output / "matches.partial.jsonl"
    previous = read_records(partial) if partial.is_file() else []
    if [r["frame"] for r in previous] != list(range(len(previous))):
        raise ValueError("Invalid partial frame ordering")
    if previous:
        # Drop an interrupted final JSON line before appending.
        partial.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous), encoding="utf-8")
    centers, by_scale = candidates()
    started = time.perf_counter()
    done = len(previous)
    masks = output / "native-masks"
    masks.mkdir(exist_ok=True)
    stop = min(len(old), limit or len(old))
    with CudaAtlas() as cuda, partial.open("a", encoding="utf-8") as sink:
        write_json(output / "cuda-build.json", cuda.build)
        for frame in range(done, stop):
            target = np.uint8(raw[frame] >= 128)
            base = np.array([[old[frame][k] for k in KEYS]], np.float64)
            baseline_metrics = cuda.score(base, target)
            base_record = score_record(base[0], baseline_metrics, 0)
            constant = target.sum() in (0, target.size)
            if constant:
                transform = view_transform(*base[0], (1444, 1080))
                subviews = []
                for y in [0.25, 0.5, 0.75]:
                    for x in [0.25, 0.5, 0.75]:
                        lon, lat = transform @ (x * 1444, y * 1080)
                        subviews.append([lon, lat, base[0, 2] * 0.45, base[0, 3]])
                geos = np.vstack([subviews, base])
                metrics = cuda.score(geos, target)
            elif baseline_metrics["objective"][0] > 0.999999:
                geos = base
                metrics = baseline_metrics
            else:
                small = (
                    np.asarray(Image.fromarray(target * 255).resize((128, 96), Image.Resampling.BOX), np.float32) / 255
                )
                middle = (
                    np.asarray(Image.fromarray(target * 255).resize((256, 192), Image.Resampling.BOX), np.float32) / 255
                )
                seeds = np.vstack([select_seeds(frame, centers, by_scale), base])
                varied = variations(seeds, [-7.5, 0, 7.5], [0.84, 1, 1.19])
                geo, values = refine_batches(cuda, varied, small, (128, 96), 128)
                winners = geo[diverse(geo, values, 12)]
                varied = variations(winners, [-3.75, 0, 3.75], [0.925, 1, 1.08])
                geo, values = refine_batches(cuda, varied, middle, (48, 36), 32)
                winners = geo[diverse(geo, values, 4)]
                # From this point every search query retains all 1,559,520 original pixels.
                varied = variations(np.vstack([winners, base]), [-1.25, 0, 1.25], [0.975, 1, 1.025])
                geo, values = refine_batches(cuda, varied, target, (24, 18), 9)
                exact = cuda.score(geo, target)
                first = geo[diverse(geo, exact["objective"], 3, overlap=0.1)]
                varied = variations(first, [-0.4, 0, 0.4], [0.992, 1, 1.008])
                second, _ = refine_batches(cuda, varied, target, (8, 6), 9)
                exact_second = cuda.score(second, target)
                fine = second[diverse(second, exact_second["objective"], 2, overlap=0.05)]
                varied = variations(fine, [-0.125, 0, 0.125], [0.9975, 1, 1.0025])
                third, _ = refine_batches(cuda, varied, target, (3, 3), 9)
                geos = np.vstack([geo, second, third, base])
                metrics = cuda.score(geos, target)
            chosen = diverse(geos, metrics["objective"], 5, overlap=0.05)
            options = [
                score_record(geos[i], metrics, int(i))
                for i in chosen
                if metrics["objective"][i] >= base_record["native_objective"] - 1e-12
            ]
            if constant:
                for o in options:
                    o["baseline_crop_frame"] = frame
            if not options:
                options = [base_record]
            # Keep a guaranteed real-photo baseline, measured under the new full-size metric.
            baseline_option = dict(base_record, baseline_photo_frame=frame)
            if all(any(abs(o[k] - baseline_option[k]) > 1e-12 for k in KEYS) for o in options):
                options.append(baseline_option)
            else:
                for o in options:
                    if all(o[k] == baseline_option[k] for k in KEYS):
                        o["baseline_photo_frame"] = frame
            winner = dict(options[0])
            geo = np.array([[winner[k] for k in KEYS]])
            mask = cuda.sample(geo, (1444, 1080))[0]
            if (mask > 1).any():
                raise ValueError("Selected mask has missing native pixels")
            mask_path = masks / f"{frame:05d}.png"
            Image.fromarray(mask * 255).save(mask_path)
            winner.update(
                frame=frame,
                time=frame / (30000 / 1001),
                match_run_id=settings["match_run_id"],
                native_mask_path=str(mask_path.resolve()),
                native_mask_sha256=digest(mask_path),
                baseline_fullres_metrics=base_record,
                native_objective_gain=winner["native_objective"] - base_record["native_objective"],
                alternatives=options[1:],
                native_pixels_verified=True,
            )
            sink.write(json.dumps(winner, ensure_ascii=False) + "\n")
            sink.flush()
            if (frame + 1) % 10 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f'full-resolution CUDA search: {frame+1}/{stop}; {elapsed:.1f}s; {elapsed/(frame+1-done):.3f}s/frame; mIoU {winner["native_miou"]:.4f}; gain {winner["native_objective_gain"]:.4f}',
                    flush=True,
                )
    if stop == len(old):
        partial.replace(output / "matches.jsonl")
        records = read_records(output / "matches.jsonl")
        write_json(
            output / "search-summary.json",
            dict(
                frames=len(records),
                elapsed_seconds=time.perf_counter() - started,
                baseline_mean_miou=float(np.mean([r["baseline_fullres_metrics"]["native_miou"] for r in records])),
                mean_miou=float(np.mean([r["native_miou"] for r in records])),
                mean_boundary_f1=float(np.mean([r["native_boundary_f1"] for r in records])),
                improved_frames=sum(r["native_objective_gain"] > 1e-7 for r in records),
                regressed_frames=sum(r["native_objective_gain"] < -1e-7 for r in records),
                evaluation_size=[1444, 1080],
                cuda_build=cuda.build,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.output, args.limit)
