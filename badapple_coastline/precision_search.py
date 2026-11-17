"""Deep CUDA search, soft contours, multiple translations and temporal reuse."""

from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from .cuda_backend import CudaAtlas, shape_metrics_from_counts, structure_metrics_from_counts
from .cuda_search import variations, score_record
from .data import digest
from .atlas import view_transform, corners
from .pipeline import read_records, write_json
from .precision_config import *
from .target_content import content, extended, border_rows


def ranking(metrics):
    return metrics.get("structure_objective", metrics["precision_objective"])


def record_score(record):
    return record.get("structure_objective", record.get("precision_objective", record["native_objective"]))


def diverse(geos, scores, count, overlap=0.12, angle_limit=5.0, scale_limit=0.06):
    """Vectorized suppression preserves geographically distinct candidates."""
    if len(geos) == 0:
        return np.empty(0, np.int64)
    scores = np.asarray(scores).copy()
    scores[~np.isfinite(scores)] = -np.inf
    selected = []
    while len(selected) < count:
        i = int(np.argmax(scores))
        if scores[i] < 0:
            break
        selected.append(i)
        lon, lat, width, angle = geos[i, :4]
        delta = (geos[:, 0] - lon + 180) % 360 - 180
        distance = np.hypot(delta * 111.32 * np.cos(np.deg2rad(lat)), (geos[:, 1] - lat) * 111.32)
        duplicate = (
            (distance < np.minimum(width, geos[:, 2]) * overlap)
            & (np.abs(np.log(geos[:, 2] / width)) < scale_limit)
            & (np.abs((geos[:, 3] - angle + 180) % 360 - 180) < angle_limit)
        )
        if geos.shape[1] == 5:
            duplicate &= geos[:, 4] == geos[i, 4]
        scores[duplicate] = -np.inf
        scores[i] = -np.inf
    return np.asarray(selected, np.int64)


class FrameScoreCache:
    """Reuse exact integer counts within one immutable full-size target.

    Geometry keys keep the full floating-point values and mirror flag. No
    coordinate rounding, candidate suppression or approximate scores are used.
    """

    def __init__(self, target, structure=False):
        self.target = np.ascontiguousarray(target, dtype=np.uint8).copy()
        self.structure = structure
        self.border = border_rows(target.shape[0]) if structure else 0
        self.query_sum = int(content(self.target, self.border).sum())
        self.rows = {}

    def score(self, cuda, geos, target, batch):
        if target.shape != self.target.shape or not np.array_equal(target, self.target):
            raise ValueError("Frame score cache belongs to another target")
        geometry, source, mirrors = CudaAtlas.split_geometry(geos)
        keys = [tuple(row) + (int(mirror),) for row, mirror in zip(source, mirrors)]
        missing = {key: i for i, key in enumerate(keys) if key not in self.rows}
        indices = list(missing.values())
        tolerance = max(1, round(3 * target.shape[1] / 1444))
        for start in range(0, len(indices), batch):
            selected = indices[start : start + batch]
            if self.structure:
                metrics = cuda.score_structure(
                    geometry[selected],
                    self.target,
                    tolerance=tolerance,
                    border_y=self.border,
                    edge_weight=FRAME_EDGE_WEIGHT,
                )
            else:
                metrics = cuda.score_shape(geometry[selected], self.target, tolerance=tolerance)
            for i, row in zip(selected, metrics["structure_counts" if self.structure else "shape_counts"]):
                self.rows[keys[i]] = row.copy()
        counts = np.array([self.rows[key] for key in keys], dtype=np.uint64).reshape(-1, 17 if self.structure else 13)
        if self.structure:
            return structure_metrics_from_counts(
                counts, self.query_sum, target.shape[1], target.shape[0], self.border, FRAME_EDGE_WEIGHT
            )
        return shape_metrics_from_counts(counts, self.query_sum, self.target.size)


def score_batches(cuda, geos, target, batch=8, cache=None, structure=False):
    if cache is not None:
        return cache.score(cuda, geos, target, batch)
    tolerance = max(1, round(3 * target.shape[1] / 1444))
    if structure:
        groups = [
            cuda.score_structure(
                geos[i : i + batch],
                target,
                tolerance=tolerance,
                border_y=border_rows(target.shape[0]),
                edge_weight=FRAME_EDGE_WEIGHT,
            )
            for i in range(0, len(geos), batch)
        ]
    else:
        groups = [
            cuda.score_shape(geos[i : i + batch], target, tolerance=tolerance) for i in range(0, len(geos), batch)
        ]
    return {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}


def refine_batches(cuda, geos, query, margin, batch, top_k=1, separation=3):
    shifted = []
    scores = []
    for start in range(0, len(geos), batch):
        current = geos[start : start + batch]
        length = len(current)
        padded = (
            np.concatenate([current, np.repeat(current[-1:], batch - length, axis=0)]) if length < batch else current
        )
        if top_k == 1:
            geo, values = cuda.refine(padded, query, margin)
        else:
            geo, values = cuda.refine_topk(padded, query, margin, top_k, separation)
        shifted.append(geo[: length * top_k])
        scores.append(values[: length * top_k])
    geos = np.concatenate(shifted)
    values = np.concatenate(scores)
    valid = values >= 0
    return geos[valid], values[valid]


def image_query(target, size):
    return np.asarray(Image.fromarray(extended(target) * 255).resize(size, Image.Resampling.BOX), np.float32) / 255


def precise_record(geo, metrics, index):
    result = score_record(geo[:4], metrics, index)
    mirrored = bool(geo[4]) if len(geo) == 5 else False
    result.update(
        mirror_horizontal=mirrored,
        footprint_lonlat=corners(view_transform(*geo[:4], (1444, 1080), mirror_horizontal=mirrored), (1444, 1080)),
    )
    result.update(
        precision_objective=float(metrics["precision_objective"][index]),
        soft_contour_f1=float(metrics["soft_contour_f1"][index]),
        soft_contour_f1_8=float(metrics["soft_contour_f1_8"][index]),
        soft_contour_f1_32=float(metrics["soft_contour_f1_32"][index]),
        precision_counts=metrics["shape_counts"][index].tolist(),
        selection_metric="0.70 balanced mIoU + 0.25 soft contour F1 (L1, 8/32 original pixels) + 0.05 boundary F1",
    )
    if "structure_objective" in metrics:
        border = int(metrics["content_border_y"][index])
        pixels = int(metrics["evaluated_pixels"][index])
        result.update(
            structure_objective=float(metrics["structure_objective"][index]),
            frame_edge_error=float(metrics["frame_edge_error"][index]),
            frame_edge_errors=metrics["frame_edge_errors"][index].tolist(),
            structure_counts=metrics["structure_counts"][index].tolist(),
            content_border_y=border,
            evaluated_pixels=pixels,
            native_content_bounds=[0, border, 1444, 1080 - border],
            selection_metric="content ROI: precision objective minus0.10 times maximum mismatch fraction across four content-frame borders",
        )
    return result


def admissible(metrics, baseline):
    valid = (
        (metrics["miou"] >= baseline["native_miou"] - 1e-12)
        & (metrics["objective"] >= baseline["native_objective"] - 1e-12)
        & (metrics["precision_objective"] >= baseline["precision_objective"] - 1e-12)
    )
    if "structure_objective" in metrics:
        valid &= metrics["structure_objective"] >= baseline["structure_objective"] - 1e-12
    return valid


def vector(record):
    return [record.get(k, False) if k == "mirror_horizontal" else record[k] for k in KEYS]


def geo_key(record):
    return tuple(round(float(v), 10) for v in vector(record))


def both_handedness(geos):
    flipped = geos.copy()
    flipped[:, 4] = 1 - flipped[:, 4]
    return np.vstack([geos, flipped])


def unique_geos(geos):
    # Keep the first occurrence/tie order while avoiding repeated FFTs of the
    # identical view returned by different descriptor profiles or neighbors.
    _, first = np.unique(geos, axis=0, return_index=True)
    return geos[np.sort(first)]


def polish(cuda, seeds, target, baseline, full_count=64, cache=None):
    """Several original-size stages; keep more than one translation maximum."""
    all_geo = []
    stages = [
        ([-0.625, 0, 0.625], [0.985, 1, 1.015], (24, 18), 3, full_count),
        ([-0.25, 0, 0.25], [0.995, 1, 1.005], (10, 8), 2, max(16, full_count // 2)),
        ([-0.10, 0, 0.10], [0.998, 1, 1.002], (4, 4), 2, max(8, full_count // 4)),
        ([-0.04, 0, 0.04], [0.9992, 1, 1.0008], (2, 2), 1, max(4, full_count // 8)),
    ]
    winners = seeds
    for angles, scales, margin, top_k, count in stages:
        varied = variations(winners, angles, scales)
        geos, values = refine_batches(cuda, varied, extended(target), margin, 6, top_k=top_k, separation=2)
        all_geo.append(geos)
        metrics = score_batches(cuda, geos, target, cache=cache, structure=True)
        # Keep candidates even if they do not yet pass the baseline floor:
        # further alignment can improve them. Final acceptance has all guards.
        chosen = diverse(geos, ranking(metrics), count, overlap=0.025, angle_limit=0.25, scale_limit=0.012)
        winners = geos[chosen]
    return np.vstack(all_geo + [seeds])


def legacy_available():
    """The optional v3 dictionary whose candidates the published v4 search also used."""
    return (LEGACY_DICTIONARY / "complete.json").is_file() or (LEGACY_DICTIONARY / "dictionary.json").is_file()


class Search:
    def __init__(self, output=OUTPUT):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.base = read_records(BASELINE / "frames.jsonl")
        self.prior = read_records(BASELINE / "matches.jsonl")
        preceding = LEGACY_OUTPUT / "global.partial.jsonl"
        self.preceding = {r["frame"]: r for r in read_records(preceding)} if preceding.is_file() else {}
        water = next(vector(r) for r in self.base if r["native_counts"][0] == 0 and r["native_counts"][2] == 0)
        self.constant_geos = {
            0: np.array([water]),
            1: np.array(
                [[-62, -4.5, 2, 0, 0], [134, -25, 2, 0, 0], [20, 22, 2, 0, 0], [15, 0, 2, 0, 0], [25, -10, 2, 0, 0]],
                np.float64,
            ),
        }
        metadata = json.loads((FRAMES / "frames.json").read_text())
        self.metadata = metadata
        self.raw = np.memmap(FRAMES / "frames.u8", dtype=np.uint8, mode="r", shape=(len(self.base), 1080, 1444))
        self.features = np.load(DICTIONARY / "queries.npy").astype(np.float32)
        self.features -= self.features.mean(axis=1, keepdims=True)
        self.features /= np.maximum(np.linalg.norm(self.features, axis=1, keepdims=True), 1e-6)
        self.photos = {geo_key(r): r["frame"] for r in self.base}
        for r in self.base:
            self.photos[geo_key(dict(r, mirror_horizontal=True))] = r["frame"]
        settings = dict(
            schema="coastline-content-precision-v4",
            matcher_schema="coastline-content-precision-v4",
            backend="cuda+rust-rayon",
            video_source=metadata["source"],
            video_source_sha256=metadata["source_sha256"],
            frames=len(self.base),
            fps=metadata["fps"],
            frames_directory=str(FRAMES.resolve()),
            source_spatial_resize=False,
            native_evaluation_size=[1444, 1080],
            source_aspect_ratio=ASPECT,
            global_search_coverage=True,
            dictionary=str(DICTIONARY.resolve()),
            dictionary_sha256=digest(DICTIONARY / "dictionary.json"),
            additional_dictionary=str(LEGACY_DICTIONARY.resolve()) if legacy_available() else None,
            additional_dictionary_sha256=digest(LEGACY_DICTIONARY / "dictionary.json") if legacy_available() else None,
            preceding_geometry_seeds_sha256=digest(preceding) if preceding.is_file() else None,
            baseline=str(BASELINE.resolve()),
            baseline_frames_sha256=digest(BASELINE / "frames.jsonl"),
            image_quality_policy="no-whiteout-v1",
            objective="0.85 balanced mean IoU + 0.15 boundary F1",
            precision_objective="0.70 mIoU + 0.25 soft contour F1 + 0.05 boundary F1",
            soft_contour="exact truncated L1 contour distance, integer exponential weights; taus8/32 original pixels",
            boundary_tolerance_original_pixels=3,
            structure_objective="precision objective -0.10*maximum content-frame border mismatch fraction",
            content_border_y=CONTENT_BORDER_Y,
            content_bounds=[0, 2, 1444, 1078],
            evaluated_pixels=1444 * 1076,
            excluded_presentation_pixels=1444 * 4,
            source_frame_modified=False,
            target_descriptor_padding="extend nearest content row through fixed top/bottom2black presentation rows; source frames unchanged",
            constant_candidate_geometries={str(k): v.tolist() for k, v in self.constant_geos.items()},
            acceptance="no decrease in freshly measured baseline content mIoU, legacy objective, precision objective or structure objective",
            historical_baseline_full_frame_mean_miou=float(np.mean([r["native_miou"] for r in self.base])),
            per_scale=PER_SCALE,
            additional_per_scale=LEGACY_PER_SCALE if legacy_available() else None,
            index_size=list(FEATURE_SIZE),
            coarse_size=[192, 144],
            middle_size=[512, 384],
            global_coarse_survivors=192,
            full_refinement_survivors=64,
            full_refinement_size=[1444, 1080],
            fft_translation_maxima=3,
            temporal_passes=2,
            allow_mirror=True,
            allow_inversion=False,
            silhouette_threshold=128,
            search="finite native geographic search with expanded seeds/scales/angles, multistage CUDA FFT, exact original-mask scoring and temporal candidate propagation",
        )
        settings["match_run_id"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self.settings = settings
        path = self.output / "run.json"
        if path.is_file() and json.loads(path.read_text()) != settings:
            raise ValueError("Output belongs to different search settings")
        write_json(path, settings)

    def load_dictionary(self):
        self.centers = np.fromfile(DICTIONARY / "centers.f64", dtype="<f8").reshape(-1, 2)
        self.by_scale = []
        sources = [(DICTIONARY, PER_SCALE // 4)]
        if legacy_available():
            # The published run also merged the regional top lists of the previous (v3) dictionary.
            current = json.loads((DICTIONARY / "dictionary.json").read_text())
            legacy = json.loads((LEGACY_DICTIONARY / "dictionary.json").read_text())
            for key in ["centers_sha256", "widths_km", "angles_clockwise", "feature_size", "source_manifest_sha256"]:
                if current[key] != legacy[key]:
                    raise ValueError("Incompatible source dictionary geometry")
            sources.append((LEGACY_DICTIONARY, LEGACY_PER_SCALE // 4))
        for scale in range(len(WIDTHS)):
            profiles = []
            for directory, count in sources:
                for pattern, mirrored in [
                    (f"{scale:02d}-[0-9][0-9].npz", False),
                    (f"{scale:02d}-[0-9][0-9]-contour.npz", False),
                    (f"{scale:02d}-[0-9][0-9]-mirror.npz", True),
                    (f"{scale:02d}-[0-9][0-9]-mirror-contour.npz", True),
                ]:
                    arrays = [np.load(p) for p in sorted((directory / "regional").glob(pattern))]
                    profiles.append(
                        (
                            np.concatenate([a["indices"] for a in arrays], axis=1),
                            np.concatenate([a["scores"] for a in arrays], axis=1),
                            mirrored,
                            count,
                        )
                    )
                    for a in arrays:
                        a.close()
            self.by_scale.append(profiles)

    def global_seeds(self, frame):
        selected = []
        for profiles in self.by_scale:
            for ids, scores, mirrored, count in profiles:
                valid = ids[frame] >= 0
                geos = geometry(ids[frame, valid], self.centers)
                values = scores[frame, valid]
                geos = np.column_stack([geos, np.full(len(geos), int(mirrored))])
                selected.extend(geos[diverse(geos, values, count, overlap=0.20, angle_limit=10, scale_limit=0.1)])
        return np.asarray(selected, dtype=np.float64).reshape(-1, 5)

    def neighbor_seeds(self, frame, records, offsets=(1, 2, 4, 8, 16, 32, 64)):
        rows = []
        for distance in offsets:
            for j in [frame - distance, frame + distance]:
                if 0 <= j < len(self.base) and self.features[frame] @ self.features[j] > 0.60:
                    r = records[j]
                    rows.append(vector(r))
                    for a in r.get("alternatives", [])[:2]:
                        rows.append(vector(a))
        return np.array(rows, dtype=np.float64).reshape(-1, 5)

    def finish(self, cuda, frame, geos, target, baseline, old=None, phase="global", cache=None):
        base_geo = np.array([vector(self.base[frame])])
        geos = np.vstack([geos, base_geo])
        geos = np.unique(geos, axis=0)
        metrics = score_batches(cuda, geos, target, cache=cache, structure=True)
        values = ranking(metrics).copy()
        values[~admissible(metrics, baseline)] = -1
        tight = diverse(geos, values, 8, overlap=0.03, angle_limit=0.3, scale_limit=0.015)
        broad = diverse(geos, values, 16, overlap=0.30, angle_limit=45, scale_limit=0.4)
        selected = sorted(set(tight.tolist() + broad.tolist()), key=lambda i: -values[i])
        options = [precise_record(geos[i], metrics, int(i)) for i in selected]
        if not options:
            options = [dict(baseline)]
        for option in options:
            if geo_key(option) in self.photos:
                option["baseline_photo_frame"] = self.photos[geo_key(option)]
        base_option = dict(baseline, baseline_photo_frame=frame)
        if all(geo_key(o) != geo_key(base_option) for o in options):
            options.append(base_option)
        if old is not None and record_score(options[0]) < record_score(old) - 1e-12:
            raise ValueError("Temporal pass regressed")
        winner = dict(options[0])
        mask = cuda.sample([vector(winner)], (1444, 1080))[0]
        if (mask > 1).any():
            raise ValueError("Chosen native mask is incomplete")
        directory = self.output / "native-masks" / phase
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{frame:05d}.png"
        Image.fromarray(mask * 255).save(path)
        # Retain evidence of the metric tradeoff without changing acceptance.
        # This lets us inspect whether the conservative legacy/mIoU floors
        # suppress a visibly better contour once real production frames exist.
        best = int(np.argmax(ranking(metrics)))
        diagnostic = None
        if ranking(metrics)[best] > record_score(winner) + 1e-6:
            diagnostic = precise_record(geos[best], metrics, best)
            extra = self.output / "native-masks" / "metric-tradeoff" / phase
            extra.mkdir(parents=True, exist_ok=True)
            extra_path = extra / f"{frame:05d}.png"
            extra_mask = cuda.sample(geos[best : best + 1], (1444, 1080))[0]
            Image.fromarray(extra_mask * 255).save(extra_path)
            diagnostic.update(
                native_mask_path=str(extra_path.resolve()),
                native_mask_sha256=digest(extra_path),
                miou_difference_from_selected=diagnostic["native_miou"] - winner["native_miou"],
                note="Diagnostic only; not selected because one or more baseline acceptance floors failed",
            )
        winner.update(
            frame=frame,
            time=frame * 1001 / 30000,
            match_run_id=self.settings["match_run_id"],
            native_mask_path=str(path.resolve()),
            native_mask_sha256=digest(path),
            native_pixels_verified=True,
            baseline_fullres_metrics=baseline,
            native_objective_gain=winner["native_objective"] - baseline["native_objective"],
            precision_objective_gain=winner["precision_objective"] - baseline["precision_objective"],
            alternatives=options[1:],
            precision_phase=phase,
        )
        if diagnostic:
            winner["metric_tradeoff_diagnostic"] = diagnostic
        if "structure_objective" in winner:
            winner["structure_objective_gain"] = winner["structure_objective"] - baseline["structure_objective"]
        return winner

    def global_frame(self, cuda, frame, recent):
        target = np.uint8(self.raw[frame] >= 128)
        base = np.array([vector(self.base[frame])])
        cache = FrameScoreCache(target, structure=True)
        metrics = score_batches(cuda, base, target, cache=cache, structure=True)
        baseline = precise_record(base[0], metrics, 0)
        region = content(target)
        if region.sum() in (0, region.size):
            candidates = np.vstack([base, self.constant_geos[int(region[0, 0])]])
            result = self.finish(cuda, frame, candidates, target, baseline, cache=cache)
            if result["native_miou"] < 1 - 1e-12:
                raise ValueError("No exact real-land/water view for constant source content")
            result["constant_content"] = True
            return result
        if metrics["structure_objective"][0] > 0.999999:
            return self.finish(cuda, frame, base, target, baseline, cache=cache)
        prior = np.array([vector(r) for r in [self.prior[frame]] + self.prior[frame].get("alternatives", [])[:4]])
        if frame in self.preceding:
            p = self.preceding[frame]
            prior = np.vstack([prior, np.array([vector(r) for r in [p] + p.get("alternatives", [])[:4]])])
        neighbors = self.neighbor_seeds(frame, recent, offsets=(1, 2, 4, 8, 16))
        seeds = unique_geos(np.vstack([self.global_seeds(frame), both_handedness(np.vstack([base, prior, neighbors]))]))
        small = image_query(target, (192, 144))
        middle = image_query(target, (512, 384))
        varied = variations(seeds, [-2.5, 0, 2.5], [0.89, 0.945, 1, 1.058, 1.12])
        geo, values = refine_batches(cuda, varied, small, (128, 96), 96, top_k=2, separation=12)
        winners = geo[diverse(geo, values, 192, overlap=0.10, angle_limit=5.0, scale_limit=0.06)]
        varied = variations(winners, [-1.25, 0, 1.25], [0.97, 1, 1.03])
        geo, values = refine_batches(cuda, varied, middle, (72, 54), 16, top_k=2, separation=8)
        binary = np.uint8(middle >= 0.5)
        metrics = score_batches(cuda, geo, binary, 48, structure=True)
        winners = geo[diverse(geo, ranking(metrics), 64, overlap=0.06, angle_limit=2.0, scale_limit=0.025)]
        geos = polish(cuda, np.vstack([winners, base, prior]), target, baseline, cache=cache)
        return self.finish(cuda, frame, geos, target, baseline, cache=cache)

    def temporal_frame(self, cuda, frame, records, phase):
        old = records[frame]
        target = np.uint8(self.raw[frame] >= 128)
        cache = FrameScoreCache(target, structure=True)
        current = np.array([vector(r) for r in [old] + old.get("alternatives", [])[:7]])
        neighbors = self.neighbor_seeds(frame, records)
        seeds = unique_geos(both_handedness(np.vstack([current, neighbors])))
        region = content(target)
        if len(neighbors) == 0 or region.sum() in (0, region.size):
            result = self.finish(cuda, frame, current, target, old["baseline_fullres_metrics"], old, phase, cache=cache)
            if region.sum() in (0, region.size):
                result["constant_content"] = True
            return result
        query = image_query(target, (512, 384))
        varied = variations(seeds, [-1.5, 0, 1.5], [0.97, 1, 1.03])
        geo, _ = refine_batches(cuda, varied, query, (64, 48), 16, top_k=2, separation=8)
        scores = score_batches(cuda, geo, np.uint8(query >= 0.5), 48, structure=True)
        winners = geo[diverse(geo, ranking(scores), 12, overlap=0.05)]
        refined = polish(
            cuda, np.vstack([winners, current]), target, old["baseline_fullres_metrics"], full_count=32, cache=cache
        )
        return self.finish(
            cuda, frame, np.vstack([refined, current]), target, old["baseline_fullres_metrics"], old, phase, cache=cache
        )

    def run(self, limit=None, pilot=None):
        if not (DICTIONARY / "complete.json").is_file() and pilot is None:
            raise ValueError("Deep dictionary is not complete")
        if (self.output / "matches.jsonl").is_file() and pilot is None:
            return
        if pilot is not None:
            # Useful before a global dictionary completes: test refinement on known and neighboring geographic views.
            rows = []
            with CudaAtlas() as cuda:
                for frame in pilot:
                    target = np.uint8(self.raw[frame] >= 128)
                    base = np.array([vector(self.base[frame])])
                    cache = FrameScoreCache(target, structure=True)
                    metrics = score_batches(cuda, base, target, cache=cache, structure=True)
                    baseline = precise_record(base[0], metrics, 0)
                    initial = both_handedness(np.vstack([base, self.neighbor_seeds(frame, self.prior)]))
                    q = image_query(target, (512, 384))
                    geo, _ = refine_batches(
                        cuda,
                        variations(initial, [-5, 0, 5], [0.85, 1, 1.18]),
                        q,
                        (160, 120),
                        16,
                        top_k=3,
                        separation=16,
                    )
                    m = score_batches(cuda, geo, np.uint8(q >= 0.5), 48, structure=True)
                    seeds = geo[diverse(geo, ranking(m), 32)]
                    geos = polish(cuda, np.vstack([seeds, base]), target, baseline, cache=cache)
                    result = self.finish(cuda, frame, geos, target, baseline, phase="pilot-content", cache=cache)
                    rows.append(result)
                    print(
                        f"precision pilot frame {frame+1}: mIoU {baseline['native_miou']:.5f} -> {result['native_miou']:.5f}; contour {baseline['soft_contour_f1']:.5f} -> {result['soft_contour_f1']:.5f}",
                        flush=True,
                    )
            (self.output / "pilot-content.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            return
        self.load_dictionary()
        started = time.perf_counter()
        path = self.output / "global.partial.jsonl"
        done = read_records(path) if path.is_file() else []
        if [r["frame"] for r in done] != list(range(len(done))):
            raise ValueError("Invalid global checkpoint")
        path.write_text("".join(json.dumps(r) + "\n" for r in done), encoding="utf-8")
        recent = self.base.copy()
        for r in done:
            recent[r["frame"]] = r
        with CudaAtlas() as cuda, path.open("a", encoding="utf-8") as sink:
            write_json(self.output / "cuda-build.json", cuda.build)
            for frame in range(len(done), min(len(self.base), limit or len(self.base))):
                r = self.global_frame(cuda, frame, recent)
                recent[frame] = r
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                sink.flush()
                status = dict(
                    phase="global",
                    complete=frame + 1,
                    total=len(self.base),
                    elapsed_seconds=time.perf_counter() - started,
                    frame_miou=r["native_miou"],
                    baseline_miou=r["baseline_fullres_metrics"]["native_miou"],
                )
                write_json(self.output / "search-status.json", status)
                if (frame + 1) % 5 == 0:
                    print(
                        f"precision global: {frame+1}/{len(self.base)}; {status['elapsed_seconds']:.1f}s; mIoU {r['native_miou']:.4f}",
                        flush=True,
                    )
            if limit and limit < len(self.base):
                return
            for iteration in range(2):
                phase = f"temporal-{iteration}"
                path = self.output / (phase + ".partial.jsonl")
                previous = read_records(path) if path.is_file() else []
                order = list(range(len(self.base)))
                if iteration == 1:
                    order.reverse()
                if [r["frame"] for r in previous] != order[: len(previous)]:
                    raise ValueError("Invalid temporal checkpoint")
                path.write_text("".join(json.dumps(r) + "\n" for r in previous), encoding="utf-8")
                for r in previous:
                    recent[r["frame"]] = r
                with path.open("a", encoding="utf-8") as temporal:
                    for number, frame in enumerate(order[len(previous) :], len(previous) + 1):
                        r = self.temporal_frame(cuda, frame, recent, phase)
                        recent[frame] = r
                        temporal.write(json.dumps(r, ensure_ascii=False) + "\n")
                        temporal.flush()
                        write_json(
                            self.output / "search-status.json",
                            dict(
                                phase=phase,
                                complete=number,
                                total=len(self.base),
                                elapsed_seconds=time.perf_counter() - started,
                            ),
                        )
                        if number % 5 == 0:
                            print(
                                f"precision {phase}: {number}/{len(self.base)}; {time.perf_counter()-started:.1f}s",
                                flush=True,
                            )
        encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recent)
        (self.output / "matches.jsonl").write_text(encoded, encoding="utf-8")
        write_json(
            self.output / "search-summary.json",
            dict(
                frames=len(recent),
                mean_miou=float(np.mean([r["native_miou"] for r in recent])),
                baseline_mean_miou=float(np.mean([r["baseline_fullres_metrics"]["native_miou"] for r in recent])),
                mean_soft_contour=float(np.mean([r["soft_contour_f1"] for r in recent])),
                baseline_mean_soft_contour=float(
                    np.mean([r["baseline_fullres_metrics"]["soft_contour_f1"] for r in recent])
                ),
                mean_structure_objective=float(np.mean([r["structure_objective"] for r in recent])),
                baseline_mean_structure_objective=float(
                    np.mean([r["baseline_fullres_metrics"]["structure_objective"] for r in recent])
                ),
                mirrored_frames=sum(bool(r.get("mirror_horizontal")) for r in recent),
                constant_content_frames=sum(bool(r.get("constant_content")) for r in recent),
                content_bounds=self.settings["content_bounds"],
                evaluated_pixels=self.settings["evaluated_pixels"],
                regressed_miou_frames=sum(
                    r["native_miou"] < r["baseline_fullres_metrics"]["native_miou"] - 1e-12 for r in recent
                ),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--pilot", type=int, nargs="+")
    a = p.parse_args()
    Search().run(a.limit, a.pilot)
