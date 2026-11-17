"""Verify and render native land/water silhouettes without acquiring photographs."""

from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .cuda_backend import CudaAtlas
from .data import digest, MASK_CREDIT
from .pipeline import read_records, write_json
from .precision_config import OUTPUT
from .precision_search import vector
from badapple_media import ffmpeg_binary, probe, read_exact, ui_font

SIZE = (1444, 1080)


def load_mask(path):
    with Image.open(path) as image:
        if image.size != SIZE:
            raise ValueError("Mask is not original size")
        pixels = np.asarray(image.convert("L")).copy()
    if not np.isin(pixels, [0, 255]).all():
        raise ValueError("Native mask contains nonbinary pixels")
    return pixels


def verify_native(output):
    settings = json.loads((output / "run.json").read_text())
    records = read_records(output / "matches.jsonl")
    if [r["frame"] for r in records] != list(range(settings["frames"])):
        raise ValueError("Missing/reordered matches")
    if digest(Path(settings["video_source"])) != settings["video_source_sha256"]:
        raise ValueError("Source video changed")
    if digest(Path(settings["dictionary"]) / "dictionary.json") != settings["dictionary_sha256"]:
        raise ValueError("Dictionary changed")
    if (
        settings.get("additional_dictionary")
        and digest(Path(settings["additional_dictionary"]) / "dictionary.json")
        != settings["additional_dictionary_sha256"]
    ):
        raise ValueError("Additional dictionary changed")
    baseline_path = Path(settings["baseline"]) / "frames.jsonl"
    if digest(baseline_path) != settings["baseline_frames_sha256"]:
        raise ValueError("Baseline changed")
    baseline = read_records(baseline_path)
    frame_dir = Path(settings["frames_directory"])
    raw = np.memmap(frame_dir / "frames.u8", dtype=np.uint8, mode="r", shape=(len(records), 1080, 1444))
    hashes = []
    started = time.perf_counter()
    with CudaAtlas() as cuda:
        for i, r in enumerate(records):
            if r["match_run_id"] != settings["match_run_id"]:
                raise ValueError("Match belongs to another run")
            geos = np.array([vector(r), vector(baseline[i])], np.float64)
            target = np.uint8(raw[i] >= 128)
            metrics = (
                cuda.score_structure(geos, target, border_y=settings["content_border_y"])
                if "structure_objective" in settings
                else cuda.score_shape(geos, target)
            )
            if metrics["missing"].any():
                raise ValueError("Missing original mask data")
            np.testing.assert_array_equal(metrics["shape_counts"][0], r["precision_counts"])
            np.testing.assert_array_equal(metrics["shape_counts"][1], r["baseline_fullres_metrics"]["precision_counts"])
            if "structure_objective" in settings:
                np.testing.assert_array_equal(metrics["structure_counts"][0], r["structure_counts"])
                np.testing.assert_array_equal(
                    metrics["structure_counts"][1], r["baseline_fullres_metrics"]["structure_counts"]
                )
                if metrics["structure_objective"][0] < metrics["structure_objective"][1] - 1e-12:
                    raise ValueError("Content structure floor regressed")
            for name in ["miou", "objective", "precision_objective"]:
                if metrics[name][0] < metrics[name][1] - 1e-12:
                    raise ValueError("Native acceptance floor regressed")
            path = Path(r["native_mask_path"])
            if digest(path) != r["native_mask_sha256"]:
                raise ValueError("Selected mask changed")
            mask = load_mask(path)
            exact = cuda.sample(geos[:1], SIZE)[0] * 255
            if not np.array_equal(mask, exact):
                raise ValueError("Saved mask differs from actual native geometry")
            hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
            if (i + 1) % 100 == 0:
                print(
                    f"monochrome native verification: {i+1}/{len(records)}; {time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    summary = dict(
        frames=len(records),
        evaluation_size=list(SIZE),
        match_run_id=settings["match_run_id"],
        mean_miou=float(np.mean([r["native_miou"] for r in records])),
        baseline_mean_miou=float(np.mean([r["baseline_fullres_metrics"]["native_miou"] for r in records])),
        mean_soft_contour_f1=float(np.mean([r["soft_contour_f1"] for r in records])),
        baseline_mean_soft_contour_f1=float(
            np.mean([r["baseline_fullres_metrics"]["soft_contour_f1"] for r in records])
        ),
        mean_precision_objective=float(np.mean([r["precision_objective"] for r in records])),
        baseline_mean_precision_objective=float(
            np.mean([r["baseline_fullres_metrics"]["precision_objective"] for r in records])
        ),
        regressed_miou_frames=0,
        regressed_legacy_objective_frames=0,
        regressed_precision_objective_frames=0,
        exact_native_mask_differences=0,
        mirrored_frames=sum(bool(r.get("mirror_horizontal")) for r in records),
        new_satellite_downloads=False,
        verification_seconds=time.perf_counter() - started,
    )
    if "structure_objective" in settings:
        summary.update(
            content_bounds=settings["content_bounds"],
            evaluated_pixels=settings["evaluated_pixels"],
            excluded_presentation_pixels=settings["excluded_presentation_pixels"],
            mean_structure_objective=float(np.mean([r["structure_objective"] for r in records])),
            baseline_mean_structure_objective=float(
                np.mean([r["baseline_fullres_metrics"]["structure_objective"] for r in records])
            ),
            mean_frame_edge_error=float(np.mean([r["frame_edge_error"] for r in records])),
            baseline_mean_frame_edge_error=float(
                np.mean([r["baseline_fullres_metrics"]["frame_edge_error"] for r in records])
            ),
            regressed_structure_objective_frames=0,
            metric_note="Baseline and new views freshly measured on the same original-pixel content ROI, excluding fixed letterbox rows",
        )
    variable = [r for r in records if r["native_counts"][4] > 0]
    if variable:
        summary.update(
            nonconstant_frames=len(variable),
            nonconstant_mean_miou=float(np.mean([r["native_miou"] for r in variable])),
            baseline_nonconstant_mean_miou=float(
                np.mean([r["baseline_fullres_metrics"]["native_miou"] for r in variable])
            ),
        )
    write_json(output / "monochrome-native-verification.json", summary)
    write_json(output / "monochrome-pixel-hashes.json", hashes)
    return settings, records, raw, hashes, summary


def encode(planes, size, count, fps, source, output, lossless=None, mirror_allowed=True):
    output = Path(output)
    partial = output.with_suffix(".partial.mp4")
    duration = f"{float(count/fps):.9f}"
    command = [
        ffmpeg_binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s:v",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(source),
    ]
    audio = ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-t", duration]
    title = "[Photosensitive Seizure Warning] Bad Apple!! - Real Land/Water Mask"
    orientation = "mirror permitted and recorded per frame" if mirror_allowed else "unmirrored baseline views"
    comment = (
        MASK_CREDIT
        + "; white=land, black=water; "
        + orientation
        + "; finite sampled geographic search; native mask pixels verified; full credits in MONOCHROME-CREDITS.txt"
    )
    metadata = ["-metadata", "title=" + title, "-metadata", "comment=" + comment]
    lossless_partial = None
    if lossless:
        lossless = Path(lossless)
        lossless_partial = lossless.with_suffix(".partial.mkv")
        command += (
            audio
            + [
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-g",
                "1",
                "-slices",
                "16",
                "-slicecrc",
                "1",
                "-threads",
                "8",
                "-pix_fmt",
                "gray",
            ]
            + metadata
            + [str(lossless_partial)]
        )
    command += (
        audio
        + [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "constqp",
            "-qp",
            "12",
            "-bf",
            "0",
            "-g",
            "30",
            "-spatial-aq",
            "0",
            "-temporal-aq",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        + metadata
        + [str(partial)]
    )
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        i = -1
        try:
            for i, plane in enumerate(planes):
                if plane.shape != (size[1], size[0]) or plane.dtype != np.uint8:
                    raise ValueError("Unexpected video plane")
                process.stdin.write(plane.tobytes())
                if (i + 1) % 100 == 0:
                    print(f"{output.stem}: {i+1}/{count}", flush=True)
            process.stdin.close()
            if process.wait():
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace"))
            if i + 1 != count:
                raise ValueError("Wrong number of supplied frames")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    partial.replace(output)
    if lossless_partial:
        lossless_partial.replace(lossless)


def audit_pixels(movie, hashes, threshold=False, crop=None, check_audio=True):
    command = [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(movie), "-an"]
    if crop:
        command += ["-vf", "crop=" + crop]
    command += ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    count = 0
    try:
        while True:
            raw = read_exact(process.stdout, np.prod(SIZE))
            if not raw:
                break
            if threshold:
                raw = (np.uint8(np.frombuffer(raw, np.uint8) >= 128) * 255).tobytes()
            if count >= len(hashes) or hashlib.sha256(raw).hexdigest() != hashes[count]:
                raise ValueError(f"Encoded mask pixels changed at frame {count}")
            count += 1
        if process.wait() != 0 or count != len(hashes):
            raise ValueError("Encoded frame/decode mismatch")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if check_audio:
        subprocess.run(
            [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(movie), "-f", "null", "-"],
            check=True,
        )
    return dict(
        frames=count,
        decode_errors=0,
        pixel_hash_differences=0,
        comparison="threshold-128 binary silhouette" if threshold else "all original grayscale pixel values",
    )


def comparison_planes(records, raw, mirror_allowed=True):
    width = SIZE[0]
    plane = np.zeros((1120, width * 2), np.uint8)
    header = Image.new("L", (width * 2, 40), 0)
    draw = ImageDraw.Draw(header)
    font = ui_font(26)
    draw.text((20, 4), "ORIGINAL SILHOUETTE", font=font, fill=255)
    draw.text(
        (width + 20, 4),
        "REAL LAND / WATER MASK" + (" (MIRROR ALLOWED)" if mirror_allowed else " - BASELINE"),
        font=font,
        fill=255,
    )
    plane[:40] = np.asarray(header)
    for i, r in enumerate(records):
        plane[40:, :width] = np.uint8(raw[i] >= 128) * 255
        plane[40:, width:] = load_mask(r["native_mask_path"])
        yield plane


def run(output=OUTPUT):
    output = Path(output)
    if (output / "monochrome-summary.json").is_file():
        return
    settings, records, raw, hashes, summary = verify_native(output)
    fps = Fraction(settings["fps"])
    source = Path(settings["video_source"])
    movie = output / "badapple-land-water.mp4"
    lossless = output / "badapple-land-water-lossless.mkv"
    comparison = output / "badapple-land-water-comparison.mp4"
    encode((load_mask(r["native_mask_path"]) for r in records), SIZE, len(records), fps, source, movie, lossless)
    exact = audit_pixels(lossless, hashes)
    visible = audit_pixels(movie, hashes, threshold=True)
    encode(comparison_planes(records, raw), (SIZE[0] * 2, 1120), len(records), fps, source, comparison)
    comparison_right = audit_pixels(comparison, hashes, True, "1444:1080:1444:40")
    target_hashes = [hashlib.sha256((np.uint8(r >= 128) * 255).tobytes()).hexdigest() for r in raw]
    comparison_left = audit_pixels(comparison, target_hashes, True, "1444:1080:0:40", check_audio=False)
    for path, size in [(movie, SIZE), (lossless, SIZE), (comparison, (SIZE[0] * 2, 1120))]:
        info = probe(path)
        if (info["width"], info["height"]) != size or info["fps"] != str(fps):
            raise ValueError("Encoded geometry/fps mismatch")
        if probe(source)["has_audio"] and not info["has_audio"]:
            raise ValueError("Audio lost")
    summary.update(
        movie=str(movie.resolve()),
        movie_sha256=digest(movie),
        lossless_sha256=digest(lossless),
        comparison_sha256=digest(comparison),
        fps=str(fps),
        has_audio=probe(movie)["has_audio"],
        duration=probe(movie)["duration"],
        lossless_pixel_audit=exact,
        mp4_silhouette_audit=visible,
        comparison_left_audit=comparison_left,
        comparison_right_audit=comparison_right,
        source_video_sha256=settings["video_source_sha256"],
        dictionary_views=json.loads((Path(settings["dictionary"]) / "dictionary.json").read_text()).get(
            "count_including_mirrors",
            json.loads((Path(settings["dictionary"]) / "dictionary.json").read_text())["count"],
        ),
    )
    credits = MASK_CREDIT + "\nhttps://doi.org/10.5281/zenodo.14899246\nhttps://creativecommons.org/licenses/by/4.0/\n"
    credits += "White=land, black=water. Original mask cropped, rotated, optionally horizontally mirrored and sampled at each exact geographic field of view. Per-frame mirror flags are recorded in matches.jsonl. No photographs were newly acquired for this monochrome edition.\n"
    if "structure_objective" in settings:
        credits += "Matching excludes the source video's fixed top/bottom 2-pixel black letterbox. Original 1444x1080 frame coordinates, unmodified source frames and full native geography masks are preserved; only descriptor/FFT padding extends the nearest content row. Content area, contours and four content-border agreements are measured without resizing.\n"
    credits += "Underlying sources: ESA WorldCover (© ESA WorldCover project 2020/2021), OpenStreetMap contributors (https://www.openstreetmap.org/copyright), Copernicus DEM.\n"
    credits += "Original Bad Apple!! music and shadow animation retain the rights of their creators. Song: Alstroemeria Records, Masayoshi Minoshima, Haruka, nomico; original composition ZUN. Shadow PV: Anira (あにら), https://www.nicovideo.jp/watch/sm8628149 .\n"
    (output / "MONOCHROME-CREDITS.txt").write_text(credits, encoding="utf-8")
    readme = """# Monochrome land/water edition

badapple-land-water.mp4: matched real land/water silhouettes only, at 1444x1080. Audio is copied from the original video.
badapple-land-water-lossless.mkv: FFV1 archival master; every decoded grayscale pixel is verified against the selected original-size masks.
badapple-land-water-comparison.mp4: original silhouette on the left, matched real geography on the right. Both panels retain all 1444x1080 pixels.

The MP4 silhouette was rethresholded at 128 and every binary frame was verified unchanged by compression. All original grayscale pixels in the lossless master were independently decoded and hashed.
matches.jsonl retains exact coordinates, scale, rotation, native metrics and provenance for every frame. monochrome-summary.json contains final measurements and file hashes.
No new satellite photographs are acquired by this stage. Full attribution is in MONOCHROME-CREDITS.txt.
"""
    if "structure_objective" in settings:
        readme += "\nMetrics use the original 1444x1076 content pixels, excluding only the fixed top/bottom 2-pixel source-video letterbox. Frame coordinates and all original 1444x1080 video pixels are preserved. Baseline views are freshly remeasured using the same content metric; historical full-frame scores are not directly comparable.\n"
    (output / "MONOCHROME-README.md").write_text(readme, encoding="utf-8")
    write_json(output / "monochrome-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path("output/coastline-fullres-cuda-v2/run.json").read_text())
    fps = Fraction(settings["fps"])
    y, x = np.indices((1080, 1444))
    rng = np.random.default_rng(478)
    planes = []
    for i in range(12):
        mask = np.uint8(x > 700 + 150 * np.sin(y / 60 + i / 3)) * 255
        if i % 3 == 0:
            mask = np.uint8(rng.random((1080, 1444)) > 0.997) * 255
        planes.append(mask)
    hashes = [hashlib.sha256(p.tobytes()).hexdigest() for p in planes]
    movie = output / "smoke.mp4"
    lossless = output / "smoke.mkv"
    encode(iter(planes), SIZE, len(planes), fps, Path(settings["video_source"]), movie, lossless)
    print(audit_pixels(lossless, hashes))
    print(audit_pixels(movie, hashes, True))
    side = output / "smoke-comparison.mp4"
    combined = [np.vstack([np.zeros((40, 2888), np.uint8), np.hstack([plane, plane])]) for plane in planes]
    encode(iter(combined), (2888, 1120), len(planes), fps, Path(settings["video_source"]), side)
    print(audit_pixels(side, hashes, True, "1444:1080:1444:40"))
    print(audit_pixels(side, hashes, True, "1444:1080:0:40", False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--smoke-test", type=Path)
    a = p.parse_args()
    smoke(a.smoke_test) if a.smoke_test else run(a.output)
