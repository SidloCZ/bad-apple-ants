"""Shared pieces of the 2560x1920/30p satellite edition: frame I/O, flash audit and audio mux."""

from __future__ import annotations
import json, subprocess, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from badapple_media import ffmpeg_binary, probe
from .data import digest
from .pipeline import write_json
from .flash_cuda import FlashCuda

SOURCE = Path("output/coastline-precision-v4")
WIDTH, HEIGHT = 2560, 1920
CREDIT_START = 6562
CREDIT_PHOTO = 6570


def fill_frame(stream, buffer):
    view = memoryview(buffer).cast("B")
    position = 0
    while position < len(view):
        count = stream.readinto(view[position:])
        if not count:
            if position:
                raise ValueError("Truncated decoded frame")
            return False
        position += count
    return True


def output_spec(count, source_fps="30000/1001"):
    return dict(
        width=WIDTH,
        height=HEIGHT,
        fps="30/1",
        progressive=True,
        drop_frame_timecode=False,
        frames=count,
        source_frames=count,
        repeated_display_frames=0,
        dropped_frames=0,
        duration_seconds=count / 30,
        source_duration_seconds=count / float(Fraction(source_fps)),
        source_fps=source_fps,
        audio_tempo=None,
        original_audio_stream_copy=None,
        content_box=[0, 0, WIDTH, HEIGHT],
        source_canvas=[1280, 960],
        scaling="Aspect-preserving 2x upscale to 2560x1920; original 4:3 layout",
    )


def source_frame_map(count):
    return np.arange(count, dtype=np.int64)






def audit(movie, output, count):
    info = probe(movie)
    if (info["width"], info["height"], info["fps"]) != (WIDTH, HEIGHT, "30"):
        raise ValueError("Output dimensions or 30.000 fps changed")
    details = json.loads(
        subprocess.check_output(
            [
                ffmpeg_binary("ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,field_order,color_space,color_transfer",
                "-of",
                "json",
                str(movie),
            ]
        )
    )["streams"][0]
    if int(details["nb_frames"]) != count or details.get("field_order") not in ("progressive", "unknown"):
        raise ValueError("Frame drop or interlaced output")
    roi_width, roi_height = 1920, 1440
    command = [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-hwaccel",
        "cuda",
        "-i",
        str(movie),
        "-an",
        "-vf",
        "crop=1920:1440:40:140",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    area_reference = 0.25 * (341 * WIDTH / 1024) * (256 * HEIGHT / 768)
    limit = int(area_reference * 0.70)
    peak = np.zeros(8, np.float64)
    minimum_y = 1.0
    bad = []
    rows = []
    start = time.perf_counter()
    buffer = np.empty(roi_width * roi_height * 3 // 2, np.uint8)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, bufsize=16 * 1024 * 1024)
        try:
            with FlashCuda(roi_width, roi_height) as auditor:
                for i in range(count):
                    if not fill_frame(process.stdout, buffer):
                        raise ValueError("Missing decoded frame")
                    values = auditor.audit_yuv420(buffer)
                    peak = np.maximum(peak, values)
                    minimum_y = min(minimum_y, float(values[5]))
                    if values[1] > limit or values[2] > limit:
                        bad.append(i)
                    rows.append(
                        [float(values[0]), int(values[1]), int(values[2]), int(values[3]), *map(float, values[4:])]
                    )
                    if (i + 1) % 100 == 0 or i + 1 == count:
                        elapsed = time.perf_counter() - start
                        write_json(
                            output / "status.json",
                            dict(
                                phase="full-resolution-flash-audit",
                                complete=i + 1,
                                total=count,
                                elapsed_seconds=elapsed,
                            ),
                        )
                        print(f"Full-pixel flash audit: {i+1}/{count}; {elapsed:.1f}s", flush=True)
                if fill_frame(process.stdout, buffer) or process.wait() != 0:
                    raise ValueError("Frame count/decode error")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    subprocess.run(
        [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(movie), "-f", "null", "-"], check=True
    )
    report = dict(
        frames=count,
        decoded_size=[WIDTH, HEIGHT],
        fps="30/1",
        all_pixels_in_audited_roi_checked=True,
        spatial_subsampling=False,
        scope="Satellite photograph panel only; original-animation comparison and mask are left untouched",
        audited_roi=[40, 140, 1960, 1580],
        audited_pixels_per_frame=roi_width * roi_height,
        whole_composite_compliance_assessed=False,
        standard="WCAG2.2 SC2.3.1 general-flash and red-flash thresholds, conservative reference-view screening",
        standard_url="https://www.w3.org/TR/WCAG22/#dfn-general-flash-and-red-flash-thresholds",
        method="Relative sRGB luminance over every 30-frame/1-second window. A pixel is a potential rapid general-flash location only if total variation >= 0.55 AND its observed all-time luminance range >= 0.10; a range below 0.10 cannot contain a qualifying opposing luminance change. All-time range overbounds every one-second range. Union of ALL saturated-red pixels over 31 frames bounds red-flash area without chromaticity exemptions.",
        reference_10_degree_field_pixels=[341 * WIDTH / 1024, 256 * HEIGHT / 768],
        normative_reference_area_pixels=area_reference,
        applied_area_limit_pixels=limit,
        area_margin=0.70,
        max_1s_luminance_variation=float(peak[0]),
        max_potential_general_flash_area_pixels=int(peak[1]),
        max_saturated_red_window_union_pixels=int(peak[2]),
        max_saturated_red_pixels_in_one_frame=int(peak[3]),
        violating_frames=bad,
        passed=not bad,
        decode_errors=0,
        max_observed_per_pixel_luminance_range=float(peak[4]),
        decoded_luminance_min=minimum_y,
        decoded_luminance_max=float(peak[6]),
        adjacent_mean_luminance_difference_max=(
            float(np.max(np.abs(np.diff(np.asarray(rows)[:, 7])))) if count > 1 else 0.0
        ),
        adjacent_mean_luminance_difference_p95=(
            float(np.percentile(np.abs(np.diff(np.asarray(rows)[:, 7])), 95)) if count > 1 else 0.0
        ),
        elapsed_seconds=time.perf_counter() - start,
        limitations="Reference viewing geometry; independent conservative implementation, not a Harding/broadcast certification or a guarantee against seizures. High-contrast patterns and motion can remain.",
    )
    write_json(output / "flash-audit.json", report)
    write_json(
        output / "flash-audit-per-frame.json",
        dict(
            columns=[
                "variation",
                "general_area",
                "red_window_area",
                "red_now",
                "all_time_pixel_range",
                "minimum_y",
                "maximum_y",
                "mean_y",
            ],
            rows=rows,
        ),
    )
    return report




def packet_hash(path, stream):
    return (
        subprocess.check_output(
            [
                ffmpeg_binary("ffmpeg"),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                f"0:{stream}:0",
                "-c",
                "copy",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ]
        )
        .decode()
        .strip()
    )


def audio_command(video, original, final, frames, mode):
    command = [
        ffmpeg_binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(video),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if mode == "original":
        command += ["-c:a", "copy"]
    elif mode == "trim":
        command += ["-c:a", "aac", "-b:a", "192k", "-t", f"{frames/30:.9f}"]
    else:
        command += ["-af", "atempo=1.001", "-c:a", "aac", "-b:a", "192k", "-t", f"{frames/30:.9f}"]
    return command + ["-timecode", "00:00:00:00", "-movflags", "+faststart", str(final)]


def finish_audio(source, output, mode="trim", assets=True):
    source = Path(source)
    output = Path(output)
    video = output / "validated-video.mp4"
    final = output / "badapple-coastline.mp4"
    summary = json.loads((output / "summary.json").read_text())
    settings = json.loads((output / "run.json").read_text())
    original = Path(json.loads((source / "run.json").read_text())["video_source"])
    write_json(output / "status.json", dict(phase="audio-mux", audio_mode=mode))
    command = audio_command(video, original, final, summary["frames"], mode)
    subprocess.run(command, check=True)
    video_hash = packet_hash(video, "v")
    if packet_hash(final, "v") != video_hash:
        raise ValueError("Validated video packets changed during audio mux")
    audio_hash = packet_hash(final, "a")
    original_audio_hash = packet_hash(original, "a")
    if mode == "original" and audio_hash != original_audio_hash:
        raise ValueError("Original audio packets changed")
    subprocess.run(
        [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(final), "-f", "null", "-"], check=True
    )
    metadata = probe(final)
    metadata.update(
        frames=summary["frames"],
        world_map=dict(enabled=True, persistent_history=True),
        display_content_box=[0, 0, WIDTH, HEIGHT],
        source_canvas=[1280, 960],
    )
    if settings.get("local_map"):
        local = settings["local_map"]
        metadata["world_map"]["map_box"] = [2 * v for v in local["native_world_box"]]
        metadata["local_map"] = dict(local, map_box=[2 * v for v in local["native_local_box"]])
    write_json(output / "badapple-coastline.verification.json", metadata)
    summary.update(
        movie_sha256=digest(final),
        validated_video_packet_hash=video_hash,
        audio_mode=mode,
        original_audio_packet_hash=original_audio_hash,
        output_audio_packet_hash=audio_hash,
        decode_errors=0,
    )
    summary["output_spec"].update(
        audio_tempo=1.001 if mode == "sync" else 1.0,
        original_audio_stream_copy=mode == "original",
        audio_codec="aac",
        audio_bitrate=None if mode == "original" else "192k",
    )
    settings.update(audio_mode=mode, audio_tempo=summary["output_spec"]["audio_tempo"])
    write_json(output / "run.json", settings)
    write_json(output / "summary.json", summary)
    if assets:
        write_json(
            output / "status.json", dict(phase="publishing-assets", complete=summary["frames"], total=summary["frames"])
        )
        subprocess.run(
            [__import__("sys").executable, "scripts/build_coastline_youtube.py", "--root", str(output)], check=True
        )
    write_json(output / "status.json", dict(phase="complete", complete=summary["frames"], total=summary["frames"]))
    print(json.dumps(summary, indent=2), flush=True)




