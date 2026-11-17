"""Fast 2560x1920/30p treatment; all original frames retained."""

from __future__ import annotations
import argparse, json, subprocess, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from badapple_media import ffmpeg_binary, probe, ui_font
from .data import digest
from .pipeline import read_records, write_json
from .flash_cuda import FlashCuda
from .footprint_map import FootprintHistory, MAP_BOX, footprint

SOURCE = Path("output/coastline-precision-v4")
OUTPUT = Path("output/coastline-precision-v4-2560x1920-30p-luminance-normalized-chroma-fixed")
LOCAL_OUTPUT = Path("output/coastline-precision-v4-2560x1920-30p-local-map-equal-panels")
WIDTH, HEIGHT = 2560, 1920
CREDIT_START = 6562
CREDIT_PHOTO = 6570
LOGO_START = 6773
LOGO_SOURCE = 6800


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
        scaling="Aspect-preserving2xupscale to2560x1920; original4:3layout",
    )


def source_frame_map(count):
    return np.arange(count, dtype=np.int64)


def prepare_credits(movie, records, output):
    assets = output / "render-assets"
    assets.mkdir(exist_ok=True)
    shots = []
    for index in [CREDIT_PHOTO, LOGO_SOURCE]:
        path = assets / f"source-{index:05d}.png"
        subprocess.run(
            [
                ffmpeg_binary("ffmpeg"),
                "-y",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                f"{index*1001/30000:.9f}",
                "-i",
                str(movie),
                "-frames:v",
                "1",
                str(path),
            ],
            check=True,
        )
        with Image.open(path) as image:
            shots.append(image.convert("RGB"))
    history = FootprintHistory()
    for i, record in enumerate(records[:CREDIT_START]):
        history.advance(record, i)
    panel = history.panel(records[CREDIT_PHOTO], CREDIT_START)
    base = shots[0]
    base.paste(panel, MAP_BOX[:2])
    draw = ImageDraw.Draw(base)
    navy = "#080f17"
    for box in [
        (996, 14, 1279, 53),
        (996, 315, 1279, 348),
        (996, 776, 1279, 799),
        (530, 878, 625, 905),
        (690, 878, 790, 905),
    ]:
        draw.rectangle(box, fill=navy)
    draw.text((1000, 22), "ENDING CREDITS", font=ui_font(17), fill="#9eafc0")
    draw.text((1000, 322), "FIXED LAND / WATER", font=ui_font(16), fill="#73e6d7")
    draw.text((1000, 780), "FIXED CREDIT VIEW", font=ui_font(12), fill="#73e6d7")
    for xy in [(535, 882), (695, 882)]:
        draw.text(xy, "n/a", font=ui_font(16, monospace=True), fill="#9eafc0")
    logo = base.copy()
    logo.paste(shots[1].crop((1000, 92, 1260, 287)), (1000, 92))
    base.save(assets / "credits-fixed.png")
    logo.save(assets / "logo-fixed.png")
    switches = sum(
        bool(records[i].get("mirror_horizontal")) != bool(records[i - 1].get("mirror_horizontal"))
        for i in range(CREDIT_START + 1, len(records))
    )
    result = dict(
        start_frame=CREDIT_START,
        start_seconds_30fps=CREDIT_START / 30,
        fixed_photo_source_frame=CREDIT_PHOTO,
        logo_start_frame=LOGO_START,
        removed_original_mirror_switches=switches,
        output_mirror_switches=0,
        geographical_view_fixed=True,
        original_credit_card_and_logo_retained=True,
        short_black_gap_held=False,
        original_comparison_preserved=True,
        photo_sha256=records[CREDIT_PHOTO]["satellite"]["image_sha256"],
    )
    return np.asarray(base), np.asarray(logo), result


def encode(movie, records, output, count, span, attempt, local_maps=None):
    start = time.perf_counter()
    spec = output_spec(count)
    path = output / f"candidate-{attempt}.mp4"
    mapping = source_frame_map(count)
    repetitions = np.bincount(mapping, minlength=count)
    source_settings = json.loads((movie.parent / "run.json").read_text())
    original_movie = Path(source_settings["video_source"])
    if digest(original_movie) != source_settings["video_source_sha256"]:
        raise ValueError("Original animation source changed")
    credits = None
    if count > CREDIT_START:
        credits = prepare_credits(movie, records, output)
    decoder_command = [
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
        "-frames:v",
        str(count),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    original_command = [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-hwaccel",
        "cuda",
        "-i",
        str(original_movie),
        "-an",
        "-vf",
        "scale=260:195:flags=lanczos",
        "-frames:v",
        str(count),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    filters = f"scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12,hwupload_cuda,scale_cuda=w={WIDTH}:h={HEIGHT}:interp_algo=lanczos"
    encoder_command = [
        ffmpeg_binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "1280x960",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-rc",
        "constqp",
        "-qp",
        "12",
        "-bf",
        "0",
        "-g",
        "60",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-an",
        "-frames:v",
        str(spec["frames"]),
        "-t",
        f'{spec["duration_seconds"]:.9f}',
        "-fps_mode",
        "passthrough",
        "-timecode",
        "00:00:00:00",
        "-movflags",
        "+faststart",
        str(path),
    ]
    buffer = np.empty((960, 1280, 3), np.uint8)
    original_buffer = np.empty((195, 260, 3), np.uint8)
    peak = np.zeros(8, np.float64)
    written = 0
    exposures = []
    source_means = []
    shadow_pixels = 0
    shadow_spread = 0
    with (
        tempfile.TemporaryFile() as decoder_log,
        tempfile.TemporaryFile() as encoder_log,
        tempfile.TemporaryFile() as original_log,
    ):
        decoder = subprocess.Popen(
            decoder_command, stdout=subprocess.PIPE, stderr=decoder_log, bufsize=16 * 1024 * 1024
        )
        original_decoder = subprocess.Popen(
            original_command, stdout=subprocess.PIPE, stderr=original_log, bufsize=1024 * 1024
        )
        encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=encoder_log, bufsize=16 * 1024 * 1024)
        try:
            with FlashCuda(span=span) as processor, FlashCuda(960, 720) as source_auditor:
                for i in range(count):
                    if not fill_frame(decoder.stdout, buffer):
                        raise ValueError("Input movie is missing a frame")
                    if not fill_frame(original_decoder.stdout, original_buffer):
                        raise ValueError("Original comparison is missing a source frame")
                    values = source_auditor.audit(buffer[70:790, 20:980])
                    peak = np.maximum(peak, values)
                    source_means.append(float(values[7]))
                    frame = credits[1 if i >= LOGO_START else 0].copy() if credits and i >= CREDIT_START else buffer
                    frame[92:287, 1000:1260] = original_buffer
                    if local_maps:
                        frame = local_maps.compose(
                            frame, i, records[i if i < CREDIT_START else CREDIT_PHOTO].get("mirror_horizontal", False)
                        )
                    processed = processor.process(frame)
                    shadows = processor.shadow_metrics()
                    if shadows[0] or shadows[1]:
                        raise ValueError(f"Blue/magenta near-black amplification at frame {i}")
                    shadow_pixels += int(shadows[2])
                    shadow_spread = max(shadow_spread, int(shadows[3]))
                    exposures.append(processor.exposure)
                    if not np.array_equal(processed[92:287, 1000:1260], original_buffer):
                        raise ValueError("Original comparison was modified by photo processing")
                    for _ in range(int(repetitions[i])):
                        encoder.stdin.write(memoryview(processed).cast("B"))
                        written += 1
                    if (i + 1) % 100 == 0 or i + 1 == count:
                        elapsed = time.perf_counter() - start
                        status = dict(
                            phase="encode-2560x1920-30p",
                            complete=written,
                            total=spec["frames"],
                            source_complete=i + 1,
                            source_total=count,
                            elapsed_seconds=elapsed,
                            fps=written / elapsed,
                            luminance_span=span,
                            image_history_mixing=False,
                        )
                        write_json(output / "status.json", status)
                        print(
                            f'2560x1920/30p encode: {written}/{spec["frames"]}; source {i+1}/{count}; {written/elapsed:.1f} fps; {elapsed:.1f}s',
                            flush=True,
                        )
            encoder.stdin.close()
            if decoder.wait() != 0 or original_decoder.wait() != 0 or encoder.wait() != 0:
                raise ValueError("FFmpeg decode/encode failed")
            if written != spec["frames"] or np.unique(mapping).size != count:
                raise ValueError("A source frame was dropped")
        except Exception:
            for process, log in [(decoder, decoder_log), (original_decoder, original_log), (encoder, encoder_log)]:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                log.seek(0)
                message = log.read().decode(errors="replace")
                if message:
                    print(message[-6000:], flush=True)
            raise
    result = dict(
        spec=spec,
        processing_seconds=time.perf_counter() - start,
        luminance_span=span,
        luminance_floor=0.008,
        luminance_target_mean=0.008 + span / 2,
        image_history_mixing=False,
        shadow_chroma_gain_limit=1.0,
        shadow_brightness_lift="Neutral luminance addition; source chroma never amplified",
        shadow_color_check_frames=count,
        near_black_source_pixels_checked=shadow_pixels,
        amplified_near_black_blue_pixels=0,
        amplified_near_black_magenta_pixels=0,
        maximum_near_black_output_channel_spread=shadow_spread,
        previous_frame_pixels_used=False,
        tone_curve="Current-frame-only exposure-normalized logarithmic luminance curve",
        source_adjacent_mean_luminance_difference_max=(
            float(np.max(np.abs(np.diff(source_means)))) if count > 1 else 0.0
        ),
        source_adjacent_mean_luminance_difference_p95=(
            float(np.percentile(np.abs(np.diff(source_means)), 95)) if count > 1 else 0.0
        ),
        original_screening_max_variation=float(peak[0]),
        original_screening_max_potential_general_area=int(peak[1]),
        original_screening_max_red_window_area=int(peak[2]),
        credits=credits[2] if credits else None,
        original_comparison_frames_verified=count,
        original_comparison_flash_or_color_processing=False,
        source_frame_map=mapping.tolist(),
    )
    if local_maps:
        manifest = output / "local-maps.json"
        result["local_map"] = dict(
            enabled=True,
            frames=count,
            manifest=str(manifest.resolve()),
            manifest_sha256=digest(manifest),
            native_local_box=local_maps.metadata["local_box"],
            native_world_box=local_maps.metadata["world_box"],
            native_mask_box=local_maps.metadata["mask_box"],
            north_up=True,
            persistent_nearby_history=True,
            native_original_box=local_maps.metadata["original_box"],
            native_footer_box=local_maps.metadata["footer_box"],
            original_and_mask_same_size=True,
            footer_only_under_left_photo=True,
            native_mask_source=True,
            elapsed_seconds=local_maps.metadata["elapsed_seconds"],
        )
    write_json(
        output / f"tone-curves-{attempt}.json",
        dict(exposure_log2=exposures, source_mean_relative_luminance=source_means),
    )
    write_json(output / f"encode-{attempt}.json", result)
    return path, result


def audit(movie, output, count):
    info = probe(movie)
    if (info["width"], info["height"], info["fps"]) != (WIDTH, HEIGHT, "30"):
        raise ValueError("Output dimensions or30.000fps changed")
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
        method="Relative sRGB luminance over every30-frame/1-second window. A pixel is a potential rapid general-flash location only if total variation>=0.55 AND its observed all-time luminance range>=0.10; a range below0.10 cannot contain a qualifying opposing luminance change. All-time range overbounds every one-second range. Union of ALL saturated-red pixels over31frames bounds red-flash area without chromaticity exemptions.",
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


def provenance(source, output, records, count, result):
    mapping = source_frame_map(count)
    settings = json.loads((source / "run.json").read_text())
    settings.update(
        schema="coastline-luminance-normalized-30p-v3",
        output_size=[WIDTH, HEIGHT],
        fps="30",
        frames=len(mapping),
        source_frames=count,
        searched_frames=len(records),
        display_content_box=[0, 0, WIDTH, HEIGHT],
        display_treatment="Satellite panel only: current-frame exposure and luminance normalization with neutral shadow lift and no chroma amplification; no previous-frame pixels or temporal image blending; original animation unchanged; fixed ending geography",
        audio_tempo=result["spec"]["audio_tempo"],
    )
    if result.get("local_map"):
        settings["local_map"] = result["local_map"]
    write_json(output / "run.json", settings)
    revised = []
    views = []
    for j, i in enumerate(mapping.tolist()):
        r = records[i]
        if i >= CREDIT_START:
            r = dict(
                records[CREDIT_PHOTO],
                stabilized_credits=True,
                display_score_valid=False,
                render_source_frame=CREDIT_PHOTO,
            )
        else:
            r = dict(r, display_score_valid=True, render_source_frame=i)
        r.update(frame=j, time=j / 30, original_source_frame=i, original_source_time=i * 1001 / 30000)
        revised.append(r)
        views.append(
            dict(
                frame=j,
                source_frame=r["render_source_frame"],
                original_source_frame=i,
                corners_lonlat=footprint(r).tolist(),
                photograph_sha256=r["satellite"]["image_sha256"],
                mirror_horizontal=bool(r.get("mirror_horizontal")),
            )
        )
    (output / "frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in revised), encoding="utf-8"
    )
    write_json(
        output / "badapple-coastline.footprints.json",
        dict(
            views=views,
            frames=len(mapping),
            ending_geography_fixed_from_source_frame=CREDIT_START,
            display_content_box=[0, 0, WIDTH, HEIGHT],
            source_map_box=result.get("local_map", {}).get("native_world_box", list(MAP_BOX)),
            history="All genuinely displayed preceding areas retained",
        ),
    )
    credit = (source / "CREDITS.txt").read_text(encoding="utf-8")
    credit += "\nSatellite-only luminance-normalized2560x1920/30pedition: all6955source frames displayed once each at30.000progressive/non-drop timecode, as requested. No frames added or dropped. Original-animation comparison decoded directly from its source and resized for display only, checked byte-for-byte before encoding; no flash, color or mirror treatment on the original comparison. Aspect-preserving2xupscale. Current-frame exposure/luminance normalization applies ONLY to the satellite photograph; no temporal image blending and no previous-frame pixels. Ending geographic view is fixed; scores n/a. Flash screening applies only to the satellite panel, not the whole composite. Scope and limits in flash-audit.json. Audio handling is recorded in summary.json.\n"
    if result.get("local_map"):
        credit += "LOCAL map: G.E. Kindermann global land/water mask,0.3arcsec, DOI10.5281/zenodo.14899246, CC BY4.0; original native10m pixels sampled with nearest neighbour in a north-up WGS84 azimuthal-equidistant projection. Gold outline is the true actual photograph footprint; cyan outlines retain all preceding nearby unique views. At least five photograph widths or20km across. No background satellite acquisition or invented geography. Detailed per-frame projection/corners/scale in local-maps.json.\n"
    (output / "CREDITS.txt").write_text(credit, encoding="utf-8")
    return revised


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
    elif mode == "unicode":
        command += ["-c:a", "aac", "-b:a", "192k", "-t", f"{frames/30:.9f}"]
    else:
        command += ["-af", "atempo=1.001", "-c:a", "aac", "-b:a", "192k", "-t", f"{frames/30:.9f}"]
    return command + ["-timecode", "00:00:00:00", "-movflags", "+faststart", str(final)]


def finish_audio(source=SOURCE, output=OUTPUT, mode="unicode", assets=True):
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
        audio_bitrate="192k" if mode == "unicode" else None,
        audio_implementation="badapple_unicode.video.render_video" if mode == "unicode" else mode,
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


def run(source=SOURCE, output=OUTPUT, max_frames=None, span=0.07, assets=True, audio_mode="unicode", local_map=False):
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    movie = source / "badapple-coastline.mp4"
    records = read_records(source / "frames.jsonl")
    count = min(len(records), max_frames or len(records))
    initial_sha = digest(movie)
    started = time.perf_counter()
    write_json(output / "status.json", dict(phase="prepare", complete=0, total=count))
    try:
        local_maps = None
        if local_map:
            from .local_map import prepare

            local_maps = prepare([records[i if i < CREDIT_START else CREDIT_PHOTO] for i in range(count)], output)
        for attempt, choice in enumerate([span, span * 0.875, span * 0.75], 1):
            candidate, result = encode(movie, records, output, count, choice, attempt, local_maps)
            report = audit(candidate, output, result["spec"]["frames"])
            write_json(output / f"flash-audit-{attempt}.json", report)
            if report["passed"]:
                break
            print(
                f"Conservative flash screening failed; narrowing luminance range for attempt{attempt+1}. No temporal image mixing.",
                flush=True,
            )
        if not report["passed"]:
            raise ValueError("Flash screening remains above the conservative limit")
        final = output / "validated-video.mp4"
        candidate.replace(final)
        revised = provenance(source, output, records, count, result)
        metadata = probe(final)
        metadata.update(
            frames=result["spec"]["frames"],
            world_map=dict(enabled=True, persistent_history=True),
            display_content_box=[0, 0, WIDTH, HEIGHT],
            source_canvas=[1280, 960],
        )
        write_json(output / "badapple-coastline.verification.json", metadata)
        if digest(movie) != initial_sha:
            raise ValueError("Original completed movie changed")
        summary = dict(
            frames=result["spec"]["frames"],
            source_frames=count,
            movie_sha256=digest(final),
            source_movie_sha256=initial_sha,
            source_movie_preserved=True,
            output_spec=result["spec"],
            flash_audit=report,
            credits=result["credits"],
            mirrored_frames=sum(bool(r.get("mirror_horizontal")) for r in revised),
            processing=result,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "summary.json", summary)
        if audio_mode == "pending":
            write_json(
                output / "status.json",
                dict(
                    phase="validated-video-awaiting-audio-choice",
                    complete=result["spec"]["frames"],
                    total=result["spec"]["frames"],
                ),
            )
            print("Satellite-only video validated; audio choice pending. No need to re-encode video.", flush=True)
        else:
            finish_audio(source, output, audio_mode, assets and count == len(records))
    except Exception as exc:
        write_json(output / "status.json", dict(phase="failed", error=str(exc), total=count))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-map", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--luminance-span", type=float, default=0.07)
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--audio-mode", choices=["pending", "original", "sync", "unicode"], default="unicode")
    parser.add_argument("--finish-audio", action="store_true")
    args = parser.parse_args()
    args.output = args.output or (LOCAL_OUTPUT if args.local_map else OUTPUT)
    if args.finish_audio:
        if args.audio_mode == "pending":
            parser.error("--finish-audio requires an audio choice")
        finish_audio(args.source, args.output, args.audio_mode, not args.skip_assets)
    else:
        run(
            args.source,
            args.output,
            args.max_frames,
            args.luminance_span,
            not args.skip_assets,
            args.audio_mode,
            args.local_map,
        )
