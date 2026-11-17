"""Compose real satellite views, reference silhouettes, coordinates and credits."""

from __future__ import annotations

import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from badapple_media import Frames, ffmpeg_binary, probe, ui_font
from .atlas import Atlas
from .data import digest


@lru_cache(maxsize=12)
def font_for(size, monospace=False):
    return ui_font(size, monospace=monospace)


@lru_cache(maxsize=12)
def fixed_footer(capture_years="2020/2021", landsat_used=False, balanced=False):
    footer = Image.new("RGB", (1280, 160), "#080f17")
    draw = ImageDraw.Draw(footer)
    muted = "#9eafc0"
    labels = [
        (20, 11, "NEAR"),
        (850, 11, "DIST km"),
        (20, 48, "LAT"),
        (300, 48, "LON"),
        (650, 48, "WIDTH km"),
        (980, 48, "ROT deg"),
        (20, 82, "FROM"),
        (255, 82, "TO"),
        (490, 82, "mIoU" if balanced else "IoU"),
        (650, 82, "MSE"),
        (855, 82, "CLOUD"),
    ]
    for x, y, text in labels:
        draw.text((x, y), text, font=font_for(17, True), fill=muted)
    imagery_credit = f"Contains modified Copernicus Sentinel data {capture_years} | Earth Search / AWS Open Data"
    if landsat_used:
        imagery_credit += " | Landsat: U.S. Geological Survey (public domain)"
    draw.text((20, 111), imagery_credit, font=font_for(12), fill=muted)
    draw.text(
        (20, 127),
        "Mask: G. E. Kindermann (2025), doi:10.5281/zenodo.14899246, CC BY 4.0; cropped/resampled/rotated | Places: Natural Earth",
        font=font_for(11),
        fill=muted,
    )
    draw.text(
        (20, 143),
        "© OpenStreetMap contributors | https://www.openstreetmap.org/copyright | © ESA WorldCover project 2020/2021 | Full notices: CREDITS.txt",
        font=font_for(11),
        fill=muted,
    )
    return footer


def clipped(draw, text, x, y, width, font, fill):
    while text and draw.textlength(text, font=font) > width:
        text = text[:-1]
    draw.text((x, y), text, font=font, fill=fill)


def compose(
    image: Image.Image,
    target,
    mask,
    record: dict,
    index: int,
    count: int,
    global_coverage: bool,
    capture_years="2020/2021",
    landsat_used=False,
    world_map_panel=None,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 960), "#080f17")
    draw = ImageDraw.Draw(canvas)
    font, small, mono = font_for(24), font_for(16), font_for(19, True)
    accent, muted = "#73e6d7", "#9eafc0"
    title = "BAD APPLE / REAL EARTH" if global_coverage else "BAD APPLE / REAL EARTH - REGIONAL PILOT"
    draw.text((20, 17), title, font=font, fill="white")
    draw.text((1000, 22), f"{index+1:05d} / {count:05d}", font=mono, fill=muted)
    canvas.paste(image.convert("RGB").resize((960, 720), Image.Resampling.LANCZOS), (20, 70))
    draw.text((1000, 62), "ORIGINAL SILHOUETTE", font=small, fill=accent)
    canvas.paste(Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255)).convert("RGB").resize((260, 195)), (1000, 92))
    draw.text((1000, 322), "MATCHED LAND / WATER", font=small, fill=accent)
    canvas.paste(Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255)).convert("RGB").resize((260, 195)), (1000, 352))
    if world_map_panel is None:
        draw.text((1000, 570), "WHITE  land", font=mono, fill="white")
        draw.text((1000, 603), "BLACK  water", font=mono, fill=muted)
        draw.text(
            (1000, 663),
            "MIRRORED VIEW" if record.get("mirror_horizontal") else "ONE GEOGRAPHIC VIEW",
            font=small,
            fill=accent,
        )
        draw.text((1000, 698), "No shape synthesis", font=small, fill=muted)
        draw.text((1000, 733), "Lakes/rivers included", font=small, fill=muted)
    else:
        from .footprint_map import MAP_BOX

        draw.text((1000, 550), "WHITE land / BLACK water", font=font_for(12), fill=muted)
        draw.text((1000, 581), "WORLD MAP / USED AREAS", font=small, fill=accent)
        canvas.paste(world_map_panel, MAP_BOX[:2])
        draw.rectangle((1002, 750, 1010, 756), outline="#ffcf6c", width=1)
        draw.text((1016, 745), "CURRENT", font=font_for(11), fill="#ffcf6c")
        draw.rectangle((1120, 750, 1128, 756), outline="#43aebc", width=1)
        draw.text((1134, 745), "HISTORY", font=font_for(11), fill=accent)
        draw.text((1000, 763), "Small boxes enlarged / Natural Earth", font=font_for(10), fill=muted)
        draw.text(
            (1000, 780),
            "MIRRORED VIEW" if record.get("mirror_horizontal") else "ONE GEOGRAPHIC VIEW",
            font=font_for(12),
            fill=accent,
        )
    canvas.paste(fixed_footer(capture_years, landsat_used, "native_miou" in record), (0, 800))
    p = record["place"]
    value_font = font_for(19, True)
    clipped(draw, p["name"], 85, 811, 365, value_font, "white")
    clipped(draw, p["country"], 475, 811, 350, value_font, "white")
    draw.text((950, 811), f"{p['distance_km']:6.0f}", font=value_font, fill="white")
    lat, lon = record["latitude"], record["longitude"]
    draw.text((70, 848), f"{abs(lat):09.5f} {'N' if lat>=0 else 'S'}", font=value_font, fill=accent)
    draw.text((350, 848), f"{abs(lon):010.5f} {'E' if lon>=0 else 'W'}", font=value_font, fill=accent)
    draw.text((755, 848), f"{record['width_km']:7.2f}", font=value_font, fill="white")
    draw.text((1070, 848), f"{record['rotation_clockwise']:5.1f}", font=value_font, fill="white")
    dates = sorted(
        {
            date[:10]
            for s in record["satellite"]["scenes"]
            for date in [s.get("start_datetime", s["datetime"]), s.get("end_datetime", s["datetime"])]
        }
    )
    date_font = font_for(16, True)
    draw.text((75, 882), dates[0] if dates else "----------", font=date_font, fill=muted)
    draw.text((290, 882), dates[-1] if dates else "----------", font=date_font, fill=muted)
    draw.text(
        (535, 882),
        f"{record.get('native_miou',record.get('native_iou',record.get('iou',0))):5.3f}",
        font=date_font,
        fill=muted,
    )
    mse = record.get("native_mse", record["mse"])
    draw.text((695, 882), f"{mse:6.4f}", font=date_font, fill=muted)
    cloud = None if record["satellite"].get("parent_cloud_metrics") else record["satellite"].get("cloud_fraction")
    draw.text((920, 882), f"{cloud:6.1%}" if cloud is not None else "   n/a", font=date_font, fill=muted)
    source_label = (
        "NO IMAGE"
        if record["satellite"].get("placeholder")
        else (
            "S2+LANDSAT"
            if record["satellite"].get("mixed_sensors")
            else (
                "LANDSAT"
                if record.get("landsat_replacement")
                else "S2 ANNUAL" if record["satellite"].get("annual_composite") else "SENTINEL-2"
            )
        )
    )
    draw.text((1060, 882), source_label, font=date_font, fill=muted)
    return canvas


def render(
    source: Path,
    frames: Frames,
    records: list[dict],
    atlas: Atlas,
    output: Path,
    encoder="libx264",
    quality_policy=None,
    world_map=False,
) -> dict:
    if len(records) != frames.count or any("satellite" not in r for r in records):
        raise ValueError("Every frame must have complete Sentinel-2 imagery before rendering")
    checked = set()
    for r in records:
        if (r["satellite"]["coverage"] != 1 and not r["satellite"].get("placeholder")) or not Path(
            r["satellite"]["path"]
        ).is_file():
            raise ValueError("Missing satellite pixels/image; refusing synthetic fill")
        if digest(Path(r["satellite"]["path"])) != r["satellite"]["image_sha256"]:
            raise ValueError("Satellite image changed; rerun imagery to verify/retrieve the cache")
        satellite = r["satellite"]
        if satellite["image_sha256"] not in checked:
            if satellite.get("placeholder"):
                with Image.open(satellite["path"]) as placeholder:
                    if np.asarray(placeholder).any():
                        raise ValueError("Unavailable imagery placeholder must be completely black")
            elif quality_policy:
                from .quality import file_quality

                if not file_quality(satellite["path"])["acceptable"]:
                    raise ValueError(f"Frame {r['frame']}: whiteout remains; repair before rendering")
            checked.add(satellite["image_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    command = [
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
        str(frames.fps),
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        encoder,
    ]
    command += (
        ["-preset", "medium", "-qp", "14", "-bf", "0", "-g", "30"]
        if encoder == "libx264"
        else [
            "-preset",
            "p5",
            "-rc",
            "constqp",
            "-qp",
            "14",
            "-bf",
            "0",
            "-g",
            "30",
            "-spatial-aq",
            "0",
            "-temporal-aq",
            "0",
        ]
    )
    command += [
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{float(frames.count / frames.fps):.9f}",
        "-movflags",
        "+faststart",
        str(partial),
    ]

    @lru_cache(maxsize=4)
    def load(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    thumbs = frames.thumbnails()
    capture_years = "/".join(
        sorted(
            {"2020", "2021"}
            | {
                s["datetime"][:4]
                for r in records
                for s in r["satellite"]["scenes"]
                if s.get("collection") != "landsat-c2-l2"
            }
        )
    )
    landsat_used = any(r.get("landsat_replacement") for r in records)
    history = None
    if world_map:
        from .footprint_map import FootprintHistory

        history = FootprintHistory()
    poster = None
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i, record in enumerate(records):
                target = (
                    np.asarray(
                        Image.fromarray(thumbs[i]).resize((260, 195), Image.Resampling.LANCZOS), dtype=np.float32
                    )
                    / 255
                )
                if "native_mask_path" in record:
                    with Image.open(record["native_mask_path"]) as image:
                        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255
                else:
                    mask, _ = atlas.sample(
                        record["longitude"],
                        record["latitude"],
                        record["width_km"],
                        record["rotation_clockwise"],
                        (260, 195),
                    )
                panel = history.panel(record, i) if history else None
                canvas = compose(
                    load(record["satellite"]["path"]),
                    target,
                    mask,
                    record,
                    i,
                    frames.count,
                    atlas.metadata["global_coverage"],
                    capture_years,
                    landsat_used,
                    panel,
                )
                process.stdin.write(canvas.tobytes())
                if i == min(frames.count - 1, frames.count // 2):
                    poster = canvas.copy()
                if (i + 1) % 100 == 0:
                    print(f"render: {i+1}/{frames.count}", flush=True)
            process.stdin.close()
            if process.wait():
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    partial.replace(output)
    if poster is not None:
        poster.save(output.with_suffix(".poster.png"))
    metadata = probe(output)
    info = json.loads(
        subprocess.check_output(
            [
                ffmpeg_binary("ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
                str(output),
            ]
        )
    )
    count = int(info["streams"][0]["nb_frames"])
    if count != frames.count or metadata["fps"] != str(frames.fps):
        raise RuntimeError("Encoded frame count or FPS mismatch")
    if probe(source)["has_audio"] and not metadata["has_audio"]:
        raise RuntimeError("Source audio was lost")
    metadata.update(
        frames=count, output=str(output.resolve()), global_search_coverage=atlas.metadata["global_coverage"]
    )
    if history:
        metadata["world_map"] = history.save(output.with_suffix(".footprints.json"))
    output.with_suffix(".verification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
