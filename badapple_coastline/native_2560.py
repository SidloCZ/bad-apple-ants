"""Native 2560x1920 edition: every panel drawn at output size, AV1 delivery.

The1280x960 compose-then-2x-upscale path blurred photographs and text, and the
tone curve amplified that intermediate H.264's compression blocks in dark water.
Here photographs and masks are read from their lossless sources, resampled once
to their display size, tone-mapped by black/white levels and encoded once.
"""

from __future__ import annotations
import argparse, json, subprocess, tempfile, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from badapple_media import ffmpeg_binary, probe
from .data import digest
from .pipeline import read_records, write_json
from .flash_cuda import FlashCuda
from .flash_reduction import SOURCE, CREDIT_START, CREDIT_PHOTO, fill_frame, audit, finish_audio
from .footprint_map import FootprintHistory, footprint, polygons, SCALE, CURRENT
from .local_map import LocalMaps, NativePointSampler, geometry, map_xy, dense_boundary, GOLD

OUTPUT = Path("output/coastline-precision-v4-2560x1920-30p-native-av1")
WIDTH, HEIGHT = 2560, 1920
NAVY = (8, 15, 23)
ACCENT = "#73e6d7"
MUTED = "#9eafc0"
RULE = (30, 45, 56)
PHOTO = (40, 140, 1960, 1580)
ORIGINAL = (2000, 184, 2520, 574)
MASK = (2000, 650, 2520, 1040)
WORLD = (2000, 1116, 2520, 1376)
LOCAL = (2000, 1488, 2520, 1880)
LOCAL_SIZE = (LOCAL[2] - LOCAL[0], LOCAL[3] - LOCAL[1])
COLUMNS = [40, 520, 1000, 1480]
HISTORY_SOFT = (67, 174, 188, 105)
FONTS = Path("C:/Windows/Fonts")


@lru_cache(maxsize=32)
def font(size, kind="regular"):
    name = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "mono": "CascadiaMono.ttf"}[kind]
    for path in [FONTS / name, FONTS / ("consola.ttf" if kind == "mono" else "arial.ttf")]:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def clipped(draw, xy, text, width, face, fill):
    while text and draw.textlength(text, font=face) > width:
        text = text[:-1]
    draw.text(xy, text, font=face, fill=fill)


class WorldMap(FootprintHistory):
    """Same persistent history, drawn natively and less densely than the1280 edition."""

    def __init__(self):
        super().__init__(size=(WORLD[2] - WORLD[0], WORLD[3] - WORLD[1]))

    def panel(self, record, index):
        if index != self.next_index:
            raise ValueError("World-map history must advance in video order")
        image = self.current(record)
        self.advance(record, index)
        return image

    def current(self, record):
        panel = Image.alpha_composite(self.base, self.history)
        draw = ImageDraw.Draw(panel)
        for polygon in polygons(footprint(record), self.internal, 10 * SCALE):
            line = [tuple(p) for p in polygon]
            line.append(line[0])
            draw.line(line, fill=(5, 15, 23, 255), width=4 * SCALE, joint="curve")
            draw.line(line, fill=CURRENT, width=2 * SCALE, joint="curve")
        return panel.convert("RGB").resize(self.size, Image.Resampling.LANCZOS)

    def advance(self, record, index):
        if index != self.next_index:
            raise ValueError("World-map history must advance in video order")
        points = footprint(record)
        key = tuple(sorted(map(tuple, points.tolist())))
        if key not in self.seen:
            draw = ImageDraw.Draw(self.history)
            for polygon in polygons(points, self.internal, 3 * SCALE):
                line = [tuple(p) for p in polygon]
                line.append(line[0])
                draw.line(line, fill=HISTORY_SOFT, width=SCALE, joint="curve")
            self.seen.add(key)
        self.views.append(
            dict(
                index=index,
                corners_lonlat=points.tolist(),
                photograph_sha256=record.get("satellite", {}).get("image_sha256"),
                mirror_horizontal=bool(record.get("mirror_horizontal")),
            )
        )
        self.next_index += 1


class NativeLocalMaps(LocalMaps):
    """LOCAL map sampled from the native10m mask directly at its display size."""

    def __init__(self, records, sampler=None):
        super().__init__(records, LOCAL_SIZE, sampler)

    def render(self, index):
        points = self.points[index]
        width, height = self.size
        image, transform, lon, lat, span = self.base(tuple(points.ravel()))
        image = image.copy()
        draw = ImageDraw.Draw(image)
        unit = self.units[index]
        dot = (self.units[self.first] * unit).sum(1)
        distances = 6371.01 * np.arccos(np.clip(dot, -1, 1))
        nearby = np.flatnonzero(
            (self.first < index) & (distances < 1.02 * (span * np.hypot(1, height / width) / 2 + self.radii) + 2)
        )
        for polygon in map_xy(transform, self.dense[nearby], span, self.size) if len(nearby) else []:
            if np.isfinite(polygon).all():
                draw.line([*map(tuple, polygon), tuple(polygon[0])], fill=(44, 118, 130), width=2)
        boundary = [*map(tuple, map_xy(transform, dense_boundary(points), span, self.size))]
        boundary.append(boundary[0])
        draw.line(boundary, fill=NAVY, width=8, joint="curve")
        draw.line(boundary, fill=GOLD, width=4, joint="curve")
        ink = "#bdcfdb"
        x = width - 28
        draw.line((x, 72, x, 42), fill=ink, width=2)
        draw.line((x - 8, 52, x, 42, x + 8, 52), fill=ink, width=2)
        draw.text((x, 36), "N", font=font(20, "semibold"), fill=ink, anchor="ms")
        target = span / 4
        power = 10 ** np.floor(np.log10(target))
        bar = max(v * power for v in [1, 2, 5] if v * power <= target)
        pixels = width * bar / span
        draw.line((18, height - 20, 18 + pixels, height - 20), fill=ink, width=3)
        draw.text((18, height - 30), f"{bar:g} km", font=font(20), fill=ink, anchor="ls")
        meta = dict(
            frame=index,
            centre_lonlat=[lon, lat],
            span_km=span,
            nearby_past_unique_views=len(nearby),
            current_corners_lonlat=points.tolist(),
            photograph_sha256=self.records[index]["satellite"]["image_sha256"],
        )
        return np.asarray(image), meta


def prepare_local_maps(records, output, workers=8):
    assets = output / "render-assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / "local-maps-native.u8"
    manifest = output / "local-maps.json"
    shape = (len(records), LOCAL_SIZE[1], LOCAL_SIZE[0], 3)
    identity = (
        digest(Path(__file__))
        + digest(Path(__file__).with_name("local_map.py"))
        + json.dumps([footprint(r).tolist() for r in records])
    )
    key = __import__("hashlib").sha256(identity.encode()).hexdigest()
    if (
        manifest.is_file()
        and path.is_file()
        and json.loads(manifest.read_text()).get("identity") == key
        and path.stat().st_size == int(np.prod(shape))
    ):
        old = json.loads(manifest.read_text())
        return np.memmap(path, np.uint8, "r", shape=shape), old
    renderer = NativeLocalMaps(records)
    frames = np.memmap(path, np.uint8, "w+", shape=shape)
    views = []
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (image, meta) in enumerate(pool.map(renderer.render, range(len(records)))):
                frames[i] = image
                views.append(meta)
                if (i + 1) % 250 == 0 or i + 1 == len(records):
                    write_json(
                        output / "status.json", dict(phase="native10m-local-maps", complete=i + 1, total=len(records))
                    )
                    print(f"Native local maps: {i+1}/{len(records)}; {time.perf_counter()-start:.1f}s", flush=True)
        frames.flush()
    finally:
        renderer.sampler.close()
    visited = np.searchsorted(renderer.first, np.arange(len(records)), side="right").tolist()
    result = dict(
        identity=key,
        frames=len(records),
        size=list(LOCAL_SIZE),
        box=list(LOCAL),
        visited_counts=visited,
        projection="WGS84 azimuthal equidistant centred on actual photograph; north up",
        native_source="G.E. Kindermann global land/water mask,0.3arcsec; lossless original pixels, nearest neighbour, sampled at display size",
        history="All preceding unique nearby photo footprints, no future views or truncation",
        views=views,
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(manifest, result)
    del frames
    return np.memmap(path, np.uint8, "r", shape=shape), result


def capture_dates(record):
    dates = sorted(
        {
            date[:10]
            for s in record["satellite"]["scenes"]
            for date in [s.get("start_datetime", s["datetime"]), s.get("end_datetime", s["datetime"])]
        }
    )
    if not dates:
        return "----------"
    return dates[0] if dates[0] == dates[-1] else f"{dates[0]} – {dates[-1]}"


def source_label(record):
    s = record["satellite"]
    return (
        "NO IMAGE"
        if s.get("placeholder")
        else (
            "S2 + LANDSAT"
            if s.get("mixed_sensors")
            else (
                "LANDSAT 8/9"
                if record.get("landsat_replacement")
                else "S2 ANNUAL" if s.get("annual_composite") else "SENTINEL-2"
            )
        )
    )


def static_layer(capture_years, landsat_used):
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    draw.text((40, 96), "BAD APPLE / REAL EARTH", font=font(46, "semibold"), fill="white", anchor="ls")
    for (x0, y0, x1, y1), title in [
        (ORIGINAL, "ORIGINAL SILHOUETTE"),
        (MASK, "MATCHED LAND / WATER"),
        (WORLD, "WORLD MAP"),
        (LOCAL, "LOCAL"),
    ]:
        draw.text((x0, y0 - 14), title, font=font(26, "semibold"), fill=ACCENT, anchor="ls")
        draw.rectangle((x0 - 1, y0 - 1, x1, y1), outline=(42, 60, 72), width=1)
    y = WORLD[3] + 30
    draw.rectangle((2000, y - 14, 2016, y - 2), outline="#ffcf6c", width=2)
    draw.text((2026, y), "current view", font=font(19), fill=MUTED, anchor="ls")
    draw.rectangle((2186, y - 14, 2202, y - 2), outline="#43aebc", width=2)
    draw.text((2212, y), "earlier views (small ones enlarged)", font=font(19), fill=MUTED, anchor="ls")
    draw.line((40, 1782, 1960, 1782), fill=RULE, width=1)
    credit = f"Contains modified Copernicus Sentinel data {capture_years}  |  Earth Search / AWS Open Data"
    if landsat_used:
        credit += "  |  Landsat: U.S. Geological Survey (public domain)"
    lines = [
        credit,
        "Mask: G. E. Kindermann (2025), doi:10.5281/zenodo.14899246, CC BY 4.0; cropped / resampled / rotated  |  Places & world map: Natural Earth",
        "© OpenStreetMap contributors  |  openstreetmap.org/copyright  |  © ESA WorldCover project 2020/2021  |  Full notices: CREDITS.txt",
    ]
    for k, text in enumerate(lines):
        draw.text((40, 1812 + 29 * k), text, font=font(19), fill=MUTED, anchor="ls")
    return np.asarray(image)


def draw_text_layer(draw, record, index, count, visited, local_span, credits):
    face = font(34, "mono")
    right = WIDTH - 40
    draw.text(
        (right, 96), "ENDING CREDITS" if credits else f"{index+1:05d} / {count:05d}", font=face, fill=MUTED, anchor="rs"
    )
    view = "FIXED VIEW" if credits else "MIRRORED VIEW" if record.get("mirror_horizontal") else "TRUE VIEW"
    small = font(21)
    draw.text(
        (right, MASK[1] - 14), view, font=small, fill="#ffcf6c" if view == "MIRRORED VIEW" else MUTED, anchor="rs"
    )
    draw.text((right, WORLD[1] - 14), f"{visited:,} places so far", font=small, fill=MUTED, anchor="rs")
    draw.text(
        (right, LOCAL[1] - 14),
        f"{local_span:.1f} km across" if local_span < 1000 else f"{local_span:.0f} km across",
        font=small,
        fill=MUTED,
        anchor="rs",
    )
    label, value, mono = font(19, "semibold"), font(31), font(31, "mono")
    p = record["place"]
    lat, lon = record["latitude"], record["longitude"]
    score = record.get("native_miou", record.get("native_iou", record.get("iou")))
    cloud = None if record["satellite"].get("parent_cloud_metrics") else record["satellite"].get("cloud_fraction")
    cells = [
        (0, 0, 2, "NEAREST MAPPED PLACE", f"{p['name']}, {p['country']}  ·  {p['distance_km']:.0f} km", value, "white"),
        (2, 0, 1, "LATITUDE", f"{abs(lat):.5f}° {'N' if lat>=0 else 'S'}", mono, ACCENT),
        (3, 0, 1, "LONGITUDE", f"{abs(lon):.5f}° {'E' if lon>=0 else 'W'}", mono, ACCENT),
        (0, 1, 1, "CAPTURED", capture_dates(record), mono, "white"),
        (
            1,
            1,
            1,
            "VIEW WIDTH  ·  ROTATION",
            f"{record['width_km']:.2f} km · {record['rotation_clockwise']:.1f}°",
            mono,
            "white",
        ),
        (2, 1, 1, "MATCH SCORE (mIoU)", "n/a" if credits else f"{score:.3f}", mono, "white"),
        (
            3,
            1,
            1,
            "IMAGERY  ·  CLOUD",
            f"{source_label(record)} · {'n/a' if cloud is None else f'{cloud:.1%}'}",
            mono,
            "white",
        ),
    ]
    for column, row, span, title, text, face, fill in cells:
        x = COLUMNS[column]
        y = 1636 + 86 * row
        draw.text((x, y), title, font=label, fill=MUTED, anchor="ls")
        clipped(draw, (x, y + 10), text, 480 * span - 24, face, fill)


def load_photo(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((PHOTO[2] - PHOTO[0], PHOTO[3] - PHOTO[1]), Image.Resampling.LANCZOS)
        )


def load_mask(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((MASK[2] - MASK[0], MASK[3] - MASK[1]), Image.Resampling.LANCZOS).convert("RGB")
        )


class Composer:
    def __init__(self, records, output, count):
        self.records = records
        self.count = count
        self.shown = [records[i if i < CREDIT_START else CREDIT_PHOTO] for i in range(count)]
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
        self.static = static_layer(capture_years, any(r.get("landsat_replacement") for r in records))
        self.local, self.local_meta = prepare_local_maps(self.shown, output)
        self.world = WorldMap()
        self.frozen = None

    def world_panel(self, i):
        if i < CREDIT_START:
            return np.asarray(self.world.panel(self.shown[i], i))
        if self.frozen is None:
            self.frozen = np.asarray(self.world.current(self.shown[i]))
        self.world.advance(self.shown[i], i)
        return self.frozen

    def compose(self, i, photo, mask, original):
        frame = self.static.copy()
        record = self.shown[i]
        frame[PHOTO[1] : PHOTO[3], PHOTO[0] : PHOTO[2]] = photo
        frame[ORIGINAL[1] : ORIGINAL[3], ORIGINAL[0] : ORIGINAL[2]] = original
        frame[MASK[1] : MASK[3], MASK[0] : MASK[2]] = mask
        frame[WORLD[1] : WORLD[3], WORLD[0] : WORLD[2]] = self.world_panel(i)
        frame[LOCAL[1] : LOCAL[3], LOCAL[0] : LOCAL[2]] = self.local[i]
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw_text_layer(
            draw, record, i, self.count, len(self.world.seen), self.local_meta["views"][i]["span_km"], i >= CREDIT_START
        )
        return np.asarray(image)


def sources(composer, pool, lookahead=24):
    """Lossless photo and mask reads, decoded ahead in a bounded window."""
    queue = deque()

    def submit(i):
        r = composer.shown[i]
        queue.append((pool.submit(load_photo, r["satellite"]["path"]), pool.submit(load_mask, r["native_mask_path"])))

    for i in range(min(lookahead, composer.count)):
        submit(i)
    for i in range(composer.count):
        photo, mask = queue.popleft()
        if i + lookahead < composer.count:
            submit(i + lookahead)
        yield photo.result(), mask.result()


def encoder_command(path, count, bitrate):
    return [
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
        f"{WIDTH}x{HEIGHT}",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-vf",
        "scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=p010le",
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-multipass",
        "fullres",
        "-rc",
        "vbr",
        "-b:v",
        f"{bitrate}M",
        "-maxrate",
        f"{bitrate*2}M",
        "-bufsize",
        f"{bitrate*4}M",
        "-rc-lookahead",
        "32",
        "-spatial-aq",
        "1",
        "-g",
        "60",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-an",
        "-frames:v",
        str(count),
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        str(path),
    ]


def encode(source, records, output, count, span, attempt, bitrate):
    start = time.perf_counter()
    path = output / f"candidate-{attempt}.mp4"
    settings = json.loads((source / "run.json").read_text())
    original_movie = Path(settings["video_source"])
    if digest(original_movie) != settings["video_source_sha256"]:
        raise ValueError("Original animation source changed")
    ow, oh = ORIGINAL[2] - ORIGINAL[0], ORIGINAL[3] - ORIGINAL[1]
    original_command = [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(original_movie),
        "-an",
        "-vf",
        f"scale={ow}:{oh}:flags=lanczos",
        "-frames:v",
        str(count),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    composer = Composer(records, output, count)
    original = np.empty((oh, ow, 3), np.uint8)
    levels = []
    shadow_pixels = 0
    shadow_spread = 0
    written = 0
    with (
        tempfile.TemporaryFile() as original_log,
        tempfile.TemporaryFile() as encoder_log,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        original_decoder = subprocess.Popen(
            original_command, stdout=subprocess.PIPE, stderr=original_log, bufsize=1024 * 1024
        )
        encoder = subprocess.Popen(
            encoder_command(path, count, bitrate), stdin=subprocess.PIPE, stderr=encoder_log, bufsize=32 * 1024 * 1024
        )
        try:
            with FlashCuda(WIDTH, HEIGHT, span=span) as processor:
                for i, (photo, mask) in enumerate(sources(composer, pool)):
                    if not fill_frame(original_decoder.stdout, original):
                        raise ValueError("Original comparison is missing a source frame")
                    frame = composer.compose(i, photo, mask, original)
                    processed = processor.process_levels(frame)
                    levels.append(list(processor.levels))
                    shadows = processor.shadow_metrics()
                    if shadows[0] or shadows[1]:
                        raise ValueError(f"Blue/magenta near-black amplification at frame {i}")
                    shadow_pixels += int(shadows[2])
                    shadow_spread = max(shadow_spread, int(shadows[3]))
                    outside = np.ones(processed.shape[:2], bool)
                    outside[PHOTO[1] : PHOTO[3], PHOTO[0] : PHOTO[2]] = False
                    if not np.array_equal(processed[outside], frame[outside]):
                        raise ValueError("Tone mapping touched pixels outside the photograph")
                    encoder.stdin.write(memoryview(processed).cast("B"))
                    written += 1
                    if (i + 1) % 100 == 0 or i + 1 == count:
                        elapsed = time.perf_counter() - start
                        write_json(
                            output / "status.json",
                            dict(
                                phase="encode-native-2560x1920-av1",
                                attempt=attempt,
                                complete=written,
                                total=count,
                                elapsed_seconds=elapsed,
                                fps=written / elapsed,
                                luminance_span=span,
                            ),
                        )
                        print(
                            f"native2560 AV1 encode: {written}/{count}; {written/elapsed:.1f} fps; {elapsed:.0f}s",
                            flush=True,
                        )
            encoder.stdin.close()
            if original_decoder.wait() != 0 or encoder.wait() != 0:
                raise ValueError("FFmpeg decode/encode failed")
        except Exception:
            for process, log in [(original_decoder, original_log), (encoder, encoder_log)]:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                log.seek(0)
                message = log.read().decode(errors="replace")
                if message:
                    print(message[-6000:], flush=True)
            raise
    composer.world.save(output / "world-map-history.json")
    spec = dict(
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
        codec="AV1 (av1_nvenc, p7, two-pass full-res VBR)",
        video_bitrate_target_mbps=bitrate,
        video_bitrate_max_mbps=bitrate * 2,
        bit_depth=10,
        pixel_format="yuv420p10le",
        content_box=[0, 0, WIDTH, HEIGHT],
        scaling="Native2560x1920 composition; photographs1444x1080->1920x1440 and masks/originals resampled once from lossless sources",
    )
    write_json(
        output / f"tone-levels-{attempt}.json",
        dict(columns=["black", "white"], gamma=0.75, floor=0.002, span=span, saturation=1.15, rows=levels),
    )
    result = dict(
        spec=spec,
        processing_seconds=time.perf_counter() - start,
        luminance_span=span,
        luminance_floor=0.002,
        tone_curve="Current-frame-only black/white levels (0.5th/99.5th luminance percentile, black<=0.01, white>=0.06), gamma0.75, into [floor,floor+span]",
        chroma="1.15x saturation within gamut; chroma gain<=1 for near-black sources, rising to3x above relative luminance0.01; saturated red limited",
        image_history_mixing=False,
        previous_frame_pixels_used=False,
        near_black_source_pixels_checked=shadow_pixels,
        amplified_near_black_blue_pixels=0,
        amplified_near_black_magenta_pixels=0,
        maximum_near_black_output_channel_spread=shadow_spread,
        pixels_outside_photo_unchanged_frames=count,
        original_comparison_flash_or_color_processing=False,
        layout=dict(photo=PHOTO, original=ORIGINAL, mask=MASK, world=WORLD, local=LOCAL),
    )
    write_json(output / f"encode-{attempt}.json", result)
    return path, result


def provenance(source, output, records, count, result, report):
    settings = json.loads((source / "run.json").read_text())
    settings.update(
        schema="coastline-native-2560-av1-v1",
        output_size=[WIDTH, HEIGHT],
        fps="30",
        frames=count,
        source_frames=count,
        display_content_box=[0, 0, WIDTH, HEIGHT],
        codec="av1",
        display_treatment=result["tone_curve"],
    )
    write_json(output / "run.json", settings)
    revised = []
    views = []
    for i in range(count):
        r = (
            dict(
                records[CREDIT_PHOTO],
                stabilized_credits=True,
                display_score_valid=False,
                render_source_frame=CREDIT_PHOTO,
            )
            if i >= CREDIT_START
            else dict(records[i], display_score_valid=True, render_source_frame=i)
        )
        r.update(frame=i, time=i / 30, original_source_frame=i, original_source_time=i * 1001 / 30000)
        revised.append(r)
        views.append(
            dict(
                frame=i,
                source_frame=r["render_source_frame"],
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
            frames=count,
            ending_geography_fixed_from_source_frame=CREDIT_START,
            world_map_box=list(WORLD),
            local_map_box=list(LOCAL),
            history="All genuinely displayed preceding areas retained",
        ),
    )
    landsat = sum(bool(r.get("landsat_replacement")) for r in revised)
    mixed = sum(bool(r["satellite"].get("mixed_sensors")) for r in revised)
    mirrored = sum(bool(r.get("mirror_horizontal")) for r in revised)
    lines = []
    for line in (source / "CREDITS.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("Landsat contributes to "):
            line = f"Landsat contributes to {landsat} displayed frames, including {mixed} frames in which real Landsat pixels fill only missing Sentinel-2 source pixels at the same coordinates."
        if line.startswith("Horizontal reflections: "):
            line = f"Horizontal reflections: {mirrored} displayed frames. Reflected views are marked MIRRORED VIEW; orientation is recorded per frame. Photographs and native masks use the same reflected geographic view."
        lines.append(line)
    lines.append(
        f"Native2560x1920/30p AV1 edition: all{count} source frames displayed once each. Every panel is drawn at output size; photographs are resampled once from their lossless1444x1080 sources, with no intermediate video. "
        "Photograph-only tone mapping uses the current frame's own black/white levels into a fixed narrow luminance window (flash reduction); no temporal image blending. "
        "Original-animation comparison is decoded from its source and resized only. Ending geographic view is fixed; scores n/a. "
        f'Flash screening covers the satellite panel only (passed={report["passed"]}); see flash-audit.json.'
    )
    lines.append(
        "LOCAL map: G.E. Kindermann global land/water mask,0.3arcsec, DOI10.5281/zenodo.14899246, CC BY4.0; native10m pixels sampled with nearest neighbour in a north-up WGS84 azimuthal-equidistant projection. Gold=actual photograph footprint; cyan=all preceding nearby views."
    )
    (output / "CREDITS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return revised


def run(source=SOURCE, output=OUTPUT, max_frames=None, span=0.09, bitrate=45, assets=True):
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(source / "frames.jsonl")
    count = min(len(records), max_frames or len(records))
    started = time.perf_counter()
    try:
        for attempt, choice in enumerate([span, span * 0.89, span * 0.78], 1):
            candidate, result = encode(source, records, output, count, choice, attempt, bitrate)
            report = audit(candidate, output, count)
            write_json(output / f"flash-audit-{attempt}.json", report)
            if report["passed"]:
                break
            print(f"Flash screening failed at span {choice:.4f}; narrowing.", flush=True)
        if not report["passed"]:
            raise ValueError("Flash screening remains above the conservative limit")
        final = output / "validated-video.mp4"
        candidate.replace(final)
        revised = provenance(source, output, records, count, result, report)
        metadata = probe(final)
        metadata.update(
            frames=count,
            display_content_box=[0, 0, WIDTH, HEIGHT],
            world_map=dict(enabled=True, persistent_history=True),
        )
        write_json(output / "badapple-coastline.verification.json", metadata)
        summary = dict(
            frames=count,
            source_frames=count,
            movie_sha256=digest(final),
            output_spec=result["spec"],
            flash_audit=report,
            mirrored_frames=sum(bool(r.get("mirror_horizontal")) for r in revised),
            landsat_frames=sum(bool(r.get("landsat_replacement")) for r in revised),
            processing=result,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "summary.json", summary)
        finish_audio(source, output, "unicode", assets and count == len(records))
    except Exception as exc:
        write_json(output / "status.json", dict(phase="failed", error=str(exc), total=count))
        raise


def stills(indices, source=SOURCE, output=OUTPUT / "preview", span=0.09):
    """Render chosen frames as PNGs (with true history) to check the layout."""
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(source / "frames.jsonl")
    count = len(records)
    settings = json.loads((source / "run.json").read_text())
    shown = [records[i if i < CREDIT_START else CREDIT_PHOTO] for i in range(count)]
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
    static = static_layer(capture_years, any(r.get("landsat_replacement") for r in records))
    local = NativeLocalMaps(shown)
    world = WorldMap()
    ow, oh = ORIGINAL[2] - ORIGINAL[0], ORIGINAL[3] - ORIGINAL[1]
    with FlashCuda(WIDTH, HEIGHT, span=span) as processor:
        for index in sorted(indices):
            while world.next_index < min(index, CREDIT_START):
                world.advance(shown[world.next_index], world.next_index)
            raw = subprocess.check_output(
                [
                    ffmpeg_binary("ffmpeg"),
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    settings["video_source"],
                    "-vf",
                    f"select=eq(n\\,{index}),scale={ow}:{oh}:flags=lanczos",
                    "-frames:v",
                    "1",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ]
            )
            original = np.frombuffer(raw, np.uint8).reshape(oh, ow, 3)
            r = shown[index]
            frame = static.copy()
            frame[PHOTO[1] : PHOTO[3], PHOTO[0] : PHOTO[2]] = load_photo(r["satellite"]["path"])
            frame[ORIGINAL[1] : ORIGINAL[3], ORIGINAL[0] : ORIGINAL[2]] = original
            frame[MASK[1] : MASK[3], MASK[0] : MASK[2]] = load_mask(r["native_mask_path"])
            frame[WORLD[1] : WORLD[3], WORLD[0] : WORLD[2]] = np.asarray(world.current(r))
            panel, meta = local.render(index)
            frame[LOCAL[1] : LOCAL[3], LOCAL[0] : LOCAL[2]] = panel
            image = Image.fromarray(frame)
            draw_text_layer(
                ImageDraw.Draw(image),
                r,
                index,
                count,
                len(world.seen) + (tuple(sorted(map(tuple, footprint(r).tolist()))) not in world.seen),
                meta["span_km"],
                index >= CREDIT_START,
            )
            Image.fromarray(processor.process_levels(np.asarray(image))).save(output / f"frame-{index:05d}.png")
            print(f"still {index}: levels {processor.levels}", flush=True)
    local.sampler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--luminance-span", type=float, default=0.09)
    parser.add_argument("--bitrate", type=int, default=45, help="AV1 target Mbps; maxrate is twice this")
    parser.add_argument("--stills", type=lambda s: [int(v) for v in s.split(",")])
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    if args.stills:
        stills(args.stills, args.source, args.output / "preview", args.luminance_span)
    else:
        run(args.source, args.output, args.max_frames, args.luminance_span, args.bitrate, not args.skip_assets)
