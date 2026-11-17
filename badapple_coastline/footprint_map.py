"""Actual photograph footprints on a world map, with persistent past views."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .atlas import view_transform, corners
from .data import digest

LAND = Path(__file__).resolve().parents[1] / "assets/coastline/ne_110m_land.geojson"
MAP_BOX = (1000, 608, 1260, 738)
CURRENT = (255, 207, 108, 255)
HISTORY = (67, 174, 188, 190)
SCALE = 3


def footprint(record):
    # A photo alternative can differ from the primary monochrome match.
    values = record.get("satellite", {}).get("footprint_lonlat", record.get("footprint_lonlat"))
    if values is None:
        values = corners(
            view_transform(
                record["longitude"],
                record["latitude"],
                record["width_km"],
                record["rotation_clockwise"],
                (1444, 1080),
                mirror_horizontal=bool(record.get("mirror_horizontal")),
            ),
            (1444, 1080),
        )
    points = np.asarray(values, np.float64)
    if points.shape == (5, 2) and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if points.shape != (4, 2) or not np.isfinite(points).all() or (np.abs(points[:, 1]) > 90 + 1e-8).any():
        raise ValueError("World map requires four valid photograph corners")
    return points


def polygons(points, size, minimum_pixels=0):
    """Unwrap and repeat at the dateline; clipping is done by the canvas."""
    width, height = size
    points = np.asarray(points, np.float64).copy()
    points[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(points[:, 0])))
    points[:, 0] -= np.floor((points[:, 0].mean() + 180) / 360) * 360
    center_geo = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center_geo[1], points[:, 0] - center_geo[0]))]
    xy = np.column_stack([(points[:, 0] + 180) * width / 360, (90 - points[:, 1]) * height / 180])
    center = xy.mean(axis=0)
    xy[:, 0] -= np.floor(center[0] / width) * width
    center = xy.mean(axis=0)
    span = float(np.ptp(xy, axis=0).max())
    if minimum_pixels and span < minimum_pixels:
        xy = center + (xy - center) * (minimum_pixels / max(span, 1e-12))
        xy += center - xy.mean(axis=0)
    return [
        xy + np.array([shift, 0])
        for shift in [-width, 0, width]
        if (xy[:, 0] + shift).max() >= 0 and (xy[:, 0] + shift).min() <= width
    ]


def background(size, land_path=LAND):
    width, height = size
    image = Image.new("RGBA", size, (10, 27, 40, 255))
    draw = ImageDraw.Draw(image)
    for lon in range(-120, 180, 60):
        x = (lon + 180) * width / 360
        draw.line([(x, 0), (x, height)], fill=(25, 48, 61, 255), width=SCALE)
    for lat in [-60, -30, 0, 30, 60]:
        y = (90 - lat) * height / 180
        draw.line([(0, y), (width, y)], fill=(25, 48, 61, 255), width=SCALE)
    collection = json.loads(Path(land_path).read_text(encoding="utf-8"))
    for feature in collection["features"]:
        geometry = feature["geometry"]
        groups = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            groups = [groups]
        elif geometry["type"] != "MultiPolygon":
            raise ValueError("Unexpected world-map geometry")
        for rings in groups:
            for i, ring in enumerate(rings):
                xy = [((lon + 180) * width / 360, (90 - lat) * height / 180) for lon, lat, *_ in ring]
                draw.polygon(xy, fill=(44, 65, 73, 255) if i == 0 else (10, 27, 40, 255))
                draw.line(xy, fill=(76, 98, 104, 255), width=max(1, SCALE // 2))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(71, 95, 109, 255), width=SCALE)
    return image


class FootprintHistory:
    def __init__(self, size=(260, 130), land_path=LAND):
        self.size = size
        self.internal = tuple(v * SCALE for v in size)
        self.base = background(self.internal, land_path)
        self.history = Image.new("RGBA", self.internal, (0, 0, 0, 0))
        self.seen = set()
        self.next_index = 0
        self.views = []
        self.land_path = Path(land_path)

    def panel(self, record, index):
        if index != self.next_index:
            raise ValueError("World-map history must advance in video order")
        points = footprint(record)
        panel = Image.alpha_composite(self.base, self.history)
        current = ImageDraw.Draw(panel)
        for polygon in polygons(points, self.internal, 6 * SCALE):
            line = [tuple(p) for p in polygon]
            line.append(line[0])
            current.line(line, fill=(5, 15, 23, 255), width=3 * SCALE, joint="curve")
            current.line(line, fill=CURRENT, width=2 * SCALE, joint="curve")
        # Commit only after drawing this frame. No future view enters history.
        self.advance(record, index)
        return panel.convert("RGB").resize(self.size, Image.Resampling.LANCZOS)

    def advance(self, record, index):
        """Accumulate a view without allocating/resampling a preview panel."""
        if index != self.next_index:
            raise ValueError("World-map history must advance in video order")
        points = footprint(record)
        key = tuple(sorted(map(tuple, points.tolist())))
        if key not in self.seen:
            draw = ImageDraw.Draw(self.history)
            for polygon in polygons(points, self.internal, 3 * SCALE):
                line = [tuple(p) for p in polygon]
                line.append(line[0])
                draw.line(line, fill=HISTORY, width=SCALE, joint="curve")
            self.seen.add(key)
        self.views.append(
            dict(
                index=index,
                frame=record.get("frame", index),
                corners_lonlat=points.tolist(),
                photograph_sha256=record.get("satellite", {}).get("image_sha256"),
                mirror_horizontal=bool(record.get("mirror_horizontal")),
            )
        )
        self.next_index += 1

    def save(self, path):
        result = dict(
            schema="world-map-footprint-history-v1",
            frames=self.next_index,
            projection="Equirectangular, longitude -180..180, latitude -90..90",
            map_box=list(MAP_BOX),
            history="All preceding photograph footprints persist; no future views",
            current_color="gold",
            history_color="cyan",
            minimum_current_box_pixels=6,
            minimum_history_box_pixels=3,
            display_note="Small boxes enlarged for visibility; true unexpanded photograph corners stored below",
            background_source="Natural Earth 1:110m land, public domain",
            background_sha256=digest(self.land_path),
            background_license_url="https://www.naturalearthdata.com/about/terms-of-use/",
            views=self.views,
        )
        Path(path).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        return dict(
            enabled=True,
            frames=self.next_index,
            footprints=str(Path(path).resolve()),
            footprints_sha256=digest(Path(path)),
            map_box=list(MAP_BOX),
            persistent_history=True,
            minimum_current_box_pixels=6,
            minimum_history_box_pixels=3,
        )


def audit_movie(movie, records):
    """Replay true history and check the map pixels in every encoded frame."""
    import subprocess
    from badapple_media import ffmpeg_binary, read_exact

    movie = Path(movie)
    metadata = json.loads(movie.with_suffix(".verification.json").read_text())["world_map"]
    saved = Path(metadata["footprints"])
    if digest(saved) != metadata["footprints_sha256"]:
        raise ValueError("World-map footprints changed")
    exported = json.loads(saved.read_text(encoding="utf-8"))
    if exported["frames"] != len(records) or len(exported["views"]) != len(records):
        raise ValueError("Incomplete map history")
    history = FootprintHistory()
    base = np.asarray(history.base.convert("RGB").resize(history.size, Image.Resampling.LANCZOS), np.int16)
    left, top, right, bottom = MAP_BOX
    width, height = right - left, bottom - top
    process = subprocess.Popen(
        [
            ffmpeg_binary("ffmpeg"),
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(movie),
            "-an",
            "-vf",
            f"crop={width}:{height}:{left}:{top}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    total = peak = current_peak = past_peak = 0.0
    count = 0
    try:
        for i, record in enumerate(records):
            raw = read_exact(process.stdout, width * height * 3)
            if not raw:
                raise ValueError("Encoded map is missing a frame")
            view = exported["views"][i]
            if (
                view["index"] != i
                or view["frame"] != record["frame"]
                or view["photograph_sha256"] != record["satellite"]["image_sha256"]
            ):
                raise ValueError("Map history differs from actual delivered photo order")
            np.testing.assert_array_equal(np.array(view["corners_lonlat"]), footprint(record))
            expected = np.asarray(history.panel(record, i), np.int16)
            actual = np.frombuffer(raw, np.uint8).reshape(height, width, 3).astype(np.int16)
            error = np.abs(actual - expected)
            mean = float(error.mean())
            peak = max(peak, mean)
            total += mean
            current = (expected[:, :, 0] > 170) & (expected[:, :, 1] > 130) & (expected[:, :, 2] < 180)
            past = (
                (expected[:, :, 1] - base[:, :, 1] > 25)
                & (expected[:, :, 2] - base[:, :, 2] > 25)
                & (expected[:, :, 0] < 150)
            )
            current_error = float(error[current].mean()) if current.any() else 0.0
            past_error = float(error[past].mean()) if past.any() else 0.0
            current_peak = max(current_peak, current_error)
            past_peak = max(past_peak, past_error)
            if mean > 8 or current_error > 30 or past_error > 30:
                raise ValueError(
                    f"Encoded world-map mismatch at frame {i}: whole={mean:.3f}, current={current_error:.3f}, history={past_error:.3f}"
                )
            count += 1
        if read_exact(process.stdout, width * height * 3) or process.wait() != 0:
            raise ValueError("Encoded map frame count/decode failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return dict(
        enabled=True,
        frames_verified=count,
        persistent_history=True,
        actual_photo_corners_verified=True,
        mean_pixel_error=total / max(1, count),
        max_frame_mean_pixel_error=peak,
        max_current_outline_mean_pixel_error=current_peak,
        max_history_outline_mean_pixel_error=past_peak,
        footprints_sha256=metadata["footprints_sha256"],
    )
