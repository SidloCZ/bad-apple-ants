"""Land/water edition: matched 10 m land/water view as the main picture, with the
satellite photograph, a north-up OpenStreetMap context map, the world map and
an animated statistics card over the original's ending credits.

Layout (2560x1920, 30p, AV1 10-bit):
  main land/water 1920x1440 at (40,140); right column 520 px wide:
  original silhouette, satellite photo (same view), world map, OpenStreetMap.
Every panel is drawn at output size from lossless sources and encoded once.
Frames are composed by a thread pool; the main thread adds the decoded original
animation and feeds av1_nvenc.

See docs/satellite.md (land & water edition) for decisions, data sources and verification.
"""

from __future__ import annotations
import argparse, json, math, os, subprocess, tempfile, time
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from badapple_media import ffmpeg_binary
from .data import digest
from .pipeline import read_records, write_json
from .flash_reduction import (
    SOURCE,
    CREDIT_START,
    CREDIT_PHOTO,
    fill_frame,
    audit,
    audio_command,
    packet_hash,
)
from .footprint_map import footprint, LAND as NE_LAND
from .native_2560 import clipped, WorldMap, capture_dates, source_label, NAVY, ACCENT, MUTED, RULE
from .native_2560 import font as _shared_font
import threading

_FONTS = threading.local()


def font(size, kind="regular"):
    """Per-thread font objects: frames are composed on several threads and a FreeType
    face must not be used by two threads at once."""
    cache = getattr(_FONTS, "cache", None)
    if cache is None:
        cache = _FONTS.cache = {}
    key = (size, kind)
    if key not in cache:
        shared = _shared_font(size, kind)
        cache[key] = type(shared)(shared.path, shared.size) if hasattr(shared, "path") else shared
    return cache[key]


from . import landwater_main as lw
from . import osm_maps
from .geostats import view_key, CONTINENT_ORDER, OPEN_OCEAN

OUTPUT = Path("output/landwater-osm")
CACHE = Path(os.environ.get("BADAPPLE_CACHE", "cache")) / "landwater-osm"
WIDTH, HEIGHT = 2560, 1920
MAIN = (40, 140, 1960, 1580)
MAIN_SIZE = (MAIN[2] - MAIN[0], MAIN[3] - MAIN[1])
ORIGINAL = (2000, 184, 2520, 574)
PHOTO = (2000, 650, 2520, 1040)
PHOTO_SIZE = (PHOTO[2] - PHOTO[0], PHOTO[3] - PHOTO[1])
WORLD = (2000, 1116, 2520, 1376)
OSM = (2000, 1488, 2520, 1880)
OSM_SIZE = (OSM[2] - OSM[0], OSM[3] - OSM[1])
CARD = (40, 140, 1960, 1790)  # statistics card over main + footer during the ending credits
LAND = (241, 238, 230)
WATER = (13, 27, 40)
GOLD = "#ffcf6c"
COLUMNS = [40, 520, 1000, 1480]
CONTINENT_COLORS = {
    "Asia": "#e9a73f",
    "North America": "#4fc3d9",
    "Europe": "#9c86ee",
    "Oceania": "#5fd08a",
    "Africa": "#ea7660",
    "South America": "#e4d260",
    "Antarctica": "#cfdbe3",
    OPEN_OCEAN: "#6f93b0",
}
PHOTO_SPAN = 0.09  # same flash-limiting luminance window as the published edition


# ---------------------------------------------------------------- photo tone
_LIN = np.where(
    np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92, ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4
).astype(np.float32)


def _encode(v):
    v = np.clip(v, 0, 1)
    return np.round(255 * np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)).astype(np.uint8)


def tone_photo(rgb, span=PHOTO_SPAN, floor=0.002, gamma=0.75, saturation0=1.15):
    """NumPy port of cuda/video_flash process_levels (current frame only): output
    relative luminance stays within [floor, floor+span] < 0.1 apart, so the photo can
    never form a WCAG general flash by itself."""
    q = _LIN[rgb]
    y0 = q @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    black = min(float(np.percentile(y0, 0.5)), 0.01)
    white = max(float(np.percentile(y0, 99.5)), 0.06)
    t = np.clip((y0 - black) / (white - black), 0, 1) ** gamma
    target = floor + span * t
    sat = np.full(y0.shape, saturation0, np.float32)
    excess = q[..., 0] - 0.6 * q.sum(-1)
    sat = np.where(excess > 0, np.minimum(sat, 1 - excess / (excess + 0.8 * y0 + 1e-12)), sat)
    limit = 1 + 2 * np.clip((y0 - 0.001) / 0.009, 0, 1)
    gain = np.minimum(limit, target / np.maximum(y0, 1e-12))
    chroma = (q - y0[..., None]) * gain[..., None]
    top = chroma.max(-1)
    bottom = chroma.min(-1)
    sat = np.where(top > 0, np.minimum(sat, (1 - target) / np.maximum(top, 1e-12)), sat)
    sat = np.where(bottom < 0, np.minimum(sat, target / np.maximum(-bottom, 1e-12)), sat)
    chroma = np.where((y0 == 0)[..., None], 0, chroma)
    out = _encode(target[..., None] + sat[..., None] * chroma)
    for _ in range(3):  # keep quantised near-blacks from becoming red-only pixels
        lin = _LIN[out]
        total = lin.sum(-1)
        fix = (total > 0) & (lin[..., 0] >= 0.72 * total)
        if not fix.any():
            break
        out[..., 1] = np.where(fix & (out[..., 1] < 255), out[..., 1] + 1, out[..., 1])
        out[..., 2] = np.where(fix & (out[..., 2] < 255), out[..., 2] + 1, out[..., 2])
    return out


def load_photo(path):
    with Image.open(path) as image:
        return tone_photo(np.asarray(image.convert("RGB").resize(PHOTO_SIZE, Image.Resampling.LANCZOS)))


# ---------------------------------------------------------------- labels
def place_text(geo):
    if geo["country"] == OPEN_OCEAN:
        return f"Open ocean{' · ' + geo['sea'] if geo.get('sea') else ''}"
    name = geo["country"]
    if geo.get("disputed_area"):
        name = f"{geo['disputed_area']['name']} (disputed)"
    elif geo.get("territory"):
        name = f"{geo['territory']} ({geo['country']})"
    if geo.get("water"):
        name += " · waters"
    return f"{name}  ·  {geo['continent']}"


def north_vector(record, size):
    t = lw.transform_for(record, size)
    v = np.array([-t.b, t.a])  # along a meridian (constant longitude)
    if t.d * v[0] + t.e * v[1] < 0:
        v = -v
    return v / np.hypot(*v)


def compass(draw, centre, direction, radius=21):
    """North on the (rotated, possibly mirrored) main picture."""
    cx, cy = centre
    dx, dy = direction
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(70, 96, 112), width=2)
    tip = (cx + dx * (radius - 3), cy + dy * (radius - 3))
    nx, ny = -dy, dx
    draw.polygon([tip, (cx + nx * 6, cy + ny * 6), (cx - nx * 6, cy - ny * 6)], fill=ACCENT)
    draw.polygon(
        [(cx - dx * (radius - 5), cy - dy * (radius - 5)), (cx + nx * 6, cy + ny * 6), (cx - nx * 6, cy - ny * 6)],
        fill=(70, 96, 112),
    )
    draw.text(
        (cx + dx * (radius + 13), cy + dy * (radius + 13)), "N", font=font(17, "semibold"), fill=ACCENT, anchor="mm"
    )


def arrow(draw, centre, direction, length, fill, width=3, label=None):
    cx, cy = centre
    dx, dy = direction
    tip = (cx + dx * length / 2, cy + dy * length / 2)
    tail = (cx - dx * length / 2, cy - dy * length / 2)
    draw.line((tail, tip), fill=fill, width=width)
    nx, ny = -dy, dx
    head = length * 0.32
    draw.polygon(
        [
            tip,
            (tip[0] - dx * head + nx * head * 0.55, tip[1] - dy * head + ny * head * 0.55),
            (tip[0] - dx * head - nx * head * 0.55, tip[1] - dy * head - ny * head * 0.55),
        ],
        fill=fill,
    )
    if label:
        draw.text((tip[0] + dx * 14, tip[1] + dy * 14), label, font=font(17, "semibold"), fill=fill, anchor="mm")


def static_layer(capture_years, landsat_used):
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    draw.text((40, 96), "BAD APPLE / REAL EARTH", font=font(46, "semibold"), fill="white", anchor="ls")
    w = draw.textlength("BAD APPLE / REAL EARTH", font=font(46, "semibold"))
    draw.text((40 + w + 24, 96), "LAND & WATER", font=font(30, "semibold"), fill=ACCENT, anchor="ls")
    for (x0, y0, x1, y1), title in [
        (ORIGINAL, "ORIGINAL SILHOUETTE"),
        (PHOTO, "SATELLITE PHOTO"),
        (WORLD, "WORLD MAP"),
        (OSM, "OPENSTREETMAP"),
    ]:
        draw.text((x0, y0 - 14), title, font=font(26, "semibold"), fill=ACCENT, anchor="ls")
        draw.rectangle((x0 - 1, y0 - 1, x1, y1), outline=(42, 60, 72), width=1)
    draw.rectangle((MAIN[0] - 1, MAIN[1] - 1, MAIN[2], MAIN[3]), outline=(42, 60, 72), width=1)
    y = WORLD[3] + 30
    draw.rectangle((2000, y - 14, 2016, y - 2), outline=GOLD, width=2)
    draw.text((2026, y), "current view", font=font(19), fill=MUTED, anchor="ls")
    draw.rectangle((2186, y - 14, 2202, y - 2), outline="#43aebc", width=2)
    draw.text((2212, y), "earlier views (small ones enlarged)", font=font(19), fill=MUTED, anchor="ls")
    draw.line((40, 1796, 1960, 1796), fill=RULE, width=1)
    lines = [
        "Land / water: G. E. Kindermann (2025), global 0.3″ land/water mask, doi:10.5281/zenodo.14899246, CC BY 4.0 — sampled, rotated and reflected as marked",
        "Map data © OpenStreetMap contributors (ODbL), openstreetmap.org/copyright  |  Basemap: Protomaps (BSD-3)  |  Boundaries, places, world map: Natural Earth",
        f"Photos: contains modified Copernicus Sentinel data {capture_years}"
        + ("  |  Landsat: U.S. Geological Survey (public domain)" if landsat_used else "")
        + "  |  © ESA WorldCover 2020/2021  |  Full notices: CREDITS.txt",
    ]
    for k, text in enumerate(lines):
        draw.text((40, 1826 + 28 * k), text, font=font(18), fill=MUTED, anchor="ls")
    return np.asarray(image)


def legend(draw):
    y = 1614
    x = 40
    for color, text in [(LAND, "land"), (WATER, "water")]:
        draw.rectangle((x, y - 16, x + 18, y + 2), fill=color, outline=(70, 90, 104))
        x += 28
        draw.text((x, y), text, font=font(20), fill=MUTED, anchor="ls")
        x += draw.textlength(text, font=font(20)) + 26
    draw.text(
        (x + 6, y),
        "10 m land/water mask at the matched position, sampled at display resolution",
        font=font(20),
        fill=MUTED,
        anchor="ls",
    )


def draw_text(draw, i, count, record, geo, visited, span_km):
    right = WIDTH - 40
    draw.text((right, 96), f"{i + 1:05d} / {count:05d}", font=font(34, "mono"), fill=MUTED, anchor="rs")
    mirrored = bool(record.get("mirror_horizontal"))
    draw.text(
        (MAIN[2], 96),
        "MIRRORED VIEW" if mirrored else "TRUE VIEW",
        font=font(24, "semibold"),
        fill=GOLD if mirrored else MUTED,
        anchor="rs",
    )
    legend(draw)
    small = font(21)
    draw.text((right, PHOTO[1] - 14), "same view", font=small, fill=MUTED, anchor="rs")
    draw.text((right, WORLD[1] - 14), f"{visited:,} views so far", font=small, fill=MUTED, anchor="rs")
    draw.text(
        (right, OSM[1] - 14),
        f"{span_km:.1f} km across" if span_km < 1000 else f"{span_km:.0f} km across",
        font=small,
        fill=MUTED,
        anchor="rs",
    )
    label, value, mono = font(19, "semibold"), font(31), font(31, "mono")
    p = record["place"]
    lat, lon = record["latitude"], record["longitude"]
    score = record.get("native_miou", record.get("native_iou", record.get("iou")))
    dates = capture_dates(record).split(" – ")[0]
    cells = [
        (0, 0, 2, "COUNTRY / TERRITORY  ·  CONTINENT", place_text(geo), value, "white"),
        (2, 0, 1, "LATITUDE", f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}", mono, ACCENT),
        (3, 0, 1, "LONGITUDE", f"{abs(lon):.5f}° {'E' if lon >= 0 else 'W'}", mono, ACCENT),
        (
            0,
            1,
            1,
            "NEAREST MAPPED PLACE",
            f"{p['name']}, {p['country']} · {p['distance_km']:.0f} km",
            font(27),
            "white",
        ),
        (
            1,
            1,
            1,
            "VIEW WIDTH  ·  ROTATION",
            f"{record['width_km']:.2f} km · {record['rotation_clockwise']:.1f}°",
            mono,
            "white",
        ),
        (2, 1, 1, "MATCH SCORE (mIoU)", f"{score:.3f}", mono, "white"),
        (3, 1, 1, "PHOTO", f"{source_label(record)} · {dates}", font(27, "mono"), "white"),
    ]
    for column, row, span, title, text, face, fill in cells:
        x = COLUMNS[column]
        y = 1662 + 82 * row
        draw.text((x, y), title, font=label, fill=MUTED, anchor="ls")
        clipped(draw, (x, y + 10), text, 480 * span - (70 if (column, row) == (1, 1) else 24), face, fill)
    compass(draw, (COLUMNS[2] - 46, 1662 + 82 + 22), north_vector(record, MAIN_SIZE))


def decorate_osm(image, span_km):
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    ink = "#d3dee6"
    x = w - 26
    draw.line((x, 58, x, 28), fill=ink, width=2)
    draw.line((x - 7, 37, x, 27, x + 7, 37), fill=ink, width=2)
    draw.text((x, 22), "N", font=font(17, "semibold"), fill=ink, anchor="ms")
    target = span_km / 4
    power = 10 ** math.floor(math.log10(target))
    bar = max(v * power for v in (1, 2, 5) if v * power <= target)
    pixels = w * bar / span_km
    draw.line((16, h - 34, 16 + pixels, h - 34), fill=ink, width=3)
    draw.text((16, h - 42), f"{bar:g} km", font=font(17), fill=ink, anchor="ls")
    text = "© OpenStreetMap contributors · Protomaps"
    f = font(14)
    tw = draw.textlength(text, font=f)
    draw.rectangle((w - tw - 14, h - 22, w, h), fill=(8, 15, 23, 190))
    draw.text((w - 7, h - 6), text, font=f, fill="#aebdca", anchor="rs")
    return image


# ---------------------------------------------------------------- statistics card
def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


@lru_cache(maxsize=1)
def world_base(size):
    w, h = size
    s = 2
    image = Image.new("RGB", (w * s, h * s), (10, 22, 33))
    draw = ImageDraw.Draw(image)
    for lon in range(-180, 181, 30):
        draw.line(((lon + 180) / 360 * w * s, 0, (lon + 180) / 360 * w * s, h * s), fill=(20, 36, 48), width=1)
    for lat in range(-60, 91, 30):
        draw.line((0, (90 - lat) / 180 * h * s, w * s, (90 - lat) / 180 * h * s), fill=(20, 36, 48), width=1)
    data = json.loads(NE_LAND.read_text(encoding="utf-8"))
    for feature in data["features"]:
        g = feature["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for k, ring in enumerate(poly):
                pts = [((x + 180) / 360 * w * s, (90 - y) / 180 * h * s) for x, y in ring]
                draw.polygon(pts, fill=(40, 56, 66) if k == 0 else (10, 22, 33))
    return image.resize(size, Image.Resampling.LANCZOS)


def stats_card(t, stats, dots):
    """t: seconds since the ending section started. dots: [(x_norm, y_norm, color, first_frame)] in video order."""
    W, H = CARD[2] - CARD[0], CARD[3] - CARD[1]
    image = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((0, 46), "STATISTICS", font=font(44, "semibold"), fill="white", anchor="ls")
    draw.text(
        (draw.textlength("STATISTICS", font=font(44, "semibold")) + 22, 46),
        "every matched frame, by where on Earth it is",
        font=font(26),
        fill=MUTED,
        anchor="ls",
    )
    k = ease(t / 1.4)
    tiles = [
        (stats["matched_frames"], "matched frames"),
        (stats["views"], "distinct views"),
        (stats["places"], "places"),
        (stats["countries"], "countries & territories"),
        (stats["continents"], "continents"),
    ]
    tile_w = (W - 4 * 20) // 5
    for n, (value, text) in enumerate(tiles):
        x = n * (tile_w + 20)
        draw.rounded_rectangle((x, 78, x + tile_w, 206), 10, fill=(14, 26, 37), outline=(34, 52, 64))
        draw.text((x + 22, 152), f"{round(value * k):,}", font=font(60, "semibold"), fill=ACCENT, anchor="ls")
        draw.text((x + 22, 188), text, font=font(22), fill=MUTED, anchor="ls")
    # continents (left column)
    top = 262
    LW = 900
    draw.text((0, top), "BY CONTINENT", font=font(24, "semibold"), fill=ACCENT, anchor="ls")
    draw.text((LW, top), "frames · places", font=font(19), fill=MUTED, anchor="rs")
    rows = sorted(stats["by_continent"], key=lambda c: (c["continent"] == OPEN_OCEAN, -c["frames"]))
    peak = max(c["frames"] for c in rows)
    for n, c in enumerate(rows):
        y = top + 24 + n * 54
        g = ease((t - 0.3 - 0.07 * n) / 1.0)
        draw.text((0, y + 30), c["continent"], font=font(25), fill="white", anchor="ls")
        x0 = 230
        length = (LW - x0 - 230) * c["frames"] / peak * g
        draw.rounded_rectangle((x0, y + 9, x0 + max(4, length), y + 37), 4, fill=CONTINENT_COLORS[c["continent"]])
        draw.text(
            (x0 + length + 12, y + 32), f"{round(c['frames'] * g):,}", font=font(25, "mono"), fill="white", anchor="ls"
        )
        draw.text((LW, y + 32), f"{c['places']:,}", font=font(21, "mono"), fill=MUTED, anchor="rs")
    # map replay of every view, in video order
    my = 790
    mw, mh = LW, LW // 2
    draw.text((0, my - 16), "EVERY VIEW, IN VIDEO ORDER", font=font(24, "semibold"), fill=ACCENT, anchor="ls")
    base = world_base((mw, mh)).copy()
    md = ImageDraw.Draw(base, "RGBA")
    shown = int(len(dots) * ease((t - 0.6) / 4.5))
    for x, y, color, _ in dots[:shown]:
        px, py = x * mw, y * mh
        md.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color + "d0")
    if 0 < shown < len(dots):
        x, y, _, frame = dots[shown - 1]
        px, py = x * mw, y * mh
        md.ellipse((px - 8, py - 8, px + 8, py + 8), outline=GOLD, width=3)
    image.paste(base, (0, my))
    draw.rectangle((0, my, mw - 1, my + mh - 1), outline=(42, 60, 72))
    present = {c["continent"] for c in stats["by_continent"]}
    for n, c in enumerate([c for c in CONTINENT_ORDER if c in present]):
        x = (n % 4) * 225
        y = my + mh + 34 + (n // 4) * 30
        draw.ellipse((x, y - 15, x + 14, y - 1), fill=CONTINENT_COLORS[c])
        draw.text((x + 22, y), c, font=font(19), fill=MUTED, anchor="ls")
    facts = [
        ("Smallest view", f"{stats['width_km_min']:.1f} km wide"),
        ("Largest view", f"{stats['width_km_max']:,.0f} km wide"),
        ("Median view", f"{stats['width_km_median']:.1f} km wide"),
        ("Area covered", f"{stats['area_union_km2'] / 1e6:.1f} million km² (overlaps removed)"),
        ("Mirrored views", f"{stats['mirrored_views']:,} of {stats['views']:,}"),
        ("Open-water views", f"{stats['water_assigned_views']:,} · nearest country ≤ 200 nmi"),
    ]
    for n, (a, b) in enumerate(facts):
        x = (n % 2) * 450
        yy = my + mh + 112 + (n // 2) * 76
        g = ease((t - 1.6 - 0.12 * n) / 0.8)
        draw.text((x, yy), a.upper(), font=font(17, "semibold"), fill=(158, 175, 192, int(255 * g)), anchor="ls")
        clipped(draw, (x, yy + 8), b, 440, font(27), (255, 255, 255, int(255 * g)))
    # every country & territory (right column), scrolled through in full
    left = 960
    RW = W - left
    countries = [c for c in stats["by_country"] if c["country"] != OPEN_OCEAN]
    ocean = next((c for c in stats["by_country"] if c["country"] == OPEN_OCEAN), None)
    draw.text(
        (left, top),
        f"ALL {len(countries)} COUNTRIES & TERRITORIES",
        font=font(24, "semibold"),
        fill=ACCENT,
        anchor="ls",
    )
    draw.text((W - 18, top), "frames · places", font=font(19), fill=MUTED, anchor="rs")
    row_h = 31
    vy0, vy1 = top + 22, my + mh + 160
    vh = vy1 - vy0
    entries = countries + ([ocean] if ocean else [])
    total_h = len(entries) * row_h
    max_scroll = max(0, total_h - vh)
    u = min(1.0, max(0.0, (t - 3.0) / 9.4))
    u = u * u * (3 - 2 * u)  # smooth start/stop, then hold on the last rows
    offset = max_scroll * u
    view = Image.new("RGB", (RW, vh), NAVY)
    vd = ImageDraw.Draw(view)
    peak = countries[0]["frames"]
    first = int(offset // row_h)
    last = min(len(entries), int((offset + vh) // row_h) + 2)
    short = {
        "United States of America": "United States",
        "Republic of the Congo": "Rep. of the Congo",
        "Democratic Republic of the Congo": "DR Congo",
        "United Republic of Tanzania": "Tanzania",
        "Federated States of Micronesia": "Micronesia",
    }
    for n in range(first, last):
        c = entries[n]
        y = n * row_h - offset
        g = ease((t - 0.9 - 0.05 * n) / 1.0)
        is_ocean = c["country"] == OPEN_OCEAN
        vd.text((0, y + 23), "" if is_ocean else f"{n + 1:>3}", font=font(18, "mono"), fill=MUTED, anchor="ls")
        clipped(vd, (50, y + 3), short.get(c["country"], c["country"]), 286, font(21), MUTED if is_ocean else "white")
        x0 = 344
        length = (RW - x0 - 190) * c["frames"] / peak * g
        vd.rounded_rectangle((x0, y + 8, x0 + max(3, length), y + 25), 3, fill=CONTINENT_COLORS[c["continent"]])
        vd.text(
            (x0 + length + 10, y + 24), f"{round(c['frames'] * g):,}", font=font(19, "mono"), fill="white", anchor="ls"
        )
        vd.text((RW - 18, y + 24), f"{c['places']:,}", font=font(18, "mono"), fill=MUTED, anchor="rs")
    alpha = np.full((vh, 1), 255, np.float32)  # fade the edges while the list is moving
    fade = 46
    if offset > 0:
        alpha[:fade, 0] = np.linspace(0, 255, fade)
    if offset < max_scroll:
        alpha[-fade:, 0] = np.linspace(255, 0, fade)
    v = np.asarray(view).astype(np.float32)
    bg = np.array(NAVY, np.float32)
    v = bg + (v - bg) * (alpha[..., None] / 255)
    image.paste(Image.fromarray(v.astype(np.uint8)), (left, vy0))
    if max_scroll:  # scroll position indicator
        draw.rectangle((W - 4, vy0, W - 1, vy1), fill=(26, 40, 51))
        th = vh * vh / total_h
        ty = vy0 + (vh - th) * (offset / max_scroll)
        draw.rectangle((W - 4, ty, W - 1, ty + th), fill=(115, 230, 215, 200))
    note = (
        "Country = the country/territory with the largest Natural Earth 1:10m land overlap of each view (de facto boundaries; "
        "no position on sovereignty is implied). Russia is split at the Urals (60° E); remote islands use their UN region. A place joins views overlapping by IoU > 0.5. "
        "Ending-credit frames (fixed view) are not counted."
    )
    words = note.split()
    line = ""
    yy = H - 50
    lines = []
    for wd in words:
        if draw.textlength(line + " " + wd, font=font(18)) > W and line:
            lines.append(line)
            line = wd
        else:
            line = (line + " " + wd).strip()
    lines.append(line)
    for n, text in enumerate(lines):
        draw.text((0, yy + 26 * n), text, font=font(18), fill=MUTED, anchor="ls")
    return np.asarray(image)


# ---------------------------------------------------------------- shared state / workers
def shown_records(records, count):
    return [records[i if i < CREDIT_START else CREDIT_PHOTO] for i in range(count)]


def prepare(output=OUTPUT, cache=CACHE, shards=6):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    records = read_records(SOURCE / "frames.jsonl")
    count = len(records)
    shown = shown_records(records, count)
    geo_path = output / "geostats.json"
    if not geo_path.is_file():
        from .geostats import build

        geo_path.write_text(json.dumps(build(records, CREDIT_START), ensure_ascii=False), encoding="utf-8")
    views, names = osm_maps.build_views(shown, OSM_SIZE)
    (output / "osm-views.json").write_text(
        json.dumps(
            dict(
                size=OSM_SIZE, label_scale=osm_maps.LABEL_SCALE, pmtiles=osm_maps.PMTILES, views=views, frame_view=names
            )
        ),
        encoding="utf-8",
    )
    prepare_main_panels(shown, cache)
    report = osm_maps.render(views, cache / "osm-base", OSM_SIZE, shards=shards)
    write_json(output / "osm-render.json", report)
    world_path = cache / "world-panels.u8"
    shape = (count, WORLD[3] - WORLD[1], WORLD[2] - WORLD[0], 3)
    marker = cache / "world-panels.json"
    identity = digest(Path(__file__).with_name("native_2560.py")) + str(len(views))
    if not (marker.is_file() and json.loads(marker.read_text()).get("identity") == identity and world_path.is_file()):
        panels = np.memmap(world_path, np.uint8, "w+", shape=shape)
        world = WorldMap()
        frozen = None
        start = time.perf_counter()
        for i in range(count):
            if i < CREDIT_START:
                panels[i] = np.asarray(world.panel(shown[i], i))
            else:
                if frozen is None:
                    frozen = np.asarray(world.current(shown[i]))
                world.advance(shown[i], i)
                panels[i] = frozen
            if (i + 1) % 1000 == 0:
                print(f"world panels {i + 1}/{count}; {time.perf_counter() - start:.0f}s", flush=True)
        panels.flush()
        del panels
        world.save(output / "world-map-history.json")
        marker.write_text(json.dumps(dict(identity=identity, shape=shape)))
    return dict(records=count, views=len(views))


def main_view_order(shown):
    order = {}
    for i, r in enumerate(shown):
        key = view_key(footprint(r))
        if key not in order:
            order[key] = (len(order), i)
    return order


def prepare_main_panels(shown, cache, threads=6, check_every=25):
    """Land fraction of every display pixel, once per unique view (lossless 8-bit PNG).
    Every `check_every`-th view is checked against its saved native 1444x1080 mask."""
    from concurrent.futures import ThreadPoolExecutor
    from .pipeline import write_json as _write

    out = Path(cache) / "main-panels"
    out.mkdir(parents=True, exist_ok=True)
    order = main_view_order(shown)
    jobs = [(k, i) for k, i in order.values() if not (out / f"{k:05d}.png").is_file()]
    renderer = lw.CoverageRenderer()
    start = time.perf_counter()
    modes = {}
    checks = []

    def one(item):
        k, i = item
        record = shown[i]
        cover, info = lw.coverage(renderer, record, MAIN_SIZE)
        Image.fromarray(np.round(cover * 255).astype(np.uint8)).save(out / f"{k:05d}.png.part.png", compress_level=6)
        (out / f"{k:05d}.png.part.png").replace(out / f"{k:05d}.png")
        result = dict(view=k, frame=i, mode=info["mode"], pixels_per_cell=info["pixels_per_cell"])
        if k % check_every == 0:
            with Image.open(record["native_mask_path"]) as m:
                native = np.asarray(m.convert("L")) > 127
            mine = (
                np.asarray(
                    Image.fromarray(np.round(cover * 255).astype(np.uint8)).resize(
                        native.shape[::-1], Image.Resampling.BOX
                    )
                )
                > 127
            )
            union = (native | mine).sum()
            result["iou_vs_native_mask"] = (
                float((native & mine).sum() / union) if union else 1.0
            )  # both empty: identical
        return result

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for n, r in enumerate(pool.map(one, jobs), 1):
                modes[r["mode"]] = modes.get(r["mode"], 0) + 1
                if "iou_vs_native_mask" in r:
                    checks.append(r)
                if n % 250 == 0 or n == len(jobs):
                    print(f"main panels {n}/{len(jobs)}; {time.perf_counter() - start:.0f}s", flush=True)
    finally:
        renderer.close()
    if jobs:
        ious = [c["iou_vs_native_mask"] for c in checks]
        _write(
            Path(cache) / "main-panels.json",
            dict(
                views=len(order),
                rendered=len(jobs),
                modes=modes,
                checked=len(checks),
                iou_min=min(ious) if ious else None,
                iou_mean=float(np.mean(ious)) if ious else None,
                checks=checks,
                seconds=time.perf_counter() - start,
                size=list(MAIN_SIZE),
                gaussian_sigma_cells=lw.SIGMA,
            ),
        )
    return len(order)


class State:
    def __init__(self, output, cache, count, slots=None):
        output = Path(output)
        cache = Path(cache)
        self.records = read_records(SOURCE / "frames.jsonl")
        self.count = count
        self.shown = shown_records(self.records, count)
        geo = json.loads((output / "geostats.json").read_text(encoding="utf-8"))
        self.stats = geo["stats"]
        self.views = geo["views"]
        self.frame_view = geo["frame_view"]
        self.geo_by_key = {view_key(v["corners_lonlat"]): v for v in self.views}
        osm = json.loads((output / "osm-views.json").read_text(encoding="utf-8"))
        self.osm = osm_maps.OsmPanels(osm["views"], osm["frame_view"], cache / "osm-base", OSM_SIZE)
        self.osm_names = osm["frame_view"]
        first = {}
        order = []
        for i, n in enumerate(self.osm_names):
            if n not in first:
                first[n] = i
                order.append(n)
        self.osm_order = order
        self.osm_first = np.array([first[n] for n in order])
        self.world = np.memmap(
            cache / "world-panels.u8",
            np.uint8,
            "r",
            shape=(len(self.records), WORLD[3] - WORLD[1], WORLD[2] - WORLD[0], 3),
        )
        years = "/".join(
            sorted(
                {"2020", "2021"}
                | {
                    s["datetime"][:4]
                    for r in self.records
                    for s in r["satellite"]["scenes"]
                    if s.get("collection") != "landsat-c2-l2"
                }
            )
        )
        self.static = static_layer(years, any(r.get("landsat_replacement") for r in self.records))
        self.main_order = main_view_order(self.shown)
        self.main_dir = cache / "main-panels"
        dots = []
        for v in sorted(self.views, key=lambda v: v["frames"][0]):
            lon, lat = v["centre_lonlat"]
            dots.append(((lon + 180) / 360, (90 - lat) / 180, CONTINENT_COLORS[v["continent"]], v["frames"][0]))
        self.dots = dots
        self.coverage_cache = {}
        self.slots = slots

    def main_panel(self, record):
        k = self.main_order[view_key(footprint(record))][0]
        with Image.open(self.main_dir / f"{k:05d}.png") as image:
            cover = np.asarray(image)
        return lw.colorize8(cover, LAND, WATER)

    def compose(self, i):
        frame = self.static.copy()
        record = self.shown[i]
        credits = i >= CREDIT_START
        frame[PHOTO[1] : PHOTO[3], PHOTO[0] : PHOTO[2]] = load_photo(record["satellite"]["path"])
        frame[WORLD[1] : WORLD[3], WORLD[0] : WORLD[2]] = self.world[i]
        k = int(np.searchsorted(self.osm_first, i, side="left"))  # views first seen before frame i
        name = self.osm_names[i]
        view = self.osm.views[name]
        osm = Image.fromarray(self.osm.render(i, self.osm_order[:k]))
        frame[OSM[1] : OSM[3], OSM[0] : OSM[2]] = np.asarray(decorate_osm(osm, view["span_km"]))
        if credits:
            frame[MAIN[1] - 1, MAIN[0] - 1 : MAIN[2] + 1] = NAVY
            frame[MAIN[1] - 1 : MAIN[3] + 1, [MAIN[0] - 1, MAIN[2]]] = NAVY
            frame[CARD[1] : CARD[3], CARD[0] : CARD[2]] = stats_card((i - CREDIT_START) / 30, self.stats, self.dots)
        else:
            frame[MAIN[1] : MAIN[3], MAIN[0] : MAIN[2]] = self.main_panel(record)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        visited = int(np.searchsorted(self.osm_first, min(i, CREDIT_START - 1), side="right"))
        if credits:
            draw.text((WIDTH - 40, 96), "ENDING CREDITS", font=font(34, "mono"), fill=MUTED, anchor="rs")
            draw.text((WIDTH - 40, PHOTO[1] - 14), "fixed view", font=font(21), fill=MUTED, anchor="rs")
            draw.text(
                (WIDTH - 40, WORLD[1] - 14), f"{visited:,} views in total", font=font(21), fill=MUTED, anchor="rs"
            )
        else:
            geo = self.views[self.frame_view[i]]
            draw_text(draw, i, self.count, record, geo, visited, view["span_km"])
        return np.asarray(image)


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
        f"{bitrate * 2}M",
        "-bufsize",
        f"{bitrate * 4}M",
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


def original_command(count):
    settings = json.loads((SOURCE / "run.json").read_text())
    movie = Path(settings["video_source"])
    if digest(movie) != settings["video_source_sha256"]:
        raise ValueError("Original animation source changed")
    ow, oh = ORIGINAL[2] - ORIGINAL[0], ORIGINAL[3] - ORIGINAL[1]
    return movie, [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(movie),
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


def encode(output, cache, count, bitrate=45, workers=12, lookahead=48):
    """Compose frames with a thread pool (PNG decode, resizing and NumPy release the
    GIL; ~19 fps with 12 threads). Threads rather than processes: on Windows the
    multiprocessing queue handles of worker processes became invalid intermittently (WinError 6)."""
    from concurrent.futures import ThreadPoolExecutor

    output = Path(output)
    path = output / "validated-video.mp4"
    start = time.perf_counter()
    movie, decode = original_command(count)
    ow, oh = ORIGINAL[2] - ORIGINAL[0], ORIGINAL[3] - ORIGINAL[1]
    state = State(output, cache, count)
    original = np.empty((oh, ow, 3), np.uint8)
    with (
        tempfile.TemporaryFile() as dlog,
        tempfile.TemporaryFile() as elog,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=dlog, bufsize=1 << 20)
        encoder = subprocess.Popen(
            encoder_command(path, count, bitrate), stdin=subprocess.PIPE, stderr=elog, bufsize=32 << 20
        )
        futures = {}
        submitted = 0
        try:
            for i in range(count):
                while submitted < count and submitted - i < lookahead:
                    futures[submitted] = pool.submit(state.compose, submitted)
                    submitted += 1
                frame = np.array(futures.pop(i).result())
                if not fill_frame(decoder.stdout, original):
                    raise ValueError("Original animation is missing a frame")
                frame[ORIGINAL[1] : ORIGINAL[3], ORIGINAL[0] : ORIGINAL[2]] = original
                encoder.stdin.write(memoryview(frame).cast("B"))
                if (i + 1) % 100 == 0 or i + 1 == count:
                    elapsed = time.perf_counter() - start
                    write_json(
                        output / "status.json",
                        dict(
                            phase="encode", complete=i + 1, total=count, fps=(i + 1) / elapsed, elapsed_seconds=elapsed
                        ),
                    )
                    print(
                        f"land/water+OSM AV1: {i + 1}/{count}; {(i + 1) / elapsed:.1f} fps; {elapsed:.0f}s", flush=True
                    )
            encoder.stdin.close()
            if decoder.wait() != 0 or encoder.wait() != 0:
                raise ValueError("FFmpeg decode/encode failed")
        except Exception:
            for f in futures.values():
                f.cancel()
            for process, log in [(decoder, dlog), (encoder, elog)]:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                log.seek(0)
                message = log.read().decode(errors="replace")
                if message:
                    print(message[-4000:], flush=True)
            raise
    return path, dict(seconds=time.perf_counter() - start, threads=workers, lookahead=lookahead, bitrate_mbps=bitrate)


def verify_main(movie, output, records, count, step=25):
    """Decode the encoded video, threshold the main panel and compare with the saved
    native 1444x1080 masks (the matching's own pixels) on every `step`-th frame."""
    shown = shown_records(records, count)
    picks = list(range(0, CREDIT_START, step))
    rows = []
    w, h = MAIN_SIZE
    command = [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(movie),
        "-an",
        "-vf",
        f"select='not(mod(n\\,{step}))*lt(n\\,{CREDIT_START})',crop={w}:{h}:{MAIN[0]}:{MAIN[1]},format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = w * h
    dark = float(np.mean(lw.colorize8(np.array([0], np.uint8), LAND, WATER)[0]))
    light = float(np.mean(lw.colorize8(np.array([255], np.uint8), LAND, WATER)[0]))
    threshold = (dark + light) / 2
    for i in picks:
        raw = process.stdout.read(size)
        if len(raw) != size:
            raise ValueError("Encoded video ended early during main-panel verification")
        shown_mask = (
            np.asarray(Image.frombytes("L", (w, h), raw).resize((1444, 1080), Image.Resampling.BOX)) > threshold
        )
        with Image.open(shown[i]["native_mask_path"]) as m:
            native = np.asarray(m.convert("L")) > 127
        union = (native | shown_mask).sum()
        rows.append(
            dict(
                frame=i,
                iou=float((native & shown_mask).sum() / union) if union else 1.0,
                native_land_fraction=float(native.mean()),
            )
        )
    process.stdout.close()
    process.wait()
    ious = np.array([r["iou"] for r in rows])
    big = np.array([r["iou"] for r in rows if 0.02 < r["native_land_fraction"] < 0.98])
    report = dict(
        frames_checked=len(rows),
        step=step,
        iou_mean=float(ious.mean()),
        iou_p05=float(np.percentile(ious, 5)),
        iou_min=float(ious.min()),
        iou_mean_excluding_tiny_or_full=float(big.mean()) if big.size else None,
        method="Encoded AV1 main panel, grey-thresholded midway between land and water colours, box-resampled to 1444x1080 and compared with the native mask PNG used for matching",
        lowest=sorted(rows, key=lambda r: r["iou"])[:10],
    )
    write_json(output / "main-panel-verification.json", report)
    return report


def credits_text(output, records, count, geo):
    source = (
        (Path("output/coastline-precision-v4-2560x1920-30p-native-av1") / "CREDITS.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    keep = [
        l
        for l in source
        if not l.startswith(("Native2560x1920/30p AV1 edition", "Native 2560x1920/30p AV1 edition", "LOCAL map:"))
    ]
    s = geo["stats"]
    keep += [
        "Land/water edition: the main picture is the matched land/water view (white = land, dark = water), sampled directly from the lossless 0.3 arc-second Kindermann mask at the 1920x1440 display size through the same affine view (rotation and reflection) as the match. Zoomed-out views use up to 4x4 area supersampling; zoomed-in views use a Gaussian-weighted (sigma 0.55 cell) contour of the 10 m grid with a one-pixel anti-aliasing ramp. Nearest-neighbour sampling of the same transform reproduces every saved native mask exactly.",
        f"OpenStreetMap panel: Map data (c) OpenStreetMap contributors, available under the Open Database License (ODbL), https://www.openstreetmap.org/copyright . Vector tiles: Protomaps planet build {Path(osm_maps.PMTILES).stem} ({osm_maps.PMTILES}; ODbL produced work), read by HTTP range requests. Style: @protomaps/basemaps 5.7.2 (BSD-3-Clause), recoloured; fonts Noto Sans (SIL OFL) and sprites from protomaps/basemaps-assets. Rendered with MapLibre GL JS 5.24.0 (BSD-3-Clause). North-up Web Mercator; gold = displayed photograph footprint, cyan = earlier nearby views.",
        'Country / continent statistics: Natural Earth 1:10m Admin 0 map units, disputed areas and marine areas (public domain), "Made with Natural Earth". Country = largest geodesic land overlap of each view (de facto boundaries; no position on sovereignty implied); views with < 2 % land use the nearest country within 200 nautical miles, otherwise "Open ocean". Russia split at the Urals (60 E). A place joins views overlapping with IoU > 0.5.',
        f"Statistics card (ending credits, frames {CREDIT_START + 1}-{count}): {s['matched_frames']} matched frames, {s['views']} distinct views, {s['places']} places, {s['countries']} countries and territories, {s['continents']} continents. Per-frame list: frames.csv.",
        "Satellite photo panel: the same photographs, resized to 520x390 and tone-mapped per frame into a relative-luminance window of 0.09 (below the 0.1 WCAG general-flash step), no temporal blending.",
    ]
    return "\n".join(keep) + "\n"


def finalize(output=OUTPUT, cache=CACHE):
    output = Path(output)
    records = read_records(SOURCE / "frames.jsonl")
    count = len(records)
    video = output / "validated-video.mp4"
    final = output / "badapple-landwater-osm.mp4"
    geo = json.loads((output / "geostats.json").read_text(encoding="utf-8"))
    original = Path(json.loads((SOURCE / "run.json").read_text())["video_source"])
    write_json(output / "status.json", dict(phase="verify-main-panel"))
    main = verify_main(video, output, records, count)
    write_json(output / "status.json", dict(phase="flash-screening"))
    audit(video, output, count)  # screens the main panel region (40,140)-(1960,1580); reported, not a gate
    flash = json.loads((output / "flash-audit.json").read_text())
    flash.update(
        scope="Main land/water panel (40,140)-(1960,1580). Informational: it reproduces the original animation's own black/white changes, so it is not expected to pass; the satellite panel is limited to a 0.09 luminance window by construction."
    )
    write_json(output / "flash-audit.json", flash)
    write_json(output / "status.json", dict(phase="audio-mux"))
    subprocess.run(audio_command(video, original, final, count, "trim"), check=True)
    if packet_hash(final, "v") != packet_hash(video, "v"):
        raise ValueError("Video packets changed during audio mux")
    subprocess.run(
        [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(final), "-f", "null", "-"], check=True
    )
    details = json.loads(
        subprocess.check_output(
            [
                ffmpeg_binary("ffprobe"),
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(final),
            ]
        )
    )["streams"]
    v = next(s for s in details if s["codec_type"] == "video")
    if int(v["nb_read_frames"]) != count or (v["width"], v["height"]) != (WIDTH, HEIGHT):
        raise ValueError("Frame count or size changed")
    (output / "CREDITS.txt").write_text(credits_text(output, records, count, geo), encoding="utf-8")
    summary = dict(
        movie=str(final),
        movie_sha256=digest(final),
        bytes=final.stat().st_size,
        streams=details,
        frames=count,
        matched_frames=CREDIT_START,
        statistics={k: v for k, v in geo["stats"].items() if k != "by_country"},
        main_panel_verification={k: v for k, v in main.items() if k != "lowest"},
        flash_main_panel_passed=flash.get("passed"),
        encode=json.loads((output / "encode.json").read_text()) if (output / "encode.json").is_file() else None,
        main_panels={
            k: v for k, v in json.loads((Path(cache) / "main-panels.json").read_text()).items() if k != "checks"
        },
        osm_render=(
            json.loads((output / "osm-render.json").read_text()) if (output / "osm-render.json").is_file() else None
        ),
    )
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", dict(phase="complete", frames=count))
    return summary


def stills(indices, output=OUTPUT, cache=CACHE):
    output = Path(output)
    out = output / "preview"
    out.mkdir(parents=True, exist_ok=True)
    records = read_records(SOURCE / "frames.jsonl")
    state = State(output, cache, len(records))
    movie = Path(json.loads((SOURCE / "run.json").read_text())["video_source"])
    ow, oh = ORIGINAL[2] - ORIGINAL[0], ORIGINAL[3] - ORIGINAL[1]
    for i in indices:
        raw = subprocess.check_output(
            [
                ffmpeg_binary("ffmpeg"),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(movie),
                "-vf",
                f"select=eq(n\\,{i}),scale={ow}:{oh}:flags=lanczos",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        t = time.perf_counter()
        frame = state.compose(i).copy()
        dt = time.perf_counter() - t
        frame[ORIGINAL[1] : ORIGINAL[3], ORIGINAL[0] : ORIGINAL[2]] = np.frombuffer(raw, np.uint8).reshape(oh, ow, 3)
        Image.fromarray(frame).save(out / f"frame-{i:05d}.png")
        print(f"still {i}: {dt:.2f}s compose", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "stills", "encode", "finalize", "all"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--frames", type=lambda s: [int(v) for v in s.split(",")])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bitrate", type=int, default=45)
    parser.add_argument("--shards", type=int, default=6)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.output, args.cache, args.shards))
    elif args.command == "stills":
        if not args.frames:
            parser.error("stills needs --frames, e.g. --frames 0,1200,6600")
        stills(args.frames, args.output, args.cache)
    elif args.command == "finalize":
        print(json.dumps(finalize(args.output, args.cache), indent=1, default=str))
    else:
        n = args.max_frames or len(read_records(SOURCE / "frames.jsonl"))
        path, result = encode(args.output, args.cache, n, args.bitrate, args.workers)
        write_json(args.output / "encode.json", result)
        print(result)
        if args.command == "all" and n == len(read_records(SOURCE / "frames.jsonl")):
            print(json.dumps(finalize(args.output, args.cache), indent=1, default=str))
