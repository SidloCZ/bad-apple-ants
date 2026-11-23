"""Render Bad Apple!! Ants composite MP4 (720p/1440p/4K), diagnostic stills, and credits."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from .geometry import max_fit_crop_w, render_crop
from .media import TARGET_FPS, extract_clip_frames

OUT_W, OUT_H = 1280, 720
MAIN_W, MAIN_H = 960, 720
SIDE_X = 960
PAD = 16
PREV_W, PREV_H = 288, 216
CHART_W, CHART_H = 288, 56
MAP_W, MAP_H = 132, 64

_EE_A1 = 1.340264
_EE_A2 = -0.081106
_EE_A3 = 0.000893
_EE_A4 = 0.003796
_EE_M = math.sqrt(3.0) / 2.0
_EE_X_MAX = (2.0 * math.sqrt(3.0) * math.pi) / (3.0 * _EE_A1)
_EE_THETA_MAX = math.pi / 3.0
_EE_Y_MAX = _EE_THETA_MAX * (
    _EE_A1
    + _EE_A2 * (_EE_THETA_MAX**2)
    + (_EE_THETA_MAX**6) * (_EE_A3 + _EE_A4 * (_EE_THETA_MAX**2))
)

RESOLUTION_SCALES = {
    "720p": 1,
    "1440p": 2,
    "4k": 3,
}

MAIN_MODES = ("grayscale", "grayscale-stretch", "color")


def _stretch_grayscale(gray_img: Image.Image, low_pct: float = 1.0, high_pct: float = 99.0) -> Image.Image:
    """Apply gentle percentile contrast stretch to an L-mode image."""
    arr = np.asarray(gray_img, dtype=np.float32)
    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi - lo < 1.0:
        return gray_img
    stretched = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(stretched, mode="L")


_FALLBACK_FONT_PATHS = (
    "NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/LeelawUI.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)

_ITALIC_FALLBACK_FONT_PATHS = (
    "C:/Windows/Fonts/segoeuii.ttf",
    "C:/Windows/Fonts/ariali.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)


class FontChain:
    """Ordered font list supporting per-character script fallback for non-Latin names."""

    def __init__(self, fonts: list[ImageFont.ImageFont]):
        self.fonts = fonts
        self._notdef_sigs: dict[int, tuple[tuple[int, int], bytes] | None] = {}
        self._char_cache: dict[str, ImageFont.ImageFont] = {}

    @classmethod
    def load(cls, assets_dir: Path, filename: str, size: int) -> FontChain:
        fonts: list[ImageFont.ImageFont] = []
        seen: set[Path] = set()

        def add_path(p: Path) -> None:
            try:
                resolved = p.resolve()
            except OSError:
                resolved = p
            if resolved in seen or not p.is_file():
                return
            try:
                fonts.append(ImageFont.truetype(str(p), size))
                seen.add(resolved)
            except OSError:
                pass

        add_path(assets_dir / "fonts" / filename)
        if "Mono" in filename:
            add_path(Path("C:/Windows/Fonts/consola.ttf"))
        fallbacks = (
            (*_ITALIC_FALLBACK_FONT_PATHS, *_FALLBACK_FONT_PATHS)
            if "Italic" in filename
            else _FALLBACK_FONT_PATHS
        )
        for rel_or_abs in fallbacks:
            p = Path(rel_or_abs)
            if not p.is_absolute():
                p = assets_dir / "fonts" / p
            add_path(p)

        if not fonts:
            fonts.append(ImageFont.load_default(size=size))
        return cls(fonts)

    def _has_glyph(self, font: ImageFont.ImageFont, ch: str) -> bool:
        if ch.isspace():
            return True
        if not isinstance(font, ImageFont.FreeTypeFont):
            return True
        fid = id(font)
        if fid not in self._notdef_sigs:
            try:
                m_nd = font.getmask("\ufffe")
                self._notdef_sigs[fid] = (m_nd.size, bytes(m_nd))
            except Exception:
                self._notdef_sigs[fid] = None
        nd_sig = self._notdef_sigs[fid]
        if nd_sig is None:
            return True
        try:
            m = font.getmask(ch)
            return (m.size, bytes(m)) != nd_sig
        except Exception:
            return False

    def resolve(self, ch: str) -> ImageFont.ImageFont:
        cached = self._char_cache.get(ch)
        if cached is not None:
            return cached
        chosen = self.fonts[0]
        for font in self.fonts:
            if self._has_glyph(font, ch):
                chosen = font
                break
        self._char_cache[ch] = chosen
        return chosen

    def runs(self, text: str) -> list[tuple[str, ImageFont.ImageFont]]:
        if not text:
            return []
        out: list[tuple[str, ImageFont.ImageFont]] = []
        cur_font = self.resolve(text[0])
        buf = [text[0]]
        for ch in text[1:]:
            f = self.resolve(ch)
            if f is cur_font:
                buf.append(ch)
            else:
                out.append(("".join(buf), cur_font))
                cur_font = f
                buf = [ch]
        out.append(("".join(buf), cur_font))
        return out

    def textlength(self, draw: ImageDraw.ImageDraw, text: str) -> float:
        return sum(float(draw.textlength(chunk, font=f)) for chunk, f in self.runs(text))

    def truncate(self, draw: ImageDraw.ImageDraw, text: str, max_w: int) -> str:
        if self.textlength(draw, text) <= max_w:
            return text
        suffix = "..."
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.textlength(draw, text[:mid] + suffix) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + suffix

    def draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        fill: tuple[int, int, int],
    ) -> None:
        x, y = float(xy[0]), float(xy[1])
        primary = self.fonts[0]
        runs = self.runs(text)
        if len(runs) == 1 and runs[0][1] is primary:
            draw.text((x, y), text, fill=fill, font=primary)
            return
        if isinstance(primary, ImageFont.FreeTypeFont):
            ascent, _ = primary.getmetrics()
            baseline_y = y + float(ascent)
            for chunk, f in runs:
                anchor = "ls" if isinstance(f, ImageFont.FreeTypeFont) else None
                draw.text((x, baseline_y if anchor else y), chunk, fill=fill, font=f, anchor=anchor)
                x += float(draw.textlength(chunk, font=f))
        else:
            for chunk, f in runs:
                draw.text((x, y), chunk, fill=fill, font=f)
                x += float(draw.textlength(chunk, font=f))


def _load_font(assets_dir: Path, filename: str, size: int) -> ImageFont.ImageFont:
    return FontChain.load(assets_dir, filename, size).fonts[0]


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    return FontChain([font]).truncate(draw, text, max_w)


def _format_taxon(item: dict) -> str:
    species = (item.get("species") or "").strip()
    if species:
        return species
    genus = (item.get("genus") or "").strip()
    if genus:
        return f"{genus} sp."
    return "Formicidae"


def _split_taxon_runs(item: dict) -> list[tuple[str, bool]]:
    """Split taxon label into (text, is_italic) runs following binomial nomenclature rules."""
    species = (item.get("species") or "").strip()
    if species:
        return [(species, True)]
    genus = (item.get("genus") or "").strip()
    if genus:
        return [(genus, True), (" sp.", False)]
    return [("Formicidae", False)]


def _draw_taxon(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    item: dict,
    font_italic: FontChain,
    font_regular: FontChain,
    max_w: int,
    fill: tuple[int, int, int],
) -> None:
    """Draw taxonomic label with italic binomial/genus and upright 'sp.' or family rank."""
    runs = _split_taxon_runs(item)
    if len(runs) == 1:
        text, is_it = runs[0]
        chain = font_italic if is_it else font_regular
        chain.draw_text(draw, xy, chain.truncate(draw, text, max_w), fill=fill)
        return

    genus, _ = runs[0]
    suffix, _ = runs[1]
    suffix_w = font_regular.textlength(draw, suffix)
    if font_italic.textlength(draw, genus) + suffix_w <= max_w:
        font_italic.draw_text(draw, xy, genus, fill=fill)
        gx = xy[0] + int(round(font_italic.textlength(draw, genus)))
        font_regular.draw_text(draw, (gx, xy[1]), suffix, fill=fill)
    elif suffix_w < max_w:
        genus_trunc = font_italic.truncate(draw, genus, max(0, int(max_w - suffix_w)))
        font_italic.draw_text(draw, xy, genus_trunc, fill=fill)
        gx = xy[0] + int(round(font_italic.textlength(draw, genus_trunc)))
        font_regular.draw_text(draw, (gx, xy[1]), suffix, fill=fill)
    else:
        font_italic.draw_text(draw, xy, font_italic.truncate(draw, genus + suffix, max_w), fill=fill)


def _project_lat_lon(
    lat: float,
    lon: float,
    map_w: int,
    map_h: int,
    margin: float = 0.0,
) -> tuple[float, float]:
    """Equal Earth projection of (lat, lon) onto pixel coordinates within map_w x map_h."""
    lat_c = min(90.0, max(-90.0, float(lat)))
    lon_c = min(180.0, max(-180.0, float(lon)))
    phi = math.radians(lat_c)
    lam = math.radians(lon_c)
    theta = math.asin(max(-1.0, min(1.0, _EE_M * math.sin(phi))))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = (2.0 * math.sqrt(3.0) * lam * math.cos(theta)) / (
        3.0 * (_EE_A1 + 3.0 * _EE_A2 * t2 + t6 * (7.0 * _EE_A3 + 9.0 * _EE_A4 * t2))
    )
    y = theta * (_EE_A1 + _EE_A2 * t2 + t6 * (_EE_A3 + _EE_A4 * t2))
    px = ((x + _EE_X_MAX) / (2.0 * _EE_X_MAX)) * float(max(1, map_w - 1))
    py = ((_EE_Y_MAX - y) / (2.0 * _EE_Y_MAX)) * float(max(1, map_h - 1))
    if margin > 0.0:
        px = min(max(margin, px), max(margin, float(map_w - 1) - margin))
        py = min(max(margin, py), max(margin, float(map_h - 1) - margin))
    return px, py


def _build_world_map_base(
    assets_dir: Path,
    map_w: int,
    map_h: int,
    scale: int,
) -> Image.Image:
    """Pre-render static Equal Earth land-only mini world map matching sidebar background."""
    ss = 2
    sw, sh = map_w * ss, map_h * ss
    bg_color = (18, 19, 22)
    land_fill = (62, 70, 84)
    land_outline = (74, 84, 98)

    hi = Image.new("RGB", (sw, sh), bg_color)
    hi_draw = ImageDraw.Draw(hi)

    geojson_path = assets_dir / "world_land_110m.geojson"
    if geojson_path.is_file():
        geo = json.loads(geojson_path.read_text(encoding="utf-8"))
        for feat in geo.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            if gtype == "Polygon":
                polys = [coords]
            elif gtype == "MultiPolygon":
                polys = coords
            else:
                continue
            for poly in polys:
                if not poly:
                    continue
                outer = [_project_lat_lon(pt[1], pt[0], sw, sh) for pt in poly[0]]
                if len(outer) >= 3:
                    hi_draw.polygon(outer, fill=land_fill, outline=land_outline)
                for hole in poly[1:]:
                    inner = [_project_lat_lon(pt[1], pt[0], sw, sh) for pt in hole]
                    if len(inner) >= 3:
                        hi_draw.polygon(inner, fill=bg_color, outline=land_outline)

    return hi.resize((map_w, map_h), Image.Resampling.LANCZOS)


def _build_glow_dot_sprite(scale: int) -> Image.Image:
    """Create a pre-rendered RGBA golden-yellow glowing marker sprite scaled for the target resolution."""
    r_glow = max(6, int(round(8.0 * scale)))
    size = 2 * r_glow + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((xx - float(r_glow)) ** 2 + (yy - float(r_glow)) ** 2)

    glow_t = np.clip(1.0 - dist / float(r_glow), 0.0, 1.0)
    alpha = 160.0 * (glow_t ** 2)
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[..., 0] = 250.0
    rgb[..., 1] = 204.0
    rgb[..., 2] = 21.0

    r_dot = 2.2 * float(scale)
    dot_cov = np.clip((r_dot + 0.6 - dist) / 1.2, 0.0, 1.0)
    alpha = np.maximum(alpha, 255.0 * dot_cov)

    r_hot = 0.9 * float(scale)
    hot_cov = np.clip((r_hot + 0.5 - dist) / 1.0, 0.0, 1.0)
    rgb[..., 0] = rgb[..., 0] * (1.0 - hot_cov) + 255.0 * hot_cov
    rgb[..., 1] = rgb[..., 1] * (1.0 - hot_cov) + 252.0 * hot_cov
    rgb[..., 2] = rgb[..., 2] * (1.0 - hot_cov) + 230.0 * hot_cov

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _draw_map_dot(
    canvas: Image.Image,
    map_x: int,
    map_y: int,
    map_w: int,
    map_h: int,
    scale: int,
    item: dict,
    glow_sprite: Image.Image,
) -> None:
    """Composite glowing yellow specimen origin dot onto the mini world map if coordinates exist."""
    lat = item.get("lat")
    lon = item.get("lon")
    if lat is None or lon is None:
        return
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return

    px_rel, py_rel = _project_lat_lon(lat_f, lon_f, map_w, map_h, margin=float(2 * scale))
    half = glow_sprite.width // 2
    gx0 = map_x + int(round(px_rel)) - half
    gy0 = map_y + int(round(py_rel)) - half
    gx1 = gx0 + glow_sprite.width
    gy1 = gy0 + glow_sprite.height

    cx0 = max(gx0, map_x)
    cy0 = max(gy0, map_y)
    cx1 = min(gx1, map_x + map_w)
    cy1 = min(gy1, map_y + map_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    sprite_crop = glow_sprite.crop((cx0 - gx0, cy0 - gy0, cx1 - gx0, cy1 - gy0))
    canvas.paste(sprite_crop, (cx0, cy0), sprite_crop)


def _format_source(item: dict) -> str:
    src = item.get("source") or ""
    view = item.get("view") or ""
    if src == "inat":
        return "Source: iNaturalist (live photo)"
    if src == "antweb":
        return f"Source: AntWeb ({view})" if view else "Source: AntWeb"
    if src == "mcz":
        return f"Source: Harvard MCZ ({view})" if view else "Source: Harvard MCZ"
    if src == "nmnh":
        return f"Source: Smithsonian NMNH ({view})" if view else "Source: Smithsonian NMNH"
    return f"Source: {src}"


def _frame_img_path(fr: dict, resolved_map: dict[str, str]) -> Path:
    item = fr["item"]
    raw_path = resolved_map.get(item["id"]) or item.get("thumb_path") or ""
    img_path = Path(raw_path.replace("\\", "/"))
    if img_path.is_file():
        return img_path
    thumb_raw = item.get("thumb_path") or ""
    p_thumb = Path(thumb_raw.replace("\\", "/"))
    return p_thumb if p_thumb.is_file() else Path(raw_path)


def _frame_crop_key(fr: dict) -> tuple:
    pose = fr["pose"]
    return (
        fr["item"]["id"],
        round(pose["cx_frac"], 6),
        round(pose["cy_frac"], 6),
        round(pose["w_frac"], 6),
        round(pose["theta"], 6),
        bool(pose["mirrored"]),
    )


def _render_panel_pair(
    img_path: Path,
    pose: dict,
    main_mode: str,
    main_w: int = MAIN_W,
    main_h: int = MAIN_H,
    prev_w: int = PREV_W,
    prev_h: int = PREV_H,
) -> tuple[Image.Image, Image.Image]:
    """Load, orient, and warp a single match photo into (main_panel, sidebar_preview)."""
    with Image.open(img_path) as raw:
        src_img = ImageOps.exif_transpose(raw).convert("RGB")
    color_crop = render_crop(
        src_img,
        main_w,
        main_h,
        pose["cx_frac"],
        pose["cy_frac"],
        pose["w_frac"],
        pose["theta"],
        bool(pose["mirrored"]),
    )
    if main_mode == "color":
        side_prev = (
            color_crop.convert("L")
            .resize((prev_w, prev_h), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        return color_crop, side_prev

    gray_l = color_crop.convert("L")
    if main_mode == "grayscale-stretch":
        gray_l = _stretch_grayscale(gray_l)
    main_panel = gray_l.convert("RGB")
    side_prev = color_crop.resize((prev_w, prev_h), Image.Resampling.LANCZOS)
    return main_panel, side_prev


def _write_credits(frames: list[dict], out_path: Path) -> None:
    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for fr in frames:
        item = fr["item"]
        iid = item["id"]
        counts[iid] = counts.get(iid, 0) + 1
        if iid not in seen:
            seen[iid] = item

    lines = [
        "Bad Apple!! Ants - Image Credits",
        "================================",
        f"Total frames: {len(frames)}",
        f"Unique ant photographs used: {len(seen)}",
        "",
        "Source video and music: Bad Apple!! (Alstroemeria Records / ZUN / Anira shadow PV).",
        "Data sources: GBIF occurrences (iNaturalist Research-Grade, AntWeb / CAS, Harvard MCZ, Smithsonian NMNH).",
        "GBIF Download Citation: GBIF.org (20 November 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.dyc97c",
        "",
        "Per-image attributions (sorted by frame count):",
        "",
    ]
    for iid, item in sorted(seen.items(), key=lambda kv: (-counts[kv[0]], _format_taxon(kv[1]))):
        taxon = _format_taxon(item)
        creator = item.get("creator") or "Unknown"
        lic = item.get("license") or "Unknown"
        src_url = item.get("gbif_url") or item.get("reference_url") or item.get("full_url") or ""
        lines.append(
            f"- [{counts[iid]:3d}f] {taxon} | {item['source']} ({item.get('view', '')}) | "
            f"Creator: {creator} | License: {lic} | {src_url}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_video(
    matches_path: Path,
    video_path: Path,
    assets_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    main_mode: str = "grayscale-stretch",
    resolution: str = "720p",
    crf: int = 18,
    preset: str = "medium",
    preview_start_s: float = 0.0,
    preview_duration_s: float = 0.0,
) -> Path:
    """Render 720p/1440p/4K MP4 with sidebar previews, NCC sparkline, mini map, credits, and stills."""
    if main_mode not in MAIN_MODES:
        raise ValueError(f"Unsupported main_mode={main_mode!r}; expected one of {MAIN_MODES}")
    if resolution not in RESOLUTION_SCALES:
        raise ValueError(
            f"Unsupported resolution={resolution!r}; expected one of {tuple(RESOLUTION_SCALES.keys())}"
        )

    s = RESOLUTION_SCALES[resolution]
    out_w, out_h = OUT_W * s, OUT_H * s
    main_w, main_h = MAIN_W * s, MAIN_H * s
    side_x = SIDE_X * s
    pad = PAD * s
    prev_w, prev_h = PREV_W * s, PREV_H * s
    chart_w, chart_h = CHART_W * s, CHART_H * s
    map_w, map_h = MAP_W * s, MAP_H * s
    border_w = max(1, s)

    payload = json.loads(matches_path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    clip_meta = payload["clip"]
    start_s = float(clip_meta["start_s"])
    duration_s = float(clip_meta["duration_s"])

    resolved_path = cache_dir / "full" / "resolved.json"
    resolved_map: dict[str, str] = (
        json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
    )

    orig_frames_u8, _ = extract_clip_frames(video_path, start_s, duration_s, cache_dir)
    if len(orig_frames_u8) != len(frames):
        raise RuntimeError(
            f"Frame count mismatch: matches={len(frames)} vs extracted={len(orig_frames_u8)}"
        )

    n_total = len(frames)
    fps_num, fps_den = (int(p) for p in TARGET_FPS.split("/"))
    fps_val = float(fps_num) / float(fps_den)
    if preview_duration_s > 0.0:
        slice_start_idx = min(n_total - 1, max(0, int(round(preview_start_s * fps_val))))
        slice_count = max(1, int(round(preview_duration_s * fps_val)))
        slice_end_idx = min(n_total, slice_start_idx + slice_count)
    else:
        slice_start_idx = 0
        slice_end_idx = n_total
    render_indices = list(range(slice_start_idx, slice_end_idx))
    if not render_indices:
        raise RuntimeError("No frames selected for rendering")

    font_label = _load_font(assets_dir, "NotoSansMono-Regular.ttf", 12 * s)
    font_taxon_italic = FontChain.load(assets_dir, "NotoSans-Italic.ttf", 15 * s)
    font_taxon_regular = FontChain.load(assets_dir, "NotoSans-Regular.ttf", 15 * s)
    font_meta = FontChain.load(assets_dir, "NotoSans-Regular.ttf", 10 * s)
    font_stats = _load_font(assets_dir, "NotoSansMono-Regular.ttf", 10 * s)

    map_base = _build_world_map_base(assets_dir, map_w, map_h, s)
    glow_sprite = _build_glow_dot_sprite(s)

    output_dir.mkdir(parents=True, exist_ok=True)
    stills_dir = output_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    for old_still in stills_dir.glob("still_*.png"):
        old_still.unlink()

    _write_credits(frames, output_dir / "CREDITS.txt")

    # Select diagnostic frames across the rendered range: 10%, 35%, 65%, 90% + best/worst textured NCC.
    still_targets: dict[int, str] = {}
    m_render = len(render_indices)
    for pct in (10, 35, 65, 90):
        rel = min(m_render - 1, max(0, int(round((pct / 100.0) * (m_render - 1)))))
        idx = render_indices[rel]
        still_targets.setdefault(idx, f"still_f{idx:04d}_p{pct}.png")
    tex_indices = [i for i in render_indices if not frames[i].get("is_solid")]
    if tex_indices:
        best_i = max(tex_indices, key=lambda i: frames[i]["score_64"])
        worst_i = min(tex_indices, key=lambda i: frames[i]["score_64"])
        still_targets[best_i] = f"still_f{best_i:04d}_best_ncc.png"
        still_targets[worst_i] = f"still_f{worst_i:04d}_worst_ncc.png"

    ncc_history = [0.0 if fr.get("is_solid") else float(fr["score_64"]) for fr in frames]
    frame_digits = max(3, len(str(n_total)))

    stem = f"badapple-ants-{resolution}-preview" if preview_duration_s > 0.0 else f"badapple-ants-{resolution}"
    mp4_path = output_dir / f"{stem}.mp4"
    part_mp4 = output_dir / f"{stem}.part.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        TARGET_FPS,
        "-i",
        "pipe:0",
    ]
    audio_ss = start_s + (slice_start_idx / fps_val if preview_duration_s > 0.0 else 0.0)
    audio_t = (slice_end_idx - slice_start_idx) / fps_val if preview_duration_s > 0.0 else duration_s
    if audio_ss > 0.0:
        cmd.extend(["-ss", f"{audio_ss:.4f}"])
    if audio_t > 0.0:
        cmd.extend(["-t", f"{audio_t:.4f}"])
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-crf",
            str(int(crf)),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(part_mp4),
        ]
    )

    last_main_panel: Image.Image | None = None
    last_side_prev: Image.Image | None = None
    side_label = "GRAYSCALE MATCH" if main_mode == "color" else "ORIGINAL PHOTO"

    with tempfile.TemporaryFile() as err_log, ThreadPoolExecutor(max_workers=4) as crop_pool:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_log)
        try:
            assert proc.stdin is not None
            pending: dict[int, Future[tuple[Image.Image, Image.Image]]] = {}
            submit_pos = 0
            last_sub_key: tuple | None = None
            prefetch_window = 16

            def _fill_prefetch() -> None:
                nonlocal submit_pos, last_sub_key
                while submit_pos < m_render and len(pending) < prefetch_window:
                    sub_t = render_indices[submit_pos]
                    sub_fr = frames[sub_t]
                    sub_key = _frame_crop_key(sub_fr)
                    if sub_key != last_sub_key:
                        img_path = _frame_img_path(sub_fr, resolved_map)
                        pending[sub_t] = crop_pool.submit(
                            _render_panel_pair,
                            img_path,
                            sub_fr["pose"],
                            main_mode,
                            main_w,
                            main_h,
                            prev_w,
                            prev_h,
                        )
                        last_sub_key = sub_key
                    submit_pos += 1

            for t in tqdm(render_indices, desc=f"render-{resolution}", unit="frame"):
                _fill_prefetch()
                fut = pending.pop(t, None)
                if fut is not None:
                    main_panel, side_prev = fut.result()
                    last_main_panel = main_panel
                    last_side_prev = side_prev
                else:
                    assert last_main_panel is not None and last_side_prev is not None
                    main_panel = last_main_panel
                    side_prev = last_side_prev

                fr = frames[t]
                item = fr["item"]
                pose = fr["pose"]

                canvas = Image.new("RGB", (out_w, out_h), (18, 19, 22))
                canvas.paste(main_panel, (0, 0))
                draw = ImageDraw.Draw(canvas)

                x0 = side_x + pad
                # 1. Original Bad Apple!! template preview
                draw.text((x0, 12 * s), "BA TEMPLATE", fill=(156, 163, 175), font=font_label)
                orig_pil = (
                    Image.fromarray(orig_frames_u8[t], mode="L")
                    .resize((prev_w, prev_h), Image.Resampling.LANCZOS)
                    .convert("RGB")
                )
                canvas.paste(orig_pil, (x0, 32 * s))
                draw.rectangle(
                    (x0 - border_w, 32 * s - border_w, x0 + prev_w + border_w - 1, 32 * s + prev_h + border_w - 1),
                    outline=(42, 45, 52),
                    width=border_w,
                )

                # 2. Secondary match preview (original color photo when main is grayscale, or vice versa)
                draw.text((x0, 258 * s), side_label, fill=(156, 163, 175), font=font_label)
                canvas.paste(side_prev, (x0, 278 * s))
                draw.rectangle(
                    (x0 - border_w, 278 * s - border_w, x0 + prev_w + border_w - 1, 278 * s + prev_h + border_w - 1),
                    outline=(42, 45, 52),
                    width=border_w,
                )

                # 3. Live NCC sparkline up to current frame t
                ncc_val = 0.0 if fr.get("is_solid") else float(fr["score_64"])
                ncc_label = "NCC SOLID" if fr.get("is_solid") else f"NCC {ncc_val:+.3f}"
                draw.text((x0, 504 * s), ncc_label, fill=(156, 163, 175), font=font_label)

                cx0, cy0 = x0, 524 * s
                cx1, cy1 = x0 + chart_w - 1, 524 * s + chart_h - 1
                draw.rectangle((cx0, cy0, cx1, cy1), fill=(24, 26, 32), outline=(42, 45, 52), width=border_w)
                chart_inner_h = chart_h - 6 * s
                chart_pad_y = 3 * s
                for ref_score in (0.5, 0.8):
                    ry = int(round(cy1 - max(0.0, min(1.0, ref_score)) * chart_inner_h - chart_pad_y))
                    draw.line((cx0 + border_w, ry, cx1 - border_w, ry), fill=(38, 42, 50), width=max(1, s))

                if t > 0:
                    pts: list[tuple[int, int]] = []
                    for px_col in range(chart_w):
                        idx = int(px_col * (n_total - 1) / max(1, chart_w - 1))
                        if idx > t:
                            break
                        val = max(0.0, min(1.0, ncc_history[idx]))
                        py = int(round(cy1 - val * chart_inner_h - chart_pad_y))
                        pts.append((cx0 + px_col, py))
                    if len(pts) >= 2:
                        draw.line(pts, fill=(56, 189, 248), width=max(2, 2 * s))
                dot_x = cx0 + int(round(t * (chart_w - 1) / max(1, n_total - 1)))
                dot_val = max(0.0, min(1.0, ncc_val))
                dot_y = int(round(cy1 - dot_val * chart_inner_h - chart_pad_y))
                dot_r = 3 * s
                draw.ellipse((dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r), fill=(250, 204, 21))

                # 4. Taxon, credit metadata, and static Equal Earth mini world map
                map_x = x0 + prev_w - map_w
                map_y = 614 * s
                meta_w = prev_w - map_w - 6 * s

                w_pack, h_pack = item["pack_dims"]
                cw_max = max_fit_crop_w(float(w_pack), float(h_pack), float(pose["theta"]), margin=0.5)
                scale_pct = int(round(100.0 * (float(pose["w_frac"]) * float(w_pack)) / max(1.0, cw_max)))
                rot_deg = int(round(math.degrees(float(pose["theta"])))) % 360
                mir_tag = " MIR" if pose["mirrored"] else ""

                source_txt = font_meta.truncate(draw, _format_source(item), meta_w)
                country_txt = font_meta.truncate(
                    draw, f"Location: {item.get('country') or 'Unknown'}", meta_w
                )
                creator_txt = font_meta.truncate(draw, item.get("creator") or "Unknown", meta_w)
                license_txt = font_meta.truncate(draw, item.get("license") or "Unknown", meta_w)
                stats_txt = (
                    f"SCALE {scale_pct:3d}%  ROT {rot_deg:3d}d{mir_tag}  F {t + 1:0{frame_digits}d}/{n_total}"
                )

                _draw_taxon(
                    draw,
                    (x0, 586 * s),
                    item,
                    font_taxon_italic,
                    font_taxon_regular,
                    prev_w,
                    fill=(243, 244, 246),
                )
                font_meta.draw_text(draw, (x0, 610 * s), source_txt, fill=(209, 213, 219))
                font_meta.draw_text(draw, (x0, 626 * s), country_txt, fill=(156, 163, 175))
                font_meta.draw_text(draw, (x0, 642 * s), creator_txt, fill=(156, 163, 175))
                font_meta.draw_text(draw, (x0, 658 * s), license_txt, fill=(156, 163, 175))

                canvas.paste(map_base, (map_x, map_y))
                _draw_map_dot(canvas, map_x, map_y, map_w, map_h, s, item, glow_sprite)

                draw.text((x0, 688 * s), stats_txt, fill=(250, 204, 21), font=font_stats)

                if t in still_targets:
                    canvas.save(stills_dir / still_targets[t])

                proc.stdin.write(canvas.tobytes())

            proc.stdin.close()
            code = proc.wait()
            if code != 0:
                err_log.seek(0)
                raise RuntimeError(f"FFmpeg encode failed: {err_log.read().decode(errors='replace')}")
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    part_mp4.replace(mp4_path)
    print(f"rendered: {mp4_path} ({mp4_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    return mp4_path
