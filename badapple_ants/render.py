"""Render 1280x720 Bad Apple!! Ants composite MP4, diagnostic stills, and credits."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections import OrderedDict
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


MAIN_MODES = ("grayscale", "grayscale-stretch", "color")


def _stretch_grayscale(gray_img: Image.Image, low_pct: float = 1.0, high_pct: float = 99.0) -> Image.Image:
    """Apply gentle percentile contrast stretch to an L-mode image."""
    arr = np.asarray(gray_img, dtype=np.float32)
    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi - lo < 1.0:
        return gray_img
    stretched = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(stretched, mode="L")


def _load_font(assets_dir: Path, filename: str, size: int) -> ImageFont.ImageFont:
    font_path = assets_dir / "fonts" / filename
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    fallback = Path("C:/Windows/Fonts/consola.ttf" if "Mono" in filename else "C:/Windows/Fonts/arial.ttf")
    if fallback.is_file():
        return ImageFont.truetype(str(fallback), size)
    return ImageFont.load_default(size=size)


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(text[:mid] + suffix, font=font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def _format_taxon(item: dict) -> str:
    species = (item.get("species") or "").strip()
    if species:
        return species
    genus = (item.get("genus") or "").strip()
    if genus:
        return f"{genus} sp."
    return "Formicidae"


def _format_source(item: dict) -> str:
    src = item.get("source") or ""
    view = item.get("view") or ""
    if src == "inat":
        return "Source: iNaturalist (live photo)"
    if src == "antweb":
        return f"Source: AntWeb ({view})" if view else "Source: AntWeb"
    return f"Source: {src}"


class _ImageLRU:
    def __init__(self, max_items: int = 48):
        self.max_items = max_items
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, path: Path) -> Image.Image:
        key = str(path)
        img = self._cache.get(key)
        if img is not None:
            self._cache.move_to_end(key)
            return img
        with Image.open(path) as raw:
            loaded = ImageOps.exif_transpose(raw).convert("RGB")
        self._cache[key] = loaded
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return loaded


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
        "Bad Apple!! Ants Prototype - Image Credits",
        "==========================================",
        f"Total frames: {len(frames)}",
        f"Unique ant photographs used: {len(seen)}",
        "",
        "Source video and music: Bad Apple!! (Alstroemeria Records / ZUN / Anira shadow PV).",
        "Data sources: GBIF occurrences from iNaturalist Research-Grade and AntWeb (California Academy of Sciences).",
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
) -> Path:
    """Render 1280x720 MP4 with sidebar previews, NCC sparkline, credits, and stills."""
    if main_mode not in MAIN_MODES:
        raise ValueError(f"Unsupported main_mode={main_mode!r}; expected one of {MAIN_MODES}")

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

    font_label = _load_font(assets_dir, "NotoSansMono-Regular.ttf", 12)
    font_taxon = _load_font(assets_dir, "NotoSans-Regular.ttf", 15)
    font_meta = _load_font(assets_dir, "NotoSans-Regular.ttf", 12)
    font_stats = _load_font(assets_dir, "NotoSansMono-Regular.ttf", 11)

    output_dir.mkdir(parents=True, exist_ok=True)
    stills_dir = output_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    for old_still in stills_dir.glob("still_*.png"):
        old_still.unlink()

    _write_credits(frames, output_dir / "CREDITS.txt")

    # Select diagnostic frames: 10%, 35%, 65%, 90% + best/worst textured NCC.
    n = len(frames)
    still_targets: dict[int, str] = {}
    for pct in (10, 35, 65, 90):
        idx = min(n - 1, max(0, int(round((pct / 100.0) * (n - 1)))))
        still_targets.setdefault(idx, f"still_f{idx:04d}_p{pct}.png")
    tex_indices = [i for i, fr in enumerate(frames) if not fr.get("is_solid")]
    if tex_indices:
        best_i = max(tex_indices, key=lambda i: frames[i]["score_64"])
        worst_i = min(tex_indices, key=lambda i: frames[i]["score_64"])
        still_targets[best_i] = f"still_f{best_i:04d}_best_ncc.png"
        still_targets[worst_i] = f"still_f{worst_i:04d}_worst_ncc.png"

    lru = _ImageLRU(max_items=48)
    ncc_history = [0.0 if fr.get("is_solid") else float(fr["score_64"]) for fr in frames]

    mp4_path = output_dir / "badapple-ants-proto.mp4"
    part_mp4 = output_dir / "badapple-ants-proto.part.mp4"
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
        f"{OUT_W}x{OUT_H}",
        "-r",
        TARGET_FPS,
        "-i",
        "pipe:0",
        "-ss",
        f"{start_s:.4f}",
        "-t",
        f"{duration_s:.4f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(part_mp4),
    ]

    last_crop_key: tuple | None = None
    last_main_panel: Image.Image | None = None
    last_side_prev: Image.Image | None = None
    side_label = "GRAYSCALE MATCH" if main_mode == "color" else "COLOR MATCH"

    with tempfile.TemporaryFile() as err_log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_log)
        try:
            assert proc.stdin is not None
            for t, fr in enumerate(tqdm(frames, desc="render", unit="frame")):
                item = fr["item"]
                pose = fr["pose"]
                img_path_str = resolved_map.get(item["id"]) or item.get("thumb_path") or ""
                img_path = Path(img_path_str)
                if not img_path.is_file():
                    img_path = Path(item["thumb_path"])

                crop_key = (
                    item["id"],
                    round(pose["cx_frac"], 6),
                    round(pose["cy_frac"], 6),
                    round(pose["w_frac"], 6),
                    round(pose["theta"], 6),
                    bool(pose["mirrored"]),
                )
                if crop_key == last_crop_key and last_main_panel is not None and last_side_prev is not None:
                    main_panel = last_main_panel
                    side_prev = last_side_prev
                else:
                    src_img = lru.get(img_path)
                    color_crop = render_crop(
                        src_img,
                        MAIN_W,
                        MAIN_H,
                        pose["cx_frac"],
                        pose["cy_frac"],
                        pose["w_frac"],
                        pose["theta"],
                        bool(pose["mirrored"]),
                    )
                    if main_mode == "color":
                        main_panel = color_crop
                        side_prev = (
                            color_crop.convert("L")
                            .resize((PREV_W, PREV_H), Image.Resampling.LANCZOS)
                            .convert("RGB")
                        )
                    else:
                        gray_l = color_crop.convert("L")
                        if main_mode == "grayscale-stretch":
                            gray_l = _stretch_grayscale(gray_l)
                        main_panel = gray_l.convert("RGB")
                        side_prev = color_crop.resize((PREV_W, PREV_H), Image.Resampling.LANCZOS)
                    last_crop_key = crop_key
                    last_main_panel = main_panel
                    last_side_prev = side_prev

                canvas = Image.new("RGB", (OUT_W, OUT_H), (18, 19, 22))
                canvas.paste(main_panel, (0, 0))
                draw = ImageDraw.Draw(canvas)

                x0 = SIDE_X + PAD
                # 1. Original frame preview
                draw.text((x0, 12), "ORIGINAL", fill=(156, 163, 175), font=font_label)
                orig_pil = (
                    Image.fromarray(orig_frames_u8[t], mode="L")
                    .resize((PREV_W, PREV_H), Image.Resampling.LANCZOS)
                    .convert("RGB")
                )
                canvas.paste(orig_pil, (x0, 32))
                draw.rectangle((x0 - 1, 31, x0 + PREV_W, 32 + PREV_H), outline=(42, 45, 52))

                # 2. Secondary match preview (color crop when main is grayscale, or vice versa)
                draw.text((x0, 258), side_label, fill=(156, 163, 175), font=font_label)
                canvas.paste(side_prev, (x0, 278))
                draw.rectangle((x0 - 1, 277, x0 + PREV_W, 278 + PREV_H), outline=(42, 45, 52))

                # 3. Live NCC sparkline up to current frame t
                ncc_val = 0.0 if fr.get("is_solid") else float(fr["score_64"])
                ncc_label = "NCC SOLID" if fr.get("is_solid") else f"NCC {ncc_val:+.3f}"
                draw.text((x0, 504), ncc_label, fill=(156, 163, 175), font=font_label)

                cx0, cy0 = x0, 524
                cx1, cy1 = x0 + CHART_W - 1, 524 + CHART_H - 1
                draw.rectangle((cx0, cy0, cx1, cy1), fill=(24, 26, 32), outline=(42, 45, 52))
                for ref_score in (0.5, 0.8):
                    ry = int(round(cy1 - max(0.0, min(1.0, ref_score)) * (CHART_H - 6) - 3))
                    draw.line((cx0 + 1, ry, cx1 - 1, ry), fill=(38, 42, 50), width=1)

                if t > 0:
                    pts: list[tuple[int, int]] = []
                    for px_col in range(CHART_W):
                        idx = int(px_col * (n - 1) / max(1, CHART_W - 1))
                        if idx > t:
                            break
                        val = max(0.0, min(1.0, ncc_history[idx]))
                        py = int(round(cy1 - val * (CHART_H - 6) - 3))
                        pts.append((cx0 + px_col, py))
                    if len(pts) >= 2:
                        draw.line(pts, fill=(56, 189, 248), width=2)
                dot_x = cx0 + int(round(t * (CHART_W - 1) / max(1, n - 1)))
                dot_val = max(0.0, min(1.0, ncc_val))
                dot_y = int(round(cy1 - dot_val * (CHART_H - 6) - 3))
                draw.ellipse((dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3), fill=(250, 204, 21))

                # 4. Taxon and credit metadata
                w_pack, h_pack = item["pack_dims"]
                cw_max = max_fit_crop_w(float(w_pack), float(h_pack), float(pose["theta"]), margin=0.5)
                scale_pct = int(round(100.0 * (float(pose["w_frac"]) * float(w_pack)) / max(1.0, cw_max)))
                rot_deg = int(round(math.degrees(float(pose["theta"])))) % 360
                mir_tag = " MIR" if pose["mirrored"] else ""

                taxon_txt = _truncate_text(draw, _format_taxon(item), font_taxon, PREV_W)
                source_txt = _truncate_text(draw, _format_source(item), font_meta, PREV_W)
                country_txt = _truncate_text(
                    draw, f"Location: {item.get('country') or 'Unknown'}", font_meta, PREV_W
                )
                creator = item.get("creator") or "Unknown"
                lic = item.get("license") or "Unknown"
                credit_txt = _truncate_text(draw, f"{creator} | {lic}", font_meta, PREV_W)
                stats_txt = (
                    f"SCALE {scale_pct:3d}%  ROT {rot_deg:3d}d{mir_tag}  F {t + 1:03d}/{n}"
                )

                draw.text((x0, 592), taxon_txt, fill=(243, 244, 246), font=font_taxon)
                draw.text((x0, 616), source_txt, fill=(209, 213, 219), font=font_meta)
                draw.text((x0, 636), country_txt, fill=(156, 163, 175), font=font_meta)
                draw.text((x0, 656), credit_txt, fill=(156, 163, 175), font=font_meta)
                draw.text((x0, 682), stats_txt, fill=(250, 204, 21), font=font_stats)

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
