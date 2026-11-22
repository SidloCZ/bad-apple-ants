"""Source video, font downloads, and clip frame extraction."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import numpy as np

VIDEO = (
    "https://raw.githubusercontent.com/bad-apple-lab/Bad-Apple/"
    "80e39ed58d754e4de420b7a1c474e53e20a71ba3/badapple.mp4",
    "badapple.mp4",
    "d8339642fd3303a0c4971e1564515e82f69ef61b700ce17b681debb19404c23a",
)

NOTO_BASE = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/"
    "412a7edb2c8c535d1066157e5b17e661fd5591fe/fonts/"
)
FONTS = [
    (
        NOTO_BASE + "NotoSans/full/ttf/NotoSans-Regular.ttf",
        "fonts/NotoSans-Regular.ttf",
        "f5f552c8c5edb61fe6efb824baf4d4de47b1a8689ab4925ff43f7bd6a4ebece5",
    ),
    (
        NOTO_BASE + "NotoSansMono/unhinted/ttf/NotoSansMono-Regular.ttf",
        "fonts/NotoSansMono-Regular.ttf",
        "87f8ce0522a6c99b743ee5fc75b4073cfdd575639119672828b7b9944b65b4f4",
    ),
]

TARGET_FPS = "30000/1001"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(url: str, path: Path, expected_sha256: str) -> Path:
    if path.is_file() and sha256_file(path) == expected_sha256:
        print(f"cached: {path}", flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "badapple-ants-proto/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)
    print(f"downloaded: {path} ({len(data) / 1e6:.2f} MB)", flush=True)
    return path


def fetch_media(assets_dir: Path) -> dict[str, Path]:
    """Download pinned badapple.mp4 and Noto fonts into assets_dir."""
    paths: dict[str, Path] = {}
    for url, rel, digest in [VIDEO, *FONTS]:
        out_path = _download_verified(url, assets_dir / rel, digest)
        paths[rel] = out_path
    return paths


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    if data and len(data) != size:
        raise RuntimeError(f"Truncated raw frame: {len(data)} / {size} bytes")
    return bytes(data)


def extract_clip_frames(
    video_path: Path,
    start_s: float,
    duration_s: float,
    cache_dir: Path,
    size: tuple[int, int] = (320, 240),
) -> tuple[np.ndarray, dict]:
    """Decode a slice of badapple.mp4 at 30000/1001 fps to (N, H, W) uint8 grayscale."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing source video: {video_path}")
    w, h = size
    key_obj = {
        "video_sha256": sha256_file(video_path),
        "start_s": round(float(start_s), 4),
        "duration_s": round(float(duration_s), 4),
        "fps": TARGET_FPS,
        "size": [w, h],
    }
    key = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:16]
    clip_dir = cache_dir / f"clip_{key}"
    meta_path = clip_dir / "meta.json"
    raw_path = clip_dir / "frames.u8"

    if meta_path.is_file() and raw_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.fromfile(raw_path, dtype=np.uint8).reshape(meta["count"], h, w)
        return arr, meta

    clip_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"fps={TARGET_FPS},scale={w}:{h}:flags=lanczos,setsar=1,format=gray"
    cmd = [ffmpeg_exe, "-v", "error", "-nostdin"]
    if start_s > 0.0:
        cmd.extend(["-ss", f"{start_s:.4f}"])
    if duration_s > 0.0:
        cmd.extend(["-t", f"{duration_s:.4f}"])
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-an",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
    )
    frame_bytes = w * h
    frames: list[np.ndarray] = []
    partial_raw = clip_dir / "frames.u8.part"
    with tempfile.TemporaryFile() as err_log, partial_raw.open("wb") as sink:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_log)
        try:
            assert proc.stdout is not None
            while True:
                raw = _read_exact(proc.stdout, frame_bytes)
                if not raw:
                    break
                sink.write(raw)
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy())
            code = proc.wait()
            if code != 0:
                err_log.seek(0)
                raise RuntimeError(err_log.read().decode(errors="replace"))
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    if not frames:
        raise RuntimeError(f"Decoded 0 frames from {video_path} at start={start_s}, dur={duration_s}")
    partial_raw.replace(raw_path)
    arr = np.stack(frames, axis=0)
    meta = {
        **key_obj,
        "count": int(arr.shape[0]),
        "width": w,
        "height": h,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"extracted {meta['count']} frames ({w}x{h} @ {TARGET_FPS} fps)", flush=True)
    return arr, meta
