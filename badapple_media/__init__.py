"""Shared video helpers: FFmpeg probing and streaming, cached frame extraction, UI fonts.

Both editions read the source animation the same way. Frames are decoded once by
FFmpeg at the source's own (fractional) frame rate and cached as raw 8-bit gray
together with a small blurred feature vector per frame. The cache key is the
SHA-256 of the source file plus every extraction setting, so a different video
or setting never reuses stale frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageFont


def ffmpeg_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is required on PATH")
    return found


def probe(path: Path) -> dict:
    result = subprocess.run(
        [ffmpeg_binary("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
    )
    info = json.loads(result.stdout)
    stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError(f"No video stream: {path}")
    rate = stream.get("avg_frame_rate", "0/0")
    if rate == "0/0":
        rate = stream.get("r_frame_rate", "0/0")
    fps = Fraction(rate)
    if fps <= 0:
        raise ValueError(f"Invalid frame rate: {rate}")
    return {
        "fps": str(fps),
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(stream.get("duration", info.get("format", {}).get("duration", 0))),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def read_exact(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes, or nothing at end of stream; a partial frame is an error."""
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            break
        data.extend(part)
    if data and len(data) != size:
        raise RuntimeError(f"Truncated raw frame: {len(data)} / {size} bytes")
    return bytes(data)


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in value.lower().split("x"))
        if width < 1 or height < 1:
            raise ValueError
        return width, height
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT with positive integers") from exc


@dataclass(frozen=True)
class FeatureConfig:
    """Per-frame feature settings.

    The project began as a glyph matcher, and all six fields are part of the frame
    cache identity, so the canvas and font fields are kept even though only the
    feature size and blur affect the frame features.
    """

    canvas_width: int = 128
    canvas_height: int = 96
    feature_width: int = 32
    feature_height: int = 24
    font_size: int = 128
    blur: float = 0.65

    def __post_init__(self):
        if min(self.canvas_width, self.canvas_height, self.feature_width, self.feature_height, self.font_size) < 1:
            raise ValueError("Sizes must be positive")
        if self.blur < 0:
            raise ValueError("blur must be nonnegative")


def feature(image: Image.Image, config: FeatureConfig) -> np.ndarray:
    small = image.convert("L").resize((config.feature_width, config.feature_height), Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(config.blur))
    return np.asarray(small, dtype=np.uint8).reshape(-1)


@dataclass
class Frames:
    directory: Path
    metadata: dict

    @property
    def count(self):
        return self.metadata["count"]

    @property
    def fps(self):
        return Fraction(self.metadata["fps"])

    def thumbnails(self):
        return np.memmap(
            self.directory / "frames.u8",
            mode="r",
            dtype=np.uint8,
            shape=(self.count, self.metadata["height"], self.metadata["width"]),
        )

    def features(self):
        return np.load(self.directory / "features.npy", allow_pickle=False)


def extract_frames(
    source: Path,
    cache: Path,
    config: FeatureConfig,
    fps: str | None = None,
    max_frames: int | None = None,
    size: tuple[int, int] = (320, 240),
) -> Frames:
    """Decode every frame once into ``cache/<key>/frames.u8`` (gray) and ``features.npy``."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    metadata = probe(source)
    rate = Fraction(fps) if fps else Fraction(metadata["fps"])
    if rate <= 0:
        raise ValueError("fps must be positive")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = {
        "schema": 1,
        "source_sha256": checksum,
        "fps": str(rate),
        "max_frames": max_frames,
        "size": size,
        "config": asdict(config),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = cache / key
    manifest_path = directory / "frames.json"
    if manifest_path.exists():
        return Frames(directory, json.loads(manifest_path.read_text(encoding="utf-8")))
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary("ffmpeg"), "-v", "error", "-nostdin", "-i", str(source), "-an",
        "-vf", f"fps={rate},scale={size[0]}:{size[1]}:flags=lanczos,setsar=1,format=gray",
    ]  # fmt: skip
    if max_frames:
        command += ["-frames:v", str(max_frames)]
    command += ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    vectors = []
    # stderr goes to a file, so a full stderr pipe cannot deadlock the raw stream.
    with tempfile.TemporaryFile() as error_log, (directory / "frames.u8").open("wb") as sink:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log)
        try:
            while True:
                raw = read_exact(process.stdout, size[0] * size[1])
                if not raw:
                    break
                sink.write(raw)
                vectors.append(feature(Image.frombytes("L", size, raw), config))
                if len(vectors) % 500 == 0:
                    print(f"extract: {len(vectors)} frames", flush=True)
            status = process.wait()
            if status:
                error_log.seek(0)
                raise RuntimeError(error_log.read().decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    if not vectors:
        raise ValueError("No frames decoded")
    np.save(directory / "features.npy", np.stack(vectors), allow_pickle=False)
    metadata.update(
        identity=identity,
        source=str(source.resolve()),
        fps=str(rate),
        count=len(vectors),
        width=size[0],
        height=size[1],
    )
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"extract: {len(vectors)} frames at {rate} fps", flush=True)
    return Frames(directory, metadata)


def ui_font(size: int, fonts_root: Path | None = None, *, monospace: bool = False) -> ImageFont.ImageFont:
    """Noto Sans (Mono) from ``fonts_root`` if present, else a system font, else Pillow's default."""
    if fonts_root and fonts_root.exists():
        preferred = list(fonts_root.rglob("NotoSansMono-Regular.ttf" if monospace else "NotoSans-Regular.ttf"))
        if preferred:
            return ImageFont.truetype(str(preferred[0]), size)
    if monospace:
        candidates = [Path("C:/Windows/Fonts/consola.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]
    else:
        candidates = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)
